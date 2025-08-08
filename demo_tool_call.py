import os
import json
import pathlib
import argparse
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- config
BASE_URL = "https://openrouter.ai/api/v1"
# Default to GLM 4.5 (tool-capable with reasoning disabled)
PREFERRED_MODEL = "z-ai/glm-4.5"
FALLBACK_MODEL = "openrouter/auto"

SYSTEM_PROMPT = (
    "You are a slide-writer agent. You MUST call the tool `write_slide` exactly once. "
    "Keep bullets short (<= 140 chars). Use at most 2 figures and only paths that start with 'figures/'."
)

USER_PROMPT = (
    "Create slide 1 titled 'European Trivia'.\n"
    "Write exactly 5 random trivia facts about Europe as concise bullets (<= 140 chars each).\n"
    "Facts should be non-obvious, diverse (geography, culture, history, science), and verifiable.\n"
    "No prose—just the slide. Figures optional and must be 'figures/...'."
)

# --- tool schema
WRITE_SLIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_slide",
        "description": "Append or update a slide in slides.json. Use concise bullets.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "slide_no": {"type": "integer", "minimum": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 120},
                "bullets": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 140},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "figures": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^figures/"},
                    "minItems": 0,
                    "maxItems": 2,
                },
            },
            "required": ["slide_no", "title", "bullets"],
        },
    },
}


def handle_write_slide(args, slides_path="slides.json"):
    # tiny validator (tool-side guardrail)
    assert isinstance(args["slide_no"], int) and args["slide_no"] >= 1
    assert isinstance(args["title"], str) and 1 <= len(args["title"]) <= 120
    assert 1 <= len(args["bullets"]) <= 5 and all(
        isinstance(b, str) and 1 <= len(b) <= 140 for b in args["bullets"]
    )
    for f in args.get("figures", []):
        if not f.startswith("figures/"):
            raise ValueError(f"Invalid figure path: {f}")

    slides = []
    p = pathlib.Path(slides_path)
    if p.exists():
        slides = json.loads(p.read_text())

    # upsert by slide_no
    found = False
    for s in slides:
        if s.get("slide_no") == args["slide_no"]:
            s.update(args)
            found = True
            break
    if not found:
        slides.append(args)

    p.write_text(json.dumps(slides, indent=2))
    return {"status": "ok", "slides_count": len(slides)}


def preflight_pick_model(api_key: str, candidates: list[str]) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}"}
    for slug in candidates:
        try:
            url = f"{BASE_URL}/models/{slug}/endpoints"
            r = httpx.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            endpoints = data.get("data", {}).get("endpoints", [])
            for ep in endpoints:
                params = ep.get("supported_parameters") or []
                if "tools" in params:
                    return slug
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Demo tool call via OpenRouter")
    parser.add_argument("--model", default=None, help="Model ID to use (overrides default)")
    parser.add_argument("--strict-tools", action="store_true", help="Fail instead of falling back if tools unsupported")
    parser.add_argument("--seed", type=int, default=7, help="Seed for determinism-ish")
    parser.add_argument("--allow-json-fallback", action="store_true", help="Allow JSON-mode fallback when tools unsupported")
    parser.add_argument(
        "--autoselect-model",
        action="store_true",
        help="Auto-pick the first candidate model that supports tools",
    )
    parser.add_argument(
        "--candidates",
        default=(
            "z-ai/glm-4.5,deepseek/deepseek-chat,openai/gpt-4o-mini,openai/gpt-4.1-mini,"
            "mistralai/mistral-large-latest,qwen/qwen3-coder,qwen/qwen3-30b-a3b-instruct-2507"
        ),
        help="Comma-separated candidate model slugs to try with --autoselect-model",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or export it.")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    # Force the tool call (so it won't send plain text)
    def call_model(model):
        extra_body = None
        # Disable reasoning for GLM 4.5 to keep schema-clean tool args
        if model.strip().lower() == "z-ai/glm-4.5":
            extra_body = {"reasoning": {"enabled": False}, "parallel_tool_calls": False}
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            tools=[WRITE_SLIDE_TOOL],
            tool_choice={"type": "function", "function": {"name": "write_slide"}},
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            seed=args.seed,  # determinism-ish
            extra_headers={  # optional, helps rankings
                "HTTP-Referer": "http://localhost",
                "X-Title": "paper2video-dev",
            },
            extra_body=extra_body,
        )

    try:
        chosen_model = args.model or PREFERRED_MODEL
        if args.autoselect_model and not args.model:
            picked = preflight_pick_model(api_key, [s.strip() for s in args.candidates.split(",") if s.strip()])
            if picked:
                print(f"Autoselect picked tool-capable model: {picked}")
                chosen_model = picked
            else:
                print("Autoselect did not find a tool-capable candidate; proceeding with default.")

        # Preflight selected model if not autoselected
        if not args.autoselect_model:
            preflight_ok = preflight_pick_model(api_key, [chosen_model]) is not None
            if not preflight_ok:
                msg = f"Selected model '{chosen_model}' does not advertise tool support on your routed provider."
                if args.strict_tools or not args.allow_json_fallback:
                    raise RuntimeError(msg)
                else:
                    print(msg)
        completion = call_model(chosen_model)
    except Exception as e:
        if args.strict_tools or not args.allow_json_fallback:
            raise
        print(f"Primary model failed ({e}). Falling back to {FALLBACK_MODEL}...")
        try:
            completion = call_model(FALLBACK_MODEL)
        except Exception as e2:
            # Final fallback: JSON mode without tools to keep demo runnable
            print(f"Fallback model also failed ({e2}). Using JSON-mode fallback.")

            json_completion = client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": (
                        "You are a slide-writer agent. Return ONLY a strict JSON object "
                        "with keys: slide_no (int), title (string), bullets (array of strings, 1-5), "
                        "and optional figures (array of 'figures/...'). No extra fields."
                    )},
                    {"role": "user", "content": USER_PROMPT},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "write_slide_args",
                        "schema": WRITE_SLIDE_TOOL["function"]["parameters"],
                        "strict": True,
                    },
                },
                temperature=0.2,
                max_tokens=256,
                seed=args.seed,
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "paper2video-dev",
                },
            )

            content = json_completion.choices[0].message.content
            try:
                json_args = json.loads(content)
            except Exception as pe:
                raise RuntimeError(f"JSON-mode produced invalid JSON: {content}") from pe
            result = handle_write_slide(json_args)
            print("JSON-mode handled:", result)
            print("Wrote/updated slides.json. Preview:\n", json.dumps(json_args, indent=2))
            return

    # Log actual routed model
    try:
        print(f"Routed model: {completion.model}")
    except Exception:
        pass

    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        raise RuntimeError("Model did not call any tool. Check tool_choice or prompts.")

    # Handle only the first call for this demo
    call = tool_calls[0]
    if call.function.name != "write_slide":
        raise RuntimeError(f"Unexpected tool: {call.function.name}")

    try:
        tool_args = json.loads(call.function.arguments)
    except Exception as pe:
        raise RuntimeError(f"Tool arguments were not valid JSON: {call.function.arguments}") from pe
    result = handle_write_slide(tool_args)
    print("Tool handled:", result)
    print("Wrote/updated slides.json. Preview:\n", json.dumps(tool_args, indent=2))


if __name__ == "__main__":
    main()
