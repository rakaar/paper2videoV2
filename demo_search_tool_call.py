import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from index.search_index import handle_search


BASE_URL = "https://openrouter.ai/api/v1"
PREFERRED_MODEL = "z-ai/glm-4.5"
FALLBACK_MODEL = "openrouter/auto"


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Return top-k chunk IDs for a query string",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    },
}


SYSTEM_PROMPT = "You are a retrieval agent. Call `search` exactly once."


def preflight_pick_model(api_key: str, candidates: list[str]) -> str | None:
    import httpx

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
    parser = argparse.ArgumentParser(description="Demo single search tool call via OpenRouter")
    parser.add_argument("query", nargs="?", default="auditory cortex", help="Query string")
    parser.add_argument("--k", type=int, default=3, help="Top-k to return")
    parser.add_argument("--paper-id", default="test_paper")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--autoselect-model", action="store_true")
    parser.add_argument(
        "--candidates",
        default=(
            "z-ai/glm-4.5,deepseek/deepseek-chat,openai/gpt-4o-mini,openai/gpt-4.1-mini,"
            "mistralai/mistral-large-latest,qwen/qwen3-coder,qwen/qwen3-30b-a3b-instruct-2507"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or export it.")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    user_prompt = f"Find chunks about: {args.query} (k={args.k})."

    def call_model(model):
        extra_body = None
        if model.strip().lower() == "z-ai/glm-4.5":
            extra_body = {"reasoning": {"enabled": False}, "parallel_tool_calls": False}
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0.0,
            max_tokens=64,
            seed=args.seed,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "paper2video-dev"},
            extra_body=extra_body,
        )

    chosen_model = args.model or PREFERRED_MODEL
    if args.autoselect_model and not args.model:
        picked = preflight_pick_model(api_key, [s.strip() for s in args.candidates.split(",") if s.strip()])
        if picked:
            print(f"Autoselect picked tool-capable model: {picked}")
            chosen_model = picked

    completion = call_model(chosen_model)
    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        raise RuntimeError("Model did not call the search tool. Check prompts and model.")
    call = tool_calls[0]
    if call.function.name != "search":
        raise RuntimeError(f"Unexpected tool: {call.function.name}")

    try:
        tool_args = json.loads(call.function.arguments)
    except Exception as pe:
        raise RuntimeError(f"Tool arguments were not valid JSON: {call.function.arguments}") from pe

    # Enforce our k override directly in handler to keep demo stable
    tool_args["k"] = int(args.k)
    result = handle_search(tool_args, paper_id=args.paper_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
