#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import httpx
from dotenv import load_dotenv


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http_client = httpx.AsyncClient()

    async def create_chat_completion(self, model: str, messages: list, **kwargs) -> dict:
        response = await self.http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages, **kwargs},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a compact Paper Card from chunk summaries for planning governance")
    ap.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory containing chunk_summaries.jsonl (default: artifacts)",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("artifacts"),
        help="Output directory for paper_card.json (default: artifacts)",
    )
    default_model = os.environ.get(
        "OPENROUTER_CARD_MODEL",
        os.environ.get(
            "OPENROUTER_GENERATOR_MODEL",
            "mistralai/mistral-small-24b-instruct-2501:free,meta-llama/llama-3.2-3b-instruct",
        ),
    )
    ap.add_argument(
        "--model", type=str, default=default_model,
        help="OpenRouter model slug(s) for card generation (comma-separated for fallback)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=200,
        help="Max tokens for card JSON response (default: 200)",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Overwrite existing paper_card.json if present")
    return ap.parse_args()


def _collect_inputs(summaries_path: Path) -> Tuple[List[dict], str, str, List[Dict[str, str]]]:
    """Load chunk summaries and prepare condensed inputs.

    Returns: (summaries_list, abstract_like_text, conclusion_like_text, all_figures)
    """
    with open(summaries_path, "r", encoding="utf-8") as f:
        summaries_list = [json.loads(line) for line in f if line.strip()]

    # Sort by (page, id) to get early/late chunks for abstract/conclusion proxies
    def _sort_key(s: dict) -> Tuple[int, str]:
        return int(s.get("page", 10**9)), str(s.get("id", ""))

    summaries_sorted = sorted(summaries_list, key=_sort_key)

    # Heuristic: first 2–3 chunks ~ abstract/intro; last 2–3 chunks ~ conclusion
    head = summaries_sorted[:3]
    tail = summaries_sorted[-3:]

    abstract_like_text = "\n\n".join(f"[{s['id']}] {s.get('gist','')}" for s in head)
    conclusion_like_text = "\n\n".join(f"[{s['id']}] {s.get('gist','')}" for s in tail)

    # Gather all unique figures (by id)
    figure_lookup: Dict[str, Dict[str, str]] = {}
    for s in summaries_list:
        for fobj in s.get("figs", []) or []:
            fid = fobj.get("id") or fobj.get("path") or f"Image-{len(figure_lookup)+1}"
            if fid not in figure_lookup:
                figure_lookup[fid] = {"id": fobj.get("id", fid), "path": fobj.get("path", ""), "caption": fobj.get("caption", "")}
    all_figures = list(figure_lookup.values())

    return summaries_list, abstract_like_text, conclusion_like_text, all_figures


def _build_prompt(abstract_like: str, conclusion_like: str, figures: List[Dict[str, str]]) -> List[Dict[str, str]]:
    figures_text = "\n".join([f"- {f.get('id','Unknown')}: {f.get('caption','')}" for f in figures]) or "(No figures detected)"

    system_prompt = (
        "You are an expert technical summarizer. Create a compact 'Paper Card' for governance of slide planning. "
        "Focus on the paper's problem, contributions, method, key results, and limitations. Return one JSON object only."
    )

    user_prompt = f"""
Use ONLY the distilled signals below to produce a Paper Card. Be concise and specific.

Abstract/Introduction-like text:
---
{abstract_like}
---

Conclusion-like text:
---
{conclusion_like}
---

Available figure captions (for grounding):
{figures_text}

Output a single JSON with exactly these keys:
{{
  "tldr": "1-2 sentence high-level summary of the paper's goal and findings",
  "contributions": ["Bullet of concrete contribution", "..."],
  "method_oneliner": "One sentence describing the core method/approach",
  "key_results": ["Key result with metric or qualitative outcome", "..."],
  "limitations": ["Notable limitation or assumption", "..."],
  "section_order": ["Overview", "Method", "Results", "Discussion", "Limitations", "Conclusion"]
}}

Rules:
- JSON only. No markdown, prose, or comments.
- Keep bullets short, precise, and evidence-backed.
- If uncertain, be conservative and omit rather than hallucinate.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def main_async(args: argparse.Namespace):
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = OpenRouterClient(api_key)
    args.outdir.mkdir(exist_ok=True)

    summaries_path = args.artifacts_dir / "chunk_summaries.jsonl"
    output_path = args.outdir / "paper_card.json"

    if output_path.exists() and not args.force:
        print(f"Paper card {output_path} already exists. Use --force to overwrite.")
        return

    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing summaries at {summaries_path}")

    summaries_list, abstract_like, conclusion_like, figures = _collect_inputs(summaries_path)

    messages = _build_prompt(abstract_like, conclusion_like, figures)

    if args.verbose:
        print("--- Paper Card Prompt (truncated) ---")
        print(messages[-1]["content"][:1000] + ("..." if len(messages[-1]["content"]) > 1000 else ""))

    models_to_try = [m.strip() for m in str(args.model).split(",") if m.strip()]
    try:
        last_err: Optional[Exception] = None
        response_text = ""
        for model_to_try in models_to_try:
            try:
                response = await client.create_chat_completion(
                    model=model_to_try,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=args.max_tokens,
                )
                response_text = response["choices"][0]["message"]["content"]
                if response_text.strip().startswith("```json"):
                    response_text = response_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                card = json.loads(response_text)
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise RuntimeError(f"All card models failed. Last error: {last_err}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate paper card: {e}")

    # Minimal validation
    required_keys = {"tldr", "contributions", "method_oneliner", "key_results", "limitations", "section_order"}
    missing = required_keys - set(card.keys())
    if missing:
        raise ValueError(f"Paper card missing keys: {missing}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"Wrote paper card to {output_path}")


def main_cli():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main_cli()
