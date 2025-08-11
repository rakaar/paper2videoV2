import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# --- Data Structures ---

class SlidePlanStep:
    def __init__(self, data: Dict[str, Any], figures: List[Dict[str, str]]):
        self.section_title: str = data.get("section_title", "")
        self.slide_topics: List[str] = data.get("slide_topics", [])
        self.plan: str = data.get("plan", "")
        self.learning_objective: str = data.get("learning_objective", "")
        self.references: List[str] = data.get("references", [])
        self.figures: List[Dict[str, str]] = figures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_title": self.section_title,
            "slide_topics": self.slide_topics,
            "plan": self.plan,
            "learning_objective": self.learning_objective,
            "references": self.references,
            "figures": self.figures,
        }

# --- OpenRouter Client ---

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

# --- Argument Parsing ---

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan slides sequentially from chunk summaries using OpenRouter")
    ap.add_argument(
        "--summaries-dir", type=Path, default=Path("artifacts"),
        help="Directory with chunk_summaries.jsonl file (default: artifacts)",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("artifacts"),
        help="Output directory (default: artifacts)",
    )
    default_model = os.environ.get(
        "OPENROUTER_PLANNER_MODEL",
        "qwen/qwen-2.5-7b-instruct:free,mistralai/mixtral-8x7b-instruct",
    )
    ap.add_argument(
        "--model", type=str, default=default_model,
        help="OpenRouter model slug for planning.",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=120,
        help="Max tokens for planner responses (default: 120)",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Force re-planning of slides")
    return ap.parse_args()

# --- Prompt Generation ---

def create_planner_prompt(
    full_chunk_summaries: str,
    previous_section_titles: List[str],
    all_figures: List[Dict[str, str]],
    paper_card: Dict[str, Any],
    next_section_title: str,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert presentation planner. Use the provided Paper Card as governance to plan exactly ONE section at a time. "
        "Follow the canonical section order strictly. Each section MUST cite evidence via references to source chunk IDs. "
        "Return one JSON object only."
    )

    previous_section_titles_str = "- " + "\n- ".join(previous_section_titles) if previous_section_titles else "(None yet)"
    figures_list_str = "\n".join([f"- {fig['id']}: {fig['caption']}" for fig in all_figures])

    user_prompt = f"""Paper Card (governance):
TL;DR: {paper_card.get('tldr', '')}
Contributions: {paper_card.get('contributions', [])}
Method (one-liner): {paper_card.get('method_oneliner', '')}
Key Results: {paper_card.get('key_results', [])}
Limitations: {paper_card.get('limitations', [])}
Section Order: {paper_card.get('section_order', [])}

You MUST plan the section titled EXACTLY: "{next_section_title}".
If the next section is "Overview" or "Introduction":
- Include 2–3 slide topics covering: problem framing & stakes; contributions; and a roadmap of upcoming sections.

Full Paper Summary with chunk IDs and figure mentions:
---
{full_chunk_summaries}
---

Available Figures:
{figures_list_str}

Titles already planned:
{previous_section_titles_str}

Hard requirements:
- `references`: at least 2 valid chunk IDs related to this section.
- `figure_ids`: OPTIONAL; if present, choose ONLY from figures mentioned in the referenced chunks.
- `learning_objective`: 1–2 sentence objective for this section.
- `section_title` MUST be exactly "{next_section_title}".

Output one JSON object only (no markdown, no commentary):
{{
  "section_title": "{next_section_title}",
  "slide_topics": ["Topic 1", "Topic 2"],
  "plan": "Detailed plan for the section, explaining flow across slide topics.",
  "learning_objective": "What the audience should learn (1–2 sentences).",
  "references": ["p001_c001", "p001_c002"],
  "figure_ids": ["Figure 1"],
  "finished": false
}}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# --- Main Logic ---

async def main_async(args: argparse.Namespace):
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = OpenRouterClient(api_key)
    args.outdir.mkdir(exist_ok=True)

    # --- Load Inputs ---
    summaries_path = args.summaries_dir / "chunk_summaries.jsonl"
    card_path = args.summaries_dir / "paper_card.json"
    output_path = args.outdir / "slide_plan.json"

    if output_path.exists() and not args.force:
        print(f"Slide plan {output_path} already exists. Use --force to overwrite.")
        return

    with open(summaries_path, "r") as f:
        summaries_list = [json.loads(line) for line in f]

    # Load governance card
    if not card_path.exists():
        raise FileNotFoundError(f"Missing paper_card.json at {card_path}. Run make_paper_card.py first.")
    with open(card_path, "r", encoding="utf-8") as f:
        paper_card = json.load(f)
    
    all_figures = []
    figure_lookup = {}
    per_chunk_figs: Dict[str, List[Dict[str, str]]] = {}
    per_chunk_claims: Dict[str, Set[str]] = {}
    for s in summaries_list:
        # Build figure list and per-chunk mappings
        figs = s.get('figs', []) or []
        per_chunk_figs[s.get('id')] = figs
        per_chunk_claims[s.get('id')] = set(s.get('claims', []) or [])
        for fig in figs:
            if fig['id'] not in figure_lookup:
                all_figures.append(fig)
                figure_lookup[fig['id']] = fig

    # Create a richer summary text that includes figure captions
    summary_lines = []
    for s in summaries_list:
        line = f"[{s['id']}]: {s['gist']}"
        if s.get('figs'):
            fig_captions = [f'{f.get("id", "Unnamed Figure")}: {f.get("caption", "No caption.")}' for f in s['figs']]
            if fig_captions:
                line += f"\n  - Figures Mentioned: {'; '.join(fig_captions)}"
        summary_lines.append(line)
    full_summary_text = "\n".join(summary_lines)

    # --- Sequential Planning Loop ---
    all_slide_plans: List[SlidePlanStep] = []
    planned_section_titles: List[str] = []
    finished = False
    # Aim to cover the governance order; allow a few retries
    desired_order: List[str] = paper_card.get("section_order", ["Overview", "Method", "Results", "Discussion", "Limitations", "Conclusion"]) or []
    max_steps = len(desired_order) + 4  # Safety break

    print(f"Starting sequential slide planning with {args.model}...")
    models_to_try = [m.strip() for m in str(args.model).split(",") if m.strip()]
    for i in range(max_steps):
        print(f"\n--- Planning Step {i+1}/{max_steps} ---")
        # Determine next required section title
        next_section_candidates = [sec for sec in desired_order if sec not in planned_section_titles]
        if not next_section_candidates:
            print("All sections from governance order have been planned. Stopping.")
            break
        next_section_title = next_section_candidates[0]

        messages = create_planner_prompt(
            full_summary_text, planned_section_titles, all_figures, paper_card, next_section_title
        )

        try:
            response_text = ""
            last_err: Optional[Exception] = None
            for model_to_try in models_to_try:
                try:
                    resp = await client.create_chat_completion(
                        model=model_to_try,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.2,
                        max_tokens=args.max_tokens,
                    )
                    response_text = resp["choices"][0]["message"]["content"]
                    if args.verbose:
                        print(f"[{model_to_try}] Raw response:\n{response_text}")
                    # Some models still wrap JSON
                    if response_text.strip().startswith("```json"):
                        response_text = response_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                    # Try to parse
                    response_data = json.loads(response_text)
                    break
                except Exception as e:
                    last_err = e
                    if args.verbose:
                        print(f"Model {model_to_try} failed: {e}")
                    continue
            else:
                raise RuntimeError(f"All planner models failed. Last error: {last_err}")

            # --- Process Response ---
            # Enforce exact section title
            returned_title = (response_data.get("section_title", "") or "").strip()
            if returned_title != next_section_title:
                print(f"Planner returned unexpected section title '{returned_title}'. Expected '{next_section_title}'. Re-asking...")
                continue

            # Validate references
            refs = [r for r in response_data.get("references", []) if isinstance(r, str) and r]
            if len(refs) < 2:
                print("Planner returned insufficient references (<2). Re-asking...")
                continue

            # Filter figure_ids to those present in referenced chunks
            allowed_figs: Set[str] = set()
            for ref in refs:
                for f in per_chunk_figs.get(ref, []) or []:
                    fid = f.get("id")
                    if fid:
                        allowed_figs.add(fid)
            filtered_fig_ids = [fid for fid in (response_data.get("figure_ids", []) or []) if fid in allowed_figs]
            response_data["figure_ids"] = filtered_fig_ids

            figures = [figure_lookup[fid] for fid in filtered_fig_ids if fid in figure_lookup]

            step_plan = SlidePlanStep(response_data, figures)
            all_slide_plans.append(step_plan)
            if step_plan.section_title:
                planned_section_titles.append(step_plan.section_title)
            print(f"Planned Section: {step_plan.section_title}")
            if step_plan.slide_topics:
                print("  - Slide Topics:")
                for topic in step_plan.slide_topics:
                    print(f"    - {topic}")

            finished = response_data.get("finished", False)
            if finished:
                print("\nModel has indicated the plan is complete.")
                break

        except Exception as e:
            print(f"An error occurred during planning step {i+1}: {e}")
            break
    else:
        print("\nReached maximum planning steps. Finishing.")

    # --- Save Final Plan ---
    final_plan = {"slides": [plan.to_dict() for plan in all_slide_plans]}
    with open(output_path, "w") as f:
        json.dump(final_plan, f, indent=2)
    
    print(f"\nSuccessfully wrote final slide plan to {output_path}")

def main_cli():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main_cli()

