import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# --- Data Structures ---

class SlidePlanStep:
    def __init__(self, data: Dict[str, Any], figures: List[Dict[str, str]]):
        self.section_title: str = data.get("section_title", "")
        self.slide_topics: List[str] = data.get("slide_topics", [])
        self.plan: str = data.get("plan", "")
        self.references: List[str] = data.get("references", [])
        self.figures: List[Dict[str, str]] = figures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_title": self.section_title,
            "slide_topics": self.slide_topics,
            "plan": self.plan,
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
    default_model = os.environ.get("OPENROUTER_PLANNER_MODEL", "deepseek/deepseek-chat")
    ap.add_argument(
        "--model", type=str, default=default_model,
        help="OpenRouter model slug for planning.",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Force re-planning of slides")
    return ap.parse_args()

# --- Prompt Generation ---

def create_planner_prompt(full_chunk_summaries: str, previous_section_titles: List[str], all_figures: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert presentation planner. Your task is to group the key ideas from a paper summary into logical sections for a slide presentation. "
        "For each section, you will provide a title and a list of topics to be covered, each topic corresponding to a single slide. "
        "Your output must be a single JSON object."
    )

    previous_section_titles_str = "- " + "\n- ".join(previous_section_titles) if previous_section_titles else "(No sections planned yet. Start with the introduction.)"
    
    figures_list_str = "\n".join([f"- {fig['id']}: {fig['caption']}" for fig in all_figures])

    user_prompt = f"""**Full Paper Summary:**
{full_chunk_summaries}

**Available Figures:**
{figures_list_str}

**Titles of Sections Already Planned:**
{previous_section_titles_str}

**Instructions:**
Based on the full summary and what has been planned so far, generate the plan for the next logical section of the presentation. The plan should have a clear purpose and cite the relevant source chunks. Avoid planning sections with titles that have already been used. If a figure from the **Available Figures** list is relevant to the plan, include its ID in the `figure_ids` list.

**Output Format:**
Return a single JSON object with the following structure. Do not include any other text or markdown.
```json
{{
  "section_title": "Title for This Section (e.g., Introduction, Methodology)",
  "slide_topics": [
      "Topic for the first slide in this section.",
      "Topic for the second slide in this section.",
      "..."
  ],
  "plan": "A detailed plan for what this entire section will cover, explaining the flow from one slide topic to the next.",
  "references": ["p001_c001", "p001_c002"],
  "figure_ids": ["Fig 1", "Image 2"],
  "finished": false
}}
```
If you believe the presentation is complete and all key topics have been covered (introduction, methods, results, conclusion), set `"finished": true`.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# --- Main Logic ---

async def main():
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = OpenRouterClient(api_key)
    args.outdir.mkdir(exist_ok=True)

    # --- Load Inputs ---
    summaries_path = args.summaries_dir / "chunk_summaries.jsonl"
    output_path = args.outdir / "slide_plan.json"

    if output_path.exists() and not args.force:
        print(f"Slide plan {output_path} already exists. Use --force to overwrite.")
        return

    with open(summaries_path, "r") as f:
        summaries_list = [json.loads(line) for line in f]
    
    all_figures = []
    figure_lookup = {}
    for s in summaries_list:
        for fig in s.get('figs', []):
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
    max_steps = 10 # Safety break

    print(f"Starting sequential slide planning with {args.model}...")
    for i in range(max_steps):
        print(f"\n--- Planning Step {i+1}/{max_steps} ---")
        messages = create_planner_prompt(full_summary_text, planned_section_titles, all_figures)

        try:
            response = await client.create_chat_completion(
                model=args.model, messages=messages
            )
            response_text = response["choices"][0]["message"]["content"]

            if args.verbose:
                print(f"Raw response:\n{response_text}")

            if response_text.strip().startswith("```json"):
                response_text = response_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            response_text = response_text.replace("\\", "\\\\")
            response_data = json.loads(response_text)

            # --- Process Response ---
            figure_ids = response_data.get("figure_ids", [])
            figures = [figure_lookup[fig_id] for fig_id in figure_ids if fig_id in figure_lookup]

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

if __name__ == "__main__":
    asyncio.run(main())
