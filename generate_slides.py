import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

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
            timeout=180,  # Increased timeout for longer generation
        )
        response.raise_for_status()
        return response.json()

# --- Argument Parsing ---

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate slide content from a plan using OpenRouter")
    ap.add_argument(
        "--ocr-dir", type=Path, default=Path("mistral_responses"), help="Root directory of the OCR output"
    )
    ap.add_argument(
        "--pdf-name", type=str, required=True, help="The name of the PDF file (e.g., 'test_paper')"
    )
    ap.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory with input files (slide_plan.json, etc.)",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("artifacts"),
        help="Output directory (default: artifacts)",
    )
    default_model = os.environ.get("OPENROUTER_GENERATOR_MODEL", "openai/gpt-4o")
    ap.add_argument(
        "--model", type=str, default=default_model,
        help="OpenRouter model slug for generation.",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Force re-generation of presentation")
    return ap.parse_args()

# --- Main Logic ---

async def main_async(args: argparse.Namespace):
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = OpenRouterClient(api_key)
    args.outdir.mkdir(exist_ok=True)

    print("Starting slide generation process...")

    # --- Load Input Files ---
    plan_path = args.artifacts_dir / "slide_plan.json"
    summaries_path = args.artifacts_dir / "chunk_summaries.jsonl"
    index_path = args.artifacts_dir / "chunk_index.jsonl"
    output_path = args.outdir / "presentation.json"

    if output_path.exists() and not args.force:
        print(f"Presentation {output_path} already exists. Use --force to overwrite.")
        return

    with open(plan_path, "r") as f:
        slide_plan = json.load(f)

    with open(summaries_path, "r") as f:
        chunk_summaries = {s['id']: s for s in (json.loads(line) for line in f)}

    with open(index_path, "r") as f:
        chunk_index = {i['id']: i for i in (json.loads(line) for line in f)}

    print(f"Loaded plan with {len(slide_plan['slides'])} slide sections.")

    # --- Helper to get full chunk text ---
    def get_chunk_text(chunk_id: str) -> str:
        index_item = chunk_index.get(chunk_id)
        if not index_item:
            return f"[Content for {chunk_id} not found]"
        
        page_num = index_item['page']
        # Path format from mistral_ocr.py: <ocr_dir>/<pdf_name>/markdown/<pdf_name>_page_<page_num>.md
        page_path = args.ocr_dir / args.pdf_name / "markdown" / f"{args.pdf_name}_page_{page_num:02d}.md"
        
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                page_content = f.read()
            return page_content[index_item['char_span'][0]:index_item['char_span'][1]]
        except FileNotFoundError:
            return f"[Content for {chunk_id} not found at {page_path}]"

    # --- Main Generation Loop ---
    final_presentation = []
    used_chunk_ids = set()
    slide_groups = slide_plan["slides"]
    for i, slide_group in enumerate(tqdm(slide_groups, desc="Generating Slides")):
        next_slide_group = slide_groups[i + 1] if i + 1 < len(slide_groups) else None

        # 1. Get full text and figures for current slide
        current_references = slide_group.get("references", [])
        current_figures = slide_group.get("figures", [])
        current_context = "\n\n---\n\n".join(get_chunk_text(ref) for ref in current_references)

        if current_figures:
            figures_text_list = []
            for fig in current_figures:
                fig_id = fig.get("id", "N/A")
                fig_caption = fig.get("caption", "No caption available.")
                figures_text_list.append(f'- Figure ID: {fig_id}\n  Caption: {fig_caption}')
            figures_text = "\n".join(figures_text_list)
        else:
            figures_text = "(No figures for this slide)"

        # 2. Get summaries of previously used chunks
        previous_summaries = [chunk_summaries[ref]['gist'] for ref in used_chunk_ids if ref in chunk_summaries]
        previous_context = "\n- ".join(previous_summaries)
        if not previous_context:
            previous_context = "(This is the first slide)"

        # 3. Create prompt
        system_prompt = (
            "You are an expert at creating slides for a detailed technical presentation based on an academic paper. "
            "Your audience is technically proficient and expects a thorough explanation. "
            "Your task is to generate the content for a single slide, ensuring it flows logically from the previous slide and sets the stage for the next one. "
            "The output must be a single, clean JSON object."
        )
        user_prompt = f'''You are creating a presentation slide. Here is the plan and the relevant information:

**Current Slide Plan:**
```json
{json.dumps(slide_group, indent=2)}
```

**Next Slide\'s Plan (for context):**
```json
{json.dumps(next_slide_group, indent=2) if next_slide_group else '(This is the final slide)'}
```

**Figures for this Slide:**
{figures_text}

**Full Text for this Slide:**
---
{current_context}
---

**Summaries of Previous Slides for Context:**
- {previous_context}

**Your Task:**
Generate the JSON for this slide with great technical detail, suitable for a knowledgeable audience.
- The `Content` should be concise, technically deep, and use bullet points.
- The `Audio` narration should be a clear, technically accurate script that explains the concepts in detail. It should flow logically from the previous slide and smoothly transition to the topic of the next slide.
- **Crucially, you must refer to the figures by their ID (e.g., \'Figure 1\') in both the `Content` and `Audio` where they are relevant to the plan.**
- The `Figures` key in the output must contain the full, unmodified JSON objects for the figures provided above.

**Output Format:**
Return a single JSON object with the following keys. Do not add any other text or markdown.
```json
{{
  "Title": "The title of the slide",
  "Content": "- Bullet point 1\n- As shown in Figure 1, ...",
  "Audio": "The narration that accompanies this slide... It should also mention Figure 1.",
  "Figures": {json.dumps(current_figures, indent=2)}
}}
```'''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if args.verbose:
            print(f"\n--- Prompt for slide {i+1} ---")
            print(user_prompt)

        # 4. Call LLM
        try:
            response = await client.create_chat_completion(
                model=args.model, messages=messages, response_format={"type": "json_object"}
            )
            response_text = response["choices"][0]["message"]["content"]
            slide_content = json.loads(response_text)
            final_presentation.append(slide_content)
            if args.verbose:
                print(f"Successfully generated content for: {slide_content.get('Title')}")
        except Exception as e:
            print(f"Error generating slide {i+1}: {e}")
            final_presentation.append({
                "Title": "Error", 
                "Content": f"Failed to generate content for slide group: {slide_group.get('slide_titles')}",
                "Audio": str(e),
                "Figures": []
            })

        # 5. Update used chunks for the next iteration
        used_chunk_ids.update(current_references)

    # --- Save Final Presentation ---
    with open(output_path, "w") as f:
        json.dump({"presentation": final_presentation}, f, indent=2)

    print(f"\nSlide generation complete. Final presentation saved to {output_path}")

def main_cli():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main_cli()

