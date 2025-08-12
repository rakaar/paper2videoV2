import argparse
import asyncio
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import deque

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# --- OpenRouter Client ---

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http_client = httpx.AsyncClient()

    async def create_chat_completion(self, model: str, messages: list, **kwargs) -> dict:
        payload = {"model": model, "messages": messages}
        # Only include kwargs whose value is not None to avoid sending nulls like response_format: null
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v
        response = await self.http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
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
    default_model = os.environ.get(
        "OPENROUTER_GENERATOR_MODEL",
        "mistralai/mistral-small-24b-instruct-2501,meta-llama/llama-3.2-3b-instruct",
    )
    ap.add_argument(
        "--model", type=str, default=default_model,
        help="OpenRouter model slug for generation.",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens for generator responses. Omit to uncap (default: unlimited)",
    )
    ap.add_argument(
        "--figure-reuse-limit", type=int, default=-1,
        help="Max number of times a single figure can be reused across the deck. -1 means unlimited (default).",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Force re-generation of presentation")
    ap.add_argument(
        "--add-cover",
        action="store_true",
        help="Prepend a cover slide (with Audio narration) using title/authors extracted from OCR via LLM (fallback heuristics).",
    )
    ap.add_argument(
        "--cover-model",
        type=str,
        default=os.environ.get("OPENROUTER_COVER_MODEL", "mistralai/mistral-small-24b-instruct-2501"),
        help="OpenRouter model slug for extracting cover metadata (default: mistral-small-24b-instruct-2501)",
    )
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
    # Allow comma-separated fallback models
    models_to_try = [m.strip() for m in str(args.model).split(",") if m.strip()]

    # Build per-chunk maps for figures and claims
    per_chunk_fig_objs: Dict[str, List[Dict[str, Any]]] = {cid: s.get("figs", []) or [] for cid, s in chunk_summaries.items()}
    per_chunk_claims: Dict[str, Set[str]] = {cid: set(s.get("claims", []) or []) for cid, s in chunk_summaries.items()}

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

    # --- Cover Slide Helpers ---
    async def _llm_extract_cover_title_authors() -> Tuple[Optional[str], Optional[str]]:
        """Use OpenRouter to extract (title, authors) from OCR page 01 markdown via JSON mode.
        Returns (None, None) on failure.
        """
        page_md = args.ocr_dir / args.pdf_name / "markdown" / f"{args.pdf_name}_page_01.md"
        try:
            text = page_md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None, None
        snippet = text[:4000]
        try:
            response = await client.create_chat_completion(
                model=args.cover_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract metadata from the first page of an academic paper. "
                            "Return strictly a JSON object with keys: title (string), authors (string). "
                            "If unsure, leave values empty."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "First page OCR Markdown follows. Extract title and authors.\n\n" + snippet,
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=None,
                temperature=0.0,
            )
            choice = (response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or "{}"
            try:
                obj = json.loads(content)
            except Exception:
                return None, None
            title = (obj.get("title") or "").strip() or None
            authors = (obj.get("authors") or "").strip() or None
            return title, authors
        except Exception:
            return None, None

    def _heuristic_extract_cover_title_authors() -> Tuple[Optional[str], Optional[str]]:
        page_md = args.ocr_dir / args.pdf_name / "markdown" / f"{args.pdf_name}_page_01.md"
        if not page_md.exists():
            return None, None
        try:
            text = page_md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None, None
        lines = text.splitlines()
        title = None
        authors = None
        for idx, line in enumerate(lines[:200]):
            if line.lstrip().startswith("#"):
                title = re.sub(r"^#+\\s*", "", line).strip()
                for j in range(idx + 1, min(idx + 10, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if cand.startswith("#"):
                        break
                    if len(cand) <= 120 and not cand.lower().startswith(("fig.", "figure ", "department")):
                        authors = cand
                        break
                break
        return title, authors

    # Prepare title registry before potential cover insertion
    seen_titles: Set[str] = set()

    # --- Optional: Prepend a Cover Slide ---
    if args.add_cover:
        cov_title, cov_authors = await _llm_extract_cover_title_authors()
        if not cov_title and not cov_authors:
            cov_title, cov_authors = _heuristic_extract_cover_title_authors()
        # Fallbacks
        if not cov_title:
            cov_title = re.sub(r"[_-]+", " ", args.pdf_name).strip().title()
        authors_text = cov_authors or "the authors"
        # Try to infer first topic for bridge
        try:
            first_topic = (
                slide_plan.get("slides")[0].get("section")
                or (slide_plan.get("slides")[0].get("slide_titles") or [None])[0]
                or "background"
            )
        except Exception:
            first_topic = "background"
        cover_slide = {
            "Title": cov_title,
            "WhyThisSlide": "Introduce the paper and agenda.",
            "BridgeFromPrevious": "",
            "Content": f"- Paper: {cov_title}\n- Authors: {authors_text}\n- Overview: motivation, method, results, and key findings",
            "Audio": (
                f"Welcome! In this talk, we will discuss '{cov_title}' by {authors_text}. "
                f"We'll briefly outline the motivation, the core method, and the main results, and then dive into details. "
                f"Let's begin with the {first_topic}."
            ),
            "BridgeToNext": f"Let's start with the {first_topic}.",
            "NewInsightAboutFigures": False,
            "Figures": [],
        }
        # Enforce unique title
        title = cover_slide.get("Title") or "Cover"
        if title in seen_titles:
            candidate = f"{title} (Cover)"
            suffix = 2
            while candidate in seen_titles:
                candidate = f"{title} (Cover-{suffix})"
                suffix += 1
            title = candidate
            cover_slide["Title"] = title
        seen_titles.add(title)
        final_presentation: List[Dict[str, Any]] = [cover_slide]
    else:
        final_presentation: List[Dict[str, Any]] = []

    # --- Main Generation Loop ---
    used_chunk_ids: Set[str] = set()
    used_claims: Set[str] = set()
    figure_use_count: Dict[str, int] = {}
    previous_slides = deque(maxlen=2)
    checkpoint_notes: List[str] = []
    # If we added a cover, seed context so first generated slide bridges naturally
    if args.add_cover and final_presentation:
        last = final_presentation[-1]
        previous_slides.append({"Title": last.get("Title", ""), "Content": last.get("Content", "")})
        checkpoint_notes.append("Cover introduced")
    slide_groups = slide_plan["slides"]
    for i, slide_group in enumerate(tqdm(slide_groups, desc="Generating Slides")):
        next_slide_group = slide_groups[i + 1] if i + 1 < len(slide_groups) else None

        # 1. Get references and full text for current slide
        current_references = slide_group.get("references", [])
        if not current_references:
            print(f"Warning: slide group {i+1} has no references; skipping generation for this group.")
            continue

        # Compose candidate figures from references (source of truth)
        figs_from_refs: List[Dict[str, Any]] = []
        for ref in current_references:
            figs_from_refs.extend(per_chunk_fig_objs.get(ref, []))

        # Respect planner intent for figures: if planner specified an empty list, attach none.
        planner_figs = slide_group.get("figures")
        suggested_ids = {f.get("id") for f in (planner_figs or []) if isinstance(f, dict)}
        current_figures: List[Dict[str, Any]] = []
        seen_fids: Set[str] = set()
        if planner_figs is not None and len(planner_figs) == 0:
            # Planner explicitly chose no figures for this slide
            current_figures = []
        else:
            for f in figs_from_refs:
                fid = f.get("id")
                if not fid or fid in seen_fids:
                    continue
                # If planner selected specific figures, restrict to that set
                if planner_figs is not None and suggested_ids and fid not in suggested_ids:
                    continue
                # Figure reuse policy: configurable; -1 means unlimited
                if args.figure_reuse_limit >= 0 and figure_use_count.get(fid, 0) >= args.figure_reuse_limit:
                    continue
                current_figures.append(f)
                seen_fids.add(fid)
                figure_use_count[fid] = figure_use_count.get(fid, 0) + 1

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

        # 2. Build de-dup signals and previous context discipline
        claims_now: Set[str] = set()
        for ref in current_references:
            claims_now |= per_chunk_claims.get(ref, set())
        novel_claims = [c for c in claims_now if c not in used_claims]
        already_covered = [c for c in used_claims if c not in set(novel_claims)]
        used_claims.update(claims_now)

        if previous_slides:
            prev_lines = []
            for ps in previous_slides:
                first_line = (ps.get("Content", "").splitlines() or [""])[0][:150]
                prev_lines.append(f"- {ps.get('Title','')}: {first_line}")
            previous_context = "\n".join(prev_lines)
        else:
            previous_context = "(This is the first slide)"
        checkpoint_context = "; ".join(checkpoint_notes[-5:]) if checkpoint_notes else ""

        # 3. Create prompt
        system_prompt = (
            "You are an expert at creating slides for a detailed technical presentation based on an academic paper. "
            "Your audience is technically proficient and expects a thorough explanation. "
            "Your task is to generate the content for a single slide, ensuring it flows logically from the previous slide and sets the stage for the next one. "
            "Return only a single, valid JSON object with the exact keys specified. Do not include code fences or any extra commentary."
        )
        
        # Add special instruction when figures are present
        figure_instruction = ""
        if current_figures:
            figure_instruction = (
                "**Special Instruction for Slides with Figures:**\n"
                "This slide includes figure candidates. Use them selectively:\n"
                "- Keep the `Content` concise (2–4 bullets) and only mention a figure if it materially supports the slide objective.\n"
                "- If a figure was already shown, mention it only when adding a genuinely new angle; otherwise omit it.\n"
                "- Put detailed technical explanations in the `Audio` narration.\n"
                "- When you do mention a figure, refer to it by its ID (e.g., 'Figure 1').\n"
                "- Set `NewInsightAboutFigures` to true only if you add genuinely new insight.\n\n"
            )
        else:
            figure_instruction = (
                "**Special Instruction (No Figures Attached):**\n"
                "Do not mention or reference any figures in `Content` or `Audio` for this slide.\n\n"
            )
        
        user_prompt = f'''You are creating a presentation slide. Here is the plan and the relevant information:

**Current Slide Plan:**
```json
{json.dumps(slide_group, indent=2)}
```

**Next Slide's Plan (for context):**
```json
{json.dumps(next_slide_group, indent=2) if next_slide_group else '(This is the final slide)'}
```

**Figures for this Slide:**
{figures_text}

{figure_instruction}**Full Text for this Slide:**
---
{current_context}
---

**Summaries of Previous Slides for Context (last 2):**
{previous_context}

**Recent Checkpoints (≤5, compact):**
{checkpoint_context}

**Claims available for this slide (prefer NOVEL):**
- Novel (aim to use these): {json.dumps(novel_claims[:6])}
- Already covered (avoid unless 1-line recap): {json.dumps(already_covered[:6])}

**Your Task:**
Generate the JSON for this slide with great technical detail, suitable for a knowledgeable audience.
- The `Content` should be concise, technically deep, and use bullet points.
- The `Audio` narration should be a clear, technically accurate script that explains the concepts in detail. It should flow logically from the previous slide and smoothly transition to the topic of the next slide.
- If and only if a figure materially supports the plan, mention it and refer to it by its ID (e.g., 'Figure 1'); otherwise, do not mention figures. 
- Do NOT include the figure JSON in your output. Set `Figures` to an empty array []; the system will attach the full, unmodified figure objects automatically.
- Avoid repeating content already covered unless a very brief recap is required to introduce a new result. Prefer NOVEL claims.

**Output Format:**
Return a single JSON object with the following keys. Do not add any other text or markdown.
```json
{{
  "Title": "The title of the slide",
  "WhyThisSlide": "1 sentence",
  "BridgeFromPrevious": "1 sentence",
  "Content": "- Bullet point 1\n- As shown in Figure 1, ...",
  "Audio": "The narration that accompanies this slide... It should also mention Figure 1.",
  "BridgeToNext": "1 sentence",
  "NewInsightAboutFigures": false,
  "Figures": []
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
        def _validate_slide_payload(payload: Dict[str, Any]) -> bool:
            required_keys = {"Title", "Content", "Audio", "Figures", "WhyThisSlide", "BridgeFromPrevious", "BridgeToNext"}
            return required_keys.issubset(payload.keys())

        async def _generate_slide(regen_note: str = "") -> Dict[str, Any]:
            extra_msg = []
            if regen_note:
                extra_msg = [{"role": "user", "content": f"IMPORTANT: {regen_note}"}]
            last_err: Optional[Exception] = None
            for model_to_try in models_to_try:
                # Try with JSON mode first, then without (some models return empty content in JSON mode)
                for use_json_mode in (True, False):
                    try:
                        response = await client.create_chat_completion(
                            model=model_to_try,
                            messages=messages + extra_msg,
                            response_format={"type": "json_object"} if use_json_mode else None,
                            temperature=0.3,
                            max_tokens=(args.max_tokens + (60 if regen_note else 0)) if args.max_tokens is not None else None,
                        )
                        # Defensive extraction of content
                        choice = (response.get("choices") or [{}])[0]
                        msg = choice.get("message") or {}
                        response_text = msg.get("content") or ""
                        if args.verbose:
                            print(f"[{model_to_try} | json_mode={use_json_mode}] Raw response len: {len(response_text)}")
                            if not response_text:
                                print(f"[{model_to_try}] Empty content; full response keys: {list(response.keys())}")
                        # Coerce to JSON dict (with minimal repairs if needed)
                        return _coerce_json_dict(response_text)
                    except Exception as e:
                        last_err = e
                        if args.verbose:
                            print(f"Model {model_to_try} failed for this slide (json_mode={use_json_mode}): {e}")
                        continue
            raise RuntimeError(f"All generator models failed. Last error: {last_err}")

        def _coerce_json_dict(text: str) -> Dict[str, Any]:
            """Attempt to parse an LLM response into a JSON dict with light repairs.
            Raises on failure so caller can retry or fallback models."""
            if not text or not isinstance(text, str):
                raise ValueError("Empty LLM content")
            s = text.strip()
            # Strip code fences if present
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*", "", s)
                s = re.sub(r"\s*```$", "", s)
            start = s.find("{")
            end = s.rfind("}")
            candidate = s[start:end + 1] if (start != -1 and end != -1 and end > start) else s
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Basic trailing comma fix and Python-literal normalization
                repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
                repaired = repaired.replace("\r", "").replace("\t", " ")
                repaired = re.sub(r'(?<![\"\w])True(?![\"\w])', 'true', repaired)
                repaired = re.sub(r'(?<![\"\w])False(?![\"\w])', 'false', repaired)
                repaired = re.sub(r'(?<![\"\w])None(?![\"\w])', 'null', repaired)
                return json.loads(repaired)

        def _as_text(value: Any) -> str:
            """Coerce various LLM-returned types to a presentable markdown string."""
            if isinstance(value, str):
                return value
            if value is None:
                return ""
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        try:
                            parts.append("- " + "; ".join(f"{k}: {v}" for k, v in item.items()))
                        except Exception:
                            parts.append("- " + str(item))
                    else:
                        parts.append("- " + str(item))
                return "\n".join(parts)
            if isinstance(value, (dict, set, tuple)):
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)
            return str(value)

        try:
            try:
                slide_content = await _generate_slide()
            except Exception as e_first:
                # One more attempt with explicit instruction to emit strict JSON
                if args.verbose:
                    print(f"Retrying slide {i+1} after parse/generation failure: {e_first}")
                slide_content = await _generate_slide(
                    "Your previous output was invalid or not strictly JSON. Return ONLY a valid JSON object with the required keys (Title, WhyThisSlide, BridgeFromPrevious, Content, Audio, BridgeToNext, NewInsightAboutFigures, Figures). Do not include markdown fences."
                )

            # Validate required fields
            if not _validate_slide_payload(slide_content):
                slide_content = await _generate_slide("The previous output missed required keys (WhyThisSlide/Bridges). Include all required keys exactly as specified and return JSON only.")

            # Normalize types: ensure core fields are strings for downstream processing
            for k in ("Title", "Content", "Audio", "WhyThisSlide", "BridgeFromPrevious", "BridgeToNext"):
                if k in slide_content:
                    slide_content[k] = _as_text(slide_content[k])

            # Ensure Figures and defaults are present and bounded
            slide_content["Figures"] = current_figures
            if "NewInsightAboutFigures" not in slide_content:
                slide_content["NewInsightAboutFigures"] = False

            # Enforce unique titles by appending a suffix if necessary
            title = slide_content.get("Title") or f"Slide {i+1}"
            if title in seen_titles:
                candidate = f"{title} (Slide {i+1})"
                suffix = 2
                while candidate in seen_titles:
                    candidate = f"{title} (Slide {i+1}-{suffix})"
                    suffix += 1
                title = candidate
                slide_content["Title"] = title
            seen_titles.add(title)

            # Simple content-similarity linter vs previous slide
            if final_presentation:
                def _tokens(s: Any) -> Set[str]:
                    s_text = s if isinstance(s, str) else _as_text(s)
                    return set(t.lower().strip(".,:;!?") for t in s_text.split())
                prev_text = (final_presentation[-1].get("Content", "") or "")
                curr_text = (slide_content.get("Content", "") or "")
                a, b = _tokens(prev_text), _tokens(curr_text)
                inter = len(a & b)
                union = max(1, len(a | b))
                jacc = inter / union
                if jacc >= 0.7:
                    slide_content = await _generate_slide("You repeated content from the previous slide; focus on NOVEL claims only and avoid overlap.")

            final_presentation.append(slide_content)
            if args.verbose:
                print(f"Successfully generated content for: {slide_content.get('Title')}")
        except Exception as e:
            print(f"Error generating slide {i+1}: {e}")
            final_presentation.append({
                "Title": "Error",
                "WhyThisSlide": "",
                "BridgeFromPrevious": "",
                "Content": f"Failed to generate content for slide group: {slide_group.get('slide_titles')}",
                "Audio": str(e),
                "BridgeToNext": "",
                "NewInsightAboutFigures": False,
                "Figures": []
            })

        # 5. Update used chunks for the next iteration
        used_chunk_ids.update(current_references)
        # Update previous slides and checkpoint notes
        last_slide = final_presentation[-1]
        previous_slides.append({"Title": last_slide.get("Title", ""), "Content": last_slide.get("Content", "")})
        # Create a compact checkpoint note (≤25 tokens)
        note_src = last_slide.get("WhyThisSlide") or last_slide.get("Title", "")
        words = (note_src or "").split()
        checkpoint_notes.append(" ".join(words[:25]))

    # --- Save Final Presentation ---
    with open(output_path, "w") as f:
        json.dump({"presentation": final_presentation}, f, indent=2)

    print(f"\nSlide generation complete. Final presentation saved to {output_path}")

def main_cli():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main_cli()

