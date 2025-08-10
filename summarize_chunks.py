#!/usr/bin/env python3
import argparse
import asyncio
import json
import ast
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChunkMeta:
    id: str
    page: int
    char_span: Tuple[int, int]
    token_span: Tuple[int, int]
    n_tokens: int
    text: str

@dataclass
class FigureInfo:
    id: str
    path: str
    caption: str

@dataclass
class SummaryResult:
    id: str
    page: int
    gist: str
    claims: List[str]
    figs: List[Dict[str, str]]
    eqs: List[str]
    key_terms: List[str]
    anchors: List[str]

# -----------------------------
# Utilities
# -----------------------------

PAGE_NUM_RE = re.compile(r"(\d+)")
CAPTION_RE = re.compile(r"^(FIG\\.?|Figure|Table)\\s+([A-Z0-9]+[:.]?)\\s*(.*)", re.IGNORECASE)
IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")

def extract_page_no(path: Path) -> Optional[int]:
    nums = PAGE_NUM_RE.findall(path.stem)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None

def extract_figures_from_pages(pages: List[Tuple[int, Path, str]], pages_dir: Path) -> List[FigureInfo]:
    figures: List[FigureInfo] = []
    fig_id_counter = 1
    for _page_no, page_path, text in pages:
        for match in IMAGE_RE.finditer(text):
            alt_text = match.group(1)
            relative_path = match.group(2)
            try:
                absolute_path = (page_path.parent / relative_path).resolve()
                final_path = absolute_path.relative_to(pages_dir.parent)
            except (ValueError, FileNotFoundError):
                final_path = Path("images") / Path(relative_path).name
            caption_search_area = text[match.end() : match.end() + 300]
            caption_lines = [line.strip() for line in caption_search_area.split("\n") if line.strip()]
            caption = alt_text
            fig_id = ""
            if caption_lines:
                cap_match = CAPTION_RE.search(caption_lines[0])
                if cap_match:
                    caption = cap_match.group(0).strip()
                    fig_id = f"{cap_match.group(1).replace('.', '').capitalize()} {cap_match.group(2).replace(':', '').replace('.', '')}"
            if not fig_id:
                fig_id = f"Image {fig_id_counter}"
                fig_id_counter += 1
            figures.append(FigureInfo(id=fig_id, path=str(final_path), caption=caption))
    seen_paths: Set[str] = set()
    unique_figures: List[FigureInfo] = []
    for fig in figures:
        if fig.path not in seen_paths:
            unique_figures.append(fig)
            seen_paths.add(fig.path)
    return unique_figures

def load_pages(pages_dir: Path) -> List[Tuple[int, Path, str]]:
    files = sorted([p for p in pages_dir.glob("*.md") if p.is_file()], key=lambda p: (extract_page_no(p) or 0, p.name))
    items: List[Tuple[int, Path, str]] = []
    for p in files:
        page_no = extract_page_no(p)
        if page_no is None:
            logging.warning(f"Skipping file without page number: {p}")
            continue
        text = p.read_text(encoding="utf-8")
        items.append((page_no, p, text))
    return items

def build_prefix_char_lengths(enc: tiktoken.Encoding, text: str) -> Tuple[List[int], List[str]]:
    tokens = enc.encode(text)
    pieces = [enc.decode([t]) for t in tokens]
    prefix_chars = [0]
    total = 0
    for piece in pieces:
        total += len(piece)
        prefix_chars.append(total)
    return prefix_chars, pieces

def chunk_page(enc: tiktoken.Encoding, page_no: int, text: str, chunk_size: int, overlap: int) -> List[ChunkMeta]:
    tokens = enc.encode(text)
    n = len(tokens)
    stride = max(1, chunk_size - overlap)
    prefix_chars, _pieces = build_prefix_char_lengths(enc, text)
    chunks: List[ChunkMeta] = []
    start_tok = 0
    chunk_idx = 1
    while start_tok < n:
        end_tok = min(n, start_tok + chunk_size)
        char_start = prefix_chars[start_tok]
        char_end = prefix_chars[end_tok]
        chunk_text = text[char_start:char_end]
        cid = f"p{page_no:03d}_c{chunk_idx:03d}"
        meta = ChunkMeta(id=cid, page=page_no, char_span=(char_start, char_end), token_span=(start_tok, end_tok), n_tokens=end_tok - start_tok, text=chunk_text)
        chunks.append(meta)
        if end_tok >= n:
            break
        start_tok += stride
        chunk_idx += 1
    return chunks

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def read_existing_summaries(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    out[obj["id"]] = obj
            except Exception:
                logging.warning("Skipping invalid JSON line in existing summaries.jsonl")
                continue
    return out

def extract_json_object(text: str) -> Optional[dict]:
    def find_first_balanced_json_object(s: str) -> Optional[str]:
        in_string, escape, depth, start = False, False, 0, -1
        for i, ch in enumerate(s):
            if in_string:
                if escape: escape = False
                elif ch == '\\': escape = True
                elif ch == '"': in_string = False
            else:
                if ch == '"': in_string = True
                elif ch == '{':
                    if depth == 0: start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1: return s[start:i + 1]
        return None
    def strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
            s = "\n".join(lines)
        return s.strip()
    def remove_trailing_commas(s: str) -> str:
        return re.sub(r",\\s*([}\]])", r"\1", s)
    text = text.strip()
    try: return json.loads(text)
    except Exception: pass
    candidate = find_first_balanced_json_object(text)
    if candidate:
        for attempt in range(3):
            try: return json.loads(candidate)
            except Exception:
                if attempt == 0: candidate = strip_code_fences(candidate)
                elif attempt == 1: candidate = remove_trailing_commas(candidate)
                else: break
        repaired = remove_trailing_commas(strip_code_fences(candidate))
        repaired = re.sub(r"\btrue\b", "True", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bfalse\b", "False", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bnull\b", "None", repaired)
        try:
            obj = ast.literal_eval(repaired)
            if isinstance(obj, dict): return obj
        except Exception: return None
    return None

# -----------------------------
# OpenRouter client
# -----------------------------

class OpenRouterClient:
    def __init__(self, model: str, timeout: float = 60.0, force_json: bool = True, max_tokens: int = 4096):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
        self.models = [m.strip() for m in str(model).split(",") if m.strip()]
        if not self.models:
            raise ValueError("No models provided for OpenRouterClient")
        self.model = self.models[0]
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"
        self.force_json = force_json
        self.max_tokens = max_tokens

    async def chat(self, prompt: str, *, http_retries: int = 5, force_json: Optional[bool] = None) -> Tuple[str, str]:
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": self.max_tokens}
        use_json = self.force_json if force_json is None else force_json
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        backoff, last_err = 1.0, None
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            for attempt in range(http_retries):
                for model_to_try in self.models:
                    payload["model"] = model_to_try
                    try:
                        response = await client.post("/chat/completions", json=payload, headers=self.headers)
                        response.raise_for_status()
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if not content:
                            tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                            if tool_calls and "function" in tool_calls[0] and "arguments" in tool_calls[0]["function"]:
                                content = tool_calls[0]["function"]["arguments"]
                        return content, ""
                    except httpx.HTTPStatusError as e:
                        try:
                            error_details = e.response.json()
                            logging.warning(f"HTTP error for model {model_to_try}: {e} - Details: {error_details}. Trying next model...")
                        except Exception:
                            logging.warning(f"HTTP error for model {model_to_try}: {e}. Trying next model...")
                        last_err = e
                        continue
                    except Exception as e:
                        logging.error(f"A non-HTTP error occurred for model {model_to_try}: {e}")
                        last_err = e
                        return "", str(last_err)
                logging.info(f"All models failed on attempt {attempt + 1}. Retrying in {backoff:.2f}s...")
                await asyncio.sleep(backoff)
                backoff *= 1.5
        return "", f"All retries failed. Last error: {last_err}"

# -----------------------------
# Summarization Prompts & Logic
# -----------------------------

GIST_PROMPT_TEMPLATE = """Please provide a concise, self-contained summary of the following text chunk from a research paper. The summary should be around 400 words and capture the key concepts, methods, and findings presented in the text. It will be used by another AI to build a presentation, so it must be clear and easy to understand.

---
{chunk_text}
---

Your summary:"""

GIST_REPROMPT_TEMPLATE = """The previous summary was too short ({word_count} words). Please try again, ensuring the summary is comprehensive and around 400 words, capturing all key details from the text provided below.

---
{chunk_text}
---

Your summary:"""

JSON_EXTRACTOR_PROMPT_TEMPLATE = """Analyze the provided text chunk summary and extract the following information into a single JSON object. The `gist` field must be an exact, verbatim copy of the summary. For `claims`, list the key assertions or findings. For `figs` and `eqs`, list the IDs of any figures or equations mentioned. For `key_terms`, list important technical terms. For `anchors`, list concepts that connect to other parts of the paper.

Available Figures for Reference:
{available_figures_json}

---
Chunk ID: {chunk_id}
Page: {page_no}
Summary:
{gist}
---

Your JSON output:
```json
{{
  "id": "{chunk_id}",
  "page": {page_no},
  "gist": "{gist}",
  "claims": [
    "Claim 1...",
    "Claim 2..."
  ],
  "figs": [
    {{"id": "Figure 2", "path": "images/path/to/fig2.jpg", "caption": "Caption of figure 2..."}}
  ],
  "eqs": ["Eq. 1"],
  "key_terms": ["Term 1", "Term 2"],
  "anchors": ["Concept 1", "Concept 2"]
}}
```"""

def validate_summary(chunk: ChunkMeta, summary: dict, provided_gist: str, allowed_figs: List[FigureInfo]) -> Tuple[Optional[SummaryResult], Optional[str]]:
    required_keys = {"id", "page", "gist", "claims", "figs", "eqs", "key_terms", "anchors"}
    if not required_keys.issubset(summary.keys()):
        return None, f"Missing keys: {required_keys - set(summary.keys())}"
    if summary.get("id") != chunk.id:
        return None, f"ID mismatch: expected {chunk.id}, got {summary.get('id')}"

    # Allow minor formatting changes; replace with provided gist if different after normalization.
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    if _normalize(summary.get("gist", "")) != _normalize(provided_gist):
        logging.warning(f"{chunk.id}: Model modified gist; replacing with provided gist.")
        summary["gist"] = provided_gist

    validated_claims = [str(c) for c in summary.get("claims", []) if isinstance(c, str) and c]
    validated_eqs = [str(e) for e in summary.get("eqs", []) if isinstance(e, str) and e]

    # Figures: accept either full dicts or string IDs like "Figure 1" / "Image 1"; fill from allowed list when possible.
    validated_figs: List[Dict[str, str]] = []
    allowed_by_path = {f.path: f for f in allowed_figs}
    allowed_by_id = {f.id: f for f in allowed_figs}

    def _match_by_id(fid: str) -> Optional[FigureInfo]:
        fid = (fid or "").strip()
        if fid in allowed_by_id:
            return allowed_by_id[fid]
        # Try swapping Figure/Image prefixes
        if fid.lower().startswith("figure "):
            alt = "Image " + fid.split(" ", 1)[1]
            return allowed_by_id.get(alt)
        if fid.lower().startswith("image "):
            alt = "Figure " + fid.split(" ", 1)[1]
            return allowed_by_id.get(alt)
        return None

    for fig_obj in summary.get("figs", []):
        if isinstance(fig_obj, dict):
            # If path is known, keep as is
            path = fig_obj.get("path")
            if path and path in allowed_by_path:
                # Ensure required keys exist
                if "caption" not in fig_obj:
                    fig_obj["caption"] = allowed_by_path[path].caption
                if "id" not in fig_obj:
                    fig_obj["id"] = allowed_by_path[path].id
                validated_figs.append({"id": fig_obj["id"], "path": path, "caption": fig_obj.get("caption", "")})
                continue
            # Try to fill from id
            match = _match_by_id(fig_obj.get("id", ""))
            if match:
                validated_figs.append({"id": match.id, "path": match.path, "caption": match.caption})
                continue
            logging.warning(f"{chunk.id}: Skipping unknown/invalid figure object: {fig_obj}")
        elif isinstance(fig_obj, str):
            match = _match_by_id(fig_obj)
            if match:
                validated_figs.append({"id": match.id, "path": match.path, "caption": match.caption})
            else:
                logging.warning(f"{chunk.id}: Skipping figure reference with no match: {fig_obj}")
        else:
            logging.warning(f"{chunk.id}: Skipping figure with unsupported type: {type(fig_obj)}")

    validated_key_terms = [str(t) for t in summary.get("key_terms", []) if isinstance(t, str) and t]
    validated_anchors = [str(a) for a in summary.get("anchors", []) if isinstance(a, str) and a]

    return (
        SummaryResult(
            id=summary["id"],
            page=summary["page"],
            gist=summary["gist"],
            claims=validated_claims,
            figs=validated_figs,
            eqs=validated_eqs,
            key_terms=validated_key_terms,
            anchors=validated_anchors,
        ),
        None,
    )

async def generate_gist_with_retry(client: OpenRouterClient, chunk_text: str, progress: tqdm) -> Tuple[Optional[str], Optional[str]]:
    progress.set_description("Generating gist (1st attempt)")
    prompt = GIST_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    gist, error = await client.chat(prompt, force_json=False)
    if error:
        return None, f"Gist generation failed: {error}"
    word_count = len(gist.split())
    if 0 < word_count < 350:
        progress.set_description(f"Gist too short ({word_count} words), re-prompting")
        re_prompt = GIST_REPROMPT_TEMPLATE.format(word_count=word_count, chunk_text=chunk_text)
        gist, error = await client.chat(re_prompt, force_json=False)
        if error:
            return None, f"Gist re-generation failed: {error}"
    if not gist:
        return None, "Gist generation returned no content"
    return gist, None

async def summarize_from_gist(client: OpenRouterClient, chunk: ChunkMeta, provided_gist: str, allowed_figs: List[FigureInfo], progress: tqdm) -> Tuple[Optional[dict], Optional[str]]:
    for attempt in range(1, 4):
        progress.set_description(f"Extracting JSON (attempt {attempt})")
        try:
            available_figures_json = json.dumps([asdict(f) for f in allowed_figs], indent=2)
            prompt = JSON_EXTRACTOR_PROMPT_TEMPLATE.format(chunk_id=chunk.id, page_no=chunk.page, gist=provided_gist, available_figures_json=available_figures_json)
            raw_content, error = await client.chat(prompt, force_json=True)
            if error:
                return None, error
            json_obj = extract_json_object(raw_content)
            if json_obj:
                return json_obj, None
            else:
                logging.warning(f"Failed to extract JSON on attempt {attempt} for {chunk.id}")
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Error during JSON extraction for {chunk.id}: {e}")
            return None, str(e)
    return None, "Failed to extract valid JSON after multiple attempts."

async def summarize_one(worker_id: int, client: OpenRouterClient, chunk: ChunkMeta, allowed_figs: List[FigureInfo], progress: tqdm) -> Tuple[str, Optional[SummaryResult], str]:
    gist, error = await generate_gist_with_retry(client, chunk.text, progress)
    if error or not gist:
        return chunk.id, None, error or "Gist generation failed."
    summary_json, error = await summarize_from_gist(client, chunk, gist, allowed_figs, progress)
    if error or not summary_json:
        return chunk.id, None, error or "JSON extraction failed."
    validated_summary, error = validate_summary(chunk, summary_json, gist, allowed_figs)
    if error or not validated_summary:
        return chunk.id, None, error or "Summary validation failed."
    return chunk.id, validated_summary, ""

async def main_async(args: argparse.Namespace):
    load_dotenv()
    enc = tiktoken.get_encoding("cl100k_base")
    ensure_outdir(args.outdir)
    pages = load_pages(args.pages_dir)
    if not pages:
        logging.error(f"No markdown pages found in {args.pages_dir}. Exiting.")
        return
    logging.info(f"Extracting figure information from all pages...")
    all_figures = extract_figures_from_pages(pages, args.pages_dir)
    logging.info(f"Found {len(all_figures)} unique figures.")
    for fig in all_figures:
        logging.info(f"  - ID: {fig.id}, Path: {fig.path}, Caption: {fig.caption[:50]}...")
    all_chunks = []
    for page_no, _path, text in pages:
        all_chunks.extend(chunk_page(enc, page_no, text, args.chunk_size, args.overlap))
    chunk_index_path = args.outdir / "chunk_index.jsonl"
    with chunk_index_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    logging.info(f"Wrote chunk index: {chunk_index_path} ({len(all_chunks)} rows)")
    summaries_path = args.outdir / "chunk_summaries.jsonl"
    existing_summaries = {} if args.force else read_existing_summaries(summaries_path)
    if existing_summaries:
        logging.info(f"Loaded {len(existing_summaries)} existing summaries from {summaries_path}")
    tasks_to_run = [c for c in all_chunks if c.id not in existing_summaries]
    if not tasks_to_run:
        logging.info("All chunks are already summarized. Use --force to re-summarize.")
        return
    logging.info(f"Summarizing {len(tasks_to_run)} chunks...")
    client = OpenRouterClient(model=os.environ.get("OPENROUTER_SUMMARIZE_MODEL", "anthropic/claude-3-haiku"))
    progress = tqdm(total=len(tasks_to_run), desc="Summarizing", unit="chunk")
    summaries = {}
    errors = {}
    async with asyncio.TaskGroup() as tg:
        workers = [asyncio.create_task(summarize_one(i, client, chunk, all_figures, progress)) for i, chunk in enumerate(tasks_to_run)]
        for task in asyncio.as_completed(workers):
            chunk_id, summary, error = await task
            if summary:
                summaries[chunk_id] = summary
            else:
                errors[chunk_id] = error
            progress.update(1)
    progress.close()
    with summaries_path.open("a", encoding="utf-8") as f:
        for chunk_id, summary in summaries.items():
            f.write(json.dumps(asdict(summary)) + "\n")
    if errors:
        logging.error(f"Encountered {len(errors)} errors during summarization:")
        for chunk_id, error_msg in errors.items():
            logging.error(f"  - {chunk_id}: {error_msg}")
    logging.info("Summarization complete.")

def main_cli():
    parser = argparse.ArgumentParser(description="Summarize paper chunks using an LLM.")
    parser.add_argument("--pages-dir", type=Path, required=True, help="Directory containing markdown page files.")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"), help="Output directory for summaries and index.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in tokens.")
    parser.add_argument("--overlap", type=int, default=128, help="Token overlap between chunks.")
    parser.add_argument("--force", action="store_true", help="Force re-summarization of all chunks.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main_cli()
