#!/usr/bin/env python3
import argparse
import asyncio
import json
import ast
import logging
import os
import re
import sys
from dataclasses import dataclass, field
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


# -----------------------------
# Utilities
# -----------------------------


PAGE_NUM_RE = re.compile(r"(\d+)")


def extract_page_no(path: Path) -> Optional[int]:
    """Extract page number from filename like page_001.md or possm_page_01.md.

    Heuristic: take the last number sequence in the stem.
    """
    nums = PAGE_NUM_RE.findall(path.stem)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


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


def chunk_page(
    enc: tiktoken.Encoding,
    page_no: int,
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[ChunkMeta]:
    tokens = enc.encode(text)
    n = len(tokens)
    stride = max(1, chunk_size - overlap)
    prefix_chars, _pieces = build_prefix_char_lengths(enc, text)

    chunks: List[ChunkMeta] = []
    start_tok = 0
    chunk_idx = 1
    while start_tok < n:
        end_tok = min(n, start_tok + chunk_size)
        # map to char spans
        char_start = prefix_chars[start_tok]
        char_end = prefix_chars[end_tok]
        chunk_text = text[char_start:char_end]
        cid = f"p{page_no:03d}_c{chunk_idx:03d}"
        meta = ChunkMeta(
            id=cid,
            page=page_no,
            char_span=(char_start, char_end),
            token_span=(start_tok, end_tok),
            n_tokens=end_tok - start_tok,
            text=chunk_text,
        )
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


def token_count(enc: tiktoken.Encoding, texts: List[str]) -> int:
    total = 0
    for s in texts:
        total += len(enc.encode(s))
    return total


def extract_json_object(text: str) -> Optional[dict]:
    """Attempt to parse a single JSON object from text with common repairs.

    Strategy:
    1) Try strict JSON parsing of the whole text.
    2) Scan for the first balanced {...} JSON object and parse strictly.
    3) If that fails, strip code fences, remove trailing commas, and try a
       Python-literal fallback via ast.literal_eval after mapping true/false/null.
    """
    def find_first_balanced_json_object(s: str) -> Optional[str]:
        in_string = False
        escape = False
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1:
                            return s[start:i + 1]
        return None
    def strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # remove opening fence line
            lines = s.splitlines()
            # drop first line
            lines = lines[1:]
            # if ends with closing fence, drop it
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines)
        return s.strip()

    def remove_trailing_commas(s: str) -> str:
        # Remove trailing commas before } or ]
        return re.sub(r",\s*([}\]])", r"\1", s)

    text = text.strip()
    # 1) Whole-text strict JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) First balanced {...} block in text
    candidate = find_first_balanced_json_object(text)
    if candidate:
        for attempt in range(3):
            try:
                return json.loads(candidate)
            except Exception:
                if attempt == 0:
                    candidate = strip_code_fences(candidate)
                elif attempt == 1:
                    candidate = remove_trailing_commas(candidate)
                else:
                    break
        # 3) Python-literal tolerant fallback
        repaired = strip_code_fences(candidate)
        repaired = remove_trailing_commas(repaired)
        # Map JSON keywords to Python for literal_eval
        repaired = re.sub(r"(?<![A-Za-z0-9_])true(?![A-Za-z0-9_])", "True", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"(?<![A-Za-z0-9_])false(?![A-Za-z0-9_])", "False", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"(?<![A-Za-z0-9_])null(?![A-Za-z0-9_])", "None", repaired)
        try:
            obj = ast.literal_eval(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# -----------------------------
# OpenRouter client
# -----------------------------


class OpenRouterClient:
    def __init__(self, model: str, timeout: float = 60.0, force_json: bool = True, max_tokens: int = 512):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
        # accept comma-separated model list for fallback
        self.models: List[str] = [m.strip() for m in str(model).split(",") if m.strip()]
        self.model = self.models[0] if self.models else str(model)
        self.headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.force_json = force_json
        self.max_tokens = max_tokens

    async def chat(self, prompt: str, *, http_retries: int = 5) -> Tuple[str, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.force_json:
            payload["response_format"] = {"type": "json_object"}
        backoff = 1.0
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(http_retries):
                # try each model in priority order per attempt
                for m in self.models:
                    payload["model"] = m
                    try:
                        resp = await client.post(self.base_url, headers=self.headers, json=payload)
                        # Prefer switching models on quota/payment errors
                        if resp.status_code in (402, 403):
                            # payment required / forbidden -> try next model immediately
                            continue
                        # switch model if not found
                        if resp.status_code == 404:
                            continue
                        if resp.status_code in (429, 500, 502, 503, 504):
                            raise httpx.HTTPStatusError("Server busy", request=resp.request, response=resp)
                        resp.raise_for_status()
                        data = resp.json()
                        msg = data.get("choices", [{}])[0].get("message", {})
                        content_val = msg.get("content", "")
                        content: str = ""
                        # content may be a string or an array (multimodal); coerce to str
                        if isinstance(content_val, str):
                            content = content_val
                        elif isinstance(content_val, list):
                            # concatenate available text fields
                            parts: List[str] = []
                            for item in content_val:
                                if isinstance(item, dict):
                                    if "text" in item and isinstance(item["text"], str):
                                        parts.append(item["text"])
                                    elif "content" in item and isinstance(item["content"], str):
                                        parts.append(item["content"])
                            content = "\n".join(parts).strip()
                        # some providers put JSON in tool/function calls
                        if not content:
                            tool_calls = msg.get("tool_calls") or []
                            if isinstance(tool_calls, list) and tool_calls:
                                try:
                                    content = tool_calls[0]["function"].get("arguments", "")
                                except Exception:
                                    pass
                        if not content and "function_call" in msg:
                            try:
                                content = msg["function_call"].get("arguments", "")
                            except Exception:
                                pass
                        # some providers include a parsed JSON object when response_format is used
                        if not content and "parsed" in msg:
                            try:
                                content = json.dumps(msg["parsed"], ensure_ascii=False)
                            except Exception:
                                pass
                        # record last used model
                        self.model = m
                        return content, resp.text
                    except httpx.HTTPStatusError as e:
                        # Fallback if model/provider doesn't support response_format
                        try:
                            code = e.response.status_code
                        except Exception:
                            code = None
                        if code == 400 and payload.get("response_format") is not None:
                            # Disable JSON mode and retry immediately for this model
                            payload.pop("response_format", None)
                            self.force_json = False
                            # retry same model without JSON mode once
                            try:
                                resp = await client.post(self.base_url, headers=self.headers, json=payload)
                                if resp.status_code in (402, 403):
                                    continue
                                if resp.status_code in (429, 500, 502, 503, 504):
                                    raise httpx.HTTPStatusError("Server busy", request=resp.request, response=resp)
                                resp.raise_for_status()
                                data = resp.json()
                                msg = data.get("choices", [{}])[0].get("message", {})
                                content = msg.get("content", "") or json.dumps(msg.get("parsed", {}))
                                self.model = m
                                return content, resp.text
                            except Exception as e2:
                                last_err = e2
                                continue
                        # 404 model not found -> try next model
                        try:
                            if e.response is not None and e.response.status_code == 404:
                                continue
                        except Exception:
                            pass
                        last_err = e
                        continue
                    except Exception as e:  # network/backoff or parsing issues
                        last_err = e
                        continue
                # if none of the models succeeded in this attempt, backoff then retry
                await asyncio.sleep(backoff)
                backoff = min(8.0, backoff * 2)
            # if all retries exhausted
            if last_err:
                raise last_err
            raise RuntimeError("OpenRouter chat failed without exception")


# Summarization logic
# -----------------------------


GIST_PROMPT_TEMPLATE = (
    "Summarize the following research paper chunk in a detailed, factual paragraph of approximately 400 words. "
    "Focus on the core concepts, claims, methods, key results, and any mentioned limitations. "
    "Your response must be a single block of text. Do not use markdown or special formatting.\n"
    "\n"
    "---\n"
    "CHUNK TEXT TO SUMMARIZE:\n"
    "{chunk_text}\n"
    "---"
)

GIST_REPROMPT_TEMPLATE = (
    "The previous summary you provided was too short at only {word_count} words. "
    "Please expand it to approximately 400 words. "
    "Ensure you elaborate on the methods used, the key results discovered, and any limitations or future work mentioned in the text. "
    "Your response must be a single block of text. Do not use markdown or special formatting.\n"
    "\n"
    "---\n"
    "ORIGINAL CHUNK TEXT TO SUMMARIZE:\n"
    "{chunk_text}\n"
    "---"
)

JSON_EXTRACTOR_PROMPT_TEMPLATE = (
    "You are creating a structured JSON metadata object from a pre-written summary (a 'gist').\n"
    "Your response must be a single, valid JSON object and nothing else.\n"
    "Follow these instructions precisely:\n"
    "1.  **id**: Repeat the provided chunk ID exactly.\n"
    "2.  **page**: Repeat the provided page number.\n"
    "3.  **gist**: **Put this exact gist into the gist field. Do not change a single character.**\n"
    "4.  **claims**: From the provided gist, extract key claims. Each must be an object `{{\"text\": \"<quote_from_gist>\"}}`.\n"
    "5.  **figs**: List figure IDs mentioned in the provided gist. Use ONLY IDs from the allowed list.\n"
    "6.  **eqs**: Extract equations from the provided gist. Each must be an object `{{\"latex\": \"<latex_code>\"}}`.\n"
    "7.  **key_terms**: List 5-10 important keywords or phrases found in the gist.\n"
    "8.  **anchors**: List 1-3 semantic anchors (e.g., `introduction`, `method`, `results`, `discussion`, `conclusion`).\n"
    "\n"
    "Here is the information:\n"
    "Chunk ID: {chunk_id}\n"
    "Page Number: {page_no}\n"
    "Allowed Figure IDs: {allowed_figure_ids}\n"
    "\n"
    "---\n"
    "GIST TO USE (COPY THIS EXACTLY):\n"
    "{gist}\n"
    "---"
)


@dataclass
class SummaryResult:
    id: str
    page: int
    gist: str
    claims: List[Dict[str, Any]] = field(default_factory=list)
    figs: List[str] = field(default_factory=list)
    eqs: List[Dict[str, Any]] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    anchors: List[str] = field(default_factory=list)


def validate_summary(
    chunk: ChunkMeta,
    summary: dict,
    allowed_figs: Set[str],
) -> Tuple[Optional[SummaryResult], Optional[str]]:
    """Validate parsed JSON against chunk text and constraints."""
    # Check required fields
    required_fields = ["id", "page", "gist", "claims", "figs", "eqs", "key_terms", "anchors"]
    for f in required_fields:
        if f not in summary:
            return None, f"Missing required field: {f}"

    # Check ID and page match
    if summary["id"] != chunk.id:
        return None, f"ID mismatch: expected {chunk.id}, got {summary['id']}"
    if summary["page"] != chunk.page:
        return None, f"Page mismatch: expected {chunk.page}, got {summary['page']}"

    # Validate claims
    validated_claims = []
    for claim in summary.get("claims", []):
        if not isinstance(claim, dict) or "text" not in claim:
            return None, "Invalid claim format"
        # char_span is now optional and not strictly verified against chunk text
        if "char_span" in claim and (not isinstance(claim["char_span"], list) or len(claim["char_span"]) != 2):
            logging.warning(f"Invalid char_span in claim: {claim}")
        validated_claims.append(claim)

    # Validate equations
    validated_eqs = []
    for eq in summary.get("eqs", []):
        if not isinstance(eq, dict) or "latex" not in eq:
            return None, "Invalid equation format"
        # char_span is now optional and not strictly verified against chunk text
        if "char_span" in eq and (not isinstance(eq["char_span"], list) or len(eq["char_span"]) != 2):
            logging.warning(f"Invalid char_span in equation: {eq}")
        validated_eqs.append(eq)

    # Validate figure IDs
    validated_figs = []
    for fig_id in summary.get("figs", []):
        if fig_id not in allowed_figs:
            logging.warning(f"{chunk.id}: Rejecting unknown figure ID: {fig_id}. Allowed: {allowed_figs}")
            continue
        validated_figs.append(fig_id)

    return (
        SummaryResult(
            id=summary["id"],
            page=summary["page"],
            gist=summary["gist"],
            claims=validated_claims,
            figs=validated_figs,
            eqs=validated_eqs,
            key_terms=summary.get("key_terms", []),
            anchors=summary.get("anchors", []),
        ),
        None,
    )


async def generate_gist_with_retry(
    client: OpenRouterClient, chunk_text: str, progress: tqdm
) -> Tuple[Optional[str], Optional[str]]:
    """Step A: Generate a ~400 word gist, with a single retry if it's too short."""
    try:
        # Initial attempt
        progress.set_description("Generating gist (1st attempt)")
        prompt = GIST_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        response = await client.get_chat_completion(
            prompt=prompt, max_tokens=700, temperature=0.2, top_p=0.9
        )
        gist = response.choices[0].message.content.strip() if response and response.choices else ""
        word_count = len(gist.split())

        # Re-prompt if too short
        if 0 < word_count < 350:
            progress.set_description(f"Gist too short ({word_count} words), re-prompting")
            re_prompt = GIST_REPROMPT_TEMPLATE.format(word_count=word_count, chunk_text=chunk_text)
            response = await client.get_chat_completion(
                prompt=re_prompt, max_tokens=700, temperature=0.2, top_p=0.9
            )
            gist = response.choices[0].message.content.strip() if response and response.choices else ""

        if not gist:
            return None, "Gist generation returned no content"
        return gist, None

    except Exception as e:
        return None, f"Gist generation failed: {e}"


async def summarize_from_gist(
    client: OpenRouterClient,
    chunk: ChunkMeta,
    provided_gist: str,
    allowed_figs: Set[str],
    progress: tqdm,
) -> Tuple[Optional[dict], Optional[str]]:
    """Step B: Create JSON from the gist, verifying the gist is copied exactly."""
    for attempt in range(1, 3):
        progress.set_description(f"Extracting JSON (attempt {attempt})")
        try:
            prompt = JSON_EXTRACTOR_PROMPT_TEMPLATE.format(
                chunk_id=chunk.id,
                page_no=chunk.page,
                allowed_figure_ids=sorted(list(allowed_figs)) if allowed_figs else "[]",
                gist=provided_gist,
            )
            response_str = await client.get_chat_completion_str(
                prompt=prompt,
                max_tokens=1024, # JSON structure is larger than gist
                temperature=0.0,
                force_json=True,
            )
            if not response_str:
                return None, "JSON extraction returned no content"

            # Verify the gist was copied exactly
            obj = json.loads(response_str)
            if obj.get("gist") != provided_gist:
                logging.warning(f"{chunk.id}: Gist mismatch, retrying JSON extraction...")
                continue # Retry Step B

            return obj, None # Success

        except Exception as e:
            logging.error(f"{chunk.id}: JSON extraction attempt {attempt} failed: {e}")
            await asyncio.sleep(1)

    return None, "Failed to extract and verify JSON after 2 attempts"


async def summarize_one(
    client: OpenRouterClient,
    enc: tiktoken.Encoding,
    chunk: ChunkMeta,
    file_lock: asyncio.Lock,
    summaries_path: Path,
    progress: tqdm,
    allowed_figs: Set[str],
) -> Tuple[Optional[dict], Optional[str]]:
    # Step A: Generate the detailed gist with retry logic
    gist, error = await generate_gist_with_retry(client, chunk.text, progress)
    if error:
        return None, error

    # Step B: Generate the structured JSON using the final gist
    obj, error = await summarize_from_gist(client, chunk, gist, allowed_figs, progress)
    if error:
        return None, error

    debug_dir = summaries_path.parent / "bad_responses"
    debug_dir.mkdir(parents=True, exist_ok=True)

    async def try_once(extra_reminder: bool = False) -> Tuple[Optional[dict], Optional[str]]:
        p = prompt
        if extra_reminder:
            p = p + "\nReminder: Return ONLY JSON, no commentary or code fences."
        try:
            content, raw_resp = await client.chat(p)
        except Exception as e:
            return None, f"OpenRouter error: {e}"
        obj = extract_json_object(content)
        if obj is None:
            # dump raw content for inspection
            try:
                (debug_dir / f"{chunk.id}.txt").write_text(content, encoding="utf-8")
                (debug_dir / f"{chunk.id}.response.json").write_text(raw_resp, encoding="utf-8")
            except Exception:
                pass
            return None, "Model did not return valid JSON"
        # Validate structure and content
        validated_summary, error = validate_summary(chunk, obj, allowed_figs)
        if error:
            return None, error
        if not validated_summary:
            return None, "Validation failed silently"

        # Check token budget on the gist (approx. 200 words is ~270 tokens)
        budget_tokens = len(enc.encode(validated_summary.gist))
        if budget_tokens > 300:
            return None, f"Token budget too high (>300): {budget_tokens}"
        if budget_tokens > 250:
            logging.warning(f"{chunk.id}: summary gist is over 250 tokens: {budget_tokens}")

        # Final validated object
        obj = validated_summary.__dict__
        return obj, None

    # First attempt
    obj, err = await try_once(extra_reminder=False)
    if obj is None:
        # Second attempt with explicit reminder
        obj, err = await try_once(extra_reminder=True)
    if obj is None:
        progress.update(1)
        return chunk.id, err or "Unknown summarization error"

    # Append to summaries.jsonl with a file lock
    line = json.dumps(obj, ensure_ascii=False)
    async with file_lock:
        with summaries_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    progress.update(1)
    return chunk.id, None


# -----------------------------
# Markdown output
# -----------------------------


def write_chunk_index(out_path: Path, chunks: List[ChunkMeta]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            row = {
                "id": ch.id,
                "page": ch.page,
                "char_span": [ch.char_span[0], ch.char_span[1]],
                "token_span": [ch.token_span[0], ch.token_span[1]],
                "n_tokens": ch.n_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summaries_md(out_path: Path, per_page_ids: Dict[int, List[str]], summaries: Dict[str, dict]) -> None:
    lines: List[str] = []
    lines.append("# Chunk Summaries")
    for page in sorted(per_page_ids.keys()):
        lines.append("")
        lines.append(f"## Page {page}")
        lines.append("| ID | gist | claims | figs | eqs | key_terms |")
        lines.append("|---|---|---|---|---|---|")
        for cid in per_page_ids[page]:
            obj = summaries.get(cid)
            if not obj:
                raise RuntimeError(f"Missing summary for chunk {cid}")
            gist = str(obj.get("gist", "")).replace("\n", " ")
            claims = "; ".join([str(x) for x in obj.get("claims", [])])
            figs = ", ".join([str(x) for x in obj.get("figs", [])])
            eqs = ", ".join([str(x) for x in obj.get("eqs", [])])
            key_terms = ", ".join([str(x) for x in obj.get("key_terms", [])])
            # add trailing cite tag inside gist cell to avoid table column drift
            gist_with_cite = (gist + f" [[cid:{cid}]]").strip()
            lines.append(
                f"| {cid} | {gist_with_cite} | {claims} | {figs} | {eqs} | {key_terms} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chunk and summarize page-wise Markdown using OpenRouter")
    ap.add_argument("--pages-dir", type=Path, required=True, help="Directory with page_XXX.md files")
    ap.add_argument("--outdir", type=Path, default=Path("artifacts"), help="Output directory (default: artifacts)")
    default_models = os.environ.get(
        "OPENROUTER_MODEL",
        # Prefer widely-available free-tier models by default; override via env/CLI
        "mistralai/mistral-7b-instruct:free,google/gemma-2-9b-it:free,qwen/qwen-2.5-7b-instruct:free",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen/qwen2.5-32b-instruct",
        help=(
            "OpenRouter model slug or comma-separated priority list. "
            "Defaults to free-tier fallbacks unless OPENROUTER_MODEL is set."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for model response (default: 1024)")
    ap.add_argument("--no-force-json", action="store_true", help="Do not request JSON mode (response_format)")
    ap.add_argument("--max-workers", type=int, default=1, help="Max concurrent requests (default: 1)")
    ap.add_argument("--chunk-size", type=int, default=800, help="Chunk size in tokens (default: 800)")
    ap.add_argument("--overlap", type=int, default=100, help="Overlap in tokens (default: 100)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Force re-summarization of all chunks")
    ap.add_argument("--figure-map", type=Path, default=None, help="Path to figure_map.json file")
    return ap.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    pages = load_pages(args.pages_dir)
    if not pages:
        logging.error("No Markdown pages found in --pages-dir")
        return 2

    ensure_outdir(args.outdir)
    chunk_index_path = args.outdir / "chunk_index.jsonl"
    summaries_path = args.outdir / "chunk_summaries.jsonl"
    summaries_md_path = args.outdir / "summaries.md"

    enc = tiktoken.get_encoding("cl100k_base")

    # Build chunks per page
    all_chunks: List[ChunkMeta] = []
    # Load figure map if provided
    figure_map: Dict[int, List[str]] = {}
    if args.figure_map and args.figure_map.exists():
        logging.info(f"Loading figure map from {args.figure_map}")
        with args.figure_map.open("r", encoding="utf-8") as f:
            raw_map = json.load(f)
            # convert keys to int
            for k, v in raw_map.items():
                try:
                    figure_map[int(k)] = v
                except ValueError:
                    logging.warning(f"Skipping invalid page number key in figure map: {k}")

    # Collect all chunks
    all_chunks: List[ChunkMeta] = []
    per_page_ids: Dict[int, List[str]] = {}
    for page_no, _path, text in pages:
        chunks = chunk_page(enc, page_no, text, args.chunk_size, args.overlap)
        all_chunks.extend(chunks)
        per_page_ids[page_no] = [c.id for c in chunks]

    # Write chunk index
    write_chunk_index(chunk_index_path, all_chunks)
    logging.info(f"Wrote chunk index: {chunk_index_path} ({len(all_chunks)} rows)")

    # Load existing summaries for resumability
    existing = read_existing_summaries(summaries_path)
    logging.info(f"Existing summaries loaded: {len(existing)}")

    # Prepare OpenRouter client
    client = OpenRouterClient(args.model, force_json=not args.no_force_json, max_tokens=args.max_tokens)

    # Create async tasks for missing chunks
    missing_chunks = [c for c in all_chunks if c.id not in existing]
    failures: List[Tuple[str, str]] = []
    if missing_chunks:
        sem = asyncio.Semaphore(max(1, args.max_workers))
        file_lock = asyncio.Lock()
        progress = tqdm(total=len(missing_chunks), desc="Summarizing", unit="chunk")

        async def worker(c: ChunkMeta):
            async with sem:
                allowed_figs_for_page = set(figure_map.get(c.page, []))
                cid, err = await summarize_one(client, enc, c, file_lock, summaries_path, progress, allowed_figs_for_page)
                if err:
                    failures.append((cid, err))

        await asyncio.gather(*(worker(c) for c in missing_chunks))
        progress.close()

    # Reload all summaries (existing + new)
    summaries = read_existing_summaries(summaries_path)

    # Validate: every chunk must have a summary
    missing = [c.id for c in all_chunks if c.id not in summaries]
    if missing:
        logging.error(f"Missing summaries for {len(missing)} chunks (e.g., {missing[:3]})")
        # Print reasons if known
        if failures:
            for cid, err in failures[:5]:
                hint = (summaries_path.parent / 'bad_responses' / f'{cid}.txt')
                extra = f" (raw: {hint})" if hint.exists() else ""
                logging.error(f"Failure {cid}: {err}{extra}")
        return 1

    # Write summaries.md
    write_summaries_md(summaries_md_path, per_page_ids, summaries)
    logging.info(f"Wrote summaries markdown: {summaries_md_path}")

    # Final quality check: row count equals number of chunks
    # Count rows generated: sum of per-page ids
    total_rows = sum(len(v) for v in per_page_ids.values())
    if total_rows != len(all_chunks):
        logging.error(
            f"Row count mismatch in summaries.md: {total_rows} vs chunks {len(all_chunks)}"
        )
        return 1

    if failures:
        logging.error(f"Encountered {len(failures)} permanent failures")
        return 1

    logging.info("All chunks summarized successfully")
    return 0


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
