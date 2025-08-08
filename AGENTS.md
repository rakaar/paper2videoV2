My plan is to work on project which does this, it takes a journal paper PDF, preferably neuroscience and ML paper. Then makes a video explaining the paper. The video is basically slides + audio. Please help me on this project planning.

# Paper2Video V2 — Project Instructions (Summary)

## Goal

Keep the **existing deterministic pipeline** (Mistral OCR → Markdown, figure crops; JSON→Marp→PNG; Sarvam TTS; ffmpeg) and add an **agentic layer** so the LLM manages context and writes slides via tool calls. Models served through **OpenRouter** (easy swapping) or orchestrated with **DeepAgents** later.

---

## Fixed Workflow (unchanged)

* **Text & figures:** Mistral OCR → **Markdown** + `figures/` (paths + captions).
* **Slides & video:** LLM JSON → **Marp** (PNG frames) → **Sarvam TTS** → **ffmpeg** MP4.

---

## Agentic Upgrade (what’s new)

* **One agent loop** (Qwen/Kimi via OpenRouter; Ollama fallback).
* Agent **does not read the whole paper**; it **pulls small chunks on demand** via tools.
* **Two-pass flow:**

  1. **Skim & Plan:** build a *Paper Card* + `slide_plan.json` (global map).
  2. **Write Slides:** fetch only relevant chunks per slide, then prune.

---

## Minimal Tool Surface (keep names short)

* `search(query, k)` → chunk IDs (vector DB).
* `open_chunk(id)` → raw text of one chunk.
* `write_slide(slide_no, title, bullets, figures)` → append/update slide JSON (validate figure paths).
* `checkpoint(scope, label, evidence_chunk_ids, keep_chunks=0)` → **summarise & prune** before moving on.
* *(Optional)* `list_segments()` / `open_segment(id, mode)` when PDFs lack headings.

> Keep tools tiny & deterministic. Validation lives in tools (e.g., reject non-existent figure paths).

---

## Context Management Policy (the “Context Governor”)

* **Working set cap:** keep **≤ 3 raw chunks** of paper text in prompt at any time.
* **Notes instead of bulk:** after each slide (or page/segment change) call **`checkpoint`** to store a ≤25-token note (e.g., “slide-3: X-method”), **then drop** raw text.
* **Evidence discipline:** each slide must list `evidence_chunk_ids` it used.
* **Hard limits:** max **12 tool steps** per run, prompt budget \~**6k tokens** (well below model window).

---

## Skim → Plan (global grasp, cheap)

Agent reads only:

* Abstract, last intro paragraph (contributions), conclusion
* Section headers / **segment summaries** (if no headings)
* Figure captions

Outputs `paper_card.json`:

* `tldr`, `contributions`, `method one-liner`, `key_results`, `limitations`
* `slide_plan`: slide titles + target sections/segments/figures

Then the writer fills slides using `search/open_chunk` under the working-set cap.

---

## Model & Provider

* **Primary:** OpenRouter (`/v1/chat/completions`) so we can switch models by changing `model` string (e.g., `kimi`, `qwen2.5-32b`, `gpt-4o`).
* **Fallback:** Ollama locally (`gemma:3n-e4b`) for offline testing.
* Same **OpenAI tools schema** everywhere.

---

## Success Criteria (A/B vs current pipeline)

* **Quality:** no duplicate topics; correct figure references; coherent narrative.
* **Budget:** prompt tokens < 6k; ≤ 12 tool steps.
* **Latency:** close to current total time (±20%).
* **Auditability:** each slide cites chunk IDs; `checkpoint` entries exist per slide.

---

## Roadmap (practical order)

1. **Indexing pre-step:** Mistral OCR → Markdown; chunk (≈800 tokens, overlap ≈100); build FAISS/Qdrant.
2. **OpenRouter hookup:** single client; model is a runtime knob.
3. **Hello-tool world:** implement `search` only; confirm one tool call end-to-end.
4. **Add full toolset:** `open_chunk`, `write_slide` (with validation), `checkpoint`.
5. **Skim & Plan pass:** output `paper_card.json` + `slide_plan.json`.
6. **Context rules:** enforce working-set cap + hard step limit.
7. **(Optional) DeepAgents:** move plan/notes/slides to its virtual FS; same tools.



## New Files (Search)

- `index/index_paper.py`: Chunk markdown (≈800/100), embed (bge-small), build FAISS + meta.
- `index/search_index.py`: `search_chunks()` + `handle_search()` returning chunk IDs only.
- `demo_search_tool_call.py`: Single tool-call demo for `search` via OpenRouter.

## Quickstart: search(query, k)

- Build index: `python -m pip install -r requirements.txt && python index/index_paper.py --paper-id test_paper --markdown-dir mistral_responses/test_paper/markdown --index-dir index`
- Direct check: `python -c "from index.search_index import search_chunks; print(search_chunks('test_paper','sparsity expression',5))"`
- Tool demo (GLM 4.5 default): `TMPDIR=./tmp python demo_search_tool_call.py "sparsity expression" --k 5 --paper-id test_paper`

Artifacts under `index/`:
- `test_paper.faiss`, `test_paper.meta.jsonl`, `embedder.json`.

Notes:
- Default model: `z-ai/glm-4.5` (reasoning disabled; single tool-call enforced).
- Set `TMPDIR=./tmp` if the environment restricts temp writes (PyTorch/ST needs it).
