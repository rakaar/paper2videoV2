# Paper2Video V2 — Developer Guide

This guide explains the full pipeline, what each script does, inputs/outputs, key arguments, environment variables, and how to run everything end-to-end. Use this as the source when unifying the pipeline into a single workflow.

## Overview of the Flow

1. **Summarize OCR pages → Chunk summaries**
   - Script: `summarize_chunks.py`
   - Reads OCR Markdown pages, tokenizes into overlapping chunks, produces per-chunk JSON summaries and a chunk index.
2. **Create governance card**
   - Script: `make_paper_card.py`
   - Builds a “paper card” that encodes TL;DR, contributions, section order, etc., guiding planning.
3. **Plan slide sections**
   - Script: `plan_slides.py`
   - Sequentially plans sections (Overview → Method → Results → …), with references to chunks and figure IDs filtered to those references.
4. **Generate slides**
   - Script: `generate_slides.py`
   - Produces slide-level JSON with Title, Content, Audio, Bridges, and Figures. Optionally prepends a cover slide extracted from OCR via LLM.
5. **Render PNG slides (Marp)**
   - Script: `generate_slide_pngs.py`
   - Converts `presentation.json` into `deck.md` and `deck.XXX.png` images.
6. **Generate audio (TTS)**
   - Script: `generate_audio.py`
   - Generates per-slide narration WAV/MP3 files (per-sentence TTS composition to avoid API issues).
7. **Stitch video**
   - Script: `stitch_video.py`
   - Combines PNGs and audio into a single MP4.

## Directory Layout

- OCR input directory (default): `mistral_responses/<paper_name>/markdown/*.md`
- Images (if referenced): `mistral_responses/<paper_name>/images/*`
- Artifacts directory (per paper recommended): `artifacts/<paper_name>/`
  - `chunk_summaries.jsonl`
  - `chunk_index.jsonl`
  - `paper_card.json`
  - `slide_plan.json`
  - `presentation.json`
  - `deck.md`
  - `pngs/` (rendered PNG slides)
  - `audio/` (narration WAV/MP3s)
  - `video.mp4` (final stitched output)

## Environment Variables

- `OPENROUTER_API_KEY` — required for all OpenRouter LLM calls.
- Optional model overrides (comma-separated for fallback):
  - `OPENROUTER_MODEL` (global fallback)
  - `OPENROUTER_SUMMARIZE_MODEL`, `OPENROUTER_EXTRACTOR_MODEL`
  - `OPENROUTER_CARD_MODEL`
  - `OPENROUTER_PLANNER_MODEL`
  - `OPENROUTER_GENERATOR_MODEL`
  - `OPENROUTER_COVER_MODEL`
- `SARVAM_API_KEY` — required by `generate_audio.py` for TTS.

Notes:
- Defaults prefer non-`:free` model slugs for reliability.
- Token limits are **uncapped by default**; pass `--max-tokens` only when you need to constrain cost.

## Script Details

### 1) Summarizer — `summarize_chunks.py`

- **Purpose:**
  - Split OCR Markdown into token chunks and generate structured JSON per chunk using OpenRouter.
- **Inputs:**
  - `--pages-dir` (required): path to markdown pages.
- **Outputs (to `--outdir`, default `artifacts/`):**
  - `chunk_summaries.jsonl` — one JSON object per chunk with fields: `id`, `page`, `gist`, `claims`, `figs`, `eqs`, `key_terms`, `anchors`.
  - `chunk_index.jsonl` — maps chunk IDs to original page spans for later text retrieval.
- **Key args:**
  - `--outdir` (default: `artifacts`)
  - `--chunk-size` (default: 1024), `--overlap` (default: 128)
  - `--model` (default: `meta-llama/llama-3.2-3b-instruct,google/gemma-2-9b-it`)
  - `--max-tokens` (default: unlimited)
  - `--force`, `--verbose`
- **Notable behavior:**
  - JSON extractor prompt includes the original `chunk_text` so the model can detect explicit `Figure/Table N` mentions.
  - Validates figure mentions by number against page-local allowed figures, populating `figs` reliably.
  - Internal `OpenRouterClient` only attaches `max_tokens` if non-None.

Example:
```bash
python summarize_chunks.py \
  --pages-dir mistral_responses/test_paper/markdown \
  --outdir artifacts/test_paper \
  --verbose --force
```

### 2) Paper Card — `make_paper_card.py`

- **Purpose:**
  - Create `paper_card.json` to guide planning with TL;DR, contributions, section order, etc.
- **Inputs:**
  - `--artifacts-dir` containing `chunk_summaries.jsonl`.
- **Outputs:**
  - `paper_card.json` in `--outdir`.
- **Key args:**
  - `--model` (default via `OPENROUTER_CARD_MODEL`)
  - `--max-tokens` (default: unlimited)
  - `--force`, `--verbose`
- **Notable behavior:**
  - Hardened JSON parsing, retries with strict JSON instruction if initial parse fails.

Example:
```bash
python make_paper_card.py \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --verbose --force
```

### 3) Planner — `plan_slides.py`

- **Purpose:**
  - Sequentially plan canonical sections and per-section topics with references and allowed figure IDs.
- **Inputs:**
  - `--summaries-dir` containing `chunk_summaries.jsonl`, `chunk_index.jsonl`, `paper_card.json`.
- **Outputs:**
  - `slide_plan.json` in `--outdir`.
- **Key args:**
  - `--model` (default via `OPENROUTER_PLANNER_MODEL`: `qwen/qwen-2.5-7b-instruct,mistralai/mixtral-8x7b-instruct`)
  - `--max-tokens` (default: unlimited)
  - `--force`, `--verbose`
- **Notable behavior:**
  - Robust JSON parsing with repairs (unquoted keys, `True/False/None` normalization) and a retry with explicit strict-JSON instructions.
  - Enforces exact `section_title`, ≥2 `references`, and filters `figure_ids` to figures only present in referenced chunks.
  - Loop continues on exceptions instead of aborting the whole plan.

Example:
```bash
python plan_slides.py \
  --summaries-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --verbose --force
```

### 4) Generator — `generate_slides.py`

- **Purpose:**
  - Generate per-slide JSON from `slide_plan.json` and source chunks. Optionally prepend a cover slide.
- **Inputs:**
  - `--artifacts-dir` containing `slide_plan.json`, `chunk_summaries.jsonl`, `chunk_index.jsonl`, `paper_card.json`
  - OCR under `--ocr-dir/<pdf-name>/markdown/*`
- **Outputs:**
  - `presentation.json` in `--outdir`.
- **Key args:**
  - `--model` (default via `OPENROUTER_GENERATOR_MODEL`)
  - `--max-tokens` (default: unlimited)
  - `--figure-reuse-limit` (default: -1 unlimited)
  - `--add-cover` (extract title/authors via LLM and create cover slide)
  - `--cover-model` (default via `OPENROUTER_COVER_MODEL`)
  - `--force`, `--verbose`
- **Notable behavior:**
  - JSON-hardening with fallback from `response_format=json_object` to plain text + light repairs; tool_calls fallback.
  - Guards against mentioning figures when none are attached.
  - Enforces unique slide titles by appending “(Slide N)” to duplicates.
  - Prefers NOVEL claims; maintains light context (last 2 slides + checkpoints) to avoid drift.

Example:
```bash
python generate_slides.py \
  --ocr-dir mistral_responses \
  --pdf-name test_paper \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --add-cover \
  --verbose --force
```

### 5) Renderer — `generate_slide_pngs.py`

- **Purpose:**
  - Render `presentation.json` to Marp `deck.md` and `deck.XXX.png` files.
- **Inputs:**
  - `--presentation-file` (required), `--output-dir`, `--paper-name`
  - Optional `--no-cover` to avoid adding an extra cover (use when generator already added one)
  - Optional `--ocr-dir` + `--cover-model` to build a cover from OCR when not using generator cover
- **Outputs:**
  - `deck.md` and `pngs/deck.XXX.png` in `--output-dir`.
- **Notable behavior:**
  - Cover extraction via OpenRouter now **uncapped** (no fixed `max_tokens`).
  - Heuristic fallback to extract title/authors if LLM parsing fails.

Example:
```bash
python generate_slide_pngs.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper \
  --paper-name test_paper \
  --no-cover
```

### 6) Audio — `generate_audio.py`

- **Purpose:**
  - Generate per-slide audio using Sarvam TTS, splitting narration per sentence to avoid API glitches.
- **Inputs:**
  - `--presentation-file`, `--output-dir`, `--paper-name`
- **Outputs:**
  - `audio/slide_XXX.wav` or `.mp3`

Example:
```bash
python generate_audio.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper/audio \
  --paper-name test_paper
```

### 7) Stitch — `stitch_video.py`

- **Purpose:**
  - Combine PNGs and WAVs to MP4 via ffmpeg.
- **Inputs:**
  - `--paper-name` or explicit `--png-dir` and `--audio-dir`
- **Outputs:**
  - `artifacts/<paper>/video.mp4`

Example:
```bash
python stitch_video.py --paper-name test_paper
```

## Integration Notes (for a unified workflow)

- **Sequence & dependencies:** summarizer → paper card → planner → generator → renderer → audio → stitch.
- **Idempotency:** most scripts support `--force` to overwrite outputs; otherwise they may exit early if outputs exist.
- **Artifacts co-location:** prefer `--outdir artifacts/<paper>` for isolation. Pass the same `--artifacts-dir` to downstream steps.
- **Cover handling:** if generator created a cover (`--add-cover`), pass `--no-cover` to renderer to avoid duplicates.
- **Figures:** generator respects planner figures and drops any not referenced; reuse can be capped with `--figure-reuse-limit`.
- **Token policy:** leave `--max-tokens` unset for unlimited outputs; set only to constrain costs.
- **Error handling:** planner/generator/card include robust JSON parsing and retries with strict-JSON instructions; planner loop continues on exceptions.
- **Models:** prefer reliable non-`:free` slugs; you may provide multiple slugs comma-separated for automatic fallback.

## End-to-End Example

```bash
# 1) Summaries
python summarize_chunks.py \
  --pages-dir mistral_responses/test_paper/markdown \
  --outdir artifacts/test_paper \
  --force

# 2) Paper card
python make_paper_card.py \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --force

# 3) Plan
python plan_slides.py \
  --summaries-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --force

# 4) Generate slides with cover
python generate_slides.py \
  --ocr-dir mistral_responses \
  --pdf-name test_paper \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --add-cover \
  --force

# 5) Render PNGs (avoid duplicate cover)
python generate_slide_pngs.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper \
  --paper-name test_paper \
  --no-cover

# 6) Generate audio WAVs/MP3s
python generate_audio.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper/audio \
  --paper-name test_paper

# 7) Stitch to MP4
python stitch_video.py --paper-name test_paper
```

## QA Checklist

- **Summaries:** `wc -l artifacts/<paper>/chunk_summaries.jsonl` matches expectations.
- **Plan:** verify each section has ≥2 `references`; figure IDs exist in referenced chunks.
- **Slides:** ensure Titles are unique; Content mentions figures only when attached; Bridges are present.
- **Renderer:** confirm number of PNGs matches slides (plus cover if applicable).
- **Audio:** number of WAV/MP3s matches slides; lengths reasonable.
- **Video:** play output MP4; check sync and transitions.
