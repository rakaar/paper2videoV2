# Paper2Video V2 — Paper Summarization

This project provides tools to process and summarize academic papers for various downstream tasks.

## Summarization Script: `summarize_chunks.py`

This is a powerful Python script designed to process academic papers in Markdown format. It intelligently chunks the text and generates detailed, structured JSON summaries for each chunk using large language models via the OpenRouter API.

### Key Features

*   **Intelligent Chunking**: Splits long Markdown files into smaller, overlapping chunks based on token count, ensuring context is preserved across chunk boundaries.
*   **Robust Summarization ("Reliable Recipe")**: Implements a sophisticated two-step process to generate high-quality summaries efficiently:
    1.  **Gist Generation**: First, it generates a concise, ~120–180 token summary (the "gist") for a text chunk, with an automatic retry mechanism if the initial summary is too short.
    2.  **Structured Data Extraction**: It then uses the generated gist to extract structured data—such as key claims, figure references, and equations—into a clean JSON object. This step is isolated from the original text to ensure the model focuses only on extraction.
*   **Validation**: Ensures the long-form gist is preserved in the final JSON; tolerates minor formatting differences by normalizing whitespace and replacing with the provided gist when needed. Accepts flexible figure references and coerces fields to the correct types.
*   **Model Flexibility**: Supports any model available through the OpenRouter API, allowing users to balance cost, speed, and quality by specifying a model at runtime.
*   **Structured Output**: Produces a `chunk_summaries.jsonl` file containing one JSON object per chunk, with fields like `id`, `page`, `gist`, `claims`, `figs`, `eqs`, `key_terms`, and `anchors`.

### Setup

1.  **Environment**: The script requires Python 3 and a virtual environment is recommended.

2.  **API Key**: Create a `.env` file in the root of the project and add your OpenRouter API key:
    ```
    OPENROUTER_API_KEY='your_api_key_here'
    ```

3.  **Dependencies**: While a `requirements.txt` is not provided, the script depends on the following packages:
    ```bash
    pip install python-dotenv httpx tqdm tiktoken
    ```

### Usage

To run the script, use the following command structure:

```bash
source .venv/bin/activate
python summarize_chunks.py --pages-dir <path_to_markdown_files> [OPTIONS]
```

**Required Arguments:**
*   `--pages-dir`: The path to the directory containing the page-wise markdown files (e.g., `page_001.md`, `page_002.md`).

**Common Options:**
*   `--force`: Force re-summarization of all chunks, even if a summary file already exists.
*   `--verbose`: Enable verbose logging to see detailed progress and potential warnings.
*   `--chunk-size`: Chunk size in tokens (default: 1024)
*   `--overlap`: Token overlap between chunks (default: 128)

**Model Selection:**
- Use the `--model` argument to specify an OpenRouter model slug, or a comma-separated list for fallback.
- Default: `meta-llama/llama-3.2-3b-instruct,google/gemma-2-9b-it:free`.
- The model can also be set via `OPENROUTER_SUMMARIZE_MODEL`, `OPENROUTER_EXTRACTOR_MODEL`, or `OPENROUTER_MODEL` environment variables.
- Token cap: `--max-tokens` (default: 180) controls response cost.
- Example:
  ```bash
  python summarize_chunks.py \
    --pages-dir mistral_responses/test_paper/markdown \
    --outdir artifacts/test_paper \
    --model meta-llama/llama-3.2-3b-instruct,google/gemma-2-9b-it:free \
    --max-tokens 180
  ```

## Slide Planner: `plan_slides.py`

Plans a sequence of slide sections from the chunk summaries. Instead of planning one slide at a time, it groups slides into logical sections, each with a title and a list of topics to be covered.

- **Prerequisite:** Run `make_paper_card.py` first to create a governance card (`paper_card.json`).
- **Inputs (from a single artifacts subdir):**
  - `chunk_summaries.jsonl`
  - `chunk_index.jsonl`
  - `paper_card.json` (from the pre-pass)
- **Output:**
  - `slide_plan.json`: A list of slide sections with fields `section_title`, `slide_topics`, `plan`, `learning_objective`, `references` (≥2 chunk IDs per section), and `figures` (filtered from referenced chunks only).
- **Env:**
  - `OPENROUTER_API_KEY` must be set in `.env`
  - Optional `OPENROUTER_PLANNER_MODEL` (default: `qwen/qwen-2.5-7b-instruct:free,mistralai/mixtral-8x7b-instruct`)
- **Model Override:** pass `--model` to override the default/env model.
- **Token Cap:** `--max-tokens` (default: 120)
- **Usage:**
  ```bash
  source .venv/bin/activate
  python plan_slides.py \
    --summaries-dir artifacts/test_paper \
    --outdir artifacts/test_paper \
    --verbose --force
  ```

Notes:
- The planner follows the canonical section order from the Paper Card (default: `Overview → Method → Results → Discussion → Limitations → Conclusion`).
- It enforces evidence: each section must include at least 2 `references` to chunk IDs.
- Figure IDs are validated against figures mentioned in the referenced chunks; unrelated figures are dropped.
- Adds `learning_objective` per section (1–2 sentences).

## Slide Generator: `generate_slides.py`

Generates per-slide content JSON (Title, Content bullets, Audio narration, Figures) using the slide plan and source text.

- **Inputs:**
  - `--artifacts-dir` containing: `paper_card.json`, `slide_plan.json`, `chunk_summaries.jsonl`, `chunk_index.jsonl`
  - OCR Markdown under `--ocr-dir/<pdf-name>/markdown/*.md` (used to pull exact chunk text spans)
- **Output:**
  - `presentation.json`
- **Env:**
  - `OPENROUTER_API_KEY` must be set in `.env`
  - Optional `OPENROUTER_GENERATOR_MODEL` (default: `mistralai/mistral-small-24b-instruct-2501,meta-llama/llama-3.2-3b-instruct`)
- **Model Override:** pass `--model` to override the default/env model.
- **Token Cap:** `--max-tokens` (default: 360)
- **Figure Reuse:** `--figure-reuse-limit` (default: -1 for unlimited reuse across the deck)
- **Usage (Option A: keep outputs isolated per paper):**
  ```bash
  source .venv/bin/activate
  python generate_slides.py \
    --ocr-dir mistral_responses \
    --pdf-name test_paper \
    --artifacts-dir artifacts/test_paper \
    --outdir artifacts/test_paper \
    --verbose --force
  ```
- **Usage (Option B: write to repo-level artifacts/presentation.json):**
  ```bash
  source .venv/bin/activate
  python generate_slides.py \
    --ocr-dir mistral_responses \
    --pdf-name test_paper \
    --artifacts-dir artifacts/test_paper \
    --outdir artifacts \
    --verbose --force
  ```

Notes:
- Figures are selected from the union of figures in the slide's `references`; planner suggestions are respected only if they intersect with referenced figures.
- Figure reuse is unlimited by default; you can cap it with `--figure-reuse-limit`.
- If no figures are attached to a slide, the generator explicitly instructs the model not to mention figures to avoid mismatches.
- The generator enforces JSON-only output, with fallback to non-JSON mode and light repairs to handle model quirks (empty content or minor JSON issues).
- Slide prompt enforces narrative continuity via `WhyThisSlide`, `BridgeFromPrevious`, and `BridgeToNext` fields.
- De-duplication: the generator tracks previously used `claims` and prefers novel claims for each slide.
- Context discipline: only the last 2 slide summaries plus compact checkpoint notes are passed to the LLM to avoid drift.

### Model reliability (generator)
- Works well: `mistralai/mistral-small-24b-instruct-2501` (consistent JSON with `response_format=json_object`).
- Caveats: `openai/gpt-oss-120b` sometimes returns empty content under JSON mode; the generator now falls back to non-JSON mode and repairs responses, but you may still prefer Mistral for reliability/cost.
- Note: slugs with `:free` can 404 on some accounts. Prefer non-`:free` slugs or verify availability.

---

## Paper Card Pre-pass: `make_paper_card.py`

Creates a governance card (`paper_card.json`) from the earliest and latest chunk summaries and figure captions to guide planning.

- **Input:**
  - `chunk_summaries.jsonl` (in an artifacts subdir)
- **Output:**
  - `paper_card.json` with keys: `tldr`, `contributions`, `method_oneliner`, `key_results`, `limitations`, and `section_order` (canonical deck order)
- **Env:**
  - `OPENROUTER_API_KEY` must be set in `.env`
  - Optional `OPENROUTER_CARD_MODEL` (default: `mistralai/mistral-small-24b-instruct-2501:free,meta-llama/llama-3.2-3b-instruct`)
- **Usage:**
  ```bash
  source .venv/bin/activate
  python make_paper_card.py \
    --artifacts-dir artifacts/test_paper \
    --outdir artifacts/test_paper \
    --verbose --force
  ```

## Audio Generation: `generate_audio.py`

Generates audio files from `presentation.json` using the Sarvam TTS API. This script includes a robust pipeline to handle texts that exceed the API's character limit:

1.  **Sentence Splitting**: The script first splits the slide's narration text into individual sentences.
2.  **Per-Sentence Audio Generation**: To work around potential API bugs with specific sentence combinations, each sentence is sent to the Sarvam API as a separate request to generate an audio chunk.
3.  **Concatenation**: The individual audio chunks for each sentence are then seamlessly concatenated into a single, complete WAV file for the slide using `ffmpeg`.

## Slide Rendering: `generate_slide_pngs.py`

Renders a PNG deck from a `presentation.json` file using Marp. This script reads the slide titles and content and creates `deck.XXX.png` images in `artifacts/<paper>/pngs/`.

- **Inputs:**
  - `--presentation-file` path to the JSON deck (from `generate_slides.py`)
  - `--output-dir` where to write `deck.md` and PNGs
  - `--paper-name` used to resolve figure paths
- **Cover Slide Tip:** If your `presentation.json` already contains a cover slide (see below), pass `--no-cover` to avoid adding another cover during rendering.
- **OCR Dir:** If you want the renderer to build a cover from OCR (not recommended when the generator already added one), pass `--ocr-dir` and omit `--no-cover`.

Example:

```bash
python generate_slide_pngs.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper \
  --paper-name test_paper \
  --no-cover
```

Notes:
- We fixed an empty-first-slide issue by not inserting a leading slide separator in the generated Marp markdown.
- Headings are sanitized to avoid stray Markdown `#` prefixes in titles.

## Cover Slide (Title + Audio) in `generate_slides.py`

You can ask the generator to prepend a cover slide that introduces the paper with title, authors, and narration (`Audio`).

- **Flags:**
  - `--add-cover` to enable cover generation
  - `--cover-model` to select the OpenRouter model for extracting title/authors from OCR page 01
- **Env:**
  - `OPENROUTER_API_KEY` must be set
  - Optional `OPENROUTER_COVER_MODEL` can override the default cover model
- **How it works:**
  - Extracts title/authors from `mistral_responses/<paper>/markdown/<paper>_page_01.md` using an LLM (OpenRouter JSON mode) with a heuristic fallback
  - Prepends a cover slide JSON with `Title`, `Content` (bullets), and `Audio` (spoken intro)
  - Seeds previous-slide context so slide 2 bridges naturally from the cover
- **Renderer tip:** When cover is generated here, pass `--no-cover` to `generate_slide_pngs.py` to avoid a duplicate cover

Example:

```bash
python generate_slides.py \
  --ocr-dir mistral_responses \
  --pdf-name test_paper \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --add-cover \
  --model mistralai/mistral-small-24b-instruct-2501 \
  --max-tokens 600 \
  --force
```

## Video Stitching: `stitch_video.py`

Combine rendered PNG slides (`deck.XXX.png`) and generated WAV files (`slide_XXX.wav`) into a single MP4 using `ffmpeg`.

- **Assumptions:**
  - PNGs: `artifacts/<paper>/pngs/deck.001.png`, `deck.002.png`, ...
  - Audio: `artifacts/<paper>/audio/slide_001.wav`, `slide_002.wav`, ...
  - Pairs matched by index; the script uses the intersection of indices found in both folders
- **Usage:**
  ```bash
  python stitch_video.py --paper-name test_paper
  # or explicit directories:
  python stitch_video.py \
    --png-dir artifacts/test_paper/pngs \
    --audio-dir artifacts/test_paper/audio \
    --output artifacts/test_paper/video.mp4
  ```
- **Troubleshooting (stale files):** If you previously generated audio without a cover and now added a cover, the first WAV may belong to the old first content slide. Regenerate audio to match the current `presentation.json`, or remove extra WAV/PNG files beyond your slide count.

## End-to-End Quickstart

```bash
# 1) Plan slides (after make_paper_card.py)
python plan_slides.py --summaries-dir artifacts/test_paper --outdir artifacts/test_paper --verbose --force

# 2) Generate slides with cover and reliable model
python generate_slides.py \
  --ocr-dir mistral_responses \
  --pdf-name test_paper \
  --artifacts-dir artifacts/test_paper \
  --outdir artifacts/test_paper \
  --add-cover \
  --model mistralai/mistral-small-24b-instruct-2501 \
  --max-tokens 600 \
  --force

# 3) Render PNGs (no extra cover)
python generate_slide_pngs.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper \
  --paper-name test_paper \
  --no-cover

# 4) Generate audio WAVs (requires SARVAM_API_KEY)
python generate_audio.py \
  --presentation-file artifacts/test_paper/presentation.json \
  --output-dir artifacts/test_paper/audio \
  --paper-name test_paper

# 5) Stitch to MP4
python stitch_video.py --paper-name test_paper
```

Tips:
- If you change the deck, regenerate audio to avoid stale WAVs.
- When a cover is added in the generator, always pass `--no-cover` to the renderer.

## Model Recommendations (as observed today)

- **Generator (stable):** `mistralai/mistral-small-24b-instruct-2501`
  - Consistently returns valid JSON with `response_format={"type":"json_object"}`
  - Use `--max-tokens 600` for longer slides
- **Generator (budget/backup):** `meta-llama/llama-3.2-3b-instruct`
  - Lower cost; acceptable for drafts, though JSON reliability can vary
- **Cover Extractor:** Same as generator by default; can override with `--cover-model` or `OPENROUTER_COVER_MODEL`
- Avoid `:free` suffixes unless verified available for your account
