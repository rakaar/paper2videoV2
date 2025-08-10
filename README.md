# Paper2Video V2 — Paper Summarization

This project provides tools to process and summarize academic papers for various downstream tasks.

## Summarization Script: `summarize_chunks.py`

This is a powerful Python script designed to process academic papers in Markdown format. It intelligently chunks the text and generates detailed, structured JSON summaries for each chunk using large language models via the OpenRouter API.

### Key Features

*   **Intelligent Chunking**: Splits long Markdown files into smaller, overlapping chunks based on token count, ensuring context is preserved across chunk boundaries.
*   **Robust Summarization ("Reliable Recipe")**: Implements a sophisticated two-step process to generate high-quality, long-form summaries, even with models that tend to be overly concise:
    1.  **Gist Generation**: First, it generates a detailed, ~400-word summary (the "gist") for a text chunk, with an automatic retry mechanism if the initial summary is too short.
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
- Use the `--model` argument to specify an OpenRouter model slug.
- Default: `openai/gpt-4o-mini`.
- The model can also be set via the `OPENROUTER_SUMMARIZE_MODEL` or `OPENROUTER_MODEL` environment variables.
- Example:
  ```bash
  python summarize_chunks.py \
    --pages-dir mistral_responses/test_paper/markdown \
    --outdir artifacts/test_paper \
    --model anthropic/claude-3-sonnet-3.5
  ```

## Slide Planner: `plan_slides.py`

Plans a sequence of slide sections from the chunk summaries. Instead of planning one slide at a time, it groups slides into logical sections, each with a title and a list of topics to be covered.

- **Inputs (from a single artifacts subdir):**
  - `chunk_summaries.jsonl`
  - `chunk_index.jsonl`
- **Output:**
  - `slide_plan.json`: Contains a list of slide sections, where each section has a `section_title`, a list of `slide_topics`, a detailed `plan`, and lists of `references` and `figures`.
- **Env:**
  - `OPENROUTER_API_KEY` must be set in `.env`
  - Optional `OPENROUTER_PLANNER_MODEL` (default: `deepseek/deepseek-chat`)
- **Model Override:** pass `--model` to override the default/env model.
- **Usage:**
  ```bash
  source .venv/bin/activate
  python plan_slides.py \
    --summaries-dir artifacts/test_paper \
    --outdir artifacts/test_paper \
    --verbose --force
  ```

## Slide Generator: `generate_slides.py`

Generates per-slide content JSON (Title, Content bullets, Audio narration, Figures) using the slide plan and source text.

- **Inputs:**
  - `--artifacts-dir` containing: `slide_plan.json`, `chunk_summaries.jsonl`, `chunk_index.jsonl`
  - OCR Markdown under `--ocr-dir/<pdf-name>/markdown/*.md` (used to pull exact chunk text spans)
- **Output:**
  - `presentation.json`
- **Env:**
  - `OPENROUTER_API_KEY` must be set in `.env`
  - Optional `OPENROUTER_GENERATOR_MODEL` (default: `openai/gpt-4o`)
- **Model Override:** pass `--model` to override the default/env model.
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

## End-to-End Pipeline: `paper_to_video.py`

This script orchestrates the full pipeline, running summarization, slide planning, and slide generation in a single command.

### Prerequisites

1.  **Activate Virtual Environment**: Before running the script, ensure you have activated your Python virtual environment.
    ```bash
    source .venv/bin/activate
    ```

2.  **API Keys**: Make sure you have a `.env` file in the project root containing your `OPENROUTER_API_KEY`. You can copy the `.env.example` file to create it.

### Usage

To run the entire pipeline for a paper, use the following command structure:

```bash
python3 paper_to_video.py --pdf-name <paper_name> [OPTIONS]
```

**Required Arguments:**
*   `--pdf-name`: The base name of the paper (e.g., `test_paper`). This is used to find the OCR output files and to name the artifact directories.

**Common Options:**
*   `--force`: Force re-processing for all steps, overwriting any existing artifacts.
*   `--verbose`: Enable detailed logging to monitor the script's progress.
*   `--summarize-model`: The OpenRouter model to use for summarization (default: `openai/gpt-4o-mini`).

### Example

To run the pipeline for the `test_paper` example, which is included in the repository:

```bash
# Ensure your venv is active and .env file is set up
.venv/bin/python3 paper_to_video.py --pdf-name test_paper --verbose --force
```

This command will create all artifacts in the `artifacts/test_paper/` directory, culminating in the final `presentation.json`.

---

## Individual Scripts

For more granular control, you can still run each script individually.

## Summarization Script: `summarize_chunks.py`
