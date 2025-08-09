# Paper2Video V2 — Paper Summarization

This project provides tools to process and summarize academic papers for various downstream tasks.

## Summarization Script: `summarize_chunks.py`

This is a powerful Python script designed to process academic papers in Markdown format. It intelligently chunks the text and generates detailed, structured JSON summaries for each chunk using large language models via the OpenRouter API.

### Key Features

*   **Intelligent Chunking**: Splits long Markdown files into smaller, overlapping chunks based on token count, ensuring context is preserved across chunk boundaries.
*   **Robust Summarization ("Reliable Recipe")**: Implements a sophisticated two-step process to generate high-quality, long-form summaries, even with models that tend to be overly concise:
    1.  **Gist Generation**: First, it generates a detailed, ~400-word summary (the "gist") for a text chunk, with an automatic retry mechanism if the initial summary is too short.
    2.  **Structured Data Extraction**: It then uses the generated gist to extract structured data—such as key claims, figure references, and equations—into a clean JSON object. This step is isolated from the original text to ensure the model focuses only on extraction.
*   **Strict Verification**: The script programmatically asserts that the model has copied the long-form gist exactly into the final JSON, preventing the model from "helpfully" shortening it.
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
*   `--model`: Specify a model slug from OpenRouter (e.g., `openai/gpt-4o`, `anthropic/claude-3-sonnet-20240229`).
*   `--force`: Force re-summarization of all chunks, even if a summary file already exists.
*   `--verbose`: Enable verbose logging to see detailed progress and potential warnings.

Overview
- Deterministic pipeline (OCR → Markdown → slides/audio/video) plus an agentic layer using OpenAI-style tools via OpenRouter.

Prereqs
- Python 3.11+ (tested on 3.12)
- Virtualenv recommended
- .env with keys (copy `.env.example`):
  - MISTRAL_API_KEY=...
  - OPENROUTER_API_KEY=...

Setup
1) Create venv and install deps
   - python -m venv .venv && . .venv/bin/activate
   - python -m pip install -U pip
   - python -m pip install -r requirements.txt

2) Optional: run OCR to (re)generate page-wise Markdown for a PDF
   - python mistral_ocr.py --pdf test_paper.pdf --out mistral_responses/test_paper
   - Output: markdown under mistral_responses/test_paper/markdown/ and figure crops under images/

Search Index (local, deterministic)
1) Build FAISS index from page-wise Markdown
   - python index/index_paper.py --paper-id test_paper \
       --markdown-dir mistral_responses/test_paper/markdown --index-dir index
   - Artifacts: index/test_paper.faiss, index/test_paper.meta.jsonl, index/embedder.json

2) Sanity check (no LLM)
   - TMPDIR=./tmp python -c "from index.search_index import search_chunks; print(search_chunks('test_paper','sparsity expression',5))"

Tool-Call Demos (OpenRouter)
1) write_slide demo (forces one tool call)
   - python demo_tool_call.py --seed 7
   - Default model: z-ai/glm-4.5 (reasoning disabled). Use --autoselect-model or --model to override.

2) search demo (forces one tool call)
   - TMPDIR=./tmp python demo_search_tool_call.py "sparsity expression" --k 5 --paper-id test_paper

Notes & Tips
- If temp writes are restricted (e.g., PyTorch/sentence-transformers complain), set TMPDIR=./tmp as shown.
- If a model slug is rejected by the router, pass --model openai/gpt-4o-mini as a known tool-capable fallback.
- Slides written by tools are stored in slides.json.

Repo Layout (key files)
- index/index_paper.py: chunk markdown (≈800/100), embed (bge-small), build FAISS + meta
- index/search_index.py: search_chunks() + handle_search() → chunk IDs only
- demo_tool_call.py: single write_slide tool-call demo via OpenRouter
- demo_search_tool_call.py: single search tool-call demo via OpenRouter
- mistral_ocr.py: OCR → Markdown + figure crops for a PDF
- mistral_responses/test_paper/...: sample OCR outputs used for indexing
