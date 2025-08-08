Paper2Video V2 — Minimal Setup & Demos

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
