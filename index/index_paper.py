import argparse
import json
import pathlib
import re
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np


def strip_markdown(md: str) -> str:
    # Remove code fences
    md = re.sub(r"```[\s\S]*?```", " ", md)
    # Remove inline code
    md = re.sub(r"`[^`]*`", " ", md)
    # Remove images ![alt](url)
    md = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", md)
    # Remove links [text](url) -> text
    md = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", md)
    # Remove footnote refs like [^1]
    md = re.sub(r"\[\^\d+\]", " ", md)
    # Remove HTML tags
    md = re.sub(r"<[^>]+>", " ", md)
    # Collapse multiple spaces/newlines
    md = re.sub(r"\s+", " ", md).strip()
    return md


def chunk_words(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def load_pages(markdown_dir: pathlib.Path) -> List[Tuple[int, str]]:
    files = sorted(markdown_dir.glob("*.md"))
    pages: List[Tuple[int, str]] = []
    for f in files:
        # Expect pattern like test_paper_page_01.md
        m = re.search(r"_page_(\d+)\.md$", f.name)
        page_no = int(m.group(1)) if m else len(pages) + 1
        pages.append((page_no, f.read_text(encoding="utf-8", errors="ignore")))
    pages.sort(key=lambda x: x[0])
    return pages


def build_index(
    paper_id: str,
    markdown_dir: pathlib.Path,
    index_dir: pathlib.Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 800,
    overlap: int = 100,
):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss is not installed. Please `pip install faiss-cpu` (or faiss-gpu)."
        ) from e

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Please `pip install sentence-transformers`."
        ) from e

    index_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pages(markdown_dir)
    metas: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []

    for page_no, md in pages:
        text = strip_markdown(md)
        chunks = chunk_words(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks, start=1):
            cid = f"{paper_id}:p{page_no}:c{i}"
            metas.append({
                "id": cid,
                "paper_id": paper_id,
                "page": page_no,
                "chunk_index": i,
                "text_len": len(ch),
            })
            chunk_texts.append(ch)

    if not chunk_texts:
        raise RuntimeError("No chunk texts produced from markdown. Check input directory.")

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    vecs = np.asarray(vecs, dtype="float32")
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Write artifacts
    with open(index_dir / f"{paper_id}.meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(index_dir / "embedder.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dim": dim,
                "normalize": True,
                "metric": "ip",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "tokenizer": "word-approx",
            },
            f,
            indent=2,
        )

    faiss.write_index(index, str(index_dir / f"{paper_id}.faiss"))


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index for a paper's markdown")
    ap.add_argument("--paper-id", default="test_paper")
    ap.add_argument(
        "--markdown-dir",
        default="mistral_responses/test_paper/markdown",
        help="Directory with page-wise markdown files",
    )
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    build_index(
        paper_id=args.paper_id,
        markdown_dir=pathlib.Path(args.markdown_dir),
        index_dir=pathlib.Path(args.index_dir),
        model_name=args.model,
        chunk_size=args.chunk,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()

