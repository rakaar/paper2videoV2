import json
import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # defer import error until use

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # defer import error until use


INDEX_DIR = pathlib.Path("index")


def _load_index(paper_id: str):
    idx_path = INDEX_DIR / f"{paper_id}.faiss"
    meta_path = INDEX_DIR / f"{paper_id}.meta.jsonl"
    embinfo_path = INDEX_DIR / "embedder.json"

    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. Please `pip install faiss-cpu` (or faiss-gpu)."
        )

    if not idx_path.exists() or not meta_path.exists() or not embinfo_path.exists():
        raise FileNotFoundError(
            f"Missing index files for '{paper_id}'. Expected: {idx_path}, {meta_path}, {embinfo_path}"
        )

    index = faiss.read_index(str(idx_path))
    metas = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    embinfo = json.loads(embinfo_path.read_text())
    return index, metas, embinfo


_model_cache: Dict[str, Any] = {}


def _get_model(name: str):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Please `pip install sentence-transformers`."
        )
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def _embed(texts: List[str], model) -> np.ndarray:
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def search_chunks(paper_id: str, query: str, k: int = 5) -> List[str]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(k, int) or k < 1 or k > 10:
        raise ValueError("k must be an integer in [1, 10]")

    index, metas, embinfo = _load_index(paper_id)
    model = _get_model(embinfo.get("model", "BAAI/bge-small-en-v1.5"))
    q = _embed([query], model)
    D, I = index.search(q, k)
    ids = [metas[i]["id"] for i in I[0] if i >= 0]
    return ids


def handle_search(args: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
    ids = search_chunks(paper_id, args["query"], args.get("k", 5))
    return {"chunk_ids": ids}

