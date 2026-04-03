"""BGE Embedding Engine — BAAI/bge-small-en-v1.5 via sentence-transformers.

Loads the model once as a module-level singleton.  All embeddings are
L2-normalised so that inner-product == cosine similarity, which is what
FAISS IndexFlatIP expects.
"""
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from jarvis.interfaces.embedding import EmbeddingEngineABC
from jarvis.infra.logger import Logger

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused for every call
# ---------------------------------------------------------------------------
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_t0 = time.perf_counter()
Logger.log("INFO", "embedding_bge", f"Loading embedding model '{_MODEL_NAME}' …")
_model: SentenceTransformer = SentenceTransformer(_MODEL_NAME)
_load_ms = (time.perf_counter() - _t0) * 1000
Logger.log("INFO", "embedding_bge", f"Model loaded in {_load_ms:.0f} ms")


def _normalise(vec: np.ndarray) -> np.ndarray:
    """Normalise to unit length (required for BGE cosine similarity)."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


class BGEEmbeddingEngine(EmbeddingEngineABC):
    """Real embedding engine using BGE-small-en-v1.5 (384 dimensions)."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string → unit-normalised float[384]."""
        t0 = time.perf_counter()
        raw: np.ndarray = _model.encode(text, normalize_embeddings=False)
        vec = _normalise(raw).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        Logger.log(
            "DEBUG", "embedding_bge",
            f"embed() — {len(text)} chars → {elapsed_ms:.1f} ms"
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts → list of unit-normalised float[384] vectors.

        More efficient than calling embed() in a loop during flush operations.
        """
        if not texts:
            return []
        t0 = time.perf_counter()
        raw: np.ndarray = _model.encode(texts, normalize_embeddings=False)
        vecs = np.array([_normalise(v) for v in raw], dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        Logger.log(
            "DEBUG", "embedding_bge",
            f"embed_batch({len(texts)} texts) — {elapsed_ms:.1f} ms"
        )
        return vecs.tolist()
