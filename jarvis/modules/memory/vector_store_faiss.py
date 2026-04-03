"""FAISS Vector Store — IndexFlatIP with persistence.

Uses faiss.IndexFlatIP (inner-product, works as cosine sim with unit vectors).
Metadata is stored in a Python dict and persisted as a .pkl file alongside
the FAISS .index file.

On init:  loads existing index + metadata from disk if available.
On upsert: appends vectors, persists both files immediately.
On search: retrieves top_k*10 from FAISS, post-filters by time, returns top_k.
"""
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from jarvis.interfaces.vector_store import VectorStoreABC, MemoryChunk
from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger

_DIMS = 384


def _resolve_paths() -> tuple[Path, Path]:
    """Return (index_path, metadata_path) from config."""
    raw_path: str = (
        config.get("storage", {}).get("vector_store_path", "data/faiss_index.bin")
    )
    index_path = Path(raw_path)
    metadata_path = index_path.with_suffix(".pkl")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return index_path, metadata_path


class FAISSVectorStore(VectorStoreABC):
    """Persistent FAISS vector store using IndexFlatIP (cosine sim)."""

    def __init__(self):
        self._index_path, self._meta_path = _resolve_paths()
        self._metadata: dict[int, dict] = {}  # faiss_id → metadata dict

        # Load or create index
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            Logger.log(
                "INFO", "vector_store_faiss",
                f"Loaded existing FAISS index: {self._index.ntotal} vectors "
                f"from {self._index_path}"
            )
        else:
            self._index = faiss.IndexFlatIP(_DIMS)
            Logger.log(
                "INFO", "vector_store_faiss",
                f"Created new FAISS IndexFlatIP({_DIMS})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, chunk: MemoryChunk) -> str:
        """Add a chunk + its vector to the index and persist to disk.

        The chunk dict must contain a 'vector' key with a 384-dim list[float].
        Returns the chunk_id (assigns a UUID4 if not already set).
        """
        chunk = dict(chunk)  # shallow copy
        chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
        chunk["chunk_id"] = chunk_id

        vector = chunk.get("vector")
        if vector is None:
            raise ValueError("upsert() requires chunk to contain a 'vector' key")

        vec_np = np.array([vector], dtype=np.float32)  # shape (1, 384)

        faiss_id = int(self._index.ntotal)
        self._index.add(vec_np)

        # Store metadata (everything except the raw vector)
        meta = {k: v for k, v in chunk.items() if k != "vector"}
        self._metadata[faiss_id] = meta

        self._persist()
        Logger.log(
            "INFO", "vector_store_faiss",
            f"Upserted chunk {chunk_id} → faiss_id {faiss_id} | "
            f"Total index size: {self._index.ntotal}"
        )
        return chunk_id

    def upsert_batch(self, chunks: list[MemoryChunk]) -> list[str]:
        """Batch upsert — more efficient for flush operations."""
        if not chunks:
            return []

        chunk_ids: list[str] = []
        vectors: list[list[float]] = []

        for chunk in chunks:
            chunk = dict(chunk)
            cid = chunk.get("chunk_id") or str(uuid.uuid4())
            chunk["chunk_id"] = cid
            chunk_ids.append(cid)

            vector = chunk.get("vector")
            if vector is None:
                raise ValueError(f"Chunk {cid} is missing 'vector'")
            vectors.append(vector)

            faiss_id = int(self._index.ntotal) + len(vectors) - 1
            meta = {k: v for k, v in chunk.items() if k != "vector"}
            self._metadata[faiss_id] = meta

        vecs_np = np.array(vectors, dtype=np.float32)
        self._index.add(vecs_np)

        self._persist()
        Logger.log(
            "INFO", "vector_store_faiss",
            f"Batch upsert: {len(chunks)} chunks | Total index: {self._index.ntotal}"
        )
        return chunk_ids

    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[MemoryChunk]:
        """Return top_k chunks most similar to query_vector.

        Optional time-range filters: {"after_ms": int, "before_ms": int}.
        Fetches top_k*10 candidates from FAISS, post-filters by timestamp,
        then returns the top_k results.
        """
        if self._index.ntotal == 0:
            Logger.log("INFO", "vector_store_faiss", "search() — index is empty")
            return []

        fetch_k = min(top_k * 10, self._index.ntotal)
        q_np = np.array([query_vector], dtype=np.float32)
        scores, faiss_ids = self._index.search(q_np, fetch_k)

        after_ms: int = filters.get("after_ms", 0)
        before_ms: int = filters.get("before_ms", int(time.time() * 1000) + 1)

        results: list[MemoryChunk] = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid < 0:
                continue  # FAISS returns -1 for padded results
            meta = self._metadata.get(int(fid))
            if meta is None:
                continue
            ts = meta.get("timestamp_start", 0)
            if after_ms <= ts <= before_ms:
                chunk: MemoryChunk = dict(meta)
                chunk["_score"] = float(score)
                results.append(chunk)
            if len(results) >= top_k:
                break

        Logger.log(
            "DEBUG", "vector_store_faiss",
            f"search() → {len(results)} results (fetched {fetch_k} candidates)"
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write FAISS index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._metadata, f)
        Logger.log(
            "DEBUG", "vector_store_faiss",
            f"Persisted index ({self._index.ntotal} vectors) to {self._index_path}"
        )
