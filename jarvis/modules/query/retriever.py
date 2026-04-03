"""Retriever — fetches and ranks relevant memory chunks.

Pipeline:
  1. Embed query_text → query_vector (BGE, 384-dim)
  2. Search LiquidBuffer in-memory (time-filter + cosine sim)
  3. Search FAISS vector store (with time filters)
  4. Merge + deduplicate by chunk_id
  5. Re-rank by similarity_score descending
  6. Return top_k
"""
import numpy as np
from typing import Any

from jarvis.infra.logger import Logger
from jarvis.modules.memory.embedding_bge import BGEEmbeddingEngine
from jarvis.modules.memory.liquid_buffer import LiquidBuffer
from jarvis.modules.memory.vector_store_faiss import FAISSVectorStore


class Retriever:
    """Fetches relevant memory chunks from both LiquidBuffer and FAISS."""

    def __init__(
        self,
        embedding_engine: BGEEmbeddingEngine | None = None,
        liquid_buffer: LiquidBuffer | None = None,
        vector_store: FAISSVectorStore | None = None,
    ):
        # These are injected by the caller; Retriever can also be used
        # as a standalone stub (e.g. smoke test) where they stay None.
        self._embedding = embedding_engine
        self._buffer = liquid_buffer
        self._store = vector_store

    def retrieve(
        self,
        query_text: str,
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[dict]:
        """Return top_k most relevant chunks for query_text.

        Args:
            query_text: Raw query string (will be embedded internally).
            filters:    Time-range dict from QueryParser:
                        {"after_ms": int|None, "before_ms": int|None}
            top_k:      Number of results to return.

        Returns:
            list[dict], each chunk includes a 'similarity_score' key.
        """
        if self._embedding is None or self._store is None:
            Logger.log("INFO", "retriever", "[retriever] not wired — returning empty")
            return []

        # 1. Embed query
        query_vector: list[float] = self._embedding.embed(query_text)
        q_np = np.array(query_vector, dtype=np.float32)

        # Normalise FAISS-style filters
        faiss_filters: dict = {}
        after_ms = filters.get("after_ms") or filters.get("time_filter", {}) and None
        before_ms = filters.get("before_ms") or None
        # Support both flat and nested time_filter dicts
        if "time_filter" in filters and isinstance(filters["time_filter"], dict):
            after_ms = filters["time_filter"].get("after_ms")
            before_ms = filters["time_filter"].get("before_ms")
        else:
            after_ms = filters.get("after_ms")
            before_ms = filters.get("before_ms")

        if after_ms is not None:
            faiss_filters["after_ms"] = after_ms
        if before_ms is not None:
            faiss_filters["before_ms"] = before_ms

        seen_ids: set[str] = set()
        results: list[dict] = []

        # 2. Search LiquidBuffer (in-memory, time-filtered, cosine sim)
        if self._buffer is not None:
            since_ms = after_ms if after_ms is not None else 0
            liquid_chunks = self._buffer.get_recent(since_ms)
            Logger.log(
                "DEBUG", "retriever",
                f"LiquidBuffer: {len(liquid_chunks)} candidates"
            )
            for chunk in liquid_chunks:
                # Apply before_ms filter manually
                ts = chunk.get("timestamp_start", 0)
                if before_ms is not None and ts > before_ms:
                    continue

                text = chunk.get("chunk_text", chunk.get("text", ""))
                if not text:
                    continue
                chunk_vec = np.array(self._embedding.embed(text), dtype=np.float32)
                score = float(np.dot(q_np, chunk_vec))

                cid = chunk.get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    enriched = dict(chunk)
                    enriched["similarity_score"] = score
                    enriched["_source"] = "liquid_buffer"
                    results.append(enriched)

        # 3. Search FAISS
        faiss_results = self._store.search(
            query_vector=query_vector,
            filters=faiss_filters,
            top_k=top_k * 2,   # fetch more, we'll trim after merge
        )
        Logger.log(
            "DEBUG", "retriever",
            f"FAISS: {len(faiss_results)} candidates"
        )
        for chunk in faiss_results:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                enriched = dict(chunk)
                enriched["similarity_score"] = chunk.get("_score", 0.0)
                enriched["_source"] = "faiss"
                results.append(enriched)

        # 4. Re-rank by similarity_score descending
        results.sort(key=lambda c: c.get("similarity_score", 0.0), reverse=True)
        top = results[:top_k]

        Logger.log(
            "INFO", "retriever",
            f"retrieve('{query_text[:50]}') → "
            f"liquid={sum(1 for r in top if r.get('_source')=='liquid_buffer')} "
            f"faiss={sum(1 for r in top if r.get('_source')=='faiss')} "
            f"total={len(top)}"
        )
        return top
