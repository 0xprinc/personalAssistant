"""Memory Manager — background thread that flushes LiquidBuffer to FAISS.

Wake interval: memory_flush_interval_seconds (config.yaml, default 60 s).
On each cycle:
  1. flush_before(now - 10 min) from LiquidBuffer
  2. embed each chunk via BGEEmbeddingEngine
  3. upsert each (chunk + vector) into FAISSVectorStore
  4. log count

flush_now() triggers an immediate flush cycle (used in tests).
"""
import threading
import time
from typing import Optional

from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger


class MemoryManager:
    """Orchestrates periodic flush from LiquidBuffer to long-term VectorStore."""

    def __init__(
        self,
        liquid_buffer=None,
        embedding_engine=None,
        vector_store=None,
    ):
        self._buffer = liquid_buffer
        self._embedding = embedding_engine
        self._store = vector_store

        self._interval: int = int(
            config.get("parameters", {}).get("memory_flush_interval_seconds", 60)
        )
        ttl_minutes: float = (
            config.get("parameters", {}).get("liquid_buffer_duration_minutes", 10)
        )
        self._ttl_ms: int = int(ttl_minutes * 60 * 1000)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background flush thread."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="MemoryManagerFlush", daemon=True
        )
        self._thread.start()
        Logger.log(
            "INFO", "memory_manager",
            f"Background flush thread started (interval={self._interval}s)"
        )

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        Logger.log("INFO", "memory_manager", "Background flush thread stopped")

    # ------------------------------------------------------------------
    # Flush API
    # ------------------------------------------------------------------

    def flush_now(self) -> int:
        """Manually trigger a flush cycle. Returns count of chunks flushed."""
        return self._flush_cycle()

    # Legacy alias for smoke test compatibility
    def flush(self) -> None:
        self.flush_now()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush_cycle()
            # Sleep in small increments so stop() is responsive
            for _ in range(self._interval * 10):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def _flush_cycle(self) -> int:
        if self._buffer is None or self._embedding is None or self._store is None:
            Logger.log("DEBUG", "memory_manager", "nothing to flush — managers not wired")
            return 0

        cutoff_ms = int(time.time() * 1000) - self._ttl_ms
        chunks = self._buffer.flush_before(cutoff_ms)

        if not chunks:
            Logger.log("DEBUG", "memory_manager", "nothing to flush — buffer empty or not expired")
            return 0

        Logger.log(
            "INFO", "memory_manager",
            f"Flushing {len(chunks)} chunks to vector store …"
        )

        # Batch embed for efficiency
        texts = [c.get("chunk_text", c.get("text", "")) for c in chunks]
        vectors = self._embedding.embed_batch(texts)

        flushed = 0
        for chunk, vector in zip(chunks, vectors):
            mem_chunk = {
                "chunk_id":       chunk.get("chunk_id", ""),
                "text":           chunk.get("chunk_text", chunk.get("text", "")),
                "vector":         vector,
                "timestamp_start": chunk.get("timestamp_start", 0),
                "timestamp_end":   chunk.get("timestamp_end", 0),
                "device_id":      chunk.get("device_id", "laptop"),
                "session_id":     chunk.get("session_id", ""),
                "confidence":     chunk.get("confidence", 1.0),
                "redacted":       chunk.get("redacted", False),
            }
            self._store.upsert(mem_chunk)
            flushed += 1

        Logger.log(
            "INFO", "memory_manager",
            f"Flush complete — embedded and stored {flushed} chunks"
        )
        return flushed
