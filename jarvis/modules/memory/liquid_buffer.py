"""Liquid Buffer — thread-safe time-based ring buffer for recent chunks.

Holds chunks from the last `liquid_buffer_duration_minutes` minutes (default
10, from config.yaml).  Eviction runs automatically on every insert call.
"""
import threading
import time
import uuid
from collections import deque
from typing import Optional

from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger


class LiquidBuffer:
    """Thread-safe in-memory ring buffer holding chunks for the last N minutes."""

    def __init__(self):
        ttl_minutes: float = (
            config.get("parameters", {}).get("liquid_buffer_duration_minutes", 10)
        )
        self._ttl_ms: int = int(ttl_minutes * 60 * 1000)
        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._insert_count = 0
        Logger.log(
            "INFO", "liquid_buffer",
            f"LiquidBuffer initialised — TTL {ttl_minutes} min ({self._ttl_ms} ms)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, chunk: dict) -> None:
        """Insert a chunk dict into the buffer.

        If chunk lacks a ``chunk_id`` it is assigned a fresh UUID4.
        Eviction of expired chunks runs automatically after each insert.
        """
        chunk = dict(chunk)  # shallow copy — do not mutate caller's dict
        if not chunk.get("chunk_id"):
            chunk["chunk_id"] = str(uuid.uuid4())
        # Record wall-clock insertion time for TTL tracking
        chunk.setdefault("_inserted_at_ms", int(time.time() * 1000))

        with self._lock:
            self._buffer.append(chunk)
            self._evict_expired()
            self._insert_count += 1
            if self._insert_count % 10 == 0:
                Logger.log(
                    "DEBUG", "liquid_buffer",
                    f"Buffer size after {self._insert_count} inserts: {len(self._buffer)}"
                )

    def get_recent(self, since_ms: int) -> list[dict]:
        """Return all chunks whose *timestamp_start* is >= since_ms."""
        with self._lock:
            return [
                c for c in self._buffer
                if c.get("timestamp_start", 0) >= since_ms
            ]

    def get_all(self) -> list[dict]:
        """Return a snapshot of the entire buffer (newest first)."""
        with self._lock:
            return list(self._buffer)

    def flush_before(self, cutoff_ms: int) -> list[dict]:
        """Remove and return all chunks whose *timestamp_start* < cutoff_ms.

        Intended to be called by MemoryManager to drain old chunks to the
        long-term vector store.
        """
        flushed: list[dict] = []
        with self._lock:
            remaining: deque = deque()
            for chunk in self._buffer:
                if chunk.get("timestamp_start", 0) < cutoff_ms:
                    flushed.append(chunk)
                else:
                    remaining.append(chunk)
            self._buffer = remaining
        Logger.log(
            "INFO", "liquid_buffer",
            f"flush_before({cutoff_ms}): flushed {len(flushed)} chunks, "
            f"{len(self._buffer)} remain"
        )
        return flushed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove chunks older than TTL. Caller must hold self._lock."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - self._ttl_ms
        # deque is ordered oldest→newest; pop from left while expired
        while self._buffer and self._buffer[0].get("_inserted_at_ms", 0) < cutoff:
            expired = self._buffer.popleft()
            Logger.log(
                "DEBUG", "liquid_buffer",
                f"Evicted expired chunk {expired.get('chunk_id', '?')}"
            )
