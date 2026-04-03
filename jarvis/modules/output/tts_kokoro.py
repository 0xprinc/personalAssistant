"""Kokoro TTS Engine — KPipeline-based speech synthesis.

Lazy-loads on first synthesise() call. Auto-unloads after 30s idle to free
1.8 GB RAM. All settings from config.yaml.
"""
import gc
import threading
import time
from typing import Optional

import numpy as np

from jarvis.interfaces.tts import TTSEngineABC
from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger


class KokoroTTS(TTSEngineABC):
    """Real Kokoro 82M TTS engine with lazy load and auto-unload."""

    SAMPLE_RATE = 24_000  # Kokoro always outputs 24 kHz

    def __init__(self):
        self._pipeline = None       # lazy-initialised
        self._lock = threading.Lock()
        self._idle_timer: Optional[threading.Timer] = None

        # Config
        self._voice: str = config.get("tts", {}).get("voice", "af_bella")
        self._lang:  str = config.get("tts", {}).get("lang_code", "a")
        self._idle_timeout: int = int(
            config.get("parameters", {}).get("tts_idle_timeout_seconds", 30)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(self, text: str) -> np.ndarray:
        """Convert text → float32 numpy array at 24 kHz.

        Returns an empty array (shape=(0,)) for blank input.
        """
        if not text or not text.strip():
            Logger.log("WARNING", "tts_kokoro", "synthesise() called with empty text — skipping")
            return np.array([], dtype=np.float32)

        self._reset_idle_timer()

        with self._lock:
            self._ensure_loaded()
            t0 = time.perf_counter()

            chunks: list[np.ndarray] = []
            # KPipeline is a generator: yields (gs, ps, audio_np) per sentence
            for _, _, audio in self._pipeline(text, voice=self._voice):
                if audio is not None and len(audio) > 0:
                    chunks.append(audio)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if not chunks:
                Logger.log("WARNING", "tts_kokoro", "KPipeline returned no audio chunks")
                return np.array([], dtype=np.float32)

            audio_full = np.concatenate(chunks).astype(np.float32)
            duration_s = len(audio_full) / self.SAMPLE_RATE
            Logger.log(
                "INFO", "tts_kokoro",
                f"Synthesised {len(text)} chars → {duration_s:.2f}s audio in {elapsed_ms:.0f} ms"
            )
            return audio_full

    def unload(self) -> None:
        """Release the Kokoro pipeline and free RAM."""
        with self._lock:
            if self._pipeline is not None:
                self._pipeline = None
                gc.collect()
                Logger.log("INFO", "tts_kokoro", "TTS unloaded, RAM released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load KPipeline if not already initialised. Caller must hold lock."""
        if self._pipeline is not None:
            return
        from kokoro import KPipeline  # deferred import — keeps RAM free at startup
        t0 = time.perf_counter()
        Logger.log("INFO", "tts_kokoro", f"Loading Kokoro (lang={self._lang}, voice={self._voice}) …")
        # Force CPU — Kokoro TTS must NOT use MPS on M1 (unstable)
        self._pipeline = KPipeline(lang_code=self._lang, device="cpu")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        Logger.log("INFO", "tts_kokoro", f"TTS loaded in {elapsed_ms:.0f} ms")

    def _reset_idle_timer(self) -> None:
        """Cancel any pending idle timer and start a fresh 30-second countdown."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self._idle_timeout, self._on_idle_timeout)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle_timeout(self) -> None:
        Logger.log(
            "INFO", "tts_kokoro",
            f"TTS idle for {self._idle_timeout}s — auto-unloading …"
        )
        self.unload()
