"""Moonshine STT module for Jarvis — Milestone 1.3.

Implements the STTEngineABC interface using the Moonshine v2 Tiny model
via the HuggingFace Transformers library (UsefulSensors/moonshine-tiny).

Backend priority:
  1. HuggingFace Transformers (torch) — primary, runs on MPS (Apple Silicon GPU)
     or CPU. Uses UsefulSensors/moonshine-tiny from HF hub.
  2. MLX runtime — if available (legacy path, kept for compatibility).
  3. Exits with helpful message if neither is available.

Input: SpeechSegment dict {'pcm_data': bytes (int16), 'start_ms': int}
       or raw int16 PCM bytes.
Output: TranscriptResult {text, start_ms, end_ms, confidence}
"""

import sys
import time
from typing import Optional

import numpy as np

from jarvis.infra.logger import Logger
from jarvis.interfaces.stt import STTEngineABC, TranscriptResult

# HuggingFace model identifier for Moonshine Tiny
HF_MODEL_ID = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000
MAX_SECONDS = 30


class MoonshineSTT(STTEngineABC):
    """Moonshine STT engine using HuggingFace Transformers.

    Loads from UsefulSensors/moonshine-tiny on first init (cached to
    ~/.cache/huggingface after download). Falls back gracefully if a
    package is missing and exits with clear instructions if nothing works.
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self._backend: str = "none"
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Try HuggingFace Transformers, then MLX, then exit."""

        # --- Primary: HuggingFace Transformers (torch) ---
        try:
            import torch
            from transformers import AutoProcessor, MoonshineForConditionalGeneration

            # Apple Silicon: prefer MPS, else CPU
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            Logger.log(
                "INFO", "stt_moonshine",
                f"Loading Moonshine Tiny from HuggingFace ({HF_MODEL_ID}) "
                f"on device={self._device} …"
            )
            self._processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
            self._model = MoonshineForConditionalGeneration.from_pretrained(
                HF_MODEL_ID
            ).to(self._device)
            self._model.eval()
            self._backend = "transformers"
            Logger.log(
                "INFO", "stt_moonshine",
                f"Moonshine Tiny loaded via HuggingFace Transformers "
                f"(device={self._device})"
            )
            return

        except ImportError as exc:
            Logger.log(
                "WARNING", "stt_moonshine",
                f"HuggingFace Transformers not available ({exc}). "
                "Trying MLX fallback …"
            )
        except Exception as exc:
            Logger.log(
                "WARNING", "stt_moonshine",
                f"HuggingFace load failed: {exc}. Trying MLX fallback …"
            )

        # --- Fallback: MLX (Apple Silicon only, legacy) ---
        try:
            import moonshine as moonshine_mlx  # type: ignore
            self._model = moonshine_mlx.load_model("moonshine/tiny")
            self._backend = "mlx"
            Logger.log("INFO", "stt_moonshine", "Moonshine loaded via MLX backend")
            return
        except ImportError:
            pass
        except Exception as exc:
            Logger.log("WARNING", "stt_moonshine", f"MLX load failed: {exc}")

        # --- Nothing worked ---
        Logger.log(
            "ERROR", "stt_moonshine",
            "Could not load Moonshine STT. Please install:\n"
            "  pip install transformers torch\n"
            "Internet is required for the first run to download the model "
            "(cached afterwards at ~/.cache/huggingface)."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio_chunk) -> TranscriptResult:
        """Transcribe a SpeechSegment dict or raw int16 PCM bytes.

        Args:
            audio_chunk: SpeechSegment dict with 'pcm_data' (int16 bytes)
                         and 'start_ms' (int), or raw int16 PCM bytes.

        Returns:
            TranscriptResult {text, start_ms, end_ms, confidence}.
        """
        # Unpack input ────────────────────────────────────────────────
        if isinstance(audio_chunk, dict) and "pcm_data" in audio_chunk:
            pcm_bytes: bytes = audio_chunk["pcm_data"]
            start_ms: int = audio_chunk.get("start_ms", int(time.time() * 1000))
        elif isinstance(audio_chunk, bytes):
            pcm_bytes = audio_chunk
            start_ms = int(time.time() * 1000)
        else:
            Logger.log(
                "WARNING", "stt_moonshine",
                "Unknown audio_chunk type — returning empty transcript"
            )
            return {"text": "", "start_ms": 0, "end_ms": 0, "confidence": 0.0}

        # Convert int16 PCM bytes → float32 [-1, 1] ──────────────────
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = (audio_int16 / 32768.0).astype(np.float32)

        # Enforce 30-second max ───────────────────────────────────────
        max_samples = MAX_SECONDS * self.sample_rate
        if len(audio_f32) > max_samples:
            audio_f32 = audio_f32[:max_samples]
            Logger.log(
                "WARNING", "stt_moonshine",
                "Audio segment exceeded 30 s — truncated to 30 s."
            )

        # Run inference ───────────────────────────────────────────────
        t0 = time.time()
        try:
            text = self._run_inference(audio_f32)
        except Exception as exc:
            Logger.log("ERROR", "stt_moonshine", f"Transcription failed: {exc}")
            return {
                "text": "", "start_ms": start_ms,
                "end_ms": start_ms, "confidence": 0.0
            }
        t1 = time.time()

        # Normalise output ────────────────────────────────────────────
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        text = str(text).strip()

        inference_ms = int((t1 - t0) * 1000)
        duration_ms = int(len(audio_f32) / self.sample_rate * 1000)
        end_ms = start_ms + duration_ms

        Logger.log(
            "INFO", "stt_moonshine",
            f"Transcribed {duration_ms} ms audio in {inference_ms} ms",
            {"text": text[:120], "inference_ms": inference_ms,
             "duration_ms": duration_ms, "backend": self._backend},
        )

        return {
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": 1.0,  # Moonshine does not expose per-token confidence
        }

    def _run_inference(self, audio_f32: np.ndarray) -> str:
        """Dispatch to the appropriate backend."""
        if self._backend == "transformers":
            return self._infer_transformers(audio_f32)
        elif self._backend == "mlx":
            return self._infer_mlx(audio_f32)
        raise RuntimeError(f"Unknown backend: {self._backend!r}")

    def _infer_transformers(self, audio_f32: np.ndarray) -> str:
        import torch

        inputs = self._processor(
            audio_f32,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self._device)

        # Build attention mask — required to avoid pad/eos token ambiguity warning
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_values,
                attention_mask=attention_mask,
                max_new_tokens=448,
            )

        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return transcription[0] if transcription else ""

    def _infer_mlx(self, audio_f32: np.ndarray) -> str:
        import moonshine as moonshine_mlx  # type: ignore
        result = moonshine_mlx.transcribe(self._model, audio_f32)
        if isinstance(result, list):
            return " ".join(str(r) for r in result)
        return str(result)
