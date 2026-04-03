"""Voice Activity Detection module using Silero VAD (PyTorch Hub).

Uses the official torch.hub.load() interface for Silero VAD, which:
- Manages LSTM state (h/c tensors) entirely internally — no manual shapes.
- Downloads and caches the model automatically to ~/.cache/torch/hub/.
- Correctly handles model state persistence across audio chunks.

Pipeline:
  AudioCapture → [int16 bytes @ 16kHz, 512 samples/block] → VadEngine (this)
             → SpeechSegment {'pcm_data': bytes, 'start_ms': int}

Configuration (from config.yaml parameters):
  sample_rate: 16000       # Must be 16000 for Silero
  block_size: 512          # 32ms per block at 16kHz (Silero's native chunk size)
  vad_threshold: 0.5       # 0.5 is the recommended activation threshold
  silence_tolerance_ms: 1200  # 1.2s silence before closing a segment (matches production use)
  prebuffer_ms: 500        # 500ms of audio kept before speech onset
"""

import queue
import time
import collections
import threading
from typing import Optional

import numpy as np

from jarvis.infra.logger import Logger
from jarvis.infra.config_manager import config


class SpeechSegment(dict):
    """A detected speech segment from VAD output.
    Keys: pcm_data (bytes, int16 @ 16kHz), start_ms (int, unix ms).
    """


class VadEngine:
    """Voice Activity Detection using Silero VAD via PyTorch Hub.

    Spawns a background worker thread that consumes int16 audio blocks
    from audio_queue, runs VAD inference, and emits SpeechSegment dicts
    onto segment_queue for downstream STT consumption.
    """

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue

        params = config.get("parameters", {})
        self.vad_threshold: float = params.get("vad_threshold", 0.5)
        self.sample_rate: int = params.get("sample_rate", 16000)
        self.block_size: int = params.get("block_size", 512)

        # Silence tolerance: how long we wait after speech drops below threshold
        # before closing and emitting the segment. Default 1200ms (1.2s) to match
        # production use (prevents clipping short pauses mid-sentence).
        silence_ms: int = params.get("silence_tolerance_ms", 1200)
        self.silence_tolerance_blocks: int = max(
            1, int((silence_ms / 1000) * self.sample_rate / self.block_size)
        )

        # Pre-buffer: audio kept from before speech onset (catches first consonant)
        prebuffer_ms: int = params.get("prebuffer_ms", 500)
        prebuffer_blocks: int = max(
            1, int((prebuffer_ms / 1000) * self.sample_rate / self.block_size)
        )
        self.pre_buffer: collections.deque = collections.deque(maxlen=prebuffer_blocks)

        self.segment_queue: queue.Queue = queue.Queue()

        self._load_model()

        self.block_count: int = 0
        self.is_speaking: bool = False
        self.silence_counter: int = 0
        self.current_segment_blocks: list = []
        self.current_segment_start_ms: int = 0

        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._process_loop, daemon=True, name="vad-worker"
        )
        self.thread.start()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Silero VAD via torch.hub (auto-downloads and caches)."""
        import torch

        Logger.log("INFO", "vad", "Loading Silero VAD via torch.hub …")
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,   # Pre-approve to avoid interactive terminal prompt
                onnx=False,
            )
            model.eval()
            self._model = model
            self._torch = torch
            Logger.log("INFO", "vad", "Silero VAD loaded successfully (PyTorch Hub)")
        except Exception as exc:
            Logger.log("ERROR", "vad", f"Failed to load Silero VAD: {exc}")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, audio_block: bytes) -> float:
        """Run Silero VAD on one 512-sample int16 audio block.

        Returns speech probability in [0, 1]. Model state is managed
        internally by the PyTorch model — no manual h/c tensors needed.
        """
        audio_int16 = np.frombuffer(audio_block, dtype=np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio_f32).unsqueeze(0)  # (1, 512)

        with self._torch.no_grad():
            confidence = self._model(tensor, self.sample_rate)

        return float(confidence.detach().item())

    def _reset_model_states(self) -> None:
        """Reset the model's internal LSTM states after a segment completes."""
        self._model.reset_states()

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        """Background thread: pull audio blocks, run VAD, emit SpeechSegments."""
        Logger.log("INFO", "vad", "VAD processing thread started")

        while not self._stop_event.is_set():
            try:
                audio_block: bytes = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.block_count += 1
            confidence = self._predict(audio_block)

            if self.block_count % 50 == 0:
                Logger.log(
                    "DEBUG", "vad", f"VAD confidence: {confidence:.3f}",
                    {"block": self.block_count,
                     "speaking": self.is_speaking,
                     "silence_ctr": self.silence_counter},
                )

            if confidence >= self.vad_threshold:
                # ── Speech active ────────────────────────────────────
                if not self.is_speaking:
                    self.is_speaking = True
                    # Prepend pre-buffer (speech onset capture)
                    self.current_segment_blocks = list(self.pre_buffer)
                    prebuf_ms = int(
                        len(self.pre_buffer) * self.block_size / self.sample_rate * 1000
                    )
                    self.current_segment_start_ms = int(time.time() * 1000) - prebuf_ms
                    Logger.log("DEBUG", "vad", "Speech onset detected")

                self.silence_counter = 0
                self.current_segment_blocks.append(audio_block)

            else:
                # ── Silence ──────────────────────────────────────────
                if self.is_speaking:
                    self.current_segment_blocks.append(audio_block)
                    self.silence_counter += 1

                    if self.silence_counter >= self.silence_tolerance_blocks:
                        # End of speech — emit the segment
                        segment_data = b"".join(self.current_segment_blocks)
                        self.segment_queue.put({
                            "pcm_data": segment_data,
                            "start_ms": self.current_segment_start_ms,
                        })
                        duration_ms = int(len(segment_data) / 2 / self.sample_rate * 1000)
                        Logger.log(
                            "INFO", "vad",
                            f"SpeechSegment emitted — {duration_ms} ms",
                        )
                        # Reset — ONLY here, never mid-speech
                        self.is_speaking = False
                        self.silence_counter = 0
                        self.current_segment_blocks = []
                        self._reset_model_states()
                else:
                    # Still in silence — keep rolling pre-buffer
                    self.pre_buffer.append(audio_block)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_speech_segment(self) -> dict:
        """Block until a SpeechSegment is ready and return it."""
        return self.segment_queue.get()

    def process(self, pcm_stream: bytes) -> bytes:
        """Stub compatibility for smoke-test path."""
        return pcm_stream

    def stop(self) -> None:
        """Signal the VAD thread to stop and wait for exit."""
        Logger.log("INFO", "vad", "VAD stopping …")
        self._stop_event.set()
        self.thread.join(timeout=5.0)
        Logger.log("INFO", "vad", "VAD stopped.")
