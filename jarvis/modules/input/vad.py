"""Voice Activity Detection module using Silero VAD v5 ONNX.

This module runs Silero VAD on each 512-sample audio block from the AudioCapture
queue, maintains a 2-second pre-buffer, and emits SpeechSegment dicts whenever a
contiguous speech region is detected and then followed by >700 ms of silence.

Key correctness notes for Silero VAD v5 ONNX:
- Input shape must be (1, samples) — NOT (samples,) or (1,1,samples)
- Audio must be float32 normalised to [-1.0, 1.0] from int16
- Hidden state h and cell state c are SEPARATE tensors, each shape (2, 1, 64)
- Sample rate is passed as a separate int64 scalar input 'sr'
- States MUST be carried across every inference call (model is stateful)
- States must NOT be reset while speech is active — only on full reset
"""

import os
import time
import queue
import collections
import threading
from pathlib import Path
from typing import TypedDict

import numpy as np
import requests
import onnxruntime as ort

from jarvis.infra.logger import Logger
from jarvis.infra.config_manager import config


class SpeechSegment(TypedDict):
    """A detected speech segment from VAD output."""
    pcm_data: bytes    # Raw int16 PCM bytes (16kHz mono)
    start_ms: int      # Unix timestamp in milliseconds of segment start


# Resolve model path relative to project root (personalass/)
# vad.py is at: personalass/jarvis/modules/input/vad.py
# So 4 parents up = personalass/
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # .../personalass


class VadEngine:
    """Voice Activity Detection using Silero VAD v5 ONNX.

    Spawns a background thread that continuously reads from audio_queue,
    runs VAD inference, and places completed SpeechSegment dicts onto
    segment_queue for downstream consumers.
    """

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue

        params = config.get("parameters", {})
        self.vad_threshold: float = params.get("vad_threshold", 0.5)
        self.sample_rate: int = params.get("sample_rate", 16000)
        self.block_size: int = params.get("block_size", 512)

        # Resolve model path from project root
        rel_model_path: str = config.get("models", {}).get(
            "vad_model_path", "models/silero_vad.onnx"
        )
        vad_model_path = str(_PKG_ROOT / rel_model_path)

        self.segment_queue: queue.Queue = queue.Queue()

        # 2-second pre-buffer — stores raw PCM bytes blocks
        self.pre_buffer_blocks = int((2.0 * self.sample_rate) / self.block_size)
        self.pre_buffer: collections.deque = collections.deque(
            maxlen=self.pre_buffer_blocks
        )

        # 700 ms silence tolerance before closing a speech segment
        self.silence_tolerance_blocks = max(
            1, int((0.7 * self.sample_rate) / self.block_size)
        )

        self._ensure_model_exists(vad_model_path)
        self._init_session(vad_model_path)
        self._reset_state()

        self.block_count = 0
        self.is_speaking = False
        self.silence_counter = 0
        self.current_segment_blocks: list[bytes] = []
        self.current_segment_start_ms = 0

        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._process_loop, daemon=True, name="vad-worker"
        )
        self.thread.start()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_model_exists(self, path: str) -> None:
        """Download the Silero VAD ONNX model if not already present."""
        if os.path.exists(path):
            Logger.log("INFO", "vad", f"Silero VAD model found at {path}")
            return

        Logger.log("INFO", "vad", f"Downloading Silero VAD model to {path} ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = (
            "https://github.com/snakers4/silero-vad/raw/master/"
            "src/silero_vad/data/silero_vad.onnx"
        )
        try:
            r = requests.get(url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            Logger.log("INFO", "vad", "Silero VAD model downloaded successfully.")
        except Exception as exc:
            Logger.log("ERROR", "vad", f"Failed to download Silero VAD: {exc}")
            raise

    def _init_session(self, path: str) -> None:
        """Load the ONNX inference session and inspect input/output names."""
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        # Discover actual input/output names — model format varies by download source
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]
        Logger.log(
            "INFO",
            "vad",
            "Silero VAD ONNX loaded",
            {"inputs": [i.name for i in self.session.get_inputs()],
             "outputs": self._output_names},
        )

    def _reset_state(self) -> None:
        """Initialise (or reset) all LSTM state tensors.

        Silero VAD is stateful — states MUST be carried across every inference
        call. Resetting between utterances is correct; resetting mid-speech
        destroys the model context and produces near-zero confidence.

        Two formats exist depending on model version:
        - v4/older: single 'state' tensor of shape (2, 1, 128)
        - v5:       separate 'h' and 'c' tensors each of shape (2, 1, 64)
        We detect the format in _init_session and reset accordingly.
        """
        if hasattr(self, '_input_names') and 'h' in self._input_names:
            # v5 format
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            # v4/older format — single state tensor
            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, audio_block: bytes) -> float:
        """Run Silero VAD inference on one audio block, return speech probability.

        Handles both the older model format (single 'state' input) and the
        v5 format (separate 'h' and 'c' inputs) by inspecting input names.
        """
        # Convert int16 PCM bytes → float32 normalised to [-1, 1]
        audio_int16 = np.frombuffer(audio_block, dtype=np.int16)
        audio_float32 = (audio_int16 / 32768.0).astype(np.float32)

        # Shape must be (1, N) for Silero VAD v5
        audio_2d = audio_float32[np.newaxis, :]  # (1, block_size)

        sr_tensor = np.array(self.sample_rate, dtype=np.int64)

        # Build inputs dict based on model format (v4 vs v5)
        if "h" in self._input_names and "c" in self._input_names:
            # Silero VAD v5 — separate hidden and cell states
            ort_inputs = {
                "input": audio_2d,
                "h": self.h,
                "c": self.c,
                "sr": sr_tensor,
            }
            ort_outs = self.session.run(None, ort_inputs)
            # Outputs: [output, hn, cn]
            out = ort_outs[0]
            self.h = ort_outs[1]
            self.c = ort_outs[2]
        else:
            # Silero VAD v4/older — single merged state tensor, output as 'stateN'
            ort_inputs = {
                "input": audio_2d,
                "state": self._state,
                "sr": sr_tensor,
            }
            ort_outs = self.session.run(None, ort_inputs)
            # Outputs: [output, stateN]
            out = ort_outs[0]
            self._state = ort_outs[1]  # carry state forward — critical for accuracy

        # out shape is typically (1, 1) — extract scalar probability
        return float(np.squeeze(out))

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
                     "silence_ctr": self.silence_counter}
                )

            if confidence > self.vad_threshold:
                # --- Speech detected ---
                if not self.is_speaking:
                    self.is_speaking = True
                    # Prepend the pre-buffer (recent silence before speech onset)
                    self.current_segment_blocks = list(self.pre_buffer)
                    prebuf_ms = int(
                        len(self.pre_buffer) * self.block_size / self.sample_rate * 1000
                    )
                    self.current_segment_start_ms = (
                        int(time.time() * 1000) - prebuf_ms
                    )
                    Logger.log("DEBUG", "vad", "Speech onset detected")

                self.silence_counter = 0
                self.current_segment_blocks.append(audio_block)

            else:
                # --- Silence detected ---
                if self.is_speaking:
                    self.current_segment_blocks.append(audio_block)
                    self.silence_counter += 1

                    if self.silence_counter >= self.silence_tolerance_blocks:
                        # Reached end-of-speech — emit segment
                        segment_data = b"".join(self.current_segment_blocks)
                        self.segment_queue.put(
                            {
                                "pcm_data": segment_data,
                                "start_ms": self.current_segment_start_ms,
                            }
                        )
                        duration_ms = int(
                            len(segment_data) / 2 / self.sample_rate * 1000
                        )
                        Logger.log(
                            "INFO",
                            "vad",
                            f"SpeechSegment emitted — {len(segment_data)} bytes "
                            f"({duration_ms} ms)",
                        )
                        # Reset for next segment — reset state ONLY here, not mid-speech
                        self.is_speaking = False
                        self.silence_counter = 0
                        self.current_segment_blocks = []
                        self._reset_state()
                else:
                    # Still in silence — keep rolling pre-buffer
                    self.pre_buffer.append(audio_block)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_speech_segment(self) -> SpeechSegment:
        """Block until a SpeechSegment is ready and return it."""
        return self.segment_queue.get()

    def process(self, pcm_stream: bytes) -> bytes:
        """Stub compatibility method for the smoke-test path."""
        return pcm_stream

    def stop(self) -> None:
        """Signal the processing thread to stop and wait for it to exit."""
        Logger.log("INFO", "vad", "VAD stopping ...")
        self._stop_event.set()
        self.thread.join(timeout=5.0)
        Logger.log("INFO", "vad", "VAD stopped.")
