"""Audio Capture module for Jarvis.

Captures continuous mono 16-bit PCM audio from the selected input device,
resamples to 16000 Hz (required by Silero VAD and Moonshine STT), and pushes
blocks onto a thread-safe queue.Queue for the VAD module to consume.

Key design:
- Captures at the device's NATIVE sample rate to avoid PortAudio resampling
  artifacts (especially with Bluetooth devices).
- Downsamples to 16kHz in-process using scipy or simple decimation.
- Each queue item is exactly block_size (512) int16 samples = 1024 bytes.

Usage:
    cap = AudioCapture(device_id=5, native_rate=48000)
    cap.start()
    # ... read from cap.audio_queue (each item = 1024 bytes of 16kHz int16)
    cap.stop()
"""

import queue
import sys
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger


def _downsample(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a 1-D int16 array from from_rate to to_rate.

    Uses scipy.signal.resample_poly if available (best quality), otherwise
    falls back to simple integer decimation (works when from_rate is an
    exact multiple of to_rate, e.g. 48000->16000 = /3).
    """
    if from_rate == to_rate:
        return data

    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(from_rate, to_rate)
        up, down = to_rate // g, from_rate // g
        return resample_poly(data.astype(np.float32), up, down).astype(np.int16)
    except ImportError:
        # Simple decimation — only exact for integer ratios
        ratio = from_rate // to_rate
        if ratio < 1:
            ratio = 1
        return data[::ratio]


class AudioCapture:
    """Continuously captures microphone audio, resampled to 16kHz.

    All heavy work is done inside the sounddevice callback which runs in its
    own C-level thread — the main Python thread is never blocked.
    """

    VAD_RATE = 16000     # Target rate required by Silero VAD and Moonshine

    def __init__(self, device_id: Optional[int] = None, native_rate: int = 48000):
        self.device_id = device_id
        self.native_rate = native_rate

        params = config.get("parameters", {})
        self.block_size: int = params.get("block_size", 512)
        # Software gain: multiply amplitude before sending to VAD.
        # Increase if VAD confidence stays near 0 (mic too quiet).
        # 1.0 = no boost. 4.0 = 4× boost, clipped to int16 range.
        self._gain: float = float(params.get("mic_gain", 4.0))

        # How many native-rate samples correspond to one 16kHz block
        self._native_block = int(self.block_size * self.native_rate / self.VAD_RATE)

        self.audio_queue: queue.Queue = queue.Queue()
        self.stream = None
        self.block_count = 0

        # Overflow buffer for leftover samples between callbacks
        self._overflow: np.ndarray = np.array([], dtype=np.int16)

    # ------------------------------------------------------------------
    # Stream callback
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddevice callback — called in a background C thread.

        indata shape: (frames, 1), dtype int16, at native_rate.
        We flatten, resample to 16kHz, split into block_size chunks,
        and push each onto the queue.
        """
        if status:
            Logger.log("WARNING", "audio_capture", f"Stream status: {status}")

        samples = indata[:, 0].copy()   # shape (frames,), int16

        # Prepend any leftover from last callback
        if len(self._overflow):
            samples = np.concatenate([self._overflow, samples])

        # Resample to VAD_RATE
        resampled = _downsample(samples, self.native_rate, self.VAD_RATE)

        # Apply software gain and clip to int16 range
        if self._gain != 1.0:
            resampled = np.clip(
                resampled.astype(np.float32) * self._gain, -32768, 32767
            ).astype(np.int16)

        # Split into block_size chunks and enqueue each
        i = 0
        while i + self.block_size <= len(resampled):
            block = resampled[i : i + self.block_size]
            self.audio_queue.put(bytes(block))
            self.block_count += 1
            i += self.block_size

        # Carry leftover samples to next callback
        self._overflow = resampled[i:].copy() if i < len(resampled) else np.array([], dtype=np.int16)

        if self.block_count % 100 == 0 and self.block_count > 0:
            rms = float(np.sqrt(np.mean(resampled.astype(np.float32) ** 2)))
            Logger.log(
                "DEBUG", "audio_capture",
                f"Captured {self.block_count} blocks | RMS={rms:.0f} (gain={self._gain}×)"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the InputStream."""
        if self.stream is not None:
            return

        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.native_rate,
                channels=1,
                dtype="int16",
                blocksize=self._native_block,
                callback=self._audio_callback,
            )
            self.stream.start()
            Logger.log(
                "INFO",
                "audio_capture",
                f"Audio capture started — device={self.device_id}, "
                f"native={self.native_rate}Hz → resample to {self.VAD_RATE}Hz, "
                f"vad_block={self.block_size} samples",
            )
        except sd.PortAudioError as exc:
            Logger.log("ERROR", "audio_capture", f"PortAudio error: {exc}")
            sys.exit(1)

    def stop(self) -> None:
        """Stop and close the InputStream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            Logger.log("INFO", "audio_capture", "Audio capture stopped.")

    # ------------------------------------------------------------------
    # Smoke test stub
    # ------------------------------------------------------------------

    def get_audio_stream(self) -> bytes:
        """Return a dummy 16kHz PCM block for smoke-test pipeline wiring."""
        return b"\x00" * (self.block_size * 2)
