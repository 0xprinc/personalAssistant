"""Audio Capture module for Jarvis.

Captures continuous 16kHz mono 16-bit PCM audio from the system microphone
using sounddevice.InputStream. Audio blocks are pushed into a thread-safe
queue.Queue so that the VAD module can consume them without blocking.

Usage:
    cap = AudioCapture(device_id=3)
    cap.start()
    # ... read via cap.audio_queue ...
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


class AudioCapture:
    """Continuously captures microphone audio into a thread-safe queue.

    All heavy work is done inside the sounddevice InputStream callback which
    runs in its own C-level thread — the main Python thread is never blocked.
    """

    def __init__(self, device_id: int = None):
        self.device_id = device_id

        params = config.get("parameters", {})
        self.sample_rate: int = params.get("sample_rate", 16000)
        self.block_size: int = params.get("block_size", 512)

        self.audio_queue: queue.Queue = queue.Queue()
        self.stream = None
        self.block_count = 0

    # ------------------------------------------------------------------
    # Stream callbacks
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddevice InputStream callback — called in a background C thread.

        indata shape: (block_size, 1) int16
        We convert to bytes and push onto the queue for the VAD thread.
        """
        if status:
            Logger.log("WARNING", "audio_capture", f"Stream status: {status}")

        # indata is a numpy array — convert to raw bytes (16-bit PCM, little-endian)
        self.audio_queue.put(bytes(indata))
        self.block_count += 1

        if self.block_count % 100 == 0:
            Logger.log(
                "DEBUG", "audio_capture", f"Captured {self.block_count} blocks"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the InputStream. Safe to call multiple times."""
        if self.stream is not None:
            return

        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self.stream.start()
            Logger.log(
                "INFO",
                "audio_capture",
                f"Audio capture started — device={self.device_id}, "
                f"rate={self.sample_rate}Hz, block={self.block_size} samples",
            )
        except sd.PortAudioError as exc:
            Logger.log(
                "ERROR",
                "audio_capture",
                f"PortAudio error: {exc}. If on macOS, run: brew install portaudio",
            )
            sys.exit(1)

    def stop(self) -> None:
        """Stop and close the InputStream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            Logger.log("INFO", "audio_capture", "Audio capture stopped.")

    # ------------------------------------------------------------------
    # Stub compatibility (used by smoke test path)
    # ------------------------------------------------------------------

    def get_audio_stream(self) -> bytes:
        """Return a dummy PCM block for smoke-test pipeline wiring."""
        return b"\x00" * (self.block_size * 2)  # block_size int16 samples = *2 bytes
