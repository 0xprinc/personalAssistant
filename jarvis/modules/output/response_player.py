"""Response Player — plays synthesised audio via sounddevice.

Accepts float32 numpy array at 24 kHz from KokoroTTS.
Blocks until playback completes (sounddevice.wait()).
"""
import numpy as np
import sounddevice as sd

from jarvis.infra.logger import Logger


class ResponsePlayer:
    """Plays audio through the default system output device."""

    def play(self, audio: np.ndarray, sample_rate: int = 24_000) -> None:
        """Play audio array through the default output device.

        Args:
            audio:       float32 numpy array from TTS.
            sample_rate: Sample rate in Hz (default 24000 — Kokoro output rate).
        """
        if audio is None or len(audio) == 0:
            Logger.log("WARNING", "response_player", "play() called with empty audio — skipping")
            return

        duration_s = len(audio) / sample_rate
        Logger.log(
            "INFO", "response_player",
            f"Playing {duration_s:.2f}s of audio at {sample_rate} Hz …"
        )

        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()  # blocks until playback complete
            Logger.log("INFO", "response_player", f"Playback complete ({duration_s:.2f}s)")
        except sd.PortAudioError as exc:
            if "no default output" in str(exc).lower() or "invalid device" in str(exc).lower():
                Logger.log(
                    "ERROR", "response_player",
                    "No audio output device found. Check system audio settings."
                )
            else:
                Logger.log("ERROR", "response_player", f"PortAudio error during playback: {exc}")
        except Exception as exc:
            Logger.log("ERROR", "response_player", f"Unexpected playback error: {exc}")

    def stop(self) -> None:
        """Immediately stop any ongoing playback."""
        try:
            sd.stop()
            Logger.log("INFO", "response_player", "Playback stopped")
        except Exception as exc:
            Logger.log("ERROR", "response_player", f"Error stopping playback: {exc}")
