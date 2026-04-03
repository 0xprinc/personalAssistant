"""Kokoro TTS stub."""
from jarvis.interfaces.tts import TTSEngineABC
from jarvis.infra.logger import Logger

class KokoroTTS(TTSEngineABC):
    """Stub implementation of Kokoro 82M TTS."""
    def synthesise(self, text: str) -> bytes:
        Logger.log("INFO", "tts_kokoro", "[tts_kokoro] stub called")
        return b'\x00' * 1024  # 24kHz PCM dummy
