"""Response Player stub."""
from jarvis.infra.logger import Logger

class ResponsePlayer:
    """Stub for CoreAudio playback."""
    def play(self, audio_stream: bytes):
        Logger.log("INFO", "response_player", "[response_player] stub called")
