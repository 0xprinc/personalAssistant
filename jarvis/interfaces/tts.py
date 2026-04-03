"""TTS Interface ABC."""
from abc import ABC, abstractmethod

class TTSEngineABC(ABC):
    """Abstract Base Class for Text-to-Speech Engines."""
    @abstractmethod
    def synthesise(self, text: str) -> bytes:
        """Convert arbitrary text back into spoken audio bytes."""
        pass
