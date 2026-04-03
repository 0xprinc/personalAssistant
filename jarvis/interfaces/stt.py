"""STT Interface ABC."""
from abc import ABC, abstractmethod
from typing import TypedDict

class TranscriptResult(TypedDict):
    text: str
    start_ms: int
    end_ms: int
    confidence: float

class STTEngineABC(ABC):
    """Abstract Base Class for Speech-to-Text Engines."""
    @abstractmethod
    def transcribe(self, audio_chunk: bytes) -> TranscriptResult:
        """Convert a chunk of audio to text."""
        pass
