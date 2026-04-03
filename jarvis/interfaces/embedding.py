"""Embedding Interface ABC."""
from abc import ABC, abstractmethod

class EmbeddingEngineABC(ABC):
    """Abstract Base Class for Embedding Engines."""
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert a text string to a 384-dimensional float vector."""
        pass
