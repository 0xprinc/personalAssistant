"""Vector Store Interface ABC."""
from abc import ABC, abstractmethod
from typing import TypedDict, Any

class MemoryChunk(TypedDict):
    chunk_id: str
    text: str
    vector: list[float]
    timestamp_start: int
    timestamp_end: int
    device_id: str
    session_id: str
    confidence: float
    redacted: bool

class VectorStoreABC(ABC):
    """Abstract Base Class for Vector Stores."""
    @abstractmethod
    def upsert(self, chunk: MemoryChunk) -> str:
        """Upsert a memory chunk and return its chunk ID."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], filters: dict[str, Any], top_k: int) -> list[MemoryChunk]:
        """Search vector store and return top-k matches."""
        pass
