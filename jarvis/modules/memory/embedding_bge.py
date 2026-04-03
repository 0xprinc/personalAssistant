"""BGE Embedding stub."""
from jarvis.interfaces.embedding import EmbeddingEngineABC
from jarvis.infra.logger import Logger

class BGEEmbeddingEngine(EmbeddingEngineABC):
    """Stub implementation of Embedding ABC."""
    def embed(self, text: str) -> list[float]:
        Logger.log("INFO", "embedding_bge", "[embedding_bge] stub called")
        return [0.0] * 384
