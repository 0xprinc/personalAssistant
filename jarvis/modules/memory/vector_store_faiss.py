"""FAISS Vector Store stub."""
import uuid
from typing import Any
from jarvis.interfaces.vector_store import VectorStoreABC, MemoryChunk
from jarvis.infra.logger import Logger

class FAISSVectorStore(VectorStoreABC):
    """Stub block for the FAISS implementation vector store."""
    def __init__(self):
        self.store = []

    def upsert(self, chunk: MemoryChunk) -> str:
        Logger.log("INFO", "vector_store_faiss", "[vector_store_faiss] upsert stub called")
        cid = str(uuid.uuid4())
        chunk["chunk_id"] = cid
        self.store.append(chunk)
        return cid

    def search(self, query_vector: list[float], filters: dict[str, Any], top_k: int) -> list[MemoryChunk]:
        Logger.log("INFO", "vector_store_faiss", "[vector_store_faiss] search stub called")
        return self.store[:top_k]
