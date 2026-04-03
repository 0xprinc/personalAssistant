"""Context Builder stub."""
from jarvis.infra.logger import Logger

class ContextBuilder:
    """Assembles LLM prompt from retrieved chunks."""
    def build(self, chunks: list[dict], query: str) -> str:
        Logger.log("INFO", "context_builder", "[context_builder] stub called")
        return f"Context: {len(chunks)} chunks. Query: {query}"
