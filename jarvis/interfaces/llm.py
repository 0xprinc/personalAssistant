"""LLM Interface ABC."""
from abc import ABC, abstractmethod
from typing import TypedDict

class LLMResponse(TypedDict):
    answer: str
    source_chunks: list[str]

class LLMEngineABC(ABC):
    """Abstract Base Class for Large Language Model Engines."""
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using context from retrieved chunks."""
        pass
