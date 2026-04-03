"""Llama LLM stub."""
from jarvis.interfaces.llm import LLMEngineABC, LLMResponse
from jarvis.infra.logger import Logger

class LlamaLLM(LLMEngineABC):
    """Stub implementation of offline Llama fallback LLM."""
    def generate(self, prompt: str) -> LLMResponse:
        Logger.log("INFO", "llm_llama", "[llm_llama] stub called")
        return {
            "answer": "this is a dummy answer from llama locally",
            "source_chunks": ["chunk1"]
        }
