"""Claude LLM stub."""
from jarvis.interfaces.llm import LLMEngineABC, LLMResponse
from jarvis.infra.logger import Logger
from jarvis.infra.config_manager import config

class ClaudeLLM(LLMEngineABC):
    """Stub implementation of Claude API LLM."""
    def generate(self, prompt: str) -> LLMResponse:
        Logger.log("INFO", "llm_claude", "[llm_claude] stub called")
        return {
            "answer": "this is a dummy answer from claude",
            "source_chunks": ["chunk1"]
        }
