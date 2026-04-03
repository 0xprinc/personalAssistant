"""Llama LLM — offline fallback via Ollama Python SDK.

Model: llama3.2:1b — pulled automatically if not present.
Checks Ollama is running before inference; raises RuntimeError if not.
"""
import re

import ollama

from jarvis.interfaces.llm import LLMEngineABC, LLMResponse
from jarvis.infra.logger import Logger


def _extract_source_chunks(prompt: str) -> list[str]:
    """Pull the raw text lines from the MEMORIES block in the prompt."""
    m = re.search(r"--- MEMORIES ---\n(.+?)\n--- END MEMORIES ---", prompt, re.DOTALL)
    if not m:
        return []
    lines = m.group(1).strip().split("\n")
    chunks = []
    for line in lines:
        text = re.sub(r"^\[\d+\] .+? — ", "", line).strip()
        if text:
            chunks.append(text)
    return chunks


class LlamaLLM(LLMEngineABC):
    """Local Llama 3.2 1B fallback via Ollama."""

    MODEL = "llama3.2:1b"

    def __init__(self):
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Verify Ollama is running and pull model if needed."""
        try:
            available = ollama.list()
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not running. Start it with `ollama serve` and retry."
            ) from exc

        model_names = [m["model"] for m in available.get("models", [])]
        if not any(self.MODEL in name for name in model_names):
            Logger.log("INFO", "llm_llama", f"Pulling model '{self.MODEL}' from Ollama …")
            try:
                ollama.pull(self.MODEL)
                Logger.log("INFO", "llm_llama", f"Model '{self.MODEL}' pulled successfully")
            except Exception as exc:
                Logger.log("ERROR", "llm_llama", f"Failed to pull '{self.MODEL}': {exc}")
                raise

    def generate(self, prompt: str) -> LLMResponse:
        Logger.log("INFO", "llm_llama", f"Calling Ollama ({self.MODEL}) …")
        try:
            response = ollama.chat(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer: str = response["message"]["content"]
            source_chunks = _extract_source_chunks(prompt)
            Logger.log(
                "INFO", "llm_llama",
                f"Ollama response received ({len(answer)} chars)"
            )
            return {"answer": answer, "source_chunks": source_chunks}

        except Exception as exc:
            Logger.log("ERROR", "llm_llama", f"Ollama inference error: {exc}")
            raise
