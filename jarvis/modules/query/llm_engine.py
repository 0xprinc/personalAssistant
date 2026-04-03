"""LLM Engine Router — selects Claude vs Llama automatically.

This is the ONLY file that imports llm_claude and llm_llama.
Everything else imports only this module.

Fallback chain:
  1. Try Claude API
  2. On APIConnectionError / RateLimitError / AuthenticationError / any network error
     → log warning + try Llama via Ollama
  3. If Llama also fails → return hardcoded "no LLM available" response
"""
import anthropic

from jarvis.infra.logger import Logger
from jarvis.interfaces.llm import LLMResponse
from jarvis.modules.query.llm_claude import ClaudeLLM
from jarvis.modules.query.llm_llama import LlamaLLM

_claude = ClaudeLLM()
_llama: LlamaLLM | None = None   # lazy-init — avoid loading Ollama at startup


def _get_llama() -> LlamaLLM:
    global _llama
    if _llama is None:
        _llama = LlamaLLM()
    return _llama


def generate(prompt: str) -> LLMResponse:
    """Generate a response, routing Claude → Llama → hardcoded fallback."""
    # --- Try Claude ---
    try:
        return _claude.generate(prompt)
    except (
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.AuthenticationError,
        RuntimeError,
    ) as exc:
        Logger.log(
            "WARNING", "llm_engine",
            f"Claude API unavailable ({type(exc).__name__}: {exc}), "
            "falling back to Llama …"
        )
    except Exception as exc:
        Logger.log(
            "WARNING", "llm_engine",
            f"Claude unexpected error ({exc}), falling back to Llama …"
        )

    # --- Try Llama ---
    try:
        return _get_llama().generate(prompt)
    except Exception as exc:
        Logger.log(
            "ERROR", "llm_engine",
            f"Llama also failed ({exc}). Returning degraded response."
        )

    # --- Hardcoded fallback ---
    return {
        "answer": "I could not reach any LLM. Please check your connection.",
        "source_chunks": [],
    }
