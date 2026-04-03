"""LLM Engine Router — cloud-only via OpenRouter.

Uses moonshotai/kimi-k2.5 via OpenRouter.
On API failure, returns a hardcoded error response (no local fallback).

This is the ONLY file that external code should import for LLM generation.
"""
from openai import APIConnectionError, RateLimitError, AuthenticationError

from jarvis.infra.logger import Logger
from jarvis.interfaces.llm import LLMResponse
from jarvis.modules.query.llm_claude import ClaudeLLM   # OpenRouter implementation

_openrouter = ClaudeLLM()


def generate(prompt: str) -> LLMResponse:
    """Generate a response via OpenRouter (moonshotai/kimi-k2.5).

    On any API/connection error, logs the failure and returns a graceful
    hardcoded response — no local model fallback (cloud-only mode).
    """
    try:
        return _openrouter.generate(prompt)
    except (APIConnectionError, RateLimitError, AuthenticationError, RuntimeError) as exc:
        Logger.log(
            "ERROR", "llm_engine",
            f"OpenRouter unavailable ({type(exc).__name__}: {exc}). "
            "Returning degraded response."
        )
    except Exception as exc:
        Logger.log("ERROR", "llm_engine", f"Unexpected LLM error: {exc}. Returning degraded response.")

    return {
        "answer": "I could not reach the LLM. Please check your OpenRouter API key and connection.",
        "source_chunks": [],
    }
