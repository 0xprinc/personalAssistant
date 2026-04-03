"""OpenRouter LLM — moonshotai/kimi-k2.5 via OpenRouter.

OpenRouter exposes an OpenAI-compatible API — we use the openai SDK with a
custom base_url. No Anthropic dependency required.

API key: config.yaml keys.openrouter_api_key or env var OPENROUTER_API_KEY.
On any API error, raises so llm_engine.py can catch and surface gracefully.
"""
import os
import re

from openai import OpenAI, APIConnectionError, RateLimitError, AuthenticationError

from jarvis.interfaces.llm import LLMEngineABC, LLMResponse
from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL = "moonshotai/kimi-k2.5"
_MAX_TOKENS = 4096   # 1024 was too small — kimi-k2.5 can truncate long summarise answers


def _get_api_key() -> str:
    key: str = config.get("keys", {}).get("openrouter_api_key", "")
    if not key or key.startswith("sk-or-xxx"):
        key = os.environ.get("OPENROUTER_API_KEY", "")
    return key


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


class ClaudeLLM(LLMEngineABC):
    """OpenRouter LLM engine using moonshotai/kimi-k2.5.

    Class kept as ClaudeLLM for interface compatibility — internally uses
    OpenRouter, not Anthropic.
    """

    def generate(self, prompt: str) -> LLMResponse:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "No OpenRouter API key found. Set keys.openrouter_api_key in "
                "config.yaml or the OPENROUTER_API_KEY environment variable."
            )

        client = OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)

        Logger.log("INFO", "llm_openrouter", f"Calling OpenRouter — model={_MODEL} …")
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            answer: str = response.choices[0].message.content or ""
            source_chunks = _extract_source_chunks(prompt)
            usage = response.usage
            Logger.log(
                "INFO", "llm_openrouter",
                f"Response received ({len(answer)} chars, "
                f"{usage.prompt_tokens if usage else '?'} in / "
                f"{usage.completion_tokens if usage else '?'} out tokens)"
            )
            return {"answer": answer, "source_chunks": source_chunks}

        except (APIConnectionError, RateLimitError, AuthenticationError) as exc:
            Logger.log("ERROR", "llm_openrouter", f"OpenRouter API error: {exc}")
            raise

        except Exception as exc:
            Logger.log("ERROR", "llm_openrouter", f"Unexpected error: {exc}")
            raise
