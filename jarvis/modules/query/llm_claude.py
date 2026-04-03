"""Claude LLM — real implementation using the Anthropic Python SDK.

Model: claude-sonnet-4-6
API key: config.yaml keys.claude_api_key or env var ANTHROPIC_API_KEY.
On any API error, raises so llm_engine.py can fall back to Llama.
"""
import os
import re

import anthropic

from jarvis.interfaces.llm import LLMEngineABC, LLMResponse
from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger


def _get_api_key() -> str:
    key: str = config.get("keys", {}).get("claude_api_key", "")
    if not key or key.startswith("sk-ant-xxx"):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key


def _extract_source_chunks(prompt: str) -> list[str]:
    """Pull the raw text lines from the MEMORIES block in the prompt."""
    m = re.search(r"--- MEMORIES ---\n(.+?)\n--- END MEMORIES ---", prompt, re.DOTALL)
    if not m:
        return []
    lines = m.group(1).strip().split("\n")
    chunks = []
    for line in lines:
        # Strip leading "[N] timestamp — " prefix
        text = re.sub(r"^\[\d+\] .+? — ", "", line).strip()
        if text:
            chunks.append(text)
    return chunks


class ClaudeLLM(LLMEngineABC):
    """Claude API LLM engine (primary)."""

    MODEL = "claude-sonnet-4-5"
    MAX_TOKENS = 1024

    def generate(self, prompt: str) -> LLMResponse:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "No Claude API key found. Set keys.claude_api_key in config.yaml "
                "or the ANTHROPIC_API_KEY environment variable."
            )

        client = anthropic.Anthropic(api_key=api_key)

        Logger.log("INFO", "llm_claude", f"Calling Claude API (model={self.MODEL}) …")
        try:
            message = client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            answer: str = message.content[0].text
            source_chunks = _extract_source_chunks(prompt)
            Logger.log(
                "INFO", "llm_claude",
                f"Claude response received ({len(answer)} chars, "
                f"{message.usage.input_tokens} in / {message.usage.output_tokens} out tokens)"
            )
            return {"answer": answer, "source_chunks": source_chunks}

        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            Logger.log("ERROR", "llm_claude", f"Claude API error: {exc}")
            raise  # re-raise so llm_engine can catch and fall back

        except anthropic.AuthenticationError as exc:
            Logger.log("ERROR", "llm_claude", f"Claude authentication error: {exc}")
            raise

        except Exception as exc:
            Logger.log("ERROR", "llm_claude", f"Unexpected Claude error: {exc}")
            raise
