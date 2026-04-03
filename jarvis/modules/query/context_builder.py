"""Context Builder — assembles LLM prompt from retrieved memory chunks.

Prompt format specified in architecture.md / prompt5.md.
Timestamps converted to human-readable: "Monday 14 Apr, 2:35 PM"
"""
from datetime import datetime
from jarvis.infra.logger import Logger


def _ms_to_human(ms: int) -> str:
    """Convert a unix-ms timestamp to 'Monday 14 Apr, 2:35 PM'."""
    if not ms:
        return "unknown time"
    try:
        dt = datetime.fromtimestamp(ms / 1000)
        return dt.strftime("%A %-d %b, %-I:%M %p")
    except Exception:
        return str(ms)


_NO_MEMORIES_PROMPT = (
    "You are Jarvis, a personal memory assistant. The user has asked: \"{query}\"\n\n"
    "No relevant memories were found for this query. "
    "Please let the user know that you could not find any relevant memories."
)


class ContextBuilder:
    """Assembles the LLM prompt from retrieved memory chunks."""

    def build(self, chunks: list[dict], query: str) -> str:
        Logger.log(
            "INFO", "context_builder",
            f"Building prompt from {len(chunks)} chunks for query: {query!r}"
        )

        if not chunks:
            return _NO_MEMORIES_PROMPT.format(query=query)

        memories_lines = []
        for i, chunk in enumerate(chunks, 1):
            # chunk_text is the key from chunker; 'text' is the key in vector store
            text = chunk.get("text", chunk.get("chunk_text", "(no text)"))
            ts_ms = chunk.get("timestamp_start", 0)
            ts_human = _ms_to_human(ts_ms)
            memories_lines.append(f"[{i}] {ts_human} — {text}")

        memories_block = "\n".join(memories_lines)

        prompt = (
            f'You are Jarvis, a personal memory assistant. The user has asked:\n'
            f'"{query}"\n\n'
            f"Here are the relevant memories retrieved from their spoken history,\n"
            f"ordered by relevance. Each memory includes when it was spoken.\n\n"
            f"--- MEMORIES ---\n"
            f"{memories_block}\n"
            f"--- END MEMORIES ---\n\n"
            f"Based only on the memories above, answer the user's question.\n"
            f"If the memories do not contain enough information to answer, say so.\n"
            f"Do not invent information not present in the memories."
        )
        return prompt
