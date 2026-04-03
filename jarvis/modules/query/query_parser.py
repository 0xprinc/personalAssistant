"""Query Parser — natural language → structured query dict.

Extracts:
  intent      : "recall" | "summarise" | "search"
  time_filter : {after_ms: int|None, before_ms: int|None}
  keywords    : list[str]

Uses Python stdlib only (datetime, re) — no dateparser dependency.
"""
import re
import time
from datetime import datetime, timedelta, timezone

from jarvis.infra.logger import Logger

# ---------------------------------------------------------------------------
# Stopwords to strip from keyword list
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "was", "i", "me", "my", "about", "what",
    "did", "do", "can", "could", "in", "on", "at", "of", "to", "for",
    "and", "or", "but", "with", "from", "it", "this", "that", "said",
    "say", "tell", "be", "are", "were", "have", "has", "had", "will",
    "would", "should", "they", "he", "she", "we", "you", "it", "there",
    "how", "when", "where", "which", "who", "any", "all",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _today_midnight_ms() -> int:
    """Today at 00:00:00 local time, as unix ms."""
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)

def _days_ago_midnight_ms(n: int) -> int:
    now = datetime.now()
    target = (now - timedelta(days=n)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(target.timestamp() * 1000)

def _days_ago_end_ms(n: int) -> int:
    now = datetime.now()
    target = (now - timedelta(days=n)).replace(hour=23, minute=59, second=59, microsecond=0)
    return int(target.timestamp() * 1000)

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def _last_weekday_ms(day_name: str) -> tuple[int, int]:
    """Return (after_ms, before_ms) for 'last <weekday>'."""
    target_wd = _WEEKDAYS.index(day_name)
    now = datetime.now()
    days_back = (now.weekday() - target_wd) % 7 or 7  # at least 1 day ago
    target = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
    after_ms = int(target.timestamp() * 1000)
    before_ms = after_ms + 86_400_000 - 1  # end of that day
    return after_ms, before_ms

# ---------------------------------------------------------------------------
# Time filter extractor
# ---------------------------------------------------------------------------

def _extract_time_filter(text: str) -> dict:
    q = text.lower()
    now_ms = _now_ms()

    # "this morning" → 6am–12pm today
    if re.search(r"\bthis morning\b", q):
        base = _today_midnight_ms()
        return {"after_ms": base + 6 * 3600 * 1000, "before_ms": base + 12 * 3600 * 1000}

    # "this afternoon" → 12pm–6pm today
    if re.search(r"\bthis afternoon\b", q):
        base = _today_midnight_ms()
        return {"after_ms": base + 12 * 3600 * 1000, "before_ms": base + 18 * 3600 * 1000}

    # "this evening" / "tonight" → 6pm–midnight today
    if re.search(r"\bthis evening\b|\btonight\b", q):
        base = _today_midnight_ms()
        return {"after_ms": base + 18 * 3600 * 1000, "before_ms": base + 86_400_000 - 1}

    # "today" (generic)
    if re.search(r"\btoday\b", q):
        base = _today_midnight_ms()
        return {"after_ms": base, "before_ms": now_ms}

    # "yesterday"
    if re.search(r"\byesterday\b", q):
        return {"after_ms": _days_ago_midnight_ms(1), "before_ms": _days_ago_end_ms(1)}

    # "last week"
    if re.search(r"\blast week\b", q):
        return {"after_ms": _days_ago_midnight_ms(7), "before_ms": _days_ago_end_ms(1)}

    # "this week"
    if re.search(r"\bthis week\b", q):
        now = datetime.now()
        monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return {"after_ms": int(monday.timestamp() * 1000), "before_ms": now_ms}

    # "last <weekday>" e.g. "last Tuesday"
    m = re.search(r"\blast (" + "|".join(_WEEKDAYS) + r")\b", q)
    if m:
        after_ms, before_ms = _last_weekday_ms(m.group(1))
        return {"after_ms": after_ms, "before_ms": before_ms}

    # "N hours ago"
    m = re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+ago", q)
    if m:
        hours = float(m.group(1))
        return {"after_ms": now_ms - int(hours * 3600 * 1000), "before_ms": now_ms}

    # "an hour ago"
    if re.search(r"\ban hour ago\b", q):
        return {"after_ms": now_ms - 3600 * 1000, "before_ms": now_ms}

    # "N minutes ago"
    m = re.search(r"(\d+)\s+minutes?\s+ago", q)
    if m:
        mins = int(m.group(1))
        return {"after_ms": now_ms - mins * 60 * 1000, "before_ms": now_ms}

    # "recently" / "just now"
    if re.search(r"\brecently\b|\bjust now\b", q):
        return {"after_ms": now_ms - 30 * 60 * 1000, "before_ms": now_ms}

    return {"after_ms": None, "before_ms": None}

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QueryParser:
    """Parses a natural-language query into structured intent + filters."""

    def parse(self, query_text: str) -> dict:
        q = query_text.strip()

        # Intent
        q_lower = q.lower()
        if re.search(r"\bsummarise\b|\bsummarize\b|\bsummary\b|\bsummarise\b", q_lower):
            intent = "summarise"
        elif re.search(r"\brecall\b|\bremember\b|\bwhat did\b|\bwhat was\b", q_lower):
            intent = "recall"
        else:
            intent = "recall"  # default

        # Time filter
        time_filter = _extract_time_filter(q)

        # Keywords — simple tokenise + stopword removal
        tokens = re.findall(r"[a-zA-Z]+", q_lower)
        keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

        result = {
            "intent": intent,
            "time_filter": time_filter,
            "keywords": keywords,
        }
        Logger.log("DEBUG", "query_parser", f"Parsed query", metadata=result)
        return result
