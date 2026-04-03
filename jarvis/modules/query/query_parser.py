"""Query Parser stub."""
from jarvis.infra.logger import Logger

class QueryParser:
    """Extracts intent and time filters from natural language."""
    def parse(self, query_text: str) -> dict:
        Logger.log("INFO", "query_parser", "[query_parser] stub called")
        return {
            "intent": "search",
            "time_filter": "last 24 hours",
            "keywords": ["test"]
        }
