"""Memory Manager stub."""
from jarvis.infra.logger import Logger

class MemoryManager:
    """Orchestrates liquid-to-long-term flush."""
    def flush(self):
        Logger.log("INFO", "memory_manager", "[memory_manager] stub called for flush")
