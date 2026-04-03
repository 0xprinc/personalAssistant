"""Liquid Buffer stub."""
from jarvis.infra.logger import Logger

class LiquidBuffer:
    """Stub for 10 min rolling RAM buffer."""
    def __init__(self):
        self.buffer = []

    def insert(self, chunk: dict):
        Logger.log("INFO", "liquid_buffer", "[liquid_buffer] insert stub called")
        self.buffer.append(chunk)

    def get_recent(self, since_ms: int) -> list[dict]:
        Logger.log("INFO", "liquid_buffer", "[liquid_buffer] get_recent stub called")
        return self.buffer
