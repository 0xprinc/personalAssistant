"""UI Layer stub."""
from jarvis.infra.logger import Logger

class UILayer:
    """Stub for Menu bar app visual display."""
    def update(self, event_type: str, data: dict):
        Logger.log("INFO", "ui", f"[ui] stub called for event {event_type}")
