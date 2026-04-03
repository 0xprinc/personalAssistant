"""Privacy Controller stub."""
from jarvis.infra.logger import Logger

class PrivacyController:
    """Decides what gets stored, redacted, or deleted."""
    def apply(self, chunk: dict) -> dict:
        Logger.log("INFO", "privacy_controller", "[privacy_controller] stub called")
        chunk["redacted"] = False
        return chunk
