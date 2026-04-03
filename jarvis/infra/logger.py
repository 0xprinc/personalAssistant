"""Logger module for Jarvis.
Outputs JSON structured logging.
"""
import json
import time

class Logger:
    @staticmethod
    def log(level: str, module: str, message: str, metadata: dict = None):
        """Constructs a JSON structured log message."""
        entry = {
            "timestamp": time.time(),
            "level": level.upper(),
            "module": module,
            "message": message,
        }
        if metadata is not None:
            entry["metadata"] = metadata
        
        # In a real environment, might target a file handler, 
        # but printing to console fulfills the base smoke-test requirement.
        print(json.dumps(entry))
