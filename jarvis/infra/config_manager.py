"""Config Manager for Jarvis.
Reads config.yaml once and makes it available globally.
"""
import yaml
from pathlib import Path

# Compute the absolute path to config.yaml relative to this script
_config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

def _load_config():
    if _config_path.exists():
        with open(_config_path, "r") as f:
            return yaml.safe_load(f)
    print(f"Warning: Configuration file not found at {_config_path}")
    return {}

config = _load_config()
