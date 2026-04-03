"""Device Priority Manager for Jarvis.

Selects the best available audio input device in this priority order:
  1. MacBook built-in microphone (best VAD accuracy — Silero is trained on it)
  2. Any wired headset microphone
  3. System default input device (Bluetooth, iPhone mic, etc. as last resort)

Bluetooth devices (AirPods, iPhone mic) are deprioritised because their
SCO audio profile produces a different spectral shape that Silero VAD was
not trained on, resulting in near-zero confidence scores.
"""

import sounddevice as sd

from jarvis.infra.logger import Logger


# Devices to prefer first (best audio quality for VAD)
_PREFERRED_KEYWORDS = ["iphone", "continuity", "external", "usb", "headset"]

# Built-in mic — decent but lower gain than iPhone Continuity
_BUILTIN_KEYWORDS = ["macbook", "built-in", "internal"]

# Bluetooth — worst (SCO codec, spectral differences)
_BLUETOOTH_KEYWORDS = ["airpods", "bluetooth", "wireless"]


class DevicePriorityManager:
    """Manages audio input device selection, preferring built-in mic."""

    def __init__(self):
        self._active_device_id: int = 0
        self._active_device_name: str = ""
        self._sample_rate: int = 48000
        self._select_best_device()

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _select_best_device(self) -> None:
        """Pick the best input device using priority rules."""
        all_devices = sd.query_devices()
        default_in = sd.default.device[0]

        candidates = []
        for idx, dev in enumerate(all_devices):
            if dev["max_input_channels"] < 1:
                continue
            name_lower = dev["name"].lower()
            native_rate = int(dev["default_samplerate"])

            # Score: 0 = best, higher = worse
            if any(k in name_lower for k in _PREFERRED_KEYWORDS):
                score = 0   # iPhone/external — best signal
            elif any(k in name_lower for k in _BUILTIN_KEYWORDS):
                score = 1   # Built-in Mac mic — good but lower gain
            elif any(k in name_lower for k in _BLUETOOTH_KEYWORDS):
                score = 3   # Bluetooth — worst
            elif idx == default_in:
                score = 2   # Other system default
            else:
                score = 2

            candidates.append((score, idx, dev["name"], native_rate))

        if not candidates:
            Logger.log("ERROR", "device_priority", "No input devices found.")
            return

        # Sort by score (ascending) then device index (ascending for stability)
        candidates.sort(key=lambda x: (x[0], x[1]))
        score, idx, name, native_rate = candidates[0]

        self._active_device_id = idx
        self._active_device_name = name
        self._sample_rate = native_rate

        Logger.log(
            "INFO",
            "device_priority",
            f"Selected audio input: [{idx}] {name} "
            f"(native={native_rate}Hz, priority_score={score})",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_source(self) -> dict:
        """Return details about the selected input device."""
        return {
            "device_id": self._active_device_id,
            "device_name": self._active_device_name,
            "sample_rate": self._sample_rate,
        }

    def get_active_source_id(self) -> str:
        """Return device ID as string (backwards compatibility)."""
        return str(self._active_device_id)

    def set_override(self, device_id: int) -> None:
        """Manually force a specific device by index."""
        try:
            dev = sd.query_devices(device_id, "input")
            self._active_device_id = device_id
            self._active_device_name = dev["name"]
            self._sample_rate = int(dev["default_samplerate"])
            Logger.log(
                "INFO",
                "device_priority",
                f"Manual override: [{device_id}] {dev['name']} "
                f"(native={self._sample_rate}Hz)",
            )
        except Exception as exc:
            Logger.log("ERROR", "device_priority", f"Override failed: {exc}")
