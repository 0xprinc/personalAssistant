"""Device Priority Manager."""
import sounddevice as sd
from jarvis.infra.logger import Logger

class DevicePriorityManager:
    """Manages audio input device selection."""
    def __init__(self):
        self._active_device_id = None
        self._active_device_name = ""
        self._sample_rate = 16000
        self._select_default_device()

    def _select_default_device(self):
        try:
            default_in = sd.default.device[0]
            device_info = sd.query_devices(default_in, 'input')
            self._active_device_id = default_in
            self._active_device_name = device_info['name']
            self._sample_rate = int(device_info['default_samplerate'])
            Logger.log("INFO", "device_priority", f"Selected default audio device: {self._active_device_name} (Index: {self._active_device_id})")
        except Exception as e:
            Logger.log("ERROR", "device_priority", f"Failed to select default device: {e}")

    def get_active_source(self) -> dict:
        """Returns details about the active input device."""
        return {
            "device_id": self._active_device_id,
            "device_name": self._active_device_name,
            "sample_rate": self._sample_rate
        }

    def set_override(self, device_id: int):
        """Manually override the active audio input device."""
        try:
            device_info = sd.query_devices(device_id, 'input')
            self._active_device_id = device_id
            self._active_device_name = device_info['name']
            self._sample_rate = int(device_info['default_samplerate'])
            Logger.log("INFO", "device_priority", f"Overridden audio device: {self._active_device_name} (Index: {self._active_device_id})")
        except Exception as e:
            Logger.log("ERROR", "device_priority", f"Failed to override device {device_id}: {e}")

    def get_active_source_id(self) -> str:
        """Backwards compatibility for interface stub."""
        return str(self._active_device_id)
