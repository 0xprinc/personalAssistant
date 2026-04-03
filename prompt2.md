You are continuing development of Jarvis. The scaffold from Milestone 1.1 is 
already complete and working. Do not modify any interfaces/, infra/, or 
main.py unless explicitly told to.

Read architecture.md and roadmap.md before starting.

---

YOUR TASK: Complete Milestone 1.2 — Audio Pipeline only.

Replace the stubs for these three modules with real implementations:
- modules/input/audio_capture.py
- modules/input/vad.py
- modules/input/device_priority.py

---

STEP 1 — Install dependencies

Run these exactly:
  pip install sounddevice numpy
  pip install onnxruntime
  pip install requests  # for downloading Silero VAD model

---

STEP 2 — modules/input/device_priority.py

Real implementation requirements:
- Use sounddevice to enumerate all available audio input devices
- Select the default input device automatically on startup
- Expose get_active_source() -> dict with keys: device_id, device_name, 
  sample_rate
- Expose set_override(device_id: int) for manual selection
- Log the selected device name and index on startup
- Must still satisfy the DevicePriorityManager interface contract from 
  interfaces/

---

STEP 3 — modules/input/audio_capture.py

Real implementation requirements:
- Use sounddevice InputStream to capture audio continuously
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit PCM (dtype=int16)
- Block size: 512 samples (~32ms per block)
- Run capture in a background thread — never block the main thread
- Push each audio block into a thread-safe queue (use queue.Queue)
- Expose start() and stop() methods
- Log captured block count every 100 blocks
- Must still satisfy the AudioCapture interface contract

---

STEP 4 — modules/input/vad.py

Real implementation requirements:
- Download Silero VAD ONNX model from:
  https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
  Save to the path specified in config.yaml under vad_model_path
  Skip download if file already exists
- Load the model using onnxruntime.InferenceSession
- Process audio blocks from the AudioCapture queue
- Maintain a 2-second pre-buffer (ring buffer of raw PCM blocks)
- Speech detection logic:
    - Run Silero VAD on each 512-sample block
    - If VAD confidence > vad_threshold (from config.yaml, default 0.5):
        mark as speech
    - If speech detected: emit a SpeechSegment containing:
        - The 2-second pre-buffer + all subsequent speech blocks
        - start_ms timestamp
    - If silence detected for > 700ms after speech: close the segment,
        emit it downstream, reset buffer
- Expose a get_speech_segment() method that blocks until a segment is ready
- Log VAD confidence score every 50 blocks at DEBUG level
- Must still satisfy the VAD interface contract

---

STEP 5 — Integration test

Update main.py to include a new test function called test_audio_pipeline() 
that:
- Initialises DevicePriorityManager, AudioCapture, and VAD
- Starts audio capture
- Listens for 10 seconds of real microphone input
- Prints each SpeechSegment detected (start_ms, duration_ms, block count)
- Stops cleanly after 10 seconds
- Does NOT replace the existing smoke test — add it as a second test

Run the integration test and confirm:
- Audio capture starts without errors
- At least one SpeechSegment is detected when you speak into the microphone
- Clean shutdown with no hanging threads

---

CONSTRAINTS:
- All config values (sample rate, block size, vad_threshold, model path) must 
  be read from config.yaml via config_manager — no hardcoded values
- No ML model loading beyond Silero VAD ONNX in this prompt
- Thread safety is non-negotiable — use queue.Queue for all inter-thread 
  communication, never shared mutable state
- All other stubs remain untouched
- If sounddevice raises a PortAudio error on M1, add a fallback that logs the 
  error and exits cleanly with a helpful message pointing to:
  brew install portaudio