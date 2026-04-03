You are continuing development of Jarvis. Milestones 1.1 through 1.6 are 
complete. The full pipeline works end-to-end: capture → STT → memory → 
query → LLM → TTS → audio playback.

Read architecture.md and roadmap.md before starting.

Scope constraint for this entire prompt: laptop only. No iOS, no phone 
fallback, no multi-device logic. Device Priority Manager is simplified 
to laptop microphone selection only.

---

YOUR TASK A: Milestone 1.7 — Device Priority Logic (Laptop Only)

Update modules/input/device_priority.py:

- Expose get_active_source() -> dict:
    {device_id: int, device_name: str, sample_rate: int, source_type: str}
  source_type is always "laptop" — hardcode this, no phone logic
- Expose set_override(device_id: int) -> None:
    - Validates device_id exists in sounddevice.query_devices()
    - If invalid: log error and keep current device unchanged
    - If valid: switch active source and log the change
- Expose list_devices() -> list[dict]:
    - Returns all available input devices:
      {id, name, sample_rate, max_input_channels}
    - Filters out devices with max_input_channels == 0
- On init: use the OS default input device. If default has 
  max_input_channels == 0, scan and pick the first valid input device.
- Log all available input devices at DEBUG level on startup
- Sync Manager stub remains untouched — do not implement it
- Must satisfy DevicePriorityManager interface contract

---

YOUR TASK B: Milestone 1.8 — macOS Menu Bar App

Create modules/output/ui.py (replace stub) and jarvis_app.py (new file).

STEP 1 — Install dependency:
  pip install rumps

STEP 2 — Menu bar app (modules/output/ui.py):

Use rumps. Menu structure:

  ● Jarvis — Active
  ─────────────────
  🎤 Pause Recording
  🔍 Query Memory...
  📋 Recent Memories     ← submenu, last 5 chunks
  ─────────────────
  ⚙️  Settings
      └─ Show Device Info
      └─ Open Log File
  ─────────────────
  Quit Jarvis

Status indicator behaviour:
- "● Jarvis — Active"    when capture is running
- "○ Jarvis — Paused"    when capture is paused
- "⏳ Jarvis — Thinking" while query is processing

Pause Recording:
- Calls audio_capture.stop() when pausing
- Calls audio_capture.start() when resuming
- Updates status item text accordingly

Query Memory dialog:
- Uses rumps.Window for text input
- Runs full query path in a background thread:
    query_parser → retriever → context_builder → llm_engine → tts → player
- Sets status to "⏳ Jarvis — Thinking..." while processing
- Restores status to "● Jarvis — Active" after answer
- Displays answer text in rumps.alert dialog
- Never blocks the main thread

Recent Memories submenu:
- Shows last 5 chunks from liquid_buffer.get_all()
- Each item: first 60 chars of chunk_text + human readable timestamp
- Refreshes on every menu open

Settings → Show Device Info:
- Calls device_priority.get_active_source()
- Logs result to console

Settings → Open Log File:
- os.system("open -a Console " + log_file_path)
- log_file_path from config.yaml

Quit Jarvis:
- audio_capture.stop()
- memory_manager.flush_now()
- tts_kokoro.unload()
- rumps.quit_application()
- All threads must exit cleanly — no hanging processes

STEP 3 — jarvis_app.py (new file, project root):

Production entry point. Startup sequence in this exact order:

  1.  config_manager.load()
  2.  logger.init()
  3.  device_priority.init()
  4.  audio_capture.init()
  5.  vad.init()
  6.  stt_moonshine.init()
  7.  embedding_bge.init()
  8.  vector_store_faiss.init()   ← loads existing FAISS index from disk
  9.  liquid_buffer.init()
  10. memory_manager.init() + start()
  11. audio_capture.start()
  12. JarvisMenuBarApp().run()     ← blocks here, must be last

Handle KeyboardInterrupt for clean shutdown at any point.

If rumps import fails (non-macOS): log error and exit with message:
  "Jarvis menu bar requires macOS. Use main.py for testing."

STEP 4 — Validation:

Run:
  python jarvis_app.py

Confirm all of these:
- Menu bar icon appears in macOS status bar
- Pause Recording toggles capture on and off, status updates correctly
- Query Memory dialog accepts text, processes in background, speaks answer
- Recent Memories submenu shows real chunks from previous test runs
- Show Device Info logs correct laptop microphone name
- Quit exits cleanly with flush_now() logged and no hanging threads

---

CONSTRAINTS:
- rumps runs on main thread only — all pipeline work in background threads
- No iOS, no phone fallback, no Sync Manager implementation
- jarvis_app.py is the production entry point going forward
- main.py remains untouched — keep it for pipeline tests
- All config values from config_manager, nothing hardcoded
- Menu bar icon title: use "𝐉" or "◈"