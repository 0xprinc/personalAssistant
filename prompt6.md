You are continuing development of Jarvis. Milestones 1.1 through 1.5 are 
complete. The full capture path and query path are wired. The LLM engine 
is using OpenRouter (Kimi k2.5) with no fallback.

Read architecture.md and roadmap.md before starting.

---

YOUR TASK: Complete Milestone 1.6 — Output Pipeline

Replace stubs for these modules with real implementations:
- modules/output/tts_kokoro.py
- modules/output/response_player.py

Do NOT touch ui.py yet — that is Milestone 1.8.

---

STEP 1 — Install dependencies

Run:
  pip install kokoro sounddevice numpy

If kokoro is not available on PyPI, install from source:
  pip install git+https://github.com/hexgrad/kokoro.git

Check available voices after install:
  python -c "from kokoro import KPipeline; print('kokoro ok')"

---

STEP 2 — modules/output/tts_kokoro.py

Real implementation requirements:
- Import KPipeline from kokoro
- Initialise pipeline lazily — do NOT load on import, only on first 
  synthesise() call. This keeps RAM free until TTS is actually needed.
- Voice selection from config.yaml tts_voice (default: "af_bella")
- Language code from config.yaml tts_lang (default: "a" for American English)
- Implement synthesise(text: str) -> np.ndarray:
    - Run KPipeline on the input text
    - Collect all audio chunks from the generator into a single numpy array
    - Return audio as float32 numpy array at 24000 Hz sample rate
    - If text is empty or whitespace: return empty numpy array, log warning
- Implement unload() -> None:
    - Set pipeline instance to None
    - Call gc.collect() to release RAM
    - Log "TTS unloaded, RAM released"
- Auto-unload: start a 30-second idle timer on every synthesise() call.
  If synthesise() is not called again within 30 seconds, call unload() 
  automatically. Reset the timer on every new call.
  Use threading.Timer for this.
- Log synthesis time in ms for every call
- Log "TTS loaded" when pipeline initialises, "TTS unloaded" when released
- Must satisfy TTSEngine ABC from interfaces/tts.py

---

STEP 3 — modules/output/response_player.py

Real implementation requirements:
- Accept np.ndarray audio at 24000 Hz from TTS engine
- Play audio through the default output device using sounddevice.play()
- Implement play(audio: np.ndarray, sample_rate: int = 24000) -> None:
    - Call sounddevice.play(audio, samplerate=sample_rate)
    - Call sounddevice.wait() to block until playback finishes
    - Log playback duration in seconds
    - If audio array is empty: log warning and return immediately
- Implement stop() -> None:
    - Call sounddevice.stop()
    - Log "Playback stopped"
- Handle sounddevice errors gracefully:
    - If no output device found: log error with message
      "No audio output device found. Check system audio settings."
    - Do not crash — return silently after logging
- Must satisfy ResponsePlayer interface contract

---

STEP 4 — Wire into query path

Update main.py query path so answers are spoken aloud:
  llm_engine.generate()
    → tts_kokoro.synthesise(answer.answer_text)
    → response_player.play(audio)

The text passed to TTS must be answer.answer_text only — not the full 
prompt, not source chunks, just the answer string.

If answer.answer_text is the hardcoded fallback string 
("I could not reach any LLM..."), do NOT pass it to TTS — log it only.

---

STEP 5 — Integration test

Add test_output_pipeline() to main.py (run via: python main.py test_output)

Test sequence:
1. Hardcode a test answer string:
     text = "Hello. I am Jarvis, your personal memory assistant. 
             The pipeline is fully wired and I can speak."
2. Call tts_kokoro.synthesise(text) and measure synthesis time
3. Call response_player.play(audio) — you should hear the voice
4. Wait 35 seconds without calling synthesise() again
5. Confirm auto-unload fires: log should show "TTS unloaded, RAM released"
6. Call synthesise() again with a second string — confirm TTS reloads 
   and speaks again
7. Print: synthesis time (ms), audio duration (seconds), unload confirmed

Also add a full end-to-end test:
  python main.py test_e2e

This test:
1. Runs capture path for 20 seconds — speak a few sentences
2. Immediately queries: "what did I just say?"
3. Passes LLM answer to TTS → plays it aloud
4. Confirms the spoken answer references what you actually said

---

CONSTRAINTS:
- Kokoro runs on CPU on M1 — do not attempt MPS/GPU for TTS
- Do not load TTS model at startup — lazy load only
- Auto-unload after 30 seconds idle is mandatory — this recovers 1.8GB RAM
- All config values (voice, language, idle timeout) from config.yaml
- Do not modify any pipeline modules from previous milestones
- sounddevice is already installed from Milestone 1.2 — do not reinstall