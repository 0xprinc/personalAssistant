You are continuing development of Jarvis. Milestones 1.1 and 1.2 are 
complete but there is one bug to fix before proceeding: VAD confidence 
is always 0.001 and no SpeechSegments are ever emitted. Fix this first, 
then integrate Moonshine STT.

Read architecture.md and roadmap.md before starting.

---

PART A — Fix Silero VAD (do this first, validate before continuing)

The VAD confidence score is always ~0.001, meaning speech is never detected.
Diagnose and fix the root cause. Known causes for this with Silero VAD ONNX:

1. Input shape mismatch — Silero VAD v5 ONNX expects input shape (1, samples)
   not (samples,) or (1, 1, samples). Print the model's expected input shape
   using:
     session.get_inputs()[0].shape
   and match your audio block shape exactly to it.

2. Audio dtype mismatch — model expects float32 normalised to [-1.0, 1.0].
   If audio is int16, convert with:
     audio_float = audio_int16.astype(np.float32) / 32768.0

3. Hidden state not preserved between chunks — Silero VAD v5 is stateful.
   It requires h and c state tensors carried across calls. Initialise them as
   zeros of shape (2, 1, 64) at session start and pass them through every
   inference call, updating them with the returned h and c values each time.

4. Sample rate input — Silero v5 requires sr (sample rate) as a separate 
   int64 input tensor with value 16000.

Fix all four of these if not already handled. After fixing, run the 10-second
integration test again and confirm VAD confidence varies when you speak vs
when you are silent. You must see at least one SpeechSegment emitted before
continuing to Part B.

Show the updated validation log before proceeding.

---

PART B — STT Integration (Milestone 1.3)

Only start this after Part A validation passes.

STEP 1 — Install dependencies

Run:
  pip install git+https://github.com/usefulsensors/moonshine.git
  pip install mlx

If mlx fails (non-Apple-Silicon machine), fall back to:
  pip install moonshine-onnx

---

STEP 2 — modules/processing/stt_moonshine.py

Real implementation requirements:
- Load Moonshine v2 Tiny model on init using MLX runtime
- Model identifier: "moonshine/tiny" 
- If MLX is unavailable, fall back to ONNX runtime version automatically
  and log a warning that MLX was not available
- Implement transcribe(audio_chunk: np.ndarray) -> TranscriptResult where:
    - audio_chunk is float32 PCM at 16kHz
    - TranscriptResult matches the interface exactly:
      {text: str, start_ms: int, end_ms: int, confidence: float}
- Process SpeechSegments emitted by VAD — not raw audio blocks
- Chunk processing: handle segments up to 30 seconds max. If a segment 
  exceeds 30 seconds, split it at silence boundaries before transcribing
- Log each transcription result: text, duration_ms, and inference time in ms
- Must still satisfy the STTEngine ABC from interfaces/stt.py

---

STEP 3 — modules/processing/text_cleaner.py

Real implementation requirements:
- Deduplicate repeated phrases — Moonshine (like Whisper) sometimes repeats
  the last phrase at the start of the next chunk. Detect and remove these.
  Strategy: compare last 5 words of previous output with first 5 words of
  current output. If overlap >= 3 words, strip the overlapping prefix.
- Basic punctuation restoration: capitalise first word of each sentence,
  ensure sentences end with punctuation if missing
- PII redaction (optional, controlled by config.yaml pii_redaction: true/false):
    - Phone numbers: replace with [PHONE]
    - Email addresses: replace with [EMAIL]
    - Credit card patterns: replace with [CARD]
- Return clean_text as a plain string
- Must satisfy the TextCleaner interface contract

---

STEP 4 — modules/processing/chunker.py

Real implementation requirements:
- Split clean text into memory chunks of approximately 50 words
- Never split mid-sentence — always break at sentence boundaries
- Each chunk must carry:
    {chunk_text: str, timestamp_start: int, timestamp_end: int}
  where timestamps are inherited from the SpeechSegment that produced them
- If a single sentence exceeds 50 words, emit it as its own chunk
- Return a list of chunk dicts

---

STEP 5 — Wire the processing layer

Update the capture path in main.py so that:
  AudioCapture → VAD → STT → TextCleaner → Chunker

Each stage hands off to the next in the same background thread pipeline.
Log the final chunk output (text + timestamps) to confirm end-to-end flow.

---

STEP 6 — Integration test

Add test_stt_pipeline() to main.py that:
- Runs the full capture → VAD → STT → TextCleaner → Chunker pipeline
- Listens for 20 seconds of real microphone input
- Prints each chunk produced: text, word count, start_ms, end_ms
- Confirms at least one chunk is produced when you speak

---

CONSTRAINTS:
- Do not touch interfaces/, infra/, or any memory/query/output stubs
- All config values from config.yaml via config_manager
- Thread safety: all inter-stage communication via queue.Queue
- If Moonshine model download fails, log the error clearly and exit with
  instructions to check internet connection
- Inference must run on MPS (Apple Silicon GPU) if available, CPU otherwise