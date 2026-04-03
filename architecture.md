# Jarvis — Architecture
*Full System Design | April 2026*

---

## 1. Design Principles

- **Modular** — Plug-and-play architecture. Every component is independently replaceable.
- **Interface-driven** — Each module exposes a typed interface contract. Swap internals freely as long as the contract is met.
- **Local-first** — All sensitive processing (audio, STT, embeddings) runs on-device by default.
- **Async pipeline** — Embedding creation never blocks the audio capture or transcription path.
- **Fail-safe** — If the LLM API is unreachable, fall back to local Llama 3.2 1B automatically.

---

## 2. End-to-End Data Flow

### Capture Path (always running)

1. CoreAudio captures 16kHz 16-bit PCM from the active microphone
2. Silero VAD detects speech segments, discards silence
3. Moonshine v2 Tiny transcribes each speech segment with timestamps
4. Text Cleaner deduplicates, punctuates, and optionally redacts PII
5. Chunker splits clean text into ~50-word memory units
6. Chunks enter the Liquid Buffer (last 10 min, in-memory)
7. Chunks also enter the Embedding Queue (async, non-blocking)
8. Embedding Engine converts chunks to 384-dim vectors
9. Vectors + metadata written to FAISS (local disk)

### Query Path (on-demand)

1. User speaks or types a query
2. Query Parser extracts intent and time filters
3. Query embedded via same Embedding Engine
4. Retriever fetches top-k chunks from FAISS + Liquid Buffer
5. Context Builder assembles a prompt with retrieved chunks
6. LLM Engine (Claude API or local Llama fallback) generates answer
7. TTS Engine (Kokoro 82M, loaded on-demand) synthesises speech
8. Response Player outputs audio

---

## 3. Full Component Map

Every module below has a defined interface contract. The internal implementation is swappable without changing any other module.

### Layer 1 — Input

| Module | What It Does | Interface Contract |
|---|---|---|
| Audio Capture | Grabs raw mic audio from active device | OUT: 16kHz PCM stream |
| VAD | Detects speech vs silence before STT | IN: PCM stream → OUT: speech/silence events + pre-buffer |
| Device Priority Manager | Selects laptop vs phone audio source | OUT: active_source_id, override API |

### Layer 2 — Processing

| Module | What It Does | Interface Contract |
|---|---|---|
| STT Engine | Converts speech audio to text | IN: audio_chunk → OUT: {text, start_ms, end_ms, confidence} |
| Text Cleaner | Dedup, punctuation, PII redaction | IN: raw_transcript → OUT: clean_text |
| Chunker | Splits text into memory units | IN: clean_text → OUT: [{chunk_text, timestamp}] |

### Layer 3 — Memory

| Module | What It Does | Interface Contract |
|---|---|---|
| Embedding Engine | Converts text chunk to float vector | IN: text → OUT: float[384] |
| Liquid Buffer | Holds last 10 min of chunks in RAM | IN: chunk → OUT: recent_chunks(since_ms) |
| Vector Store | Persists and searches embeddings on disk | IN: (vector, metadata) → OUT: top_k_chunks(query_vector, filters) |
| Memory Manager | Orchestrates liquid-to-long-term flush | Internal — talks to Buffer and Vector Store only |

### Layer 4 — Query

| Module | What It Does | Interface Contract |
|---|---|---|
| Query Parser | Extracts intent and time filters from natural language | IN: query_text → OUT: {intent, time_filter, keywords} |
| Retriever | Fetches relevant memory chunks | IN: query_vector + filters → OUT: ranked_chunks[] |
| Context Builder | Assembles LLM prompt from retrieved chunks | IN: chunks[] + query → OUT: formatted_prompt_string |
| LLM Engine | Generates natural language answer | IN: prompt → OUT: {answer_text, source_chunks[]} |

### Layer 5 — Output

| Module | What It Does | Interface Contract |
|---|---|---|
| TTS Engine | Converts answer text to speech audio | IN: text → OUT: audio_stream (24kHz) |
| Response Player | Plays synthesised audio | IN: audio_stream → OUT: speaker |
| UI Layer | Visual display of transcripts, answers, memory stats | IN: events (transcript, answer, memory_count) → OUT: rendered UI |

### Layer 6 — Infrastructure

| Module | What It Does | Interface Contract |
|---|---|---|
| Config Manager | All model choices, paths, API keys — single source of truth | OUT: config object consumed by every module at init |
| Logger | Timestamped structured event log | IN: (level, module, message, metadata) → OUT: log file + console |
| Privacy Controller | Decides what gets stored, redacted, or deleted | Applied as middleware before anything hits the vector store |
| Sync Manager | Phase 2 multi-device sync | IN: local DB state → OUT: synced remote state |

---

## 4. Model Assignments

| Module | MVP Choice | Swap Candidate | RAM (MVP) |
|---|---|---|---|
| Audio Capture | CoreAudio (macOS native) | PortAudio (cross-platform) | ~0 MB |
| VAD | Silero VAD (ONNX) | WebRTC VAD | ~5 MB |
| STT Engine | Moonshine v2 Tiny (MLX) | Faster-Whisper, Apple STT | ~450 MB |
| Embedding Engine | BGE-small-en-v1.5 | all-MiniLM-L6-v2, EmbeddingGemma | ~130 MB |
| Vector Store | FAISS (file-based) | Qdrant, ChromaDB, memvid | ~150 MB |
| LLM Engine | Claude API (cloud default) | Llama 3.2 1B Q4 (offline fallback) | API: ~0 MB / Local: ~700 MB |
| TTS Engine | Kokoro 82M (on-demand) | Piper TTS, CosyVoice 2 | ~1.8 GB (unloaded when idle) |

---

## 5. Vector Store Schema

Each memory chunk stored in the vector database carries the following metadata:

| Field | Type | Description |
|---|---|---|
| chunk_id | UUID | Unique identifier for this memory chunk |
| text | string | The raw transcribed text of this chunk |
| vector | float[384] | BGE-small embedding of the text |
| timestamp_start | unix ms | When this chunk started being spoken |
| timestamp_end | unix ms | When this chunk finished |
| device_id | string | Source device (laptop / phone) |
| session_id | UUID | Groups chunks from the same continuous speaking session |
| confidence | float 0–1 | STT confidence score for this chunk |
| redacted | boolean | Whether PII was detected and redacted |

---

## 6. RAM Budget — M1 Air 8GB

| Component | Resident RAM | Notes |
|---|---|---|
| macOS baseline | ~2.5 GB | System reserve |
| Moonshine v2 Tiny (STT) | ~450 MB | Always resident |
| Silero VAD | ~5 MB | Always resident |
| BGE-small-en-v1.5 (embeddings) | ~130 MB | Always resident |
| FAISS vector store | ~150 MB | Grows with memory count |
| Liquid buffer + app logic | ~100 MB | 10 min rolling window |
| Kokoro 82M (TTS) | ~1.8 GB | Loaded on-demand, unloaded after 30s idle |
| Llama 3.2 1B Q4 (offline LLM) | ~700 MB | Only loaded when offline, via Ollama |
| **TOTAL (active query, TTS loaded)** | **~5.8 GB** | **Fits in 8 GB with headroom** |

---

## 7. Interface Contracts

All modules communicate through typed interfaces. Rule: if your replacement returns the same shape, plug it in and nothing else changes.

### STT Engine
```
transcribe(audio_chunk: PCMAudio) -> TranscriptResult
TranscriptResult: { text: string, start_ms: int, end_ms: int, confidence: float }
```

### Embedding Engine
```
embed(text: string) -> float[384]
```

### Vector Store
```
upsert(chunk: MemoryChunk) -> chunk_id
search(query_vector: float[384], filters: QueryFilters, top_k: int) -> MemoryChunk[]
```

### LLM Engine
```
generate(prompt: string) -> LLMResponse
LLMResponse: { answer: string, source_chunks: string[] }
```

### TTS Engine
```
synthesise(text: string) -> AudioStream
```

---

## 8. Suggested Directory Structure

```
jarvis/
├── config/
│   └── config.yaml                  # Single source of truth for all settings
├── interfaces/
│   ├── stt.py                       # STT ABC
│   ├── embedding.py                 # Embedding ABC
│   ├── vector_store.py              # Vector Store ABC
│   ├── llm.py                       # LLM Engine ABC
│   └── tts.py                       # TTS ABC
├── modules/
│   ├── input/
│   │   ├── audio_capture.py         # CoreAudio tap
│   │   ├── vad.py                   # Silero VAD
│   │   └── device_priority.py      # Laptop vs phone selection
│   ├── processing/
│   │   ├── stt_moonshine.py         # Moonshine v2 Tiny implementation
│   │   ├── text_cleaner.py          # Dedup, punctuation, PII
│   │   └── chunker.py               # ~50-word chunk splitter
│   ├── memory/
│   │   ├── embedding_bge.py         # BGE-small-en-v1.5 implementation
│   │   ├── liquid_buffer.py         # 10-min in-memory ring buffer
│   │   ├── vector_store_faiss.py    # FAISS implementation
│   │   └── memory_manager.py        # Flush orchestration
│   ├── query/
│   │   ├── query_parser.py          # Intent + time filter extraction
│   │   ├── retriever.py             # FAISS + buffer search
│   │   ├── context_builder.py       # Prompt assembly
│   │   └── llm_claude.py            # Claude API implementation
│   │   └── llm_llama.py             # Llama 3.2 1B fallback
│   └── output/
│       ├── tts_kokoro.py            # Kokoro 82M implementation
│       ├── response_player.py       # CoreAudio playback
│       └── ui.py                    # Menu bar app
└── infra/
    ├── config_manager.py
    ├── logger.py
    └── privacy_controller.py
```