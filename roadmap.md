# Jarvis — Roadmap
*Build Plan | April 2026*

---

## Phase Overview

| Phase | Name | Goal | Target |
|---|---|---|---|
| 1 | MVP — Passive Memory Core | Single device, always-on, queryable | 8 weeks |
| 2 | Multi-Device + Call Recording | Phone fallback, meeting capture | 12 weeks post-MVP |
| 3 | Product + Monetisation | Consumer polish, freemium launch | TBD |

---

## Phase 1 — MVP

Goal: a working single-device passive memory system on MacBook Air M1 that continuously records, embeds, and answers queries about your spoken history.

---

### 1.1 — Project Scaffold

- Initialise repo with modular directory structure (one folder per module/layer)
- Set up Config Manager — single `config.yaml` consumed by all modules
- Set up Logger with structured JSON output
- Define all interface contracts as Python abstract base classes (ABCs) in `interfaces/`
- Write stub implementations for every module — system must wire together end-to-end before any real model is integrated

---

### 1.2 — Audio Pipeline

- Implement CoreAudio tap via `sounddevice` for continuous 16kHz PCM capture
- Integrate Silero VAD (ONNX) — speech/silence detection with 2-second pre-buffer
- Wire VAD output to STT trigger
- Unit test: VAD correctly segments speech in a noisy environment

---

### 1.3 — STT Integration

- Integrate Moonshine v2 Tiny via MLX runtime
- Implement streaming transcription — process 5-second audio chunks with overlap
- Implement Text Cleaner — deduplication of repeated phrases (common in streaming STT), basic punctuation restoration
- Optional: PII redaction via regex patterns (phone numbers, emails, names)
- Unit test: transcription accuracy on 60 seconds of varied speech

---

### 1.4 — Memory Pipeline

- Implement Chunker — splits clean transcript into ~50-word chunks with timestamp metadata
- Implement Liquid Buffer — in-memory ring buffer holding last 10 minutes of chunks
- Integrate BGE-small-en-v1.5 as Embedding Engine via `sentence-transformers` or ONNX Runtime
- Implement async Embedding Queue — chunks processed in background without blocking capture path
- Integrate FAISS as Vector Store — local index file persisted to disk
- Implement Memory Manager — orchestrates 10-minute flush from Liquid Buffer to FAISS
- Integration test: 30 minutes of speech, verify chunks are retrievable by semantic query

---

### 1.5 — Query Pipeline

- Implement Query Parser — extract time filters ("Tuesday evening", "last week") and intent
- Implement Retriever — embed query, search FAISS + Liquid Buffer, return ranked top-k chunks
- Implement Context Builder — format retrieved chunks into LLM prompt with timestamps
- Integrate Claude API as LLM Engine (primary)
- Integrate Llama 3.2 1B via Ollama as offline fallback — auto-switch when API unreachable
- Integration test: query returns accurate answer about content spoken 2 hours ago

---

### 1.6 — Output Pipeline

- Integrate Kokoro 82M as TTS Engine — load on-demand, unload after 30 seconds idle
- Implement Response Player using CoreAudio
- Wire full query path: spoken query in → answer spoken out

---

### 1.7 — Device Priority Logic

- Implement Device Priority Manager — detect available audio sources
- Default to laptop microphone when available
- Expose manual override API for future phone fallback (Phase 2)

---

### 1.8 — macOS App Shell

- Minimal menu bar app (Swift or Python with `rumps`) — start/stop/status indicator
- Memory viewer: scrollable list of recent transcribed chunks with timestamps
- Query input: text field + voice trigger
- Settings panel: API key entry, model selection dropdowns, privacy controls

---

### Phase 1 Exit Criteria

- Continuous recording runs for 8+ hours without memory leak or crash
- Semantic query returns correct answer for content spoken 24 hours ago
- Total resident RAM under 3 GB when TTS is not loaded
- No audio, transcripts, or embeddings leave the device except the LLM query prompt

---

## Phase 2 — Multi-Device + Call Recording

### 2.1 — Phone Companion App

- iOS app that captures audio and sends transcript chunks to laptop over local network
- Device Priority Manager updated to handle phone fallback
- Seamless handoff: phone activates automatically when laptop microphone is unavailable

### 2.2 — Multi-Device Vector Sync

- Migrate Vector Store from FAISS to Qdrant (supports distributed deployment)
- Implement Sync Manager — bidirectional sync of Qdrant collections across devices
- Conflict resolution: last-write-wins per chunk_id

### 2.3 — System Audio Capture

- macOS: integrate BlackHole virtual audio device for mic + speaker capture
- Windows: integrate VB-Audio Cable
- Two-sided recording enables capturing both sides of phone/computer calls

### 2.4 — Meeting Bot Integration

- Google Meet / Zoom bot attendee (similar to Fireflies.ai architecture)
- Automatic meeting transcription piped into the same memory pipeline
- Legal consent handling — disclosure prompt before joining any meeting

---

## Phase 3 — Product + Monetisation

### 3.1 — Consumer Polish

- Full-featured macOS app replacing the menu bar stub
- Timeline view: visual memory map by day/week
- Memory search UI with source highlighting
- Onboarding flow for first-time setup

### 3.2 — Freemium Launch

- Free tier: local-only, fully private, no account required
- Paid tier: cloud sync, cross-device, priority LLM responses
- Stripe billing integration
- Privacy policy explicitly covering what does and does not leave the device

### 3.3 — Enterprise Tier

- On-prem deployment for teams
- Meeting knowledge capture across the organisation
- Admin dashboard for team memory management
- SSO and role-based access control

---

## Immediate Next Steps

Start here before writing any feature code:

1. Create repo with modular directory structure matching the architecture layer map
2. Define all interface ABCs in `interfaces/` — one file per module
3. Write stub implementations for every module — verify the full pipeline wires together with no-op stubs
4. Integrate Moonshine v2 Tiny + Silero VAD — validate microphone capture on M1
5. Integrate BGE-small-en-v1.5 + FAISS — validate embedding and retrieval roundtrip
6. Wire Claude API as LLM Engine — first end-to-end query test
7. Add Kokoro TTS — first full voice-in, voice-out test