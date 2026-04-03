# Jarvis — Description & Background
*Passive Personal Memory System | April 2026*

---

## 1. What Is This

Jarvis is a passive, always-on personal memory system that continuously listens through your microphone, transcribes everything you say, converts it into vector embeddings in real time, and stores them in a persistent local database.

You can then semantically query your own spoken history — asking questions like "what was I talking about Tuesday evening?" — and get back accurate, context-aware answers synthesised by an LLM.

The core privacy promise: all recording, transcription, and embedding creation happens entirely on your device. The only optional network call is the LLM query at response time, which can also be run locally when offline.

---

## 2. The Problem It Solves

Human memory is lossy. We forget conversations, ideas, and decisions within hours. Existing solutions either:

- Require manual note-taking (friction kills adoption)
- Store data in the cloud (privacy risk, requires subscription)
- Are triggered by wake words (miss passive context)
- Only record structured meetings, not ambient personal thought

Jarvis solves this by being passive — it runs in the background, capturing everything without requiring any deliberate action from the user. The result is a complete, searchable record of your own spoken thought.

---

## 3. Memory Model

### Long-Term Memory
Everything from installation up to 10 minutes ago. Stored as vector embeddings in a persistent local database (FAISS for MVP, Qdrant for Phase 2). Semantically searchable by natural language query.

### Liquid Memory (Short-Term Buffer)
The last 10 minutes held in an active in-memory buffer. Not yet embedded — available immediately for fast recall with near-zero latency. Flushed to the vector database on a rolling basis every 10 minutes.

### Why Two Tiers
Embedding creation has a small processing cost. Forcing every sentence through the embedding pipeline in real time would create latency. The liquid buffer absorbs the last 10 minutes as raw text, making recent context instantly accessible while the embedding queue processes asynchronously in the background.

---

## 4. Device Priority Logic

- If laptop and phone are both active: capture from laptop
- If only phone is active: capture from phone (Phase 2)
- Manual override: user can force phone-only mode at any time
- Multi-device vector sync: Phase 2 concern, not MVP

---

## 5. Privacy Architecture

Privacy is the primary product differentiator. The system is designed so that sensitive data never has to leave the device:

- Audio captured and processed entirely on-device
- Transcription runs locally via Moonshine v2 Tiny (MLX-optimised for Apple Silicon)
- Embeddings generated locally via BGE-small-en-v1.5 — no cloud embedding API
- Vector database stored on local disk — no sync by default
- Optional PII redaction layer before anything hits the vector store
- LLM query step: defaults to Claude API (cloud) but falls back to Llama 3.2 1B locally when offline

The freemium model preserves this: local-only is free and private. Optional cloud sync or cross-device features are paid add-ons that the user consciously opts into.

---

## 6. Target Hardware — MVP

| Attribute | Value |
|---|---|
| Device | MacBook Air M1 |
| RAM | 8 GB unified memory |
| OS | macOS 12+ (Monterey or later) |
| AI runtime | MLX (Apple Silicon native) |
| Total AI pipeline RAM budget | ~3 GB (leaves headroom for macOS + app shell) |

---

## 7. Monetisation Strategy

### Free Tier
Fully local, fully private. No account required. All core functionality available: continuous recording, embedding, semantic search, local LLM fallback.

### Paid Tier
Optional cloud sync for cross-device access. Mobile app with phone-side capture. Priority cloud LLM responses. Team/enterprise on-prem deployment for meeting knowledge capture.

### Non-Negotiable Constraint
The core privacy promise is never compromised for revenue. Cloud features are always strictly opt-in.

---

## 8. Future Scope (Post-MVP)

- Two-sided call recording (Google Meet, Zoom) via bot attendee — similar to Fireflies.ai
- System audio capture (mic + speaker) using BlackHole on Mac or VB-Audio on Windows
- Multi-device vector DB sync (Qdrant distributed deployment)
- Mobile companion app for phone-side capture
- Enterprise on-prem tier

Note: Two-sided recording carries legal disclosure requirements in many jurisdictions. This must be handled explicitly in Phase 2 with consent flows.