You are continuing development of Jarvis. Milestones 1.1, 1.2, and 1.3 
are complete. STT is confirmed working and producing accurate transcripts.

Read architecture.md and roadmap.md before starting.

---

PRE-TASK A — Fix test harness sleep bug

In main.py, find time.sleep(2000) in test_stt_pipeline() and replace with 
time.sleep(20). This is a seconds value, not milliseconds. 2000 seconds is 
33 minutes — clearly wrong.

---

PRE-TASK B — Fix chunker minimum word threshold

Chunker is emitting single-word and short utterances as individual chunks:
"Yo", "Handle handle", "Hello hello". These must not become memory chunks.

Fix chunker.py so that:
- Chunks accumulate until >= 50 words
- A chunk is only emitted when one of these is true:
    1. Accumulated text has reached >= 50 words
    2. The SpeechSegment has ended and remaining accumulated text is >= 10 words
- Fragments under 10 words are held in a carry-over buffer and prepended 
  to the next incoming transcript
- Utterances under 3 words ("Yo", "Uh", "Okay", "Hello hello") are 
  discarded entirely — do not add to carry-over buffer either
- The long transcript ("You are a developer or a recruiter...") is a perfect 
  example of a valid chunk — that should pass through as one chunk intact

Do not continue to the main task until chunker produces zero sub-10-word 
chunks in a test run.

---

YOUR TASK: Complete Milestone 1.4 — Memory Pipeline

Replace stubs for these modules with real implementations:
- modules/memory/liquid_buffer.py
- modules/memory/embedding_bge.py
- modules/memory/vector_store_faiss.py
- modules/memory/memory_manager.py

---

STEP 1 — Install dependencies

Run:
  pip install sentence-transformers faiss-cpu numpy

---

STEP 2 — modules/memory/liquid_buffer.py

Real implementation requirements:
- Thread-safe in-memory ring buffer holding chunks from the last 10 minutes
- Use collections.deque with time-based eviction
- Each entry stores the full chunk dict:
    {chunk_id, chunk_text, timestamp_start, timestamp_end, session_id}
- chunk_id: generate as UUID4 on insert
- session_id: UUID4 generated at AudioCapture start, shared across all 
  chunks from the same continuous recording session
- Expose:
    insert(chunk: dict) -> None
    get_recent(since_ms: int) -> list[dict]
    get_all() -> list[dict]
    flush_before(cutoff_ms: int) -> list[dict]  # removes and returns chunks older than cutoff
- Eviction: chunks older than liquid_buffer_ttl_minutes (from config.yaml, 
  default 10) are automatically evicted on every insert call
- Log buffer size every 10 inserts at DEBUG level
- Must satisfy LiquidBuffer interface contract

---

STEP 3 — modules/memory/embedding_bge.py

Real implementation requirements:
- Load BGE-small-en-v1.5 via sentence-transformers on init:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
- Implement embed(text: str) -> list[float] returning 384-dimensional vector
- Implement embed_batch(texts: list[str]) -> list[list[float]] for batch 
  processing during flush operations
- Normalise all embeddings to unit length — BGE requires this for cosine 
  similarity to work correctly:
    vector = vector / np.linalg.norm(vector)
- Load model as module-level singleton — load once, reuse across all calls
- Log model load time on init
- Log embedding inference time per call at DEBUG level
- Must satisfy EmbeddingEngine ABC from interfaces/embedding.py

---

STEP 4 — modules/memory/vector_store_faiss.py

Real implementation requirements:
- Use faiss.IndexFlatIP (inner product index — works correctly with 
  unit-normalised vectors for cosine similarity)
- Index dimensions: 384
- Load existing index from disk on init if file exists, create new if not
- Persist index to disk after every upsert batch using path from 
  config.yaml vector_store_path
- Store metadata in a Python dict keyed by integer FAISS index ID:
    metadata_store[faiss_id] = {
        chunk_id, text, timestamp_start, timestamp_end,
        device_id, session_id, confidence, redacted
    }
- Persist metadata_store as a .pkl file alongside the FAISS index
- Implement:
    upsert(chunk: dict, vector: list[float]) -> str  # returns chunk_id
    search(query_vector: list[float], top_k: int, filters: dict) -> list[dict]
- search() must support optional time-range filters:
    {"after_ms": int, "before_ms": int}
  Fetch top_k * 10 from FAISS, then post-filter by timestamp, return top_k
- Log total index size after every upsert batch
- Must satisfy VectorStore ABC from interfaces/vector_store.py

---

STEP 5 — modules/memory/memory_manager.py

Real implementation requirements:
- Background thread waking every memory_flush_interval_seconds (config.yaml,
  default 60)
- On each wake cycle:
    1. Call liquid_buffer.flush_before(now_ms - 10_minutes_in_ms)
    2. For each flushed chunk: call embedding_bge.embed(chunk.chunk_text)
    3. Call vector_store.upsert(chunk, vector)
    4. Log count of chunks flushed and embedded this cycle
- Expose flush_now() for manual triggering in tests
- Handle empty flush gracefully — log "nothing to flush" and skip
- Must satisfy MemoryManager interface contract

---

STEP 6 — Wire into capture path

Update main.py capture path:
  Chunker → LiquidBuffer (synchronous, every chunk)
           → EmbeddingQueue (async via MemoryManager background thread)

LiquidBuffer.insert() is called immediately on every chunk in the capture 
thread. MemoryManager handles embedding and persistence asynchronously.

---

STEP 7 — Integration test

Add test_memory_pipeline() to main.py that:
- Runs full capture path for 30 seconds (fix: use time.sleep(30) not 2000)
- After capture stops, calls memory_manager.flush_now() immediately
- Embeds a hardcoded test query string using embedding_bge.embed():
    query = "developer recruiter startup reaching out"
- Calls vector_store.search(query_vector, top_k=3, filters={})
- Prints each retrieved chunk: text, timestamp_start, timestamp_end, 
  similarity score
- Confirms FAISS .index file and metadata .pkl file exist on disk
- Stops all threads cleanly with no hanging processes

After test completes, restart Python and run test_memory_pipeline() again 
without speaking. Confirm the previously stored chunks are loaded from disk 
and are still searchable. This validates persistence.

---

CONSTRAINTS:
- Do not touch interfaces/, infra/, or query/output stubs
- All config values from config.yaml via config_manager
- Thread safety: queue.Queue for all inter-thread communication
- FAISS index and metadata .pkl must survive process restart
- BGE model downloads automatically on first run — no manual steps needed
- sentence-transformers will cache the model locally after first download