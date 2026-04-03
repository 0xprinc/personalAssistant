You are continuing development of Jarvis. Milestones 1.1 through 1.4 are 
complete. The full capture path is working: audio → VAD → STT → chunker → 
liquid buffer → FAISS vector store with persistence confirmed.

Read architecture.md and roadmap.md before starting.

---

YOUR TASK: Complete Milestone 1.5 — Query Pipeline

Replace stubs for these modules with real implementations:
- modules/query/query_parser.py
- modules/query/retriever.py
- modules/query/context_builder.py
- modules/query/llm_claude.py
- modules/query/llm_llama.py

---

STEP 1 — Install dependencies

Run:
  pip install anthropic
  pip install ollama

---

STEP 2 — modules/query/query_parser.py

Real implementation requirements:
- Accept a natural language query string
- Extract and return a structured dict:
    {
      intent: str,          # "recall" | "summarise" | "search"
      time_filter: {
        after_ms: int | None,
        before_ms: int | None
      },
      keywords: list[str]
    }
- Time filter extraction must handle natural language time references:
    "this morning"     → after_ms = today 6am, before_ms = today 12pm
    "yesterday"        → after_ms = yesterday 0:00, before_ms = yesterday 23:59
    "last Tuesday"     → compute correct day relative to today
    "an hour ago"      → after_ms = now - 1hr, before_ms = now
    "last week"        → after_ms = 7 days ago 0:00, before_ms = yesterday 23:59
    No time reference  → both None (search all time)
- Use Python stdlib only for time parsing (datetime, re) — no dateparser 
  dependency
- keywords: extract meaningful nouns and verbs, strip stopwords
  (a, the, is, was, I, me, my, about, what, did, do)
- intent: default to "recall" unless query contains "summarise" or "summary"
- Log parsed result at DEBUG level
- Must satisfy QueryParser interface contract

---

STEP 3 — modules/query/retriever.py

Real implementation requirements:
- Accept: query_text (str), filters (dict), top_k (int, default 5)
- Pipeline:
    1. Embed query_text using embedding_bge.embed()
    2. Search liquid_buffer.get_recent() for chunks matching time filter
       (in-memory search — compare chunk timestamps against filter)
    3. Search vector_store_faiss.search() with query_vector and filters
    4. Merge results from both sources, deduplicate by chunk_id
    5. Re-rank merged results by similarity score descending
    6. Return top_k chunks as list[dict]
- Each returned chunk must include a similarity_score field
- For liquid buffer results, compute cosine similarity manually:
    score = np.dot(query_vector, embed(chunk.chunk_text))
- Log: query text, number of results from each source, final count returned
- Must satisfy Retriever interface contract

---

STEP 4 — modules/query/context_builder.py

Real implementation requirements:
- Accept: chunks (list[dict]), query (str)
- Build a prompt string in this exact format:

    You are Jarvis, a personal memory assistant. The user has asked:
    "{query}"

    Here are the relevant memories retrieved from their spoken history,
    ordered by relevance. Each memory includes when it was spoken.

    --- MEMORIES ---
    [1] {timestamp_human_readable} — {chunk_text}
    [2] {timestamp_human_readable} — {chunk_text}
    ...
    --- END MEMORIES ---

    Based only on the memories above, answer the user's question.
    If the memories do not contain enough information to answer, say so.
    Do not invent information not present in the memories.

- timestamp_human_readable: convert unix ms to "Monday 14 Apr, 2:35 PM"
- If no chunks are provided, return a prompt that tells the LLM no relevant 
  memories were found and to say so to the user
- Must satisfy ContextBuilder interface contract

---

STEP 5 — modules/query/llm_claude.py

Real implementation requirements:
- Use the anthropic Python SDK
- API key from config.yaml claude_api_key (or env var ANTHROPIC_API_KEY 
  as fallback)
- Model: claude-sonnet-4-6 (latest, best quality/cost balance)
- max_tokens: 1024
- Implement generate(prompt: str) -> LLMResponse where:
    LLMResponse: {answer: str, source_chunks: list[str]}
- source_chunks: extract chunk texts from the prompt's MEMORIES section 
  and include them for attribution
- On API error (rate limit, network): log the error, raise an exception 
  so llm_engine.py can catch it and fall back to Llama
- Must satisfy LLMEngine ABC from interfaces/llm.py

---

STEP 6 — modules/query/llm_llama.py

Real implementation requirements:
- Use Ollama Python SDK for local inference
- Model: llama3.2:1b (pull automatically if not present)
- Check Ollama is running before attempting inference:
    import ollama; ollama.list()
  If Ollama is not running, log a clear error and raise RuntimeError
- Implement generate(prompt: str) -> LLMResponse with same interface as 
  llm_claude.py
- This is the offline fallback — called automatically when Claude API 
  is unreachable
- Must satisfy LLMEngine ABC from interfaces/llm.py

---

STEP 7 — modules/query/llm_engine.py (new file — router)

Create a new file that is NOT a stub — this is the router that selects 
Claude vs Llama automatically:

- Try Claude API first
- If anthropic.APIConnectionError or anthropic.RateLimitError or any 
  network error: log warning "Claude API unavailable, falling back to Llama"
  and call llm_llama.generate() instead
- If Ollama also fails: log error and return a hardcoded LLMResponse:
    {answer: "I could not reach any LLM. Please check your connection.",
     source_chunks: []}
- Expose generate(prompt: str) -> LLMResponse as the single entry point
  used by everything else — nothing outside this file should import 
  llm_claude or llm_llama directly

---

STEP 8 — Wire the full query path

Update main.py so the query path is:
  user_query (str)
    → query_parser.parse()
    → retriever.retrieve()
    → context_builder.build()
    → llm_engine.generate()
    → print answer

---

STEP 9 — Integration test

Add test_query_pipeline() to main.py (run via: python main.py test_query)

Test sequence:
1. Skip capture — use whatever chunks are already in the FAISS index from 
   previous test runs (they are persisted on disk)
2. Run these three hardcoded queries through the full query pipeline:
     Query 1: "what did I say about developers and recruiters?"
     Query 2: "what was I talking about most recently?"
     Query 3: "summarise everything I said today"
3. For each query, print:
     - Parsed intent and time filter
     - Number of chunks retrieved
     - Final LLM answer
4. Confirm Claude API is called for Query 1 and 2
5. To test fallback: temporarily set claude_api_key to "invalid_key" in 
   config.yaml, run Query 3, confirm Llama fallback activates and returns 
   an answer

---

CONSTRAINTS:
- Do not touch interfaces/, infra/, audio pipeline, or memory modules
- claude_api_key must be read from config.yaml — never hardcoded
- ANTHROPIC_API_KEY environment variable is acceptable fallback for the key
- All config values from config_manager
- llm_engine.py is the only file that imports llm_claude and llm_llama
- The query path must work even if FAISS index has zero chunks — return 
  "no memories found" gracefully