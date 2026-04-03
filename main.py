"""Jarvis main entrypoint.

Run modes:
    python main.py              — smoke test (all stubs, verifies pipeline wiring)
    python main.py test         — audio pipeline test (10 s mic, VAD only)
    python main.py test_stt     — full STT pipeline test (20 s, AudioCapture→VAD→STT→Cleaner→Chunker)
    python main.py test_memory  — memory pipeline test (30 s capture + FAISS persistence check)
"""

import os
import sys
import time
import queue
import threading
import uuid

from jarvis.infra.config_manager import config
from jarvis.infra.logger import Logger
from jarvis.infra.privacy_controller import PrivacyController

# Layer 1 — Input
from jarvis.modules.input.audio_capture import AudioCapture
from jarvis.modules.input.vad import VadEngine
from jarvis.modules.input.device_priority import DevicePriorityManager

# Layer 2 — Processing
from jarvis.modules.processing.stt_moonshine import MoonshineSTT
from jarvis.modules.processing.text_cleaner import TextCleaner
from jarvis.modules.processing.chunker import Chunker

# Layer 3 — Memory
from jarvis.modules.memory.embedding_bge import BGEEmbeddingEngine
from jarvis.modules.memory.liquid_buffer import LiquidBuffer
from jarvis.modules.memory.vector_store_faiss import FAISSVectorStore
from jarvis.modules.memory.memory_manager import MemoryManager

# Layer 4 — Query
from jarvis.modules.query.query_parser import QueryParser
from jarvis.modules.query.retriever import Retriever
from jarvis.modules.query.context_builder import ContextBuilder
from jarvis.modules.query.llm_claude import ClaudeLLM

# Layer 5 — Output
from jarvis.modules.output.tts_kokoro import KokoroTTS
from jarvis.modules.output.response_player import ResponsePlayer
from jarvis.modules.output.ui import UILayer


# ---------------------------------------------------------------------------
# Smoke test — runs entirely with stubs, no hardware required
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Wire every module end-to-end using lightweight stubs to verify no import/API errors.

    This test does NOT load any ML models or open any audio streams.
    It proves the pipeline is correctly wired before any real hardware is used.
    """
    Logger.log("INFO", "main", "Starting Jarvis smoke test")

    # Infrastructure
    device_mgr = DevicePriorityManager()
    privacy = PrivacyController()

    # Layer 1 — use stub methods only (no real audio or VAD thread)
    audio_cap = AudioCapture()
    cleaner = TextCleaner()
    chunker = Chunker()

    # Layer 3 — memory (stubs)
    embedding = BGEEmbeddingEngine()
    buffer = LiquidBuffer()
    vector_store = FAISSVectorStore()
    memory_mgr = MemoryManager()

    # Layer 4 — query (stubs)
    parser = QueryParser()
    retriever = Retriever()
    ctx_builder = ContextBuilder()
    llm = ClaudeLLM()

    # Layer 5 — output (stubs)
    tts = KokoroTTS()
    player = ResponsePlayer()
    ui = UILayer()

    Logger.log("INFO", "main", "=== Starting Capture Path ===")
    device_id = device_mgr.get_active_source_id()

    # Dummy STT result — skip model load entirely in smoke test
    dummy_text = "This is a smoke test. The pipeline is fully wired."
    dummy_start_ms = int(time.time() * 1000)
    dummy_end_ms = dummy_start_ms + 3000
    stt_res = {"text": dummy_text, "start_ms": dummy_start_ms, "end_ms": dummy_end_ms, "confidence": 1.0}
    Logger.log("INFO", "main", "[stt_moonshine] stub bypass in smoke test")

    # Text cleaning + chunking
    clean_txt = cleaner.clean(stt_res["text"])
    chunks = chunker.split(
        clean_txt or "smoke test fallback.",
        stt_res["start_ms"],
        stt_res["end_ms"],
    )

    for c in chunks:
        buffer.insert(c)
        vec = embedding.embed(c["chunk_text"])

        mem_chunk = {
            "chunk_id": "",
            "text": c["chunk_text"],
            "vector": vec,
            "timestamp_start": c["timestamp_start"],
            "timestamp_end": c["timestamp_end"],
            "device_id": device_id,
            "session_id": "smoke_test_session",
            "confidence": stt_res["confidence"],
            "redacted": False,
        }

        mem_chunk = privacy.apply(mem_chunk)
        vector_store.upsert(mem_chunk)

    memory_mgr.flush()

    Logger.log("INFO", "main", "=== Starting Query Path ===")
    query_text = "What was I talking about?"
    ui.update("query_received", {"query": query_text})
    parsed_query = parser.parse(query_text)
    q_vec = embedding.embed(str(parsed_query))
    ranked = retriever.retrieve(q_vec, {})
    prompt = ctx_builder.build(ranked, query_text)
    llm_res = llm.generate(prompt)
    ui.update("answer_generated", {"answer": llm_res["answer"]})
    audio_ans = tts.synthesise(llm_res["answer"])
    player.play(audio_ans)

    Logger.log("INFO", "main", "Smoke test complete. Exiting without errors.")


# ---------------------------------------------------------------------------
# Audio pipeline test — real microphone, VAD only (no STT)
# ---------------------------------------------------------------------------

def test_audio_pipeline() -> None:
    """10-second live mic test: Clean VAD output."""
    # Suppress JSON logs for this clean test
    Logger.log = lambda *args, **kwargs: None

    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    print(f"🎤 Using Microphone: {source['device_name']} (Native: {source['sample_rate']}Hz)\n")

    audio_cap = AudioCapture(device_id=source["device_id"], native_rate=source["sample_rate"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)

    print("🟢 Pipeline Ready. Please speak!\n")
    audio_cap.start()

    test_running = [True]

    def _monitor_vad():
        was_speaking = False
        while test_running[0]:
            try:
                # Segment was emitted (speech ended)
                seg = vad.segment_queue.get(timeout=0.05)
                n_samples = len(seg["pcm_data"]) / 2
                duration_ms = n_samples / 16000 * 1000
                print(f"⏹️  [VAD] Speech ended! Captured {duration_ms:.0f}ms of audio.\n")
                was_speaking = False
            except queue.Empty:
                # Poll is_speaking state to show when speech starts
                if vad.is_speaking and not was_speaking:
                    print("🗣️  [VAD] Detecting speech... listening...", flush=True)
                    was_speaking = True

    t = threading.Thread(target=_monitor_vad, daemon=True)
    t.start()

    time.sleep(15)

    test_running[0] = False
    vad.stop()
    audio_cap.stop()
    print("🛑 Test complete.")


# ---------------------------------------------------------------------------
# Full STT pipeline test — AudioCapture → VAD → STT → TextCleaner → Chunker
# ---------------------------------------------------------------------------

def test_stt_pipeline() -> list[str]:
    """20-second live mic test: Clean STT transcription output.
    Returns:
        List of all clean transcribed strings detected during the 20 seconds.
    """
    # Suppress JSON logs for this clean test
    Logger.log = lambda *args, **kwargs: None

    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    print(f"🎤 Using Microphone: {source['device_name']} (Native: {source['sample_rate']}Hz)\n")

    audio_cap = AudioCapture(device_id=source["device_id"], native_rate=source["sample_rate"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)
    stt = MoonshineSTT()
    cleaner = TextCleaner()

    print("🟢 Pipeline Ready. Please speak!\n")
    audio_cap.start()

    test_running = [True]
    collected_transcripts = []

    def _processing_loop():
        was_speaking = False
        while test_running[0]:
            try:
                segment = vad.segment_queue.get(timeout=0.05)
                # Segment obtained, speech ended
                print("⏳ Transcribing...")
                transcript = stt.transcribe(segment)
                clean_text = cleaner.clean(transcript["text"])
                if clean_text:
                    print(f"\n📝 [Transcript]: {clean_text}\n")
                    collected_transcripts.append(clean_text)
                else:
                    print("❌ [Transcript]: (Empty or unintelligible)\n")
                was_speaking = False
            except queue.Empty:
                if vad.is_speaking and not was_speaking:
                    print("🗣️  [VAD] Detecting speech... listening...", flush=True)
                    was_speaking = True

    t = threading.Thread(target=_processing_loop, daemon=True)
    t.start()

    time.sleep(20)

    test_running[0] = False
    try:
        vad.stop()
        audio_cap.stop()
    except Exception:
        pass
    print("🛑 Test complete.")
    return collected_transcripts


# ---------------------------------------------------------------------------
# Memory pipeline test — full capture path + FAISS persistence check
# ---------------------------------------------------------------------------

def test_memory_pipeline() -> None:
    """30-second live mic test: AudioCapture → VAD → STT → Cleaner → Chunker
    → LiquidBuffer → MemoryManager → FAISS.

    After capture stops, flushes all pending chunks to FAISS, runs a test
    semantic query, and confirms FAISS + metadata files exist on disk.
    Also validates that restarting without speech still queries previously
    stored chunks (persistence check).
    """
    print("\n" + "=" * 60)
    print("🧠  Jarvis Memory Pipeline Test")
    print("=" * 60)

    # --- Build the full pipeline ---
    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    print(f"🎤 Microphone: {source['device_name']} ({source['sample_rate']} Hz)\n")

    session_id = str(uuid.uuid4())
    device_id = source.get("device_id", "laptop")

    audio_cap = AudioCapture(device_id=source["device_id"], native_rate=source["sample_rate"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)
    stt = MoonshineSTT()
    cleaner = TextCleaner()
    chunker = Chunker()

    buffer = LiquidBuffer()
    embedding = BGEEmbeddingEngine()
    vector_store = FAISSVectorStore()
    memory_mgr = MemoryManager(
        liquid_buffer=buffer,
        embedding_engine=embedding,
        vector_store=vector_store,
    )
    memory_mgr.start()

    print("🟢 Pipeline ready — please speak for 30 seconds!\n")
    audio_cap.start()

    test_running = [True]
    chunk_count = [0]

    def _processing_loop():
        was_speaking = False
        while test_running[0]:
            try:
                segment = vad.segment_queue.get(timeout=0.05)
                print("⏳ Transcribing …")
                transcript = stt.transcribe(segment)
                clean_text = cleaner.clean(transcript["text"])
                if not clean_text:
                    print("❌ (Empty or unintelligible)\n")
                    was_speaking = False
                    continue

                print(f"\n📝 [{transcript['start_ms']} ms] {clean_text}\n")
                chunks = chunker.split(
                    clean_text,
                    transcript["start_ms"],
                    transcript["end_ms"],
                )
                for c in chunks:
                    c["session_id"] = session_id
                    c["device_id"] = device_id
                    c["confidence"] = transcript.get("confidence", 1.0)
                    c["redacted"] = False
                    # STEP 6: insert into LiquidBuffer synchronously on every chunk
                    buffer.insert(c)
                    chunk_count[0] += 1
                    print(f"   💾 Chunk inserted → buffer (total: {chunk_count[0]})")
                was_speaking = False
            except queue.Empty:
                if vad.is_speaking and not was_speaking:
                    print("🗣️  Speech detected … listening …", flush=True)
                    was_speaking = True

    t = threading.Thread(target=_processing_loop, daemon=True)
    t.start()

    time.sleep(30)

    print("\n⏹️  Capture stopped after 30 seconds.")
    test_running[0] = False
    try:
        vad.stop()
        audio_cap.stop()
    except Exception:
        pass

    # Give processing loop 2 s to finish any in-flight transcription
    t.join(timeout=5)

    # --- Flush all buffered chunks to FAISS immediately ---
    print("\n🔄 Flushing LiquidBuffer → FAISS …")
    # Force flush_before a far-future cutoff to drain everything
    far_future_ms = int(time.time() * 1000) + 10_000_000
    chunks_to_flush = buffer.flush_before(far_future_ms)
    print(f"   Drained {len(chunks_to_flush)} chunks from buffer for direct flush")

    flushed = 0
    if chunks_to_flush:
        texts = [c.get("chunk_text", c.get("text", "")) for c in chunks_to_flush]
        vectors = embedding.embed_batch(texts)
        for chunk, vector in zip(chunks_to_flush, vectors):
            mem_chunk = {
                "chunk_id":       chunk.get("chunk_id", ""),
                "text":           chunk.get("chunk_text", chunk.get("text", "")),
                "vector":         vector,
                "timestamp_start": chunk.get("timestamp_start", 0),
                "timestamp_end":   chunk.get("timestamp_end", 0),
                "device_id":      chunk.get("device_id", "laptop"),
                "session_id":     chunk.get("session_id", ""),
                "confidence":     chunk.get("confidence", 1.0),
                "redacted":       chunk.get("redacted", False),
            }
            vector_store.upsert(mem_chunk)
            flushed += 1

    memory_mgr.stop()
    print(f"✅ Flushed {flushed} chunks to FAISS (index size: {vector_store._index.ntotal})\n")

    # --- Semantic search test ---
    query = "developer recruiter startup reaching out"
    print(f"🔍 Test query: \"{query}\"")
    q_vec = embedding.embed(query)
    results = vector_store.search(q_vec, top_k=3, filters={})

    print(f"\n📚 Top {len(results)} results:")
    if results:
        for i, r in enumerate(results, 1):
            score = r.get("_score", 0.0)
            text = r.get("text", r.get("chunk_text", "(no text)"))
            ts_start = r.get("timestamp_start", 0)
            ts_end = r.get("timestamp_end", 0)
            print(f"\n  [{i}] score={score:.4f}  {ts_start}ms → {ts_end}ms")
            print(f"       {text}")
    else:
        print("  (No results — no speech was recorded during the 30s window)")

    # --- Persistence validation ---
    print("\n📁 Checking persistence …")
    from jarvis.infra.config_manager import config as cfg
    index_path_str: str = cfg.get("storage", {}).get("vector_store_path", "data/faiss_index.bin")
    from pathlib import Path
    index_path = Path(index_path_str)
    meta_path = index_path.with_suffix(".pkl")

    index_ok = index_path.exists()
    meta_ok = meta_path.exists()
    print(f"   FAISS index  ({index_path}): {'✅ EXISTS' if index_ok else '❌ MISSING'}")
    print(f"   Metadata pkl ({meta_path}): {'✅ EXISTS' if meta_ok else '❌ MISSING'}")

    if index_ok and meta_ok:
        print("\n✅ Persistence validated — restart Python and run test_memory_pipeline()")
        print("   without speaking to confirm old chunks are still searchable.")
    else:
        print("\n⚠️  Persistence check FAILED — files not written correctly.")

    print("\n🛑 Memory pipeline test complete.\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "test":
            test_audio_pipeline()
        elif arg == "test_stt":
            results = test_stt_pipeline()
            print("\n--- FINAL OUTPUT ---")
            print(results)
        elif arg == "test_memory":
            test_memory_pipeline()
        else:
            run_smoke_test()
    else:
        run_smoke_test()
