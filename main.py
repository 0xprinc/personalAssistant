"""Jarvis main entrypoint.

Run modes:
    python main.py              — smoke test (all stubs, verifies pipeline wiring)
    python main.py test         — audio pipeline test (10 s mic, VAD only)
    python main.py test_stt     — full STT pipeline test (20 s, AudioCapture→VAD→STT→Cleaner→Chunker)
    python main.py test_memory  — memory pipeline test (30 s capture + FAISS persistence check)
    python main.py test_query   — query pipeline test (uses existing FAISS index, no capture)
    python main.py test_output  — TTS + playback test (auto-unload validation)
    python main.py test_e2e     — end-to-end: capture → query → speak answer aloud
"""

import sys
import time
import queue
import threading
import uuid

import numpy as np

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
import jarvis.modules.query.llm_engine as llm_engine

# Layer 5 — Output
from jarvis.modules.output.tts_kokoro import KokoroTTS
from jarvis.modules.output.response_player import ResponsePlayer
from jarvis.modules.output.ui import UILayer

# Hardcoded fallback answer text — do NOT pass to TTS
_LLM_FALLBACK_ANSWER = "I could not reach the LLM. Please check your OpenRouter API key and connection."


def _speak(tts: KokoroTTS, player: ResponsePlayer, answer: str) -> None:
    """Synthesise and play answer text — skips TTS on hardcoded fallback."""
    if not answer or answer.strip() == _LLM_FALLBACK_ANSWER:
        Logger.log("WARNING", "main", "LLM fallback answer — skipping TTS playback")
        print(f"\n⚠️  LLM unavailable: {answer}")
        return
    audio = tts.synthesise(answer)
    player.play(audio)


# ---------------------------------------------------------------------------
# Smoke test — stubs only, no hardware
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    Logger.log("INFO", "main", "Starting Jarvis smoke test")

    device_mgr = DevicePriorityManager()
    privacy = PrivacyController()
    audio_cap = AudioCapture()
    cleaner = TextCleaner()
    chunker = Chunker()

    embedding = BGEEmbeddingEngine()
    buffer = LiquidBuffer()
    vector_store = FAISSVectorStore()
    memory_mgr = MemoryManager()

    parser = QueryParser()
    retriever = Retriever()
    ctx_builder = ContextBuilder()

    tts = KokoroTTS()
    player = ResponsePlayer()
    ui = UILayer()

    Logger.log("INFO", "main", "=== Starting Capture Path ===")
    device_id = device_mgr.get_active_source_id()

    dummy_text = "This is a smoke test. The pipeline is fully wired."
    dummy_start_ms = int(time.time() * 1000)
    dummy_end_ms = dummy_start_ms + 3000
    stt_res = {"text": dummy_text, "start_ms": dummy_start_ms, "end_ms": dummy_end_ms, "confidence": 1.0}

    clean_txt = cleaner.clean(stt_res["text"])
    chunks = chunker.split(clean_txt or "smoke test fallback.", stt_res["start_ms"], stt_res["end_ms"])

    for c in chunks:
        buffer.insert(c)
        vec = embedding.embed(c["chunk_text"])
        mem_chunk = {
            "chunk_id": "", "text": c["chunk_text"], "vector": vec,
            "timestamp_start": c["timestamp_start"], "timestamp_end": c["timestamp_end"],
            "device_id": device_id, "session_id": "smoke_test_session",
            "confidence": stt_res["confidence"], "redacted": False,
        }
        mem_chunk = privacy.apply(mem_chunk)
        vector_store.upsert(mem_chunk)

    memory_mgr.flush()

    Logger.log("INFO", "main", "=== Starting Query Path ===")
    query_text = "What was I talking about?"
    ui.update("query_received", {"query": query_text})
    parsed_query = parser.parse(query_text)
    ranked = retriever.retrieve(query_text, parsed_query["time_filter"] or {})
    prompt = ctx_builder.build(ranked, query_text)
    llm_res = llm_engine.generate(prompt)
    ui.update("answer_generated", {"answer": llm_res["answer"]})
    # Smoke test: do NOT invoke real TTS (lazy-load, avoid model download during smoke)
    Logger.log("INFO", "main", f"[smoke] LLM answer (TTS skipped): {llm_res['answer'][:80]}")

    Logger.log("INFO", "main", "Smoke test complete. Exiting without errors.")


# ---------------------------------------------------------------------------
# Audio pipeline test (VAD only)
# ---------------------------------------------------------------------------

def test_audio_pipeline() -> None:
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
                seg = vad.segment_queue.get(timeout=0.05)
                n_samples = len(seg["pcm_data"]) / 2
                duration_ms = n_samples / 16000 * 1000
                print(f"⏹️  [VAD] Speech ended! Captured {duration_ms:.0f}ms of audio.\n")
                was_speaking = False
            except queue.Empty:
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
# STT pipeline test
# ---------------------------------------------------------------------------

def test_stt_pipeline() -> list[str]:
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
# Memory pipeline test
# ---------------------------------------------------------------------------

def test_memory_pipeline() -> None:
    print("\n" + "=" * 60)
    print("🧠  Jarvis Memory Pipeline Test")
    print("=" * 60)

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
    memory_mgr = MemoryManager(liquid_buffer=buffer, embedding_engine=embedding, vector_store=vector_store)
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
                chunks = chunker.split(clean_text, transcript["start_ms"], transcript["end_ms"])
                for c in chunks:
                    c["session_id"] = session_id
                    c["device_id"] = device_id
                    c["confidence"] = transcript.get("confidence", 1.0)
                    c["redacted"] = False
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
    t.join(timeout=5)

    print("\n🔄 Flushing LiquidBuffer → FAISS …")
    far_future_ms = int(time.time() * 1000) + 10_000_000
    chunks_to_flush = buffer.flush_before(far_future_ms)
    flushed = 0
    if chunks_to_flush:
        texts = [c.get("chunk_text", c.get("text", "")) for c in chunks_to_flush]
        vectors = embedding.embed_batch(texts)
        for chunk, vector in zip(chunks_to_flush, vectors):
            mem_chunk = {
                "chunk_id": chunk.get("chunk_id", ""), "text": chunk.get("chunk_text", chunk.get("text", "")),
                "vector": vector, "timestamp_start": chunk.get("timestamp_start", 0),
                "timestamp_end": chunk.get("timestamp_end", 0), "device_id": chunk.get("device_id", "laptop"),
                "session_id": chunk.get("session_id", ""), "confidence": chunk.get("confidence", 1.0),
                "redacted": chunk.get("redacted", False),
            }
            vector_store.upsert(mem_chunk)
            flushed += 1

    memory_mgr.stop()
    print(f"✅ Flushed {flushed} chunks to FAISS (index size: {vector_store._index.ntotal})\n")
    print("🛑 Memory pipeline test complete.\n")


# ---------------------------------------------------------------------------
# Query pipeline test (no capture, uses existing FAISS index)
# ---------------------------------------------------------------------------

def test_query_pipeline() -> None:
    print("\n" + "=" * 60)
    print("🔍  Jarvis Query Pipeline Test")
    print("=" * 60 + "\n")

    embedding = BGEEmbeddingEngine()
    vector_store = FAISSVectorStore()
    buffer = LiquidBuffer()
    parser = QueryParser()
    retriever = Retriever(embedding_engine=embedding, liquid_buffer=buffer, vector_store=vector_store)
    ctx_builder = ContextBuilder()

    print(f"📦 FAISS index loaded: {vector_store._index.ntotal} vectors\n")

    test_queries = [
        "what did I say about developers and recruiters?",
        "what was I talking about most recently?",
        "summarise everything I said today",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 55}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 55}")
        parsed = parser.parse(query)
        print(f"  Intent: {parsed['intent']}  |  Keywords: {parsed['keywords']}")
        chunks = retriever.retrieve(query_text=query, filters=parsed["time_filter"] or {}, top_k=5)
        print(f"  Chunks retrieved: {len(chunks)}")
        prompt = ctx_builder.build(chunks, query)
        print(f"\n  🤖 Calling LLM …")
        llm_res = llm_engine.generate(prompt)
        print(f"\n  ✅ Answer:\n  {llm_res['answer']}\n")

    print("=" * 60)
    print("🛑 Query pipeline test complete.\n")


# ---------------------------------------------------------------------------
# Output pipeline test — TTS synthesis + auto-unload validation
# ---------------------------------------------------------------------------

def test_output_pipeline() -> None:
    print("\n" + "=" * 60)
    print("🔊  Jarvis Output Pipeline Test")
    print("=" * 60 + "\n")

    tts = KokoroTTS()
    player = ResponsePlayer()

    # --- First synthesis ---
    text1 = (
        "Hello. I am Jarvis, your personal memory assistant. "
        "The pipeline is fully wired and I can speak."
    )
    print(f"📢 Synthesising: \"{text1}\"\n")
    t0 = time.perf_counter()
    audio1 = tts.synthesise(text1)
    synth_ms = (time.perf_counter() - t0) * 1000
    duration_s = len(audio1) / 24000 if len(audio1) > 0 else 0.0
    print(f"   ✅ Synthesis complete: {synth_ms:.0f}ms | Audio: {duration_s:.2f}s\n")
    player.play(audio1)

    # --- Wait for auto-unload (30s idle timer) ---
    print(f"\n⏳ Waiting 35 seconds for auto-unload …")
    for remaining in range(35, 0, -5):
        time.sleep(5)
        print(f"   {remaining - 5}s remaining …")

    print("\n✅ Auto-unload window passed — check logs above for 'TTS unloaded, RAM released'\n")

    # --- Second synthesis (forces reload) ---
    text2 = "Auto-unload confirmed. Jarvis is reloading the TTS engine now."
    print(f"📢 Synthesising again (forces TTS reload): \"{text2}\"\n")
    t0 = time.perf_counter()
    audio2 = tts.synthesise(text2)
    synth_ms2 = (time.perf_counter() - t0) * 1000
    duration_s2 = len(audio2) / 24000 if len(audio2) > 0 else 0.0
    print(f"   ✅ Reload + synthesis: {synth_ms2:.0f}ms | Audio: {duration_s2:.2f}s\n")
    player.play(audio2)

    print("\n📊 Results summary:")
    print(f"   First synthesis:   {synth_ms:.0f}ms → {duration_s:.2f}s audio")
    print(f"   Second synthesis:  {synth_ms2:.0f}ms → {duration_s2:.2f}s audio (includes reload)")
    print("\n🛑 Output pipeline test complete.\n")


# ---------------------------------------------------------------------------
# End-to-end test — capture → query → speak aloud
# ---------------------------------------------------------------------------

def test_e2e() -> None:
    print("\n" + "=" * 60)
    print("🔁  Jarvis End-to-End Test (Capture → Query → Speak)")
    print("=" * 60 + "\n")

    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    print(f"🎤 Microphone: {source['device_name']} ({source['sample_rate']} Hz)\n")

    session_id = str(uuid.uuid4())
    device_id = source.get("device_id", "laptop")

    # Build full pipeline
    audio_cap = AudioCapture(device_id=source["device_id"], native_rate=source["sample_rate"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)
    stt = MoonshineSTT()
    cleaner = TextCleaner()
    chunker = Chunker()

    embedding = BGEEmbeddingEngine()
    buffer = LiquidBuffer()
    vector_store = FAISSVectorStore()

    parser = QueryParser()
    retriever = Retriever(embedding_engine=embedding, liquid_buffer=buffer, vector_store=vector_store)
    ctx_builder = ContextBuilder()

    tts = KokoroTTS()
    player = ResponsePlayer()

    print("🟢 Speak for 20 seconds, then Jarvis will answer what you said.\n")
    audio_cap.start()

    test_running = [True]
    spoken_chunks: list[dict] = []

    def _capture_loop():
        was_speaking = False
        while test_running[0]:
            try:
                segment = vad.segment_queue.get(timeout=0.05)
                print("⏳ Transcribing …")
                transcript = stt.transcribe(segment)
                clean_text = cleaner.clean(transcript["text"])
                if not clean_text:
                    was_speaking = False
                    continue
                print(f"📝 {clean_text}")
                chunks = chunker.split(clean_text, transcript["start_ms"], transcript["end_ms"])
                for c in chunks:
                    c["session_id"] = session_id
                    c["device_id"] = device_id
                    c["confidence"] = transcript.get("confidence", 1.0)
                    c["redacted"] = False
                    buffer.insert(c)
                    spoken_chunks.append(c)
                was_speaking = False
            except queue.Empty:
                if vad.is_speaking and not was_speaking:
                    print("🗣️  Listening …", flush=True)
                    was_speaking = True

    cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()

    time.sleep(20)
    test_running[0] = False
    try:
        vad.stop()
        audio_cap.stop()
    except Exception:
        pass
    cap_thread.join(timeout=5)

    print(f"\n⏹️  Capture stopped. Collected {len(spoken_chunks)} chunks.\n")

    # --- Query with what was just captured ---
    query = "what did I just say?"
    print(f"🔍 Query: \"{query}\"\n")

    parsed = parser.parse(query)
    # Ensure we search very recent content
    parsed["time_filter"] = {"after_ms": int(time.time() * 1000) - 120_000, "before_ms": None}

    chunks = retriever.retrieve(query_text=query, filters=parsed["time_filter"], top_k=5)
    print(f"   Retrieved {len(chunks)} chunks from memory")

    prompt = ctx_builder.build(chunks, query)
    print("   🤖 Calling LLM …")
    llm_res = llm_engine.generate(prompt)
    answer = llm_res["answer"]

    print(f"\n   ✅ Answer: {answer}\n")

    # --- Speak the answer ---
    print("🔊 Speaking answer …\n")
    _speak(tts, player, answer)

    print("🛑 End-to-end test complete.\n")


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
        elif arg == "test_query":
            test_query_pipeline()
        elif arg == "test_output":
            test_output_pipeline()
        elif arg == "test_e2e":
            test_e2e()
        else:
            run_smoke_test()
    else:
        run_smoke_test()
