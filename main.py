"""Jarvis main entrypoint.

Run modes:
    python main.py              — smoke test (all stubs, verifies pipeline wiring)
    python main.py test         — audio pipeline test (10 s mic, VAD only)
    python main.py test_stt     — full STT pipeline test (20 s, AudioCapture→VAD→STT→Cleaner→Chunker)
"""

import sys
import time
import queue
import threading

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
    """10-second live mic test: AudioCapture → VAD → print SpeechSegments."""
    Logger.log("INFO", "main", "=== Starting Audio Pipeline Integration Test ===")

    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    audio_cap = AudioCapture(device_id=source["device_id"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)

    audio_cap.start()

    test_running = [True]

    def _consume_segments():
        while test_running[0]:
            try:
                seg = vad.segment_queue.get(timeout=1.0)
                # len(pcm_data) / 2  = number of int16 samples (2 bytes each)
                n_samples = len(seg["pcm_data"]) / 2
                duration_ms = n_samples / 16000 * 1000
                block_count = n_samples / 512
                print(
                    f"SpeechSegment — start_ms={seg['start_ms']} "
                    f"duration_ms={duration_ms:.1f}  blocks={block_count:.1f}"
                )
            except queue.Empty:
                pass

    t = threading.Thread(target=_consume_segments, daemon=True)
    t.start()

    Logger.log("INFO", "main", "Listening for 10 seconds… please speak into the mic!")
    time.sleep(10)

    test_running[0] = False
    vad.stop()
    audio_cap.stop()
    Logger.log("INFO", "main", "Audio pipeline test stopped cleanly.")


# ---------------------------------------------------------------------------
# Full STT pipeline test — AudioCapture → VAD → STT → TextCleaner → Chunker
# ---------------------------------------------------------------------------

def test_stt_pipeline() -> None:
    """20-second live mic test: full capture → transcription → chunked output."""
    Logger.log("INFO", "main", "=== Starting STT Pipeline Integration Test ===")

    device_mgr = DevicePriorityManager()
    source = device_mgr.get_active_source()
    audio_cap = AudioCapture(device_id=source["device_id"])
    vad = VadEngine(audio_queue=audio_cap.audio_queue)
    stt = MoonshineSTT()
    cleaner = TextCleaner()
    chunker = Chunker()

    audio_cap.start()

    test_running = [True]

    def _processing_loop():
        Logger.log("INFO", "main", "STT processing pipeline thread started")
        while test_running[0]:
            try:
                segment = vad.segment_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                transcript = stt.transcribe(segment)
                clean_text = cleaner.clean(transcript["text"])
                if not clean_text:
                    continue

                chunks = chunker.split(clean_text, transcript["start_ms"], transcript["end_ms"])
                for chunk in chunks:
                    words = len(chunk["chunk_text"].split())
                    print(
                        f"Chunk: '{chunk['chunk_text']}' | "
                        f"words={words} | "
                        f"start={chunk['timestamp_start']} | "
                        f"end={chunk['timestamp_end']}"
                    )
            except Exception as exc:
                Logger.log("ERROR", "main", f"Processing pipeline error: {exc}")

    t = threading.Thread(target=_processing_loop, daemon=True)
    t.start()

    Logger.log("INFO", "main", "Listening for 20 seconds… please speak into the mic!")
    time.sleep(20)

    test_running[0] = False
    try:
        vad.stop()
        audio_cap.stop()
    except Exception:
        pass
    Logger.log("INFO", "main", "STT pipeline test stopped cleanly.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "test":
            test_audio_pipeline()
        elif arg == "test_stt":
            test_stt_pipeline()
        else:
            run_smoke_test()
    else:
        run_smoke_test()
