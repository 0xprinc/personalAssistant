"""Chunker Module."""
import re
from jarvis.infra.logger import Logger

class Chunker:
    def __init__(self):
        self.target_words = 50

    def split(self, clean_text: str, start_ms: int, end_ms: int) -> list[dict]:
        if not clean_text:
            return []
            
        # Segment by sentence bounds
        sentences = re.split(r'(?<=[.!?]) +', clean_text)
        
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            if current_word_count + words > self.target_words and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_text": chunk_text,
                    "timestamp_start": start_ms,
                    "timestamp_end": end_ms
                })
                current_chunk_sentences = [sentence]
                current_word_count = words
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += words
                
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_text": chunk_text,
                "timestamp_start": start_ms,
                "timestamp_end": end_ms
            })
            
        Logger.log("INFO", "chunker", f"Split transcript into {len(chunks)} memory chunks")
        return chunks
