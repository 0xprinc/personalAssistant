"""Text Cleaner Module."""
import re
from jarvis.infra.logger import Logger
from jarvis.infra.config_manager import config

class TextCleaner:
    def __init__(self):
        self.last_five_words = []
        app_params = config.get("parameters", {})
        self.pii_redaction_enabled = app_params.get("pii_redaction", False)

    def _deduplicate(self, text: str) -> str:
        current_words = text.split()
        if not current_words or not self.last_five_words:
            self.last_five_words = current_words[-5:] if len(current_words) >= 5 else current_words
            return text
            
        first_current = current_words[:5]
        max_overlap = 0
        
        prefix_len = min(len(self.last_five_words), len(first_current))
        for i in range(3, prefix_len + 1):
            if self.last_five_words[-i:] == first_current[:i]:
                max_overlap = i
                
        if max_overlap >= 3:
            current_words = current_words[max_overlap:]
            Logger.log("DEBUG", "text_cleaner", f"Stripped {max_overlap} overlapping words")
            
        self.last_five_words = current_words[-5:] if len(current_words) >= 5 else current_words
        return " ".join(current_words)

    def _restore_punctuation(self, text: str) -> str:
        if not text:
            return text
        text = text.strip()
        # Capitalize
        text = text[0].upper() + text[1:]
        # Append dot if missing
        if text[-1] not in ['.', '!', '?']:
            text += '.'
        return text

    def _redact_pii(self, text: str) -> str:
        if not self.pii_redaction_enabled:
            return text
        
        phone_pattern = r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b'
        text = re.sub(phone_pattern, '[PHONE]', text)
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        card_pattern = r'\b(?:\d{4}[\s-]?){4}\b'
        text = re.sub(card_pattern, '[CARD]', text)
        
        return text

    def clean(self, transcript: str) -> str:
        if not transcript.strip():
            return ""
            
        text = self._deduplicate(transcript)
        if not text.strip():
            return ""
            
        text = self._restore_punctuation(text)
        text = self._redact_pii(text)
        
        Logger.log("INFO", "text_cleaner", "Cleaned transcript", {"clean_text": text})
        return text
