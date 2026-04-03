You are building a macOS application called Jarvis — a passive, always-on 
personal memory system. Read the three attached documents carefully before 
writing a single line of code:

- description_and_background.md  
- architecture.md  
- roadmap.md

These are your source of truth. Do not deviate from the architecture defined 
in them.

---

YOUR TASK FOR THIS PROMPT:

Complete roadmap milestone 1.1 — Project Scaffold only. Nothing else.

1. Create the full directory structure exactly as specified in architecture.md 
   section 8.

2. Create config/config.yaml with placeholder values for every setting the 
   system will need: model paths, chunk size, liquid buffer duration, API keys, 
   device selection, VAD threshold, embedding dimensions, vector store path, 
   TTS idle timeout. Add a comment on every line explaining what it controls.

3. Create interfaces/ as Python abstract base classes (ABCs). One file per 
   module. Every ABC must enforce the exact interface contract defined in 
   architecture.md section 7. The contracts are non-negotiable — any 
   implementation that satisfies the ABC must be a valid drop-in replacement.
   Files to create:
   - interfaces/stt.py
   - interfaces/embedding.py
   - interfaces/vector_store.py
   - interfaces/llm.py
   - interfaces/tts.py

4. Create a stub implementation for every module listed in architecture.md 
   section 3. Each stub must:
   - Import and implement its corresponding ABC
   - Accept the correct inputs
   - Return dummy data of the correct type and shape
   - Log a message via the Logger saying "[MODULE_NAME] stub called"
   - NOT raise NotImplementedError — it must return valid dummy output so the 
     full pipeline can be wired together

5. Create infra/config_manager.py that loads config.yaml and exposes a single 
   config object imported by every other module.

6. Create infra/logger.py with structured JSON logging. Every log entry must 
   include: timestamp, level, module name, message, and optional metadata dict.

7. Create a main.py that imports every stub, wires the full capture path and 
   query path end-to-end using stubs, and runs without errors. This is a 
   smoke test — it proves the architecture is correctly connected before any 
   real model is integrated.

---

CONSTRAINTS:

- Language: Python 3.11+
- No real models, no pip installs of ML libraries yet — stubs only in this prompt
- Every file must have a module-level docstring explaining its role in the system
- Follow the layer naming from architecture.md exactly: input, processing, 
  memory, query, output, infra
- main.py must run cleanly with: python main.py

When done, run python main.py and confirm it exits without errors.