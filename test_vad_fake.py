import numpy as np
from jarvis.modules.input.vad import VadEngine
import queue

q = queue.Queue()
vad = VadEngine(q)

fake_noise = (np.random.randn(512) * 10000).astype(np.int16).tobytes()
fake_speech = (np.sin(np.linspace(0, 2*np.pi*440, 512)) * 20000).astype(np.int16).tobytes()

print("Noise conf:", vad._predict(fake_noise))
print("Noise conf:", vad._predict(fake_noise))
print("Speech conf:", vad._predict(fake_speech))
print("Speech conf:", vad._predict(fake_speech))
print("Speech conf:", vad._predict(fake_speech))
print("Speech conf:", vad._predict(fake_speech))
