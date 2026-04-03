import onnxruntime as ort
session = ort.InferenceSession("models/silero_vad.onnx", providers=['CPUExecutionProvider'])
for i in session.get_inputs():
    print(i.name, i.shape, i.type)
