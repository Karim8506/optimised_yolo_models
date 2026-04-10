#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
VIDEO = "video.mp4"
SOURCE_FPS = 30.0
IMGSZ = 320

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(VIDEO)
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW

    t0 = time.perf_counter()
    session.run(None, {input_name: img})
    inference_times.append((time.perf_counter() - t0) * 1000)

cap.release()

avg_inference = sum(inference_times) / len(inference_times)
fps = 1000.0 / avg_inference
rtf = fps / SOURCE_FPS

print(f"Avg inference   : {avg_inference:.1f} ms")
print(f"FPS             : {fps:.2f}")
print(f"Real-time factor: {rtf:.3f}  ({'real-time' if rtf >= 1 else f'{1/rtf:.2f}x slower than real-time'})")
