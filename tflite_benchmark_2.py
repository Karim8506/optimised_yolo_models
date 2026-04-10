#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_int8.tflite"
VIDEO = "video.mp4"
SOURCE_FPS = 30.0

interpreter = Interpreter(model_path=MODEL, num_threads=4)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

IMGSZ      = input_details["shape"][1]
input_type = input_details["dtype"]

cap = cv2.VideoCapture(VIDEO)
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if input_type == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        scale, zero_point = input_details["quantization"]
        img = (img / (scale * 255.0) + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details["index"], img[np.newaxis])

    t0 = time.perf_counter()
    interpreter.invoke()
    inference_times.append((time.perf_counter() - t0) * 1000)

cap.release()

avg_inference = sum(inference_times) / len(inference_times)
fps = 1000.0 / avg_inference
rtf = fps / SOURCE_FPS

print(f"Avg inference   : {avg_inference:.1f} ms")
print(f"FPS             : {fps:.2f}")
print(f"Real-time factor: {rtf:.3f}  ({'real-time' if rtf >= 1 else f'{1/rtf:.2f}x slower than real-time'})")