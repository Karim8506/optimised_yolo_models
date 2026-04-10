#!/usr/bin/env python3
import time
import sys
import cv2
import numpy as np
import ncnn

MODEL_PARAM = sys.argv[1] if len(sys.argv) > 1 else "model.param"
MODEL_BIN   = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
VIDEO       = "video.mp4"
SOURCE_FPS  = 30.0
IMGSZ       = 320

net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 4
net.load_param(MODEL_PARAM)
net.load_model(MODEL_BIN)

cap = cv2.VideoCapture(VIDEO)
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_RGB, IMGSZ, IMGSZ)
    mean_vals = [0, 0, 0]
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    ex = net.create_extractor()

    t0 = time.perf_counter()
    ex.input("in0", mat_in)
    _, _ = ex.extract("out0")
    inference_times.append((time.perf_counter() - t0) * 1000)

cap.release()

avg_inference = sum(inference_times) / len(inference_times)
fps = 1000.0 / avg_inference
rtf = fps / SOURCE_FPS

print(f"Avg inference   : {avg_inference:.1f} ms")
print(f"FPS             : {fps:.2f}")
print(f"Real-time factor: {rtf:.3f}  ({'real-time' if rtf >= 1 else f'{1/rtf:.2f}x slower than real-time'})")
