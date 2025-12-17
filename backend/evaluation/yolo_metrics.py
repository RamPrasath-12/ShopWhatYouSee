"""
YOLOv8 Inference Performance Evaluation (Review-1)
-------------------------------------------------
This script benchmarks YOLOv8 inference performance
using a pre-fine-tuned fashion detection model.

Metrics reported (Review-1):
- Average inference time (ms)
- Min / Max inference time
- Number of detections per frame

NOTE:
Precision, Recall, and mAP are not computed here
because no labeled dataset (e.g., DeepFashion2)
is used at this stage. Full evaluation is planned
for upcoming reviews after fine-tuning.
"""

import time
import cv2
import numpy as np
import os
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------

# Path to YOLO weights (DO NOT use torch.load)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "yolo", "detect.pt")
)

# Use a saved frame from your video pipeline
TEST_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sample_frame.jfif")
)

RUNS = 5   # number of repeated inferences

# -----------------------------
# SANITY CHECKS
# -----------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(
        f"Test image not found: {TEST_IMAGE_PATH}\n"
        f"👉 Save one frame as 'sample_frame.jpg' in backend/evaluation/"
    )

# -----------------------------
# LOAD MODEL (SAFE WAY)
# -----------------------------

print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)   # ✅ SAFE (Ultralytics handles PyTorch security)

# -----------------------------
# LOAD TEST IMAGE
# -----------------------------

frame = cv2.imread(TEST_IMAGE_PATH)
if frame is None:
    raise RuntimeError("Failed to load test image")

# -----------------------------
# RUN INFERENCE BENCHMARK
# -----------------------------

inference_times = []
detection_counts = []

print("\nRunning YOLO inference benchmark...\n")

for i in range(RUNS):
    start_time = time.time()

    results = model(frame)[0]   # single-image inference

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    inference_times.append(elapsed_ms)
    detection_counts.append(len(results.boxes))

    print(
        f"Run {i+1}: "
        f"Inference Time = {elapsed_ms:.2f} ms | "
        f"Detections = {len(results.boxes)}"
    )

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------

print("\n===== YOLOv8 INFERENCE METRICS (REVIEW-1) =====")
print(f"Model Used            : {os.path.basename(MODEL_PATH)}")
print(f"Test Image Resolution : {frame.shape[1]}x{frame.shape[0]}")
print(f"Average Inference Time: {np.mean(inference_times):.2f} ms")
print(f"Min Inference Time    : {np.min(inference_times):.2f} ms")
print(f"Max Inference Time    : {np.max(inference_times):.2f} ms")
print(f"Avg Detections/Frame  : {np.mean(detection_counts):.2f}")
print("==============================================")
