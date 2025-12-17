"""
One-time script to extract YOLO garment crops
--------------------------------------------
- Uses YOLOv8 model directly
- Saves cropped garment images for AG-MAN evaluation
- NOT part of runtime / Flask pipeline

Run once, then stop.
"""

import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "data/yolo/detect.pt"
INPUT_IMAGE = "evaluation/sample_frame.jfif"   # one frame from video
OUTPUT_DIR = "evaluation/sample_crops"
CONF_THRESHOLD = 0.4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# -----------------------------
# LOAD IMAGE
# -----------------------------

frame = cv2.imread(INPUT_IMAGE)
if frame is None:
    raise RuntimeError(f"Failed to load image: {INPUT_IMAGE}")

# -----------------------------
# RUN DETECTION
# -----------------------------

print("Running detection...")
results = model(frame)[0]

crop_count = 0

for idx, box in enumerate(results.boxes):
    conf = float(box.conf[0])
    if conf < CONF_THRESHOLD:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    filename = f"{cls_name}_{crop_count}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(save_path, crop)
    crop_count += 1

    print(f"Saved crop: {filename} (conf={conf:.2f})")

print(f"\nDone. Total crops saved: {crop_count}")
print(f"Location: {OUTPUT_DIR}")
