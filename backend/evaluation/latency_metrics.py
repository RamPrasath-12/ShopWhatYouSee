"""
End-to-End Latency Evaluation (Review-1)
---------------------------------------
Measures latency of each module in the
ShopWhatYouSee pipeline and total system latency.

Pipeline:
Frame → YOLO → AG-MAN → Scene → LLM → Retrieval
"""

import time
import base64
import cv2
import numpy as np

from models.yolo_detector import YoloDetector
from models.agman_extractor import process_crop_base64
from models.scene_context import SceneContextDetector
from models.gemini_reasoner import GeminiReasoner
from models.product_retrieval import search_products
from config import YOLO_WEIGHTS, YOLO_CONF_THRESH

# -----------------------------
# LOAD MODELS
# -----------------------------

yolo = YoloDetector(YOLO_WEIGHTS)
scene_detector = SceneContextDetector()
llm = GeminiReasoner()

# -----------------------------
# UTILITIES
# -----------------------------

def cv2_to_data_url(img):
    _, buffer = cv2.imencode(".jpg", img)  # encoding as JPG is fine
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"



# -----------------------------
# CONFIG
# -----------------------------

TEST_IMAGE_PATH = "evaluation/sample_frame.jpg"

# -----------------------------
# MAIN EVALUATION
# -----------------------------

def evaluate_latency():
    print("\nRunning end-to-end latency evaluation...\n")

    # -----------------------------
    # Load frame
    # -----------------------------
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        raise RuntimeError("Failed to load test frame")

    t_start = time.time()

    # -----------------------------
    # 1. YOLO Detection
    # -----------------------------
    t0 = time.time()
    detections = yolo.infer(frame)
    detections = [d for d in detections if d["conf"] >= YOLO_CONF_THRESH]
    t1 = time.time()

    if not detections:
        print("No detections found. Exiting.")
        return

    # Use first detected item
    item = detections[0]

    # -----------------------------
    # 2. AG-MAN Attributes
    # -----------------------------
    t2 = time.time()
    agman_result = process_crop_base64(item["cropped_image"])
    t3 = time.time()

    # -----------------------------
    # 3. Scene Context
    # -----------------------------
    t4 = time.time()
    scene_result = scene_detector.infer(cv2_to_data_url(frame))

    t5 = time.time()

    # -----------------------------
    # 4. LLM Reasoning
    # -----------------------------
    t6 = time.time()
    llm_output = llm.reason(
        category=item["class"],
        agman_attributes={
            "color_hex": agman_result["attributes"]["color_hex"],
            "pattern": agman_result["attributes"]["pattern"],
            "sleeve_length": agman_result["attributes"]["sleeve_length"]
        },
        scene=scene_result.get("scene_label", ""),
        user_query=""
    )
    t7 = time.time()

    # -----------------------------
    # 5. Product Retrieval
    # -----------------------------
    t8 = time.time()
    products = search_products({
        **(llm_output.get("final_filters", {})),
        "category": item["class"]
    })
    t9 = time.time()

    t_end = time.time()

    # -----------------------------
    # REPORT
    # -----------------------------
    print("===== LATENCY METRICS =====")
    print(f"YOLO Detection        : {(t1 - t0) * 1000:.2f} ms")
    print(f"AG-MAN Extraction     : {(t3 - t2) * 1000:.2f} ms")
    print(f"Scene Context         : {(t5 - t4) * 1000:.2f} ms")
    print(f"LLM Reasoning         : {(t7 - t6) * 1000:.2f} ms")
    print(f"Product Retrieval     : {(t9 - t8) * 1000:.2f} ms")
    print("-------------------------------------")
    print(f"End-to-End Latency    : {(t_end - t_start) * 1000:.2f} ms")
    print(f"Products Retrieved   : {len(products)}")
    print("=====================================")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    evaluate_latency()
