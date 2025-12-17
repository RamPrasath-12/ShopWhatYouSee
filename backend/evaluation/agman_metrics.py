"""
AG-MAN Attribute Extraction Evaluation (Review-1)
-------------------------------------------------
This script evaluates the AG-MAN module in terms of:
- Inference latency
- Embedding dimensionality
- Embedding similarity stability
- Attribute output consistency

NOTE:
No labeled attribute dataset is used at this stage.
Fine-tuning and quantitative accuracy evaluation
will be performed in future reviews.
"""

import time
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.agman_extractor import process_crop_base64
import base64

# -----------------------------
# CONFIGURATION
# -----------------------------

CROPS_DIR = "evaluation/sample_crops"  # folder with detected crops
RUNS_PER_IMAGE = 3

# -----------------------------
# LOAD CROPS
# -----------------------------

def load_crops():
    images = []
    for file in os.listdir(CROPS_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(CROPS_DIR, file))
            if img is not None:
                images.append(img)
    return images


def cv2_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# -----------------------------
# MAIN EVALUATION
# -----------------------------

def evaluate_agman():
    images = load_crops()
    assert len(images) > 0, "No crop images found!"

    inference_times = []
    embeddings = []

    print("\nRunning AG-MAN attribute extraction...\n")

    for idx, img in enumerate(images):
        img_b64 = cv2_to_base64(img)

        for r in range(RUNS_PER_IMAGE):
            start = time.time()
            result = process_crop_base64(img_b64)
            end = time.time()

            inference_times.append((end - start) * 1000)

            emb = np.array(result["embedding"])
            embeddings.append(emb)

            print(
                f"Image {idx+1}, Run {r+1} | "
                f"Time: {(end-start)*1000:.2f} ms | "
                f"Pattern: {result['attributes'].get('pattern')} | "
                f"Sleeve: {result['attributes'].get('sleeve_length')}"
            )

    embeddings = np.array(embeddings)

    # -----------------------------
    # EMBEDDING SIMILARITY
    # -----------------------------
    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        sims.append(sim)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    print("\n===== AG-MAN METRICS (REVIEW-1) =====")
    print(f"Embedding Dimension      : {embeddings.shape[1]}")
    print(f"Avg Inference Time (ms)  : {np.mean(inference_times):.2f}")
    print(f"Min Inference Time (ms)  : {np.min(inference_times):.2f}")
    print(f"Max Inference Time (ms)  : {np.max(inference_times):.2f}")
    print(f"Avg Cosine Similarity    : {np.mean(sims):.4f}")
    print(f"Min Cosine Similarity    : {np.min(sims):.4f}")
    print(f"Max Cosine Similarity    : {np.max(sims):.4f}")
    print("====================================")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    evaluate_agman()
