# ------------------------------------------------------------
# Review 1: Pretrained Model Configurations
# ------------------------------------------------------------

# ---------------------------
# YOLOv8 Configuration
# 
#---------------------------
# YOLO_WEIGHTS="yolov8s.pt"
# ===============================
# YOLO Configuration (Inference)
# ===============================

YOLO_MODELS = [
    # "data/yolo/best.pt",                  # Large model 1
    "data/yolo/yolov8x_best_100 .pt",     # Medium model
    "data/yolo/best_yolov8m_27.pt",       # Small model 1
    "data/yolo/detect.pt"                 # Small model 2
]

# Lower confidence = better recall (important for garments)
# Increased to 0.35 to reduce false positives (hand → shoe etc.)
YOLO_CONF_THRESH = 0.35

# Moderate NMS IoU (don’t suppress accessories)
YOLO_IOU = 0.45

# MUST match training resolution
YOLO_INPUT_SIZE = 960


# ---------------------------
# AG-MAN Attribute Extractor (Pretrained or Public weights)
# ---------------------------
AGMAN_WEIGHTS = "backend/data/agman/agman_pretrained.pth"

# ---------------------------
# Scene Context Detector (Places365)
# ---------------------------
PLACES365_WEIGHTS = "backend/data/scene/places365.pth"
PLACES365_INPUT_SIZE = 224

# ---------------------------
# LLM Reasoner (NO FINETUNING FOR REVIEW 1)
# ---------------------------
LLM_MODEL = "mistral-7b-instruct"   # or GPT-3.5 API
LLM_USE_API = True
OPENAI_API_KEY = ""        # optional for review 1
CONFIDENCE_THRESHOLD = 0.5

# ---------------------------
# Product Retrieval System
# ---------------------------
FAISS_INDEX_PATH = "backend/data/embeddings/sample_index.faiss"
PRODUCT_EMB_PATH = "backend/data/embeddings/sample_embeddings.npy"
PRODUCT_IDS_PATH  = "backend/data/embeddings/sample_ids.npy"

# ---------------------------
# Local Product Database
# ---------------------------
PRODUCT_DB_PATH = "backend/data/db/products.db"


import os

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBu5XSBAJVssQqfv06yJUh0SVXFO3AsJDs")
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyD1YRfxaXnTeQDDkCW3m2kYfiHy8s8ycsg")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyBvHydmgRp04qfflrzOvMNiJYtfIwy_fDA")

# LLM generation controls
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_TOKENS = 400

HOST = "0.0.0.0"
PORT = 5000
