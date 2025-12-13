# ------------------------------------------------------------
# Review 1: Pretrained Model Configurations
# ------------------------------------------------------------

# ---------------------------
# YOLOv8 Configuration
# ---------------------------
YOLO_WEIGHTS = "data/yolo/detect.pt"   # Pretrained model
YOLO_CONF_THRESH = 0.35
YOLO_IOU = 0.45
YOLO_INPUT_SIZE = 640

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

HOST = "0.0.0.0"
PORT = 5000
