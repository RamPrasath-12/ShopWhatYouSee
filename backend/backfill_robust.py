"""
Robust Backfill Script - Standalone Feature Extraction
Does NOT depend on fragile external modules.
"""
import psycopg2
import json
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}
IMAGES_DIR = Path("../data/images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# STANDALONE EXTRACTOR
# =============================================================================

# 1. Load Model (ResNet50 - Generic ImageNet is robust and standard)
# This matches the 'v4' extractor logic which used weights="IMAGENET1K_V1"
print(f"Loading ResNet50 on {DEVICE}...")
model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity() # 2048-dim
model.to(DEVICE)
model.eval()

# 2. Transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            emb = model(img_t)
            
        emb = emb.cpu().numpy().flatten()
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()
    except Exception as e:
        logger.error(f"Embedding error {image_path}: {e}")
        return None

def get_colors(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for speed
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Simple KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        counts = np.bincount(labels.flatten())
        order = np.argsort(counts)[::-1]
        
        primary = centers[order[0]].astype(int)
        secondary = centers[order[1]].astype(int) if len(order) > 1 else None
        
        p_hex = '#{:02x}{:02x}{:02x}'.format(*primary)
        s_hex = '#{:02x}{:02x}{:02x}'.format(*secondary) if secondary is not None else None
        
        return p_hex, s_hex
    except Exception as e:
        return None, None

# =============================================================================
# MAIN LOOP
# =============================================================================

def backfill():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Check work todo
    # We update rows where embedding IS NULL
    cur.execute("SELECT product_id FROM products WHERE embedding IS NULL")
    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} products to process.")
    
    success = 0
    failed = 0
    
    # Process in chunks of commit
    for idx, (pid,) in enumerate(tqdm(rows)):
        p_path = IMAGES_DIR / f"{pid}.jpg"
        
        if not p_path.exists():
            continue
            
        # Extract features
        emb = get_embedding(p_path)
        p_hex, s_hex = get_colors(p_path)
        
        if emb:
            # Update DB
            emb_bytes = psycopg2.Binary(json.dumps(emb).encode('utf-8'))
            
            # attributes json
            attrs = {
                "color_hex": p_hex,
                "secondary_color_hex": s_hex,
                "pattern": "solid" # Default for speed, user can update later
            }
            
            try:
                cur.execute("""
                    UPDATE products 
                    SET embedding = %s,
                        base_colour = COALESCE(NULLIF(base_colour, ''), %s),
                        colour1 = %s,
                        article_attributes = %s
                    WHERE product_id = %s
                """, (emb_bytes, p_hex, s_hex, json.dumps(attrs), pid))
                
                success += 1
            except Exception as e:
                print(f"Update failed for {pid}: {e}")
                failed += 1
        else:
            failed += 1
            
        # Commit every 50
        if idx % 50 == 0:
            conn.commit()
            
    conn.commit()
    cur.close()
    conn.close()
    print(f"Done! Updated {success}, Failed {failed}")

if __name__ == "__main__":
    backfill()
