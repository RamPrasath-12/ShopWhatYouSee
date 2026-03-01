"""
Finetuned AGMAN Backfill Script
Process images to extract 512-dim embeddings using Finetuned AGMAN model.
"""
import os
import gc
import json
import psycopg2
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from sklearn.cluster import KMeans
import logging
import sys

# Add models path to sys.path to ensure imports work
sys.path.append(os.path.dirname(__file__))

from models.agman_loader import refine_embedding
# Ensure we don't hold onto the agman model global variable if possible?
# agman_loader loads it globally. Ideally, we load it once.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 50 

# ============================================================================
# UTILS
# ============================================================================
COLOR_NAMES = {
    "Black": [(0, 0, 0), (50, 50, 50)],
    "White": [(230, 230, 230), (255, 255, 255)],
    "Grey": [(80, 80, 80), (180, 180, 180)],
    "Red": [(150, 0, 0), (255, 80, 80)],
    "Blue": [(0, 0, 120), (100, 100, 255)],
    "Navy Blue": [(0, 0, 60), (50, 50, 120)],
    "Green": [(0, 100, 0), (100, 200, 100)],
    "Yellow": [(200, 200, 0), (255, 255, 100)],
    "Orange": [(200, 100, 0), (255, 180, 50)],
    "Pink": [(200, 100, 150), (255, 180, 210)],
    "Purple": [(100, 0, 150), (180, 100, 220)],
    "Brown": [(80, 40, 0), (180, 120, 80)],
    "Beige": [(180, 160, 130), (240, 220, 190)],
    "Olive": [(80, 80, 0), (150, 150, 80)],
    "Maroon": [(80, 0, 0), (150, 50, 50)],
    "Teal": [(0, 100, 100), (100, 180, 180)],
    "Multi": None
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_color_name(r, g, b):
    min_dist = float('inf')
    best_color = "Multi"
    for name, ranges in COLOR_NAMES.items():
        if ranges is None: continue
        min_rgb, max_rgb = ranges
        if (min_rgb[0] <= r <= max_rgb[0] and min_rgb[1] <= g <= max_rgb[1] and min_rgb[2] <= b <= max_rgb[2]):
            return name
        center = ((min_rgb[0]+max_rgb[0])//2, (min_rgb[1]+max_rgb[1])//2, (min_rgb[2]+max_rgb[2])//2)
        dist = ((r-center[0])**2 + (g-center[1])**2 + (b-center[2])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            best_color = name
    return best_color

def hex_to_color_name(hex_color):
    try:
        r, g, b = hex_to_rgb(hex_color)
        return rgb_to_color_name(r, g, b)
    except:
        return "Multi"

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
print(f"Loading ResNet50 (Base) on {DEVICE}...")
base_model = models.resnet50(weights="IMAGENET1K_V1")
base_model.fc = torch.nn.Identity()
base_model = base_model.to(DEVICE)
base_model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_base_embedding(img_path):
    # Returns 2048-dim embedding via ResNet
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = base_model(img_t).squeeze().cpu().numpy()
        return emb
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def get_colors(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3)
        
        mask = ~((pixels.sum(axis=1) > 700) | (pixels.sum(axis=1) < 60))
        pixels = pixels[mask]
        
        if len(pixels) < 50: pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=min(4, len(pixels)), random_state=42, n_init=3)
        kmeans.fit(pixels)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_idx = np.argsort(-counts)
        colors = []
        for idx in sorted_idx[:2]:
            center = kmeans.cluster_centers_[idx].astype(int)
            colors.append("#{:02x}{:02x}{:02x}".format(*center))
        while len(colors) < 2: colors.append("#000000")
        return colors[0], colors[1]
    except: return None, None

def get_pattern(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "solid"
        img = cv2.resize(img, (100, 100))
        std_dev = np.std(img)
        if std_dev > 40: return "patterned"
        return "solid"
    except: return "solid"

def get_sleeve(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        aspect = w/h if h>0 else 1.0
        if aspect > 1.4: return "long"
        elif aspect > 1.15: return "three_quarter"
        else: return "short"
    except: return None

# ============================================================================
# BACKFILL
# ============================================================================
def ensure_columns(conn):
    columns = [
        ("embedding", "TEXT"),
        ("primary_color", "VARCHAR(50)"),
        ("secondary_color", "VARCHAR(50)"),
        ("agman_pattern", "VARCHAR(50)"),
        ("agman_sleeve", "VARCHAR(50)")
    ]
    cur = conn.cursor()
    for col_name, col_type in columns:
        try:
            cur.execute(f"ALTER TABLE products ADD COLUMN {col_name} {col_type}")
            conn.commit()
            print(f"Added column: {col_name}")
        except:
            conn.rollback()
    cur.close()

def backfill():
    print("Connecting to DB...")
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_columns(conn)
    
    cur = conn.cursor()
    # Find products needing updates (either no embedding, OR embedding is old 2048-dim if we are re-running)
    # Actually, let's just process NULL embeddings or NULL colors
    cur.execute("SELECT product_id, yolo_category FROM products WHERE embedding IS NULL OR primary_color IS NULL")
    products = cur.fetchall()
    total = len(products)
    print(f"Found {total} products to process")
    
    processed = 0
    errors = 0
    
    for i, (pid, cat) in enumerate(products):
        try:
            img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_DIR, f"{pid}") # specific fix 

            if not os.path.exists(img_path):
                # errors+=1
                continue

            # 1. Base Embedding (2048)
            base_emb = get_base_embedding(img_path)
            
            # 2. Refined AGMAN Embedding (512)
            agman_emb = None
            if base_emb is not None:
                agman_emb = refine_embedding(base_emb) # Uses loaded AGMAN model
            
            # 3. Attributes
            primary, secondary = get_colors(img_path)
            pattern = get_pattern(img_path)
            sleeve = get_sleeve(img_path) if cat and cat.lower() in ['shirt','t_shirt','jacket','top'] else None
            
            # Update DB
            p_name = hex_to_color_name(primary) if primary else None
            s_name = hex_to_color_name(secondary) if secondary else None
            emb_json = json.dumps(agman_emb) if agman_emb else None
            
            cur.execute("""
                UPDATE products SET 
                    embedding = %s, primary_color = %s, secondary_color = %s, 
                    agman_pattern = %s, agman_sleeve = %s 
                WHERE product_id = %s
            """, (emb_json, p_name, s_name, pattern, sleeve, pid))
            
            processed += 1
            if processed % BATCH_SIZE == 0:
                conn.commit()
                # Clear memory aggressively
                gc.collect()
                if DEVICE == 'cuda': torch.cuda.empty_cache()
                print(f"Progress: {processed}/{total}")
        
        except Exception as e:
            errors += 1
            if errors < 5: print(f"Error {pid}: {e}")
            
    conn.commit()
    cur.close()
    conn.close()
    print("Backfill Complete!")

if __name__ == "__main__":
    backfill()
