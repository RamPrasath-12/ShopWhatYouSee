# AGMAN Embeddings & Attributes Extraction for Kaggle
# ======================================================
# Upload this to Kaggle, add your images as a dataset, and run with GPU
# Output: CSV file to download and import into PostgreSQL
#
# INSTRUCTIONS:
# 1. Create new Kaggle notebook
# 2. Add dataset: Upload your images folder (data/images) as a dataset
# 3. Enable GPU: Settings > Accelerator > GPU P100
# 4. Copy-paste this code and run
# 5. Download the output CSV

!pip install -q torch torchvision scikit-learn opencv-python-headless

import os
import gc
import csv
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from sklearn.cluster import KMeans
from tqdm import tqdm

# ============================================================================
# CONFIG - UPDATE THIS PATH TO YOUR KAGGLE DATASET
# ============================================================================
IMAGES_DIR = "/kaggle/input/shopwhatyousee-images/images"  # UPDATE THIS!
OUTPUT_CSV = "/kaggle/working/agman_attributes.csv"
BATCH_SIZE = 100  # Higher batch for GPU

# Check if path exists
if not os.path.exists(IMAGES_DIR):
    print(f"⚠️ Images directory not found: {IMAGES_DIR}")
    print("Available datasets:")
    for d in os.listdir("/kaggle/input"):
        print(f"  /kaggle/input/{d}")
    raise FileNotFoundError("Update IMAGES_DIR to your dataset path")

# ============================================================================
# COLOR NAME MAPPING
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
    "Cream": [(240, 230, 200), (255, 250, 230)],
    "Charcoal": [(40, 40, 40), (80, 80, 80)],
    "Burgundy": [(100, 0, 30), (150, 50, 80)],
    "Rust": [(150, 60, 20), (200, 100, 60)],
    "Mustard": [(200, 180, 0), (240, 200, 50)],
    "Khaki": [(180, 180, 100), (200, 200, 140)],
    "Lavender": [(180, 150, 200), (230, 200, 255)],
    "Coral": [(255, 100, 80), (255, 150, 120)],
    "Turquoise": [(0, 180, 180), (100, 230, 230)],
}

def rgb_to_color_name(r, g, b):
    min_dist = float('inf')
    best_color = "Multi"
    for name, ranges in COLOR_NAMES.items():
        if ranges is None: continue
        min_rgb, max_rgb = ranges
        if (min_rgb[0] <= r <= max_rgb[0] and 
            min_rgb[1] <= g <= max_rgb[1] and 
            min_rgb[2] <= b <= max_rgb[2]):
            return name
        center = ((min_rgb[0] + max_rgb[0])//2, 
                  (min_rgb[1] + max_rgb[1])//2, 
                  (min_rgb[2] + max_rgb[2])//2)
        dist = ((r - center[0])**2 + (g - center[1])**2 + (b - center[2])**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_color = name
    return best_color

def hex_to_color_name(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb_to_color_name(r, g, b)
    except:
        return "Multi"

# ============================================================================
# LOAD MODEL
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(img_t).squeeze().cpu().numpy()
        return emb
    except:
        return None

def get_colors(img_path):
    try:
        img = cv2.imread(img_path)
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
            hex_color = "#{:02x}{:02x}{:02x}".format(center[0], center[1], center[2])
            color_name = hex_to_color_name(hex_color)
            colors.append(color_name)
        
        while len(colors) < 2: colors.append("Multi")
        return colors[0], colors[1]
    except:
        return None, None

def get_pattern(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "solid"
        img = cv2.resize(img, (100, 100))
        std_dev = np.std(img)
        
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        horiz = magnitude[center_h-5:center_h+5, :]
        vert = magnitude[:, center_w-5:center_w+5]
        
        horiz_energy = np.mean(horiz)
        vert_energy = np.mean(vert)
        
        if horiz_energy > 4.0 and vert_energy > 4.0: return "checked"
        elif horiz_energy > 4.0 or vert_energy > 4.0: return "striped"
        elif std_dev > 40: return "patterned"
        else: return "solid"
    except:
        return "solid"

def get_sleeve(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None: return None
        h, w = img.shape[:2]
        aspect = w / h if h > 0 else 1.0
        if aspect > 1.4: return "long"
        elif aspect > 1.15: return "three_quarter"
        else: return "short"
    except:
        return None

# ============================================================================
# MAIN PROCESSING
# ============================================================================
print(f"📂 Scanning images in {IMAGES_DIR}...")
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"📸 Found {len(image_files)} images")

results = []
embeddings = {}

for i, filename in enumerate(tqdm(image_files, desc="Processing")):
    try:
        product_id = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGES_DIR, filename)
        
        # Extract features
        embedding = get_embedding(img_path)
        primary_color, secondary_color = get_colors(img_path)
        pattern = get_pattern(img_path)
        sleeve = get_sleeve(img_path)
        
        results.append({
            'product_id': product_id,
            'primary_color': primary_color or 'Multi',
            'secondary_color': secondary_color or 'Multi',
            'agman_pattern': pattern or 'solid',
            'agman_sleeve': sleeve or ''
        })
        
        if embedding is not None:
            embeddings[product_id] = embedding
        
        # Clear memory periodically
        if i % BATCH_SIZE == 0:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error {filename}: {e}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print(f"\n💾 Saving {len(results)} results to CSV...")

# Save attributes CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['product_id', 'primary_color', 'secondary_color', 'agman_pattern', 'agman_sleeve'])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Attributes saved to: {OUTPUT_CSV}")

# Save embeddings as numpy file
embeddings_path = "/kaggle/working/agman_embeddings.npz"
np.savez_compressed(embeddings_path, **embeddings)
print(f"✅ Embeddings saved to: {embeddings_path}")

print(f"\n🎉 DONE! Download these files:")
print(f"   1. {OUTPUT_CSV}")
print(f"   2. {embeddings_path}")
print(f"\nThen run the import script locally to update your database.")
