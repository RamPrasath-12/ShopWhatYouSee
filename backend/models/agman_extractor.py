# backend/models/agman_extractor.py

import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from sklearn.cluster import KMeans

# ---------------------------------------------
# Device setup
# ---------------------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------------------------------------
# # Load ResNet50 backbone (ImageNet pretrained)
# # Embedding size = 2048
# # ---------------------------------------------
# resnet = models.resnet50(pretrained=True)
# resnet.fc = torch.nn.Identity()  # remove top classifier → output 2048D
# resnet = resnet.to(DEVICE)
# resnet.eval()

# # ---------------------------------------------
# # Standard ImageNet preprocessing
# # ---------------------------------------------
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])


# # ---------------------------------------------
# # Convert Base64 to PIL Image
# # ---------------------------------------------
# def b64_to_pil(b64str):
#     if "," in b64str:
#         _, b64data = b64str.split(",", 1)
#     else:
#         b64data = b64str

#     img_bytes = base64.b64decode(b64data)
#     return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# # ---------------------------------------------
# # 2048-D Embedding Extraction
# # ---------------------------------------------
# def extract_embedding(pil_img):
#     img_t = transform(pil_img).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         emb = resnet(img_t)  # (1, 2048)

#     emb = emb.cpu().numpy().flatten()
#     emb = emb / np.linalg.norm(emb)  # normalize
#     return emb


# # ---------------------------------------------
# # Dominant color (KMeans)
# # ---------------------------------------------
# def dominant_color_kmeans(pil_img, k=3):
#     img = np.array(pil_img)
#     pixels = img.reshape(-1, 3).astype(np.float32) / 255.0

#     kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)

#     counts = np.bincount(kmeans.labels_)
#     dominant = kmeans.cluster_centers_[np.argmax(counts)]
#     rgb = (dominant * 255).astype(int).tolist()

#     hex_color = '#%02x%02x%02x' % tuple(rgb)
#     return hex_color, rgb


# # ---------------------------------------------
# # ADVANCED PATTERN DETECTOR (FFT-based)
# # Detects: solid, striped, checked, pattern
# # ---------------------------------------------
# def pattern_detector(pil_img):
#     img = np.array(pil_img)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # FFT transform
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)
#     magnitude = np.abs(fshift)

#     # Normalize
#     mag_norm = magnitude / np.max(magnitude)

#     # 1️⃣ Solid → very low overall frequency
#     if mag_norm.mean() < 0.05:
#         return "solid"

#     # 2️⃣ Stripes → strong horizontal or vertical peaks
#     vertical_energy = np.sum(mag_norm[:, mag_norm.shape[1] // 2] > 0.35)
#     horizontal_energy = np.sum(mag_norm[mag_norm.shape[0] // 2, :] > 0.35)

#     if vertical_energy + horizontal_energy > 40:
#         return "striped"

#     # 3️⃣ Checked / Plaid → grid-like repeated peaks
#     peak_count = np.sum(mag_norm > 0.35)
#     if peak_count > 300:
#         return "checked"

#     # 4️⃣ Otherwise → generic patterned texture
#     return "pattern"


# # ---------------------------------------------
# # Sleeve length (simple heuristic)
# # ---------------------------------------------
# def sleeve_length_estimator(pil_img):
#     w, h = pil_img.size
#     ratio = h / w

#     if ratio > 1.6:
#         return "long"
#     elif ratio > 1.1:
#         return "three_quarter"
#     else:
#         return "short"


# # ---------------------------------------------
# # MAIN API FUNCTION
# # ---------------------------------------------
# def process_crop_base64(b64img):
#     pil_img = b64_to_pil(b64img)

#     # 1. Embedding
#     embedding = extract_embedding(pil_img)

#     # 2. Attributes
#     color_hex, color_rgb = dominant_color_kmeans(pil_img)
#     pattern = pattern_detector(pil_img)  # UPDATED
#     sleeve = sleeve_length_estimator(pil_img)

#     attributes = {
#         "color_hex": color_hex,
#         "color_rgb": color_rgb,
#         "pattern": pattern,
#         "sleeve_length": sleeve
#     }

#     return {
#         "attributes": attributes,
#         "embedding": embedding.tolist()
#     }


###-------Review1 code end---------------#####




# backend/models/agman_extractor.py
#  =====================================================
# AG-MAN Review-2 Extractor (ConvNeXt – Finetuned)
# =====================================================

# import io
# import base64
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# import timm
# from PIL import Image

# # -----------------------------------------------------
# # DEVICE
# # -----------------------------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------------------------------
# # ATTRIBUTE LISTS (MUST MATCH TRAINING)
# # -----------------------------------------------------
# SLEEVE_LIST   = ["short", "long", "sleeveless"]
# PATTERN_LIST  = ["solid", "striped", "checked", "printed", "floral", "other"]
# LENGTH_LIST   = ["short", "midi", "long"]
# FIT_LIST      = ["slim", "regular", "loose", "other"]
# MATERIAL_LIST = ["cotton", "denim", "silk", "polyester", "wool", "other"]

# # -----------------------------------------------------
# # MODEL DEFINITION (MATCHES CHECKPOINT)
# # -----------------------------------------------------
# class AGMAN(nn.Module):
#     def __init__(self, num_colors=2):
#         super().__init__()

#         self.backbone = timm.create_model(
#             "convnext_base",
#             pretrained=False,
#             num_classes=0
#         )

#         feat_dim = 1024

#         self.heads = nn.ModuleDict({
#             "sleeve":   nn.Linear(feat_dim, len(SLEEVE_LIST)),
#             "pattern":  nn.Linear(feat_dim, len(PATTERN_LIST)),
#             "length":   nn.Linear(feat_dim, len(LENGTH_LIST)),
#             "fit":      nn.Linear(feat_dim, len(FIT_LIST)),
#             "material": nn.Linear(feat_dim, len(MATERIAL_LIST)),

#             # RGB REGRESSION (EXACT MATCH TO TRAINING)
#             "color": nn.Sequential(
#                 nn.Linear(feat_dim, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, num_colors * 3)
#             )
#         })

#     def forward(self, x):
#         feat = self.backbone(x)
#         feat = torch.nn.functional.normalize(feat, dim=1)

#         out = {k: head(feat) for k, head in self.heads.items()}
#         out["color"] = out["color"].view(-1, 2, 3)   # (B,2,3)
#         out["embedding"] = feat
#         return out

# # -----------------------------------------------------
# # LOAD CHECKPOINT (SAFE LOAD)
# # -----------------------------------------------------
# MODEL_PATH = "data/agman/combined_fashionnet_final.pth"

# model = AGMAN().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# model_state = model.state_dict()
# filtered = {k: v for k, v in checkpoint.items()
#             if k in model_state and model_state[k].shape == v.shape}

# model_state.update(filtered)
# model.load_state_dict(model_state, strict=False)
# model.eval()

# print(f"✅ AG-MAN loaded ({len(filtered)} layers matched)")

# # -----------------------------------------------------
# # IMAGE PREPROCESSING
# # -----------------------------------------------------
# transform = T.Compose([
#     T.Resize((224,224)),
#     T.ToTensor(),
#     T.Normalize(
#         mean=[0.485,0.456,0.406],
#         std=[0.229,0.224,0.225]
#     )
# ])

# # -----------------------------------------------------
# # HELPERS
# # -----------------------------------------------------
# def b64_to_pil(b64):
#     if "," in b64:
#         b64 = b64.split(",",1)[1]
#     return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

# def rgb01_to_hex(rgb):
#     rgb = np.clip(rgb, 0, 1)
#     r, g, b = (rgb * 255).astype(int)
#     return "#{:02x}{:02x}{:02x}".format(r, g, b)

# def decode_logits(logits, labels):
#     probs = torch.softmax(logits, dim=1)
#     idx = probs.argmax(1).item()
#     return {
#         "label": labels[idx],
#         "confidence": round(probs[0, idx].item(), 4)
#     }

# # -----------------------------------------------------
# # MAIN API
# # -----------------------------------------------------
# def process_crop_base64(b64img):
#     img = b64_to_pil(b64img)
#     x = transform(img).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         out = model(x)

#     # ---- COLOR (RGB REGRESSION)
#     colors_rgb = out["color"][0].cpu().numpy()  # (2,3)
#     colors_hex = [rgb01_to_hex(c) for c in colors_rgb]

#     attributes = {
#         "colors": [
#             {"hex": colors_hex[i], "confidence": 1.0}
#             for i in range(len(colors_hex))
#         ],
#         "sleeve":   decode_logits(out["sleeve"], SLEEVE_LIST),
#         "pattern":  decode_logits(out["pattern"], PATTERN_LIST),
#         "length":   decode_logits(out["length"], LENGTH_LIST),
#         "fit":      decode_logits(out["fit"], FIT_LIST),
#         "material": decode_logits(out["material"], MATERIAL_LIST)
#     }

#     return {
#         "attributes": attributes,
#         "embedding": out["embedding"][0].cpu().numpy().tolist()
#     }




##################
#review 2 code
####################

# import io
# import base64
# import numpy as np
# from PIL import Image
# import cv2
# import torch
# import torchvision.transforms as T
# import torchvision.models as models
# from sklearn.cluster import KMeans

# # ---------------------------------------------
# # Device
# # ---------------------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------------------------------------
# # ResNet50 backbone (Embedding ONLY)
# # ---------------------------------------------
# resnet = models.resnet50(pretrained=True)
# resnet.fc = torch.nn.Identity()  # 2048-D
# resnet = resnet.to(DEVICE)
# resnet.eval()

# # ---------------------------------------------
# # ImageNet preprocessing
# # ---------------------------------------------
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# # ---------------------------------------------
# # Base64 → PIL
# # ---------------------------------------------
# def b64_to_pil(b64str):
#     if "," in b64str:
#         _, b64data = b64str.split(",", 1)
#     else:
#         b64data = b64str
#     img_bytes = base64.b64decode(b64data)
#     return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# # ---------------------------------------------
# # Embedding extraction (for FAISS only)
# # ---------------------------------------------
# def extract_embedding(pil_img):
#     img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         emb = resnet(img_t)
#     emb = emb.cpu().numpy().flatten()
#     return (emb / np.linalg.norm(emb)).tolist()

# # ---------------------------------------------
# # Foreground mask (simple, fast, review-safe)
# # ---------------------------------------------
# def improved_foreground_mask(img_rgb):
#     hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
#     _, _, v = cv2.split(hsv)

#     # adaptive threshold to remove background
#     mask = cv2.adaptiveThreshold(
#         v, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21, 2
#     )

#     # clean noise
#     kernel = np.ones((5,5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     return mask


# # ---------------------------------------------
# # Dominant color (LAB, masked, NO mapping)
# # ---------------------------------------------

# def extract_primary_secondary_color(pil_img, k=3):
#     img = np.array(pil_img)
#     mask = improved_foreground_mask(img)

#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     pixels = lab[mask > 0]

#     if len(pixels) < 800:
#         return None, None

#     pixels = pixels[(pixels[:,0] > 15) & (pixels[:,0] < 95)]
#     if len(pixels) < 800:
#         return None, None

#     pixels = pixels.astype(np.float32)
#     K = min(k, len(pixels))

#     _, labels, centers = cv2.kmeans(
#         pixels, K, None,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
#         5, cv2.KMEANS_PP_CENTERS
#     )

#     counts = np.bincount(labels.flatten())
#     order = np.argsort(counts)[::-1]

#     def lab_to_hex(lab_color):
#         rgb = cv2.cvtColor(
#             np.uint8([[lab_color]]), cv2.COLOR_LAB2RGB
#         )[0][0]
#         return '#%02x%02x%02x' % tuple(int(x) for x in rgb)

#     primary_lab = centers[order[0]]
#     primary_hex = lab_to_hex(primary_lab)

#     secondary_hex = None
#     if len(order) > 1 and counts[order[1]] > 0.25 * counts[order[0]]:
#         secondary_lab = centers[order[1]]
#         secondary_hex = lab_to_hex(secondary_lab)

#     return primary_hex, secondary_hex


# # ---------------------------------------------
# # Pattern detector (ONLY for fabric)
# # ---------------------------------------------
# def detect_pattern(pil_img):
#     img = np.array(pil_img)
#     mask = improved_foreground_mask(img)

#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray = gray[mask > 0]

#     if gray.size < 2000:
#         return None

#     gray = cv2.resize(gray, (128, 128))

#     f = np.fft.fftshift(np.fft.fft2(gray))
#     mag = np.abs(f)
#     mag /= (mag.max() + 1e-6)

#     mean_energy = mag.mean()

#     if mean_energy < 0.08:
#         return "solid"

#     vertical = np.sum(mag[:, mag.shape[1]//2] > 0.35)
#     horizontal = np.sum(mag[mag.shape[0]//2, :] > 0.35)

#     if vertical + horizontal > 30:
#         return "striped"

#     if np.sum(mag > 0.35) > 250:
#         return "checked"

#     return "patterned"

# # ---------------------------------------------
# # Sleeve estimator (CATEGORY-GATED)
# # ---------------------------------------------
# def estimate_sleeve(pil_img):
#     img = np.array(pil_img)
#     h, w, _ = img.shape

#     # Focus on upper half of garment
#     upper = img[:h//2, :, :]
#     gray = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)

#     edges = cv2.Canny(gray, 50, 150)
#     vertical_density = edges.sum(axis=1)

#     coverage = np.count_nonzero(vertical_density) / len(vertical_density)

#     if coverage > 0.75:
#         return "long"
#     elif coverage > 0.45:
#         return "three_quarter"
#     else:
#         return "short"


# # ---------------------------------------------
# # CATEGORY RULES
# # ---------------------------------------------
# UPPER_WEAR = {
#     "Shirt", "T_shirt", "Blouse", "Blazer"
# }

# FABRIC_ITEMS = {
#     "Shirt", "T_shirt", "Blouse", "Blazer",
#     "Pant", "Skirt", "Leggings", "Churidhar",
#     "Saree", "Dhoti", "Shawl"
# }

# # ---------------------------------------------
# # MAIN ENTRY POINT
# # ---------------------------------------------
# def process_crop_base64(b64img, category):
#     pil_img = b64_to_pil(b64img)

#     embedding = extract_embedding(pil_img)

#     primary_color, secondary_color = extract_primary_secondary_color(pil_img)

#     pattern = detect_pattern(pil_img) if category in FABRIC_ITEMS else None
#     sleeve = estimate_sleeve(pil_img) if category in UPPER_WEAR else None

#     attributes = {
#         "color_hex": primary_color,
#         "secondary_color_hex": secondary_color,
#         "pattern": pattern,
#         "sleeve": sleeve
#     }

#     return {
#         "attributes": attributes,
#         "embedding": embedding
#     }




# =============================================================================
# AG-MAN EXTRACTOR v4 - OPTIMIZED FOR SPEED
# =============================================================================
# Performance optimizations:
# - Removed slow GrabCut (was called 3x per image)
# - Fast center-crop + skin masking instead
# - Reduced image size for analysis (128x128)
# - Fewer KMeans iterations
# - Cached color space conversions
# =============================================================================

import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from .agman_loader import refine_embedding

# ---------------------------------------------
# Device
# ---------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------
# ResNet50 backbone (Embedding ONLY)
# ---------------------------------------------
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.fc = torch.nn.Identity()  # 2048-D
resnet = resnet.to(DEVICE)
resnet.eval()

# ---------------------------------------------
# ImageNet preprocessing
# ---------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------
# Base64 → PIL
# ---------------------------------------------
def b64_to_pil(b64str):
    if "," in b64str:
        _, b64data = b64str.split(",", 1)
    else:
        b64data = b64str
    img_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# ---------------------------------------------
# Embedding extraction (for FAISS only)
# ---------------------------------------------
def extract_embedding(pil_img):
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(img_t)
    emb = emb.cpu().numpy().flatten()
    return (emb / np.linalg.norm(emb)).tolist()


# =============================================================================
# CATEGORY DEFINITIONS (ALL 32 YOLO CATEGORIES)
# =============================================================================

# Upper body garments with sleeves
UPPER_WEAR = {
    "shirt", "tshirt", "t_shirt", "blouse", "blazer", "jacket", "shawl"
}

# Lower body garments
LOWER_WEAR = {
    "pant", "shorts", "skirt", "leggings"
}

# Full body / traditional wear
FULL_BODY = {
    "churidhar", "dhoti", "saree"
}

# All fabric items (for pattern detection)
FABRIC_ITEMS = UPPER_WEAR | LOWER_WEAR | FULL_BODY

# Accessories (color only, no pattern/sleeve)
ACCESSORIES = {
    "bag", "bangle", "belt", "bracelet", "cap", "earring", "glass",
    "hairclip", "necklace", "purse", "ring", "tie", "watch"
}

# Footwear (color only)
FOOTWEAR = {
    "footwear_flats", "footwear_heels", "footwear_sandals", "footwear_shoes"
}

# All known categories
ALL_CATEGORIES = FABRIC_ITEMS | ACCESSORIES | FOOTWEAR


# =============================================================================
# FAST SKIN DETECTION (Optimized - single pass)
# =============================================================================

def create_skin_mask_fast(img_rgb):
    """
    Fast skin detection using YCrCb only (faster than dual-colorspace).
    Returns mask where non-skin = 255, skin = 0.
    """
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    # YCrCb skin detection (covers most skin tones)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower, upper)
    
    # Invert: 255 = keep (non-skin), 0 = remove (skin)
    return cv2.bitwise_not(skin_mask)


# =============================================================================
# FAST COLOR EXTRACTION (No GrabCut)
# =============================================================================

def extract_primary_secondary_color_fast(img_rgb, k=4):
    """
    Fast color extraction using center crop + skin removal.
    No GrabCut - uses simple center region instead.
    """
    h, w = img_rgb.shape[:2]
    
    # Step 1: Use center 60% of image (fast approximation of foreground)
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    center_crop = img_rgb[y1:y2, x1:x2]
    
    # Step 2: Downsample for speed (max 100x100)
    ch, cw = center_crop.shape[:2]
    if ch > 100 or cw > 100:
        scale = min(100/ch, 100/cw)
        center_crop = cv2.resize(center_crop, (int(cw*scale), int(ch*scale)))
    
    # Step 3: Remove skin pixels
    skin_mask = create_skin_mask_fast(center_crop)
    
    # Step 4: Convert to LAB and extract masked pixels
    lab = cv2.cvtColor(center_crop, cv2.COLOR_RGB2LAB)
    pixels = lab[skin_mask > 0]
    
    # Fallback if too few pixels
    if len(pixels) < 100:
        pixels = lab.reshape(-1, 3)
    
    # Filter shadows (keep lights/whites)
    pixels = pixels[pixels[:, 0] > 20]
    
    if len(pixels) < 100:
        # Early exit: always return 4 values (primary, secondary, conf, sec_conf)
        return None, None, 0.0, 0.0
    
    # Subsample for speed (max 2000 pixels)
    if len(pixels) > 2000:
        indices = np.random.choice(len(pixels), 2000, replace=False)
        pixels = pixels[indices]
    
    pixels = pixels.astype(np.float32)
    
    # Fast KMeans (fewer iterations)
    K = min(k, max(2, len(pixels) // 300))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 10 iterations
    
    try:
        _, labels, centers = cv2.kmeans(
            pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS  # 3 attempts
        )
    except cv2.error:
        # Early exit: always return 4 values (primary, secondary, conf, sec_conf)
        return None, None, 0.0, 0.0
    
    # Count clusters
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    
    # Calculate color confidence based on:
    # 1. Dominance of primary cluster (how much of image it covers)
    # 2. Separation from secondary cluster
    total_pixels = len(pixels)
    primary_coverage = counts[order[0]] / total_pixels
    
    # Base confidence from coverage (0.5-0.9 range)
    color_confidence = 0.5 + (primary_coverage * 0.5)
    
    # Boost confidence if secondary is very different or very small
    if len(order) > 1:
        secondary_coverage = counts[order[1]] / total_pixels
        if secondary_coverage < 0.15:
            color_confidence = min(0.95, color_confidence + 0.1)
    
    def lab_to_hex(lab_color):
        lab_uint8 = np.uint8([[lab_color]])
        rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)[0][0]
        return '#%02x%02x%02x' % tuple(rgb)
    
    primary_hex = lab_to_hex(centers[order[0]])
    
    # Secondary color
    secondary_hex = None
    secondary_confidence = 0.0
    if len(order) > 1:
        if counts[order[1]] >= 0.15 * counts[order[0]]:
            color_diff = float(np.linalg.norm(centers[order[0]] - centers[order[1]]))
            if color_diff > 20:
                secondary_hex = lab_to_hex(centers[order[1]])
                secondary_confidence = float(0.4 + (color_diff / 100))  # Cast to native float
    
    # Assertion: guarantee caller always receives exactly 4 values
    result = (primary_hex, secondary_hex, color_confidence, secondary_confidence)
    assert len(result) == 4, f"extract_primary_secondary_color_fast must return 4 values, got {len(result)}"
    return result


# =============================================================================
# FAST PATTERN DETECTION (No GrabCut)
# =============================================================================

def detect_pattern_fast(img_rgb):
    """
    Fast pattern detection using center crop + FFT.
    No GrabCut - uses simple center region.
    
    Returns:
        tuple: (pattern_type, confidence)
    """
    h, w = img_rgb.shape[:2]
    
    # Use center 60%
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    center = img_rgb[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    
    if gray.size < 500:
        return "solid", 0.5  # Low confidence for small images
    
    # Texture variance (fast check)
    std_dev = np.std(gray)
    
    # ---- SOLID DETECTION (raised threshold) ----
    # std_dev < 25 means very uniform texture → almost certainly solid
    if std_dev < 25:
        confidence = min(0.95, 0.6 + (25 - std_dev) / 40)
        return "solid", confidence
    
    # Resize to 64x64 for fast FFT
    gray_small = cv2.resize(gray, (64, 64))
    
    # FFT analysis
    f = np.fft.fftshift(np.fft.fft2(gray_small.astype(np.float32)))
    magnitude = np.abs(f)
    magnitude = np.log(magnitude + 1)
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    
    cy, cx = 32, 32
    
    # ---- MASK OUT DC COMPONENT ----
    # The center 5x5 region is DC spillover, NOT pattern signal.
    # Measuring it gives false positives on solid/dark colors.
    dc_mask = magnitude.copy()
    dc_mask[cy-2:cy+3, cx-2:cx+3] = 0  # Zero out DC region
    
    # ---- STRIPE DETECTION (with DC exclusion) ----
    # Vertical stripes → horizontal FFT energy (along cx column band, excluding DC)
    v_band = dc_mask[:, cx-3:cx+3]
    h_band = dc_mask[cy-3:cy+3, :]
    v_energy = np.sum(v_band > 0.35)
    h_energy = np.sum(h_band > 0.35)
    
    # Stripes need STRONG directional energy AND should be much stronger
    # on one axis vs the other (real stripes are directional)
    max_energy = max(v_energy, h_energy)
    min_energy = min(v_energy, h_energy) + 1  # avoid div-by-zero
    directional_ratio = max_energy / min_energy
    
    if max_energy > 30 and directional_ratio > 2.0 and std_dev > 30:
        # High energy + strong directionality + enough texture variance
        stripe_confidence = min(0.95, 0.6 + max_energy / 120)
        return "striped", stripe_confidence
    
    # ---- CHECK / PLAID DETECTION ----
    outer = dc_mask.copy()
    outer[cy-10:cy+10, cx-10:cx+10] = 0
    check_score = np.sum(outer > 0.30)
    if check_score > 80:
        check_confidence = min(0.90, 0.5 + check_score / 300)
        return "checked", check_confidence
    
    if std_dev > 35:
        # High texture variance but no clear stripe/check = generic pattern
        return "patterned", 0.65
    
    return "solid", 0.70


# =============================================================================
# FAST SLEEVE DETECTION (No GrabCut)
# =============================================================================

def estimate_sleeve_fast(img_rgb):
    """
    Fast sleeve estimation using aspect ratio + edge analysis.
    No GrabCut dependency.
    
    Returns:
        tuple: (sleeve_type, confidence)
    """
    h, w = img_rgb.shape[:2]
    
    # Method 1: Aspect ratio (reliable for YOLO crops)
    aspect = w / h if h > 0 else 1.0
    
    if aspect > 1.4:
        return "long", 0.85  # High confidence for extreme aspect
    if aspect > 1.15:
        return "three_quarter", 0.75
    
    # Method 2: Edge analysis on upper portion (fast)
    upper = img_rgb[:h//2, :]
    
    # Downsample for speed
    uh, uw = upper.shape[:2]
    if uw > 100:
        upper = cv2.resize(upper, (100, int(uh * 100 / uw)))
        uw = 100
    
    gray = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # Check left/right quadrants
    left = edges[:, :uw//4]
    right = edges[:, 3*uw//4:]
    
    left_density = np.mean(left > 0) * 100
    right_density = np.mean(right > 0) * 100
    avg_density = (left_density + right_density) / 2
    
    if avg_density > 12:
        # Higher density = more certain about longer sleeves
        confidence = min(0.90, 0.6 + avg_density / 50)
        return "three_quarter", confidence
    elif avg_density > 6:
        confidence = min(0.85, 0.55 + avg_density / 40)
        return "short", confidence
    else:
        # Very low density = confident sleeveless
        confidence = min(0.88, 0.7 + (6 - avg_density) / 20)
        return "sleeveless", confidence


# =============================================================================
# MAIN ENTRY POINT (Optimized)
# =============================================================================

def process_crop_base64(b64img, category):
    """
    Main function to extract attributes from cropped fashion item.
    Optimized for speed - no GrabCut, fast algorithms.
    
    PRODUCTION-GRADE: Returns structured attributes with confidence scores
    for use in HARD/SOFT constraint classification.
    
    Args:
        b64img: Base64 encoded image string
        category: Fashion category (from YOLO detection)
    
    Returns:
        dict with 'attributes' (structured with confidence) and 'embedding'
    """
    pil_img = b64_to_pil(b64img)
    img_rgb = np.array(pil_img)
    
    # Normalize category name
    category_normalized = category.lower().strip().replace(" ", "_")
    
    # Handle common variations
    category_map = {
        "t_shirt": "tshirt",
        "t-shirt": "tshirt",
        "jacket": "jacket",
        "coat": "jacket",
    }
    category_normalized = category_map.get(category_normalized, category_normalized)
    
    # Extract embedding (always needed, can't optimize much)
    embedding = extract_embedding(pil_img)
    refined_embedding = refine_embedding(embedding)
    
    # =========================================================================
    # STRUCTURED ATTRIBUTE EXTRACTION WITH CONFIDENCE SCORES
    # =========================================================================
    confidence_scores = []  # Track all confidences for extraction_quality
    
    # Extract colors (fast version) - now returns confidence
    primary_color, secondary_color, color_confidence, secondary_confidence = extract_primary_secondary_color_fast(img_rgb)
    confidence_scores.append(color_confidence)
    
    # Map hex colors to color names for better search matching
    try:
        from utils.color_utils import hex_to_color_name
        primary_color_name = hex_to_color_name(primary_color) if primary_color else None
        secondary_color_name = hex_to_color_name(secondary_color) if secondary_color else None
        print(f"[AGMAN] Color mapped: {primary_color} → {primary_color_name}")
    except Exception as e:
        print(f"[AGMAN] Color mapping failed: {e}")
        primary_color_name = None
        secondary_color_name = None
    
    # Extract pattern (only for fabric items, fast version) - now returns (value, confidence)
    pattern_value = None
    pattern_confidence = 0.0
    if category_normalized in FABRIC_ITEMS:
        pattern_value, pattern_confidence = detect_pattern_fast(img_rgb)
        confidence_scores.append(pattern_confidence)
    
    # Extract sleeve (only for upper wear, fast version) - now returns (value, confidence)
    sleeve_value = None
    sleeve_confidence = 0.0
    if category_normalized in UPPER_WEAR:
        sleeve_value, sleeve_confidence = estimate_sleeve_fast(img_rgb)
        confidence_scores.append(sleeve_confidence)
    
    # =========================================================================
    # STRUCTURED OUTPUT FORMAT FOR PRODUCTION RETRIEVAL
    # Each attribute has: value, confidence, source (always "visual" from AGMAN)
    # =========================================================================
    attributes = {
        # Color - structured
        "color": {
            "value": primary_color_name,
            "hex": primary_color,
            "confidence": round(color_confidence, 3),
            "source": "visual"
        },
        # Secondary color (if detected)
        "secondary_color": {
            "value": secondary_color_name,
            "hex": secondary_color,
            "confidence": round(secondary_confidence, 3),
            "source": "visual"
        } if secondary_color else None,
        # Pattern - structured (new format)
        "pattern_structured": {
            "value": pattern_value,
            "confidence": round(pattern_confidence, 3),
            "source": "visual"
        } if pattern_value else None,
        # Sleeve - structured (new format)
        "sleeve_structured": {
            "value": sleeve_value,
            "confidence": round(sleeve_confidence, 3),
            "source": "visual"
        } if sleeve_value else None,
        
        # =====================================================================
        # LEGACY FLAT ATTRIBUTES (for backward compatibility)
        # These match what the frontend expects
        # =====================================================================
        "color_hex": primary_color,
        "color_name": primary_color_name,
        "secondary_color_hex": secondary_color,
        "secondary_color_name": secondary_color_name,
        "pattern": pattern_value,    # Flat string for frontend
        "sleeve": sleeve_value,      # Flat string for frontend
        "sleeve_length": sleeve_value,  # Alias for frontend compatibility
    }
    
    # Calculate overall extraction quality (min of all confidences)
    # This indicates how reliable the entire extraction is
    extraction_quality = min(confidence_scores) if confidence_scores else 0.5
    attributes["extraction_quality"] = round(extraction_quality, 3)
    
    # ===== ENHANCED AGMAN LOGGING =====
    print(f"\n{'='*60}")
    print(f"[AGMAN] 🧬 FINETUNED EMBEDDING EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"[AGMAN] Category: {category}")
    print(f"[AGMAN] Embedding dims: {len(refined_embedding)}")
    print(f"[AGMAN] Top 5 embedding values: {refined_embedding[:5]}")
    print(f"[AGMAN] Embedding norm: {sum(v**2 for v in refined_embedding)**0.5:.4f}")
    print(f"[AGMAN] Attributes extracted (with confidence):")
    print(f"        Color: {primary_color_name} ({primary_color}) [conf: {color_confidence:.2f}]")
    print(f"        Pattern: {pattern_value} [conf: {pattern_confidence:.2f}]")
    print(f"        Sleeve: {sleeve_value} [conf: {sleeve_confidence:.2f}]")
    print(f"        Extraction Quality: {extraction_quality:.2f}")
    print(f"{'='*60}\n")
    
    return {
        "attributes": attributes,
        "embedding": refined_embedding
    }