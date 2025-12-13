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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------
# Load ResNet50 backbone (ImageNet pretrained)
# Embedding size = 2048
# ---------------------------------------------
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove top classifier → output 2048D
resnet = resnet.to(DEVICE)
resnet.eval()

# ---------------------------------------------
# Standard ImageNet preprocessing
# ---------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------
# Convert Base64 to PIL Image
# ---------------------------------------------
def b64_to_pil(b64str):
    if "," in b64str:
        _, b64data = b64str.split(",", 1)
    else:
        b64data = b64str

    img_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---------------------------------------------
# 2048-D Embedding Extraction
# ---------------------------------------------
def extract_embedding(pil_img):
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = resnet(img_t)  # (1, 2048)

    emb = emb.cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)  # normalize
    return emb


# ---------------------------------------------
# Dominant color (KMeans)
# ---------------------------------------------
def dominant_color_kmeans(pil_img, k=3):
    img = np.array(pil_img)
    pixels = img.reshape(-1, 3).astype(np.float32) / 255.0

    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    rgb = (dominant * 255).astype(int).tolist()

    hex_color = '#%02x%02x%02x' % tuple(rgb)
    return hex_color, rgb


# ---------------------------------------------
# ADVANCED PATTERN DETECTOR (FFT-based)
# Detects: solid, striped, checked, pattern
# ---------------------------------------------
def pattern_detector(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # FFT transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Normalize
    mag_norm = magnitude / np.max(magnitude)

    # 1️⃣ Solid → very low overall frequency
    if mag_norm.mean() < 0.05:
        return "solid"

    # 2️⃣ Stripes → strong horizontal or vertical peaks
    vertical_energy = np.sum(mag_norm[:, mag_norm.shape[1] // 2] > 0.35)
    horizontal_energy = np.sum(mag_norm[mag_norm.shape[0] // 2, :] > 0.35)

    if vertical_energy + horizontal_energy > 40:
        return "striped"

    # 3️⃣ Checked / Plaid → grid-like repeated peaks
    peak_count = np.sum(mag_norm > 0.35)
    if peak_count > 300:
        return "checked"

    # 4️⃣ Otherwise → generic patterned texture
    return "pattern"


# ---------------------------------------------
# Sleeve length (simple heuristic)
# ---------------------------------------------
def sleeve_length_estimator(pil_img):
    w, h = pil_img.size
    ratio = h / w

    if ratio > 1.6:
        return "long"
    elif ratio > 1.1:
        return "three_quarter"
    else:
        return "short"


# ---------------------------------------------
# MAIN API FUNCTION
# ---------------------------------------------
def process_crop_base64(b64img):
    pil_img = b64_to_pil(b64img)

    # 1. Embedding
    embedding = extract_embedding(pil_img)

    # 2. Attributes
    color_hex, color_rgb = dominant_color_kmeans(pil_img)
    pattern = pattern_detector(pil_img)  # UPDATED
    sleeve = sleeve_length_estimator(pil_img)

    attributes = {
        "color_hex": color_hex,
        "color_rgb": color_rgb,
        "pattern": pattern,
        "sleeve_length": sleeve
    }

    return {
        "attributes": attributes,
        "embedding": embedding.tolist()
    }
