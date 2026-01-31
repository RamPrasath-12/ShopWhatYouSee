
import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------------------------
# Device
# ---------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------
# ResNet50 backbone (Embedding ONLY)
# ---------------------------------------------
resnet = models.resnet50(pretrained=True)
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

# ---------------------------------------------
# FIXED: Skin removal (less aggressive)
# ---------------------------------------------
def create_skin_mask(img_rgb):
    """
    Creates mask where skin pixels = 0, non-skin = 255.
    Less aggressive to preserve more garment pixels.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    # HSV skin detection (narrower range)
    lower_skin_hsv = np.array([0, 25, 80], dtype=np.uint8)
    upper_skin_hsv = np.array([25, 180, 255], dtype=np.uint8)
    skin_hsv = cv2.inRange(img_hsv, lower_skin_hsv, upper_skin_hsv)
    
    # YCrCb skin detection (tighter bounds)
    lower_skin_ycrcb = np.array([0, 140, 85], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 170, 120], dtype=np.uint8)
    skin_ycrcb = cv2.inRange(img_ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
    
    # Combine - pixel must be skin in BOTH to be filtered
    skin_mask = cv2.bitwise_and(skin_hsv, skin_ycrcb)
    
    # Small erosion to remove false positives
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    
    # Invert: 255 = keep (non-skin), 0 = remove (skin)
    return cv2.bitwise_not(skin_mask)

# ---------------------------------------------
# FIXED: Color extraction (handles white better)
# ---------------------------------------------
def extract_primary_secondary_color(pil_img, k=5):
    """
    Extracts dominant colors with proper white/light color handling.
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    
    # Use center 70% of image (avoid edges where background often is)
    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    img_center = img[y1:y2, x1:x2]
    
    # Create skin mask
    skin_mask = create_skin_mask(img_center)
    
    # Convert to LAB for better color separation
    lab = cv2.cvtColor(img_center, cv2.COLOR_RGB2LAB)
    
    # Get pixels (skin-filtered)
    pixels = lab[skin_mask > 0]
    
    if len(pixels) < 300:
        # Fallback: use all pixels if skin filtering is too aggressive
        pixels = lab.reshape(-1, 3)
    
    # Don't filter out whites/lights - they're valid garment colors!
    # Only remove extreme darks (likely shadows/background)
    pixels = pixels[pixels[:, 0] > 15]
    
    if len(pixels) < 300:
        return None, None
    
    # Sample for efficiency
    if len(pixels) > 5000:
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[indices]
    
    pixels = pixels.astype(np.float32)
    
    # KMeans clustering
    K = min(k, max(2, len(pixels) // 100))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort by frequency
    order = np.argsort(counts)[::-1]
    
    def lab_to_hex(lab_color):
        """Convert LAB to RGB hex"""
        lab_uint8 = np.uint8([[lab_color]])
        rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)[0][0]
        return '#%02x%02x%02x' % tuple(rgb)
    
    # Primary color (most frequent cluster)
    primary_hex = lab_to_hex(centers[order[0]])
    
    # Secondary color (must be at least 15% of primary)
    secondary_hex = None
    if len(order) > 1:
        primary_count = counts[order[0]]
        secondary_count = counts[order[1]]
        
        if secondary_count >= 0.15 * primary_count:
            # Check color difference (avoid near-duplicates)
            primary_lab = centers[order[0]]
            secondary_lab = centers[order[1]]
            
            color_diff = np.linalg.norm(primary_lab - secondary_lab)
            
            if color_diff > 15:  # Significant color difference
                secondary_hex = lab_to_hex(secondary_lab)
    
    return primary_hex, secondary_hex

# ---------------------------------------------
# Pattern detector (unchanged core logic)
# ---------------------------------------------
def detect_pattern(pil_img):
    """
    Detects patterns: solid, striped, checked, or patterned.
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    
    # Use center region
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    img_center = img[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(img_center, cv2.COLOR_RGB2GRAY)
    
    if gray.size < 500:
        return None
    
    # Calculate texture variance
    std_dev = np.std(gray)
    
    # Very uniform = solid
    if std_dev < 20:
        return "solid"
    
    # Resize for FFT
    gray_resized = cv2.resize(gray, (128, 128))
    
    # FFT analysis
    f = np.fft.fftshift(np.fft.fft2(gray_resized))
    magnitude = np.abs(f)
    magnitude = np.log(magnitude + 1)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    cy, cx = 64, 64
    
    # Check for stripes
    vertical_strip = magnitude[:, cx-4:cx+4]
    horizontal_strip = magnitude[cy-4:cy+4, :]
    
    v_energy = np.sum(vertical_strip > 0.4)
    h_energy = np.sum(horizontal_strip > 0.4)
    
    if v_energy > 35 or h_energy > 35:
        return "striped"
    
    # Check for checks
    outer = magnitude.copy()
    outer[cy-15:cy+15, cx-15:cx+15] = 0
    high_freq = np.sum(outer > 0.35)
    
    if high_freq > 180:
        return "checked"
    
    if std_dev > 28:
        return "patterned"
    
    return "solid"

# ---------------------------------------------
# FIXED: Sleeve detection (more robust)
# ---------------------------------------------
def estimate_sleeve(pil_img):
    """
    Estimates sleeve length for cropped garment images.
    Analyzes the aspect ratio and coverage of the garment.
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    
    # For very small images, return None
    if h < 50 or w < 50:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create a mask of the garment (non-background)
    # Use Otsu's thresholding to separate garment from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Analyze left and right thirds (where sleeves typically are)
    left_third = binary[:, :w//3]
    right_third = binary[:, 2*w//3:]
    
    def analyze_sleeve_region(region):
        """Analyze how far down the sleeve extends"""
        h_region = region.shape[0]
        
        # For each row, check if there's significant content
        row_coverage = []
        for i in range(h_region):
            row = region[i, :]
            coverage = np.sum(row > 0) / len(row) if len(row) > 0 else 0
            row_coverage.append(coverage)
        
        # Find where the sleeve ends (coverage drops below threshold)
        sleeve_extent = 0
        for i, cov in enumerate(row_coverage):
            if cov > 0.2:  # At least 20% of row has content
                sleeve_extent = i
        
        # Return as fraction of total height
        return sleeve_extent / h_region if h_region > 0 else 0
    
    left_extent = analyze_sleeve_region(left_third)
    right_extent = analyze_sleeve_region(right_third)
    
    # Average both sides
    avg_extent = (left_extent + right_extent) / 2
    
    # Also consider aspect ratio - tall narrow crops are often long sleeves
    aspect_ratio = h / w if w > 0 else 1
    
    # Debug logging
    print(f"🔍 Sleeve Detection Debug:")
    print(f"   Image size: {w}x{h}, Aspect ratio: {aspect_ratio:.2f}")
    print(f"   Left extent: {left_extent:.2f}, Right extent: {right_extent:.2f}")
    print(f"   Average extent: {avg_extent:.2f}")
    
    # Adjusted thresholds for cropped garments
    # Short sleeves: extend less than 40% down the image
    # Three-quarter: 40-70%
    # Long: more than 70%
    
    if avg_extent < 0.4 or aspect_ratio < 0.8:
        result = "short"
    elif avg_extent < 0.7:
        result = "three_quarter"
    else:
        result = "long"
    
    print(f"   → Result: {result}")
    return result

# ---------------------------------------------
# CATEGORY RULES
# ---------------------------------------------
UPPER_WEAR = {
    "shirt", "t_shirt", "tshirt", "blouse", "blazer", "coat",
    "jacket", "top", "sweater", "hoodie", "cardigan", "dress"
}

FABRIC_ITEMS = {
    "shirt", "t_shirt", "tshirt", "blouse", "blazer", "coat",
    "jacket", "top", "sweater", "hoodie", "cardigan", "dress",
    "pant", "skirt", "leggings", "churidhar",
    "saree", "dhoti", "shawl", "shorts"
}

# ---------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------
def process_crop_base64(b64img, category):
    """
    Main function to extract attributes from cropped fashion item.
    
    Args:
        b64img: Base64 encoded image string
        category: Fashion category (from YOLO)
    
    Returns:
        dict with 'attributes' and 'embedding'
    """
    pil_img = b64_to_pil(b64img)
    
    # Normalize category name
    category_normalized = category.lower().strip().replace(" ", "_")
    
    # Handle variations
    if category_normalized in ["t_shirt", "tshirt"]:
        category_normalized = "tshirt"
    
    # Extract embedding
    embedding = extract_embedding(pil_img)
    
    # Extract colors (always)
    primary_color, secondary_color = extract_primary_secondary_color(pil_img)
    
    # Extract pattern (only for fabric items)
    pattern = None
    if category_normalized in FABRIC_ITEMS:
        pattern = detect_pattern(pil_img)
    
    # Extract sleeve (only for upper wear)
    sleeve = None
    if category_normalized in UPPER_WEAR:
        sleeve = estimate_sleeve(pil_img)
    
    attributes = {
        "color_hex": primary_color,
        "secondary_color_hex": secondary_color,
        "pattern": pattern,
        "sleeve_length": sleeve
    }
    
    return {
        "attributes": attributes,
        "embedding": embedding
    }