# backend/models/yolo_detector.py
# =============================================================================
# YOLO DETECTOR v2 - Fixed & Optimized
# =============================================================================
# Fixes:
# - Complete CANONICAL_MAP for all 32 categories
# - Separate T-shirt from Shirt
# - Body part filtering (hand, face, person → skip)
# - Cross-class deduplication (overlapping boxes)
# - Higher confidence threshold for accuracy
# =============================================================================

import cv2
import base64
import numpy as np
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Classes to DROP completely (invalid for fashion retrieval)
DROP_CLASSES = {
    "dress",        # Not in our product DB
    "person",       # Body, not fashion item
    "hand",         # Body part
    "face",         # Body part
    "arm",          # Body part
    "leg",          # Body part
    "head",         # Body part
    "body",         # Body part
    "human",        # Body
}

# Complete mapping from YOLO output → Canonical category names
# All 32 valid categories are listed here
CANONICAL_MAP = {
    # Upper wear
    "shirt": "shirt",
    "t-shirt": "tshirt",
    "tshirt": "tshirt",
    "t_shirt": "tshirt",
    "blouse": "blouse",
    "blazer": "blazer",
    "jacket": "jacket",
    "coat": "jacket",
    "shawl": "shawl",
    
    # Lower wear
    "pants": "pant",
    "pant": "pant",
    "trousers": "pant",
    "shorts": "shorts",
    "short": "shorts",
    "skirt": "skirt",
    "leggings": "pant",      # Map leggings → pant (DB category)
    "legging": "pant",       # Map legging → pant (DB category)
    
    # Full body / Traditional
    "churidhar": "churidhar",
    "churidar": "churidhar",
    "dhoti": "dhoti",
    "saree": "saree",
    "sari": "saree",
    
    # Accessories
    "bag": "bag",
    "handbag": "bag",
    "purse": "purse",
    "bangle": "bangle",
    "bracelet": "bracelet",
    "belt": "belt",
    "cap": "cap",
    "hat": "cap",           # Map hat → cap
    "earring": "earring",
    "earrings": "earring",
    "glass": "glass",
    "glasses": "glass",
    "sunglass": "glass",
    "sunglasses": "glass",
    "hairclip": "hairclip",
    "hairclips": "hairclip",
    "necklace": "necklace",
    "ring": "ring",
    "tie": "tie",
    "watch": "watch",
    
    # Footwear
    "shoe": "footwear_shoes",
    "shoes": "footwear_shoes",
    "footwear_shoes": "footwear_shoes",
    "footwear_sandals": "footwear_sandals",
    "sandal": "footwear_sandals",
    "sandals": "footwear_sandals",
    "footwear_heels": "footwear_heels",
    "heel": "footwear_heels",
    "heels": "footwear_heels",
    "footwear_flats": "footwear_flats",
    "flat": "footwear_flats",
    "flats": "footwear_flats",
}

# =============================================================================
# YOLO → DB CATEGORY MAPPING
# =============================================================================
# Maps YOLO canonical output names to the EXACT category names stored in the DB.
# Fixes singular/plural, case, and semantic mismatches.
# Applied immediately after YOLO detection, before retrieval.
# DO NOT modify DB categories — normalize at detection layer only.
YOLO_TO_DB_CATEGORY = {
    "shirt":            "shirts",
    "cap":              "caps",
    "glass":            "glasses",
    "earring":          "earrings",
    "jacket":           "Jacket",
    "footwear_shoes":   "Footwear_shoes",
    "footwear_sandals": "Footwear_sandals",
    "footwear_heels":   "Footwear_heels",
    "footwear_flats":   "Footwear_flats",
    # Categories that already match DB exactly:
    # tshirt, blazer, shorts, pant, skirt, churidhar, dhoti,
    # necklace, belt, tie, watch, bag, etc.
}

# Footwear categories (for special deduplication)
FOOTWEAR_CATEGORIES = {
    "footwear_shoes", "footwear_sandals", "footwear_heels", "footwear_flats"
}

# Upper wear categories (for body part confusion filtering)
UPPER_CATEGORIES = {"shirt", "tshirt", "blouse", "blazer", "jacket", "shawl"}

# Lower wear categories
LOWER_CATEGORIES = {"pant", "shorts", "skirt", "leggings"}


# =============================================================================
# YOLO DETECTOR CLASS
# =============================================================================

class YoloDetector:
    def __init__(self, model_paths, conf_thresh=0.35):
       
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.models = []
        self.conf_thresh = conf_thresh
        self._stored_paths = model_paths  # Store for potential reload

        # Load models
        for path in model_paths:
            try:
                model = YOLO(path)
                self.models.append(model)
                print(f"[YOLO] [OK] Model loaded: {path}")
            except Exception as e:
                print(f"[YOLO] [ERR] Failed to load model {path}: {e}")

        if not self.models:
            raise RuntimeError("[YOLO] No valid models loaded")
    
    # -------------------------------------------------------------------------
    # Memory Management - Unload and Reload
    # -------------------------------------------------------------------------
    
    def unload(self):
        """
        NO-OP: Models stay loaded permanently for fast inference.
        Previously unloaded to free RAM for local LLM, but LLM now uses
        Gemini API (external), so keeping YOLO loaded is safe and eliminates
        the 8-24s reload penalty per request.
        """
        pass  # Intentional no-op
    
    def ensure_loaded(self):
        """
        Safety check: if models were somehow lost, reload them.
        Under normal operation this is a no-op since unload() is disabled.
        """
        if not self.models and hasattr(self, '_stored_paths'):
            print("[YOLO] Models missing — reloading...")
            for path in self._stored_paths:
                try:
                    model = YOLO(path)
                    self.models.append(model)
                    print(f"[YOLO] [OK] Model reloaded: {path}")
                except Exception as e:
                    print(f"[YOLO] [ERR] Failed to reload model {path}: {e}")
        return bool(self.models)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def cv2_to_base64(self, img):
        """Convert OpenCV image to base64 string."""
        try:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode()
        except Exception:
            return None

    def compute_iou(self, a, b):
        """Compute Intersection over Union between two boxes."""
        xA = max(a["x1"], b["x1"])
        yA = max(a["y1"], b["y1"])
        xB = min(a["x2"], b["x2"])
        yB = min(a["y2"], b["y2"])

        if xB <= xA or yB <= yA:
            return 0.0

        inter = (xB - xA) * (yB - yA)
        areaA = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        areaB = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

        return inter / float(areaA + areaB - inter + 1e-6)

    def is_valid_box_ratio(self, x1, y1, x2, y2, category):
        """
        Check if bounding box has valid aspect ratio for the category.
        Helps filter out body parts mistaken for fashion items.
        """
        w = x2 - x1
        h = y2 - y1
        
        if w < 20 or h < 20:
            return False
            
        aspect = w / h if h > 0 else 1
        
        # Footwear should be wider than tall
        if category in FOOTWEAR_CATEGORIES:
            if aspect < 0.5:  # Too tall for shoes
                return False
        
        # Upper wear typically has aspect ratio 0.5-2.0
        if category in UPPER_CATEGORIES:
            if aspect < 0.3 or aspect > 3.0:
                return False
        
        return True

    # -------------------------------------------------------------------------
    # Main Inference
    # -------------------------------------------------------------------------

    def infer(self, frame):
        """
        Run inference on frame using all loaded models.
        Returns deduplicated list of detections.
        """
        raw = []
        frame_h, frame_w = frame.shape[:2]

        for model_idx, model in enumerate(self.models):
            try:
                # Run YOLO inference
                results = model(frame, verbose=False)
                res = results[0]
            except Exception as e:
                print(f"[YOLO] [ERR] Inference failed on model {model_idx}: {e}")
                continue

            if not hasattr(res, "boxes") or res.boxes is None:
                continue

            for box in res.boxes:
                try:
                    conf = float(box.conf[0])
                    if conf < self.conf_thresh:
                        continue

                    cls_idx = int(box.cls[0])
                    cls_name = model.names.get(cls_idx, "").lower().strip()

                    # Drop unwanted classes
                    if cls_name in DROP_CLASSES:
                        continue

                    # Map to canonical name
                    canonical = CANONICAL_MAP.get(cls_name)
                    if not canonical:
                        # Unknown class - skip
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Clamp to frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_w, x2)
                    y2 = min(frame_h, y2)
                    
                    # Validate box ratio
                    if not self.is_valid_box_ratio(x1, y1, x2, y2, canonical):
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_b64 = self.cv2_to_base64(crop)
                    if crop_b64 is None:
                        continue

                    # Map YOLO canonical → DB category name
                    db_category = YOLO_TO_DB_CATEGORY.get(canonical, canonical)

                    raw.append({
                        "class": db_category,
                        "conf": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cropped_image": crop_b64,
                        "model_idx": model_idx,
                    })

                except Exception as e:
                    print(f"⚠️ Skipped invalid detection: {e}")
                    continue

        return self._dedupe_and_merge(raw)

    # -------------------------------------------------------------------------
    # Deduplication / Merging
    # -------------------------------------------------------------------------

    def _dedupe_and_merge(self, detections):
        """
        Remove duplicate detections using IoU-based NMS.
        Handles both same-class and cross-class duplicates.
        """
        if not detections:
            return []
            
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x["conf"], reverse=True)
        final = []

        for det in sorted_dets:
            keep = True

            for existing in final:
                iou = self.compute_iou(det, existing)

                # Same class - stricter threshold
                if det["class"] == existing["class"]:
                    if iou >= 0.5:
                        keep = False
                        break
                
                # Cross-class overlap (different classes, same location)
                # This catches body parts being detected as different items
                else:
                    if iou >= 0.7:
                        # Keep the one with higher confidence (already in final)
                        keep = False
                        break
                    
                    # Special case: footwear overlap
                    if det["class"] in FOOTWEAR_CATEGORIES and existing["class"] in FOOTWEAR_CATEGORIES:
                        if iou >= 0.4:
                            keep = False
                            break

            if keep:
                # Remove model_idx before returning (internal use only)
                clean_det = {k: v for k, v in det.items() if k != "model_idx"}
                final.append(clean_det)

        return final
