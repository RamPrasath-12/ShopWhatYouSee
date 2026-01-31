# backend/models/yolo_detector.py

import cv2
import base64
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_paths, conf_thresh=0.4):
        """
        Initialize ensemble detector with multiple YOLO models.
        
        Args:
            model_paths: List of paths to YOLO model weights, or single path string
            conf_thresh: Confidence threshold for detections
        """
        # Support both single model and ensemble
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        
        self.models = []
        self.conf_thresh = conf_thresh
        
        print(f"🔄 Loading {len(model_paths)} YOLO model(s) for ensemble detection...")
        for i, path in enumerate(model_paths):
            try:
                model = YOLO(path)
                self.models.append(model)
                print(f"✅ Model {i+1}/{len(model_paths)}: {path}")
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
        
        if not self.models:
            raise ValueError("No YOLO models loaded successfully!")
        
        print(f"✅ Ensemble detector ready with {len(self.models)} model(s)\n")

    def cv2_to_base64(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    def compute_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1['x1'], box1['y1'], box1['x2'], box1['y2']
        x2_min, y2_min, x2_max, y2_max = box2['x1'], box2['y1'], box2['x2'], box2['y2']
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def remove_duplicates(self, detections, iou_threshold=0.5):
        """
        Remove duplicate detections using NMS.
        Keep the detection with highest confidence when boxes overlap significantly.
        More aggressive settings to handle ensemble duplicates.
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while detections:
            # Take the highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections that overlap significantly with the best one
            remaining = []
            for det in detections:
                iou = self.compute_iou(best, det)
                
                # Suppress if:
                # 1. Same class and IoU > 0.3 (very aggressive for same class)
                # 2. High overlap (>0.5) regardless of class (likely same object)
                should_suppress = False
                if det['class'].lower() == best['class'].lower() and iou >= 0.3:
                    should_suppress = True
                elif iou >= 0.5:  # Cross-class suppression for high overlap
                    should_suppress = True
                
                if not should_suppress:
                    remaining.append(det)
            
            detections = remaining
        
        return keep

    def infer(self, frame):
        """
        Run inference using all models and combine results.
        Removes duplicates using NMS.
        """
        all_detections = []
        
        # Run inference on each model
        for model_idx, model in enumerate(self.models):
            try:
                results = model(frame)[0]
                
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_thresh:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    # Filter out dress category
                    if cls_name.lower() == "dress":
                        continue
                    
                    # Rename hat to cap
                    if cls_name.lower() == "hat":
                        cls_name = "cap"

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    cropped_b64 = self.cv2_to_base64(crop)

                    all_detections.append({
                        "class": cls_name,
                        "conf": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cropped_image": cropped_b64,
                        "model_idx": model_idx
                    })
            except Exception as e:
                print(f"⚠️  Model {model_idx} inference failed: {e}")
                continue
        
        # Remove duplicates using NMS (aggressive threshold for better deduplication)
        unique_detections = self.remove_duplicates(all_detections, iou_threshold=0.3)
        
        print(f"📊 Ensemble: {len(all_detections)} total → {len(unique_detections)} after NMS")
        
        return unique_detections
