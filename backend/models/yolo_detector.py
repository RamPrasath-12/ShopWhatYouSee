# backend/models/yolo_detector.py

import cv2
import base64
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights_path, conf_thresh=0.4):
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh

    def cv2_to_base64(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    def infer(self, frame):
        # YOLOv8 handles internal preprocessing
        results = self.model(frame)[0]

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cropped_b64 = self.cv2_to_base64(crop)

            detections.append({
                "class": cls_name,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cropped_image": cropped_b64
            })
        
    

        return detections
