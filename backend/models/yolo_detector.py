# backend/models/yolo_detector.py

import cv2
import base64
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def cv2_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def infer(self, frame):
        results = self.model(frame)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # Crop the detected item
            crop = frame[y1:y2, x1:x2]

            # Convert crop to base64
            cropped_b64 = self.cv2_to_base64(crop)

            det = {
                "class": cls_name,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cropped_image": cropped_b64
            }

            detections.append(det)

        return detections
