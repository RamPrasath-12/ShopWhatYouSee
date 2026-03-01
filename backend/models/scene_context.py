# backend/models/scene_context.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import json
import os

class SceneContextDetector:
    def __init__(self, model_path="data/scene/places365.pth.tar"):
        # Load class labels
        classes_path = os.path.join("models", "places365_classes.json")
        with open(classes_path, "r") as f:
            self.classes = json.load(f)

        # Load ResNet18 WITHOUT pretrained fc layer
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

        # Replace fc layer to match Places365 (365 classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 365)

        # Load Places365 checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Preprocessing transforms
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def infer(self, img_b64):
        """
        Perform scene prediction from a base64 image
        """
        # Decode Base64 image
        header, data = img_b64.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

        # Preprocess
        tensor = self.tf(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        label = self.classes[idx]
        confidence = float(probs[idx])

        return {
            "scene_label": label,
            "confidence": confidence,
            "places_available": True
        }

# ------------------------------
# Module-level wrapper function
# ------------------------------
# So you can import directly in app.py: from models.scene_context import infer
scene_detector = SceneContextDetector()

def infer(img_b64):
    return scene_detector.infer(img_b64)
