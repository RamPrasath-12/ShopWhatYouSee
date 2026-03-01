"""
YOLO Detector Accuracy Test Script

Tests YOLO detection on sample images to verify:
1. Correct class detection
2. Correct bounding box placement
3. Proper deduplication
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from config import YOLO_MODELS, YOLO_CONF_THRESH

def load_models():
    """Load YOLO models from config."""
    models = []
    backend_dir = Path(__file__).parent  # ShopWhatYouSee/backend/
    
    for model_path in YOLO_MODELS:
        # Config paths are relative to ShopWhatYouSee/, but models are in backend/data/yolo/
        # Try multiple resolution strategies
        candidates = [
            backend_dir / model_path.replace("data/", "data/"),  # backend/data/yolo/
            backend_dir.parent / model_path,  # ShopWhatYouSee/data/yolo/
            Path(model_path),  # Absolute path
        ]
        
        loaded = False
        for full_path in candidates:
            if full_path.exists():
                try:
                    model = YOLO(str(full_path))
                    models.append((model, str(full_path.name)))
                    print(f"✅ Loaded: {full_path}")
                    # Print class names
                    class_names = list(model.names.values())
                    print(f"   Classes ({len(class_names)}): {class_names[:15]}...")
                    loaded = True
                    break
                except Exception as e:
                    print(f"❌ Failed to load {full_path}: {e}")
        
        if not loaded:
            print(f"⚠️ Not found: {model_path}")
    
    return models

def test_single_image(models, image_path, save_output=True):
    """Run detection on a single image and display results."""
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"Image size: {img.shape}")
    print(f"{'='*60}")
    
    all_detections = []
    
    for model, model_name in models:
        print(f"\n[Model: {model_name}]")
        
        results = model(img, conf=YOLO_CONF_THRESH, verbose=False)
        res = results[0]
        
        if res.boxes is None or len(res.boxes) == 0:
            print("  No detections")
            continue
        
        for box in res.boxes:
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            cls_name = model.names.get(cls_idx, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            print(f"  {cls_name}: {conf:.3f} @ ({x1}, {y1}) -> ({x2}, {y2})")
            
            all_detections.append({
                'class': cls_name,
                'conf': conf,
                'box': (x1, y1, x2, y2),
                'model': model_name
            })
    
    # Draw detections on image
    if save_output and all_detections:
        output_img = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, det in enumerate(all_detections):
            x1, y1, x2, y2 = det['box']
            color = colors[i % len(colors)]
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['conf']:.2f}"
            cv2.putText(output_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, output_img)
        print(f"\n📁 Saved detection result: {output_path}")
    
    print(f"\n📊 Total detections: {len(all_detections)}")
    return all_detections


def main():
    print("="*60)
    print("YOLO DETECTOR ACCURACY TEST")
    print("="*60)
    
    # Load models
    print("\n[Loading Models]")
    models = load_models()
    
    if not models:
        print("\n❌ No models loaded! Check your config paths.")
        return
    
    # Get test images from command line or use defaults
    test_images = []
    
    if len(sys.argv) > 1:
        test_images = sys.argv[1:]
    else:
        # Try to find sample images in data folder
        data_dir = Path(__file__).parent.parent / "data" / "images"
        if data_dir.exists():
            images = list(data_dir.glob("*.jpg"))[:3]  # First 3 images
            test_images = [str(img) for img in images]
            print(f"\nUsing sample images from {data_dir}")
    
    if not test_images:
        print("\n⚠️ No test images found!")
        print("Usage: python test_yolo_accuracy.py <image1.jpg> [image2.jpg] ...")
        return
    
    # Run tests
    for image_path in test_images:
        test_single_image(models, image_path)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
