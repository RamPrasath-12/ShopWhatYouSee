import requests
import base64
import json
import os
import cv2
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_IMAGE_PATH = "" # Will be set dynamically

def get_test_image():
    """Get a valid image path from the database"""
    import sqlite3
    conn = sqlite3.connect('../data/products.db')
    c = conn.cursor()
    c.execute("SELECT image_path FROM products WHERE processing_status='completed' ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        b64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_data}"

def print_step(step, content):
    print("\n" + "="*60)
    print(f"STEP {step}")
    print("="*60)
    print(content)

def verify_pipeline():
    print("🚀 STARTING PIPELINE VERIFICATION")
    
    # 1. Get Test Image
    img_rel_path = get_test_image()
    if not img_rel_path:
        print("❌ Error: No processed images found in database.")
        return
        
    # Handle path differences (DB has relative path or full path?)
    # DB has: "data/images/123.jpg" or "dataset_images/123.jpg"
    # Actual file is in "d:/Final_Year_Project/ShopWhatYouSee/data/images/"
    # Let's construct full path
    
    # Assuming standard structure
    basename = os.path.basename(img_rel_path)
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images', basename)
    
    if not os.path.exists(full_path):
        print(f"❌ Error: Image file not found at {full_path}")
        # Try to find it
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images', basename)
    
    print_step(1, f"Test Image: {full_path}")
    b64_image = encode_image(full_path)
    print(f"Base64 encoded (len={len(b64_image)})")

    # 2. Call /detect (YOLO)
    print_step(2, "Sending to /detect (YOLO)...")
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/detect", json={"image": b64_image})
        print(f"Time: {time.time()-start:.2f}s")
        
        if resp.status_code != 200:
            print(f"❌ Error: {resp.text}")
            return
            
        detections = resp.json().get("detections", [])
        print(f"Detections Found: {len(detections)}")
        for i, d in enumerate(detections):
            box = f"[{d.get('x1')}, {d.get('y1')}, {d.get('x2')}, {d.get('y2')}]"
            print(f"  [{i}] {d['class']} (conf: {d['conf']:.2f}) - Box: {box}")
            
        if not detections:
            print("⚠️ No detections found. Using default full image for next step.")
            selected_box = None
            detected_category = "clothing"
        else:
            # Pick first detection
            selected_box = detections[0]
            detected_category = selected_box['class']
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure app.py is running!")
        return

    # 3. Call /extract-attributes (AGMAN)
    print_step(3, f"Sending to /extract-attributes (AGMAN)...")
    # Note: simulate cropping on frontend? Or send full image?
    # app.py expects payload base64. process_crop_base64 handles it.
    # Usually frontend crops. Here I'll send the full image for simplicity,
    # assuming process_crop_base64 handles resizing.
    
    try:
        start = time.time()
        payload = {
            "image": b64_image, # Ideally crop this to detection box
            "category": detected_category
        }
        resp = requests.post(f"{BASE_URL}/extract-attributes", json=payload)
        print(f"Time: {time.time()-start:.2f}s")
        
        if resp.status_code != 200:
            print(f"❌ Error: {resp.text}")
            return
            
        agman_result = resp.json()
        embeddings = agman_result.get("embedding")
        attrs = agman_result.get("attributes", {})
        
        print(f"Attributes Extracted:")
        print(json.dumps(attrs, indent=2))
        print(f"Embedding extracted: {len(embeddings) if embeddings else 0} dimensions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 4. Call /search (Product Retrieval)
    print_step(4, f"Sending to /search (Recommendation)...")
    
    # Construct filters from extracted attributes
    filters = {
        "category": detected_category,
        "color": attrs.get("color_hex"),
        "pattern": attrs.get("pattern"),
        "sleeve": attrs.get("sleeve")
    }
    
    # Clean None values
    filters = {k: v for k, v in filters.items() if v}
    
    print(f"Search Query Filters: {json.dumps(filters, indent=2)}")
    
    try:
        start = time.time()
        payload = {
            "filters": filters,
            "detected_category": detected_category
        }
        resp = requests.post(f"{BASE_URL}/search", json=payload)
        print(f"Time: {time.time()-start:.2f}s")
        
        if resp.status_code != 200:
            print(f"❌ Error: {resp.text}")
            return
            
        results = resp.json().get("products", [])
        print(f"\nRecommended Products ({len(results)}):")
        
        for i, p in enumerate(results[:5]):
            print(f"  [{i+1}] {p['name']} - ${p['price']} ({p['brand']})")
            print(f"      Image: {p['image_url']}")
            
        print("\n✅ PIPELINE VERIFIED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_pipeline()
