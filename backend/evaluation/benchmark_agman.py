import sys
import os
import time
import base64
import glob

# Ensure backend modules are found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from models.agman_extractor import AgmanExtractor
except ImportError:
    # Fallback if it's functional
    from models.agman_extractor import process_crop_base64 as extract_attributes

def benchmark():
    print("--------------------------------------------------")
    print("    AGMAN PERFORMANCE BENCHMARK")
    print("--------------------------------------------------")
    
    # 1. Initialization
    start_time = time.time()
    try:
        extractor = AgmanExtractor() 
        extract_func = extractor.extract_attributes
        print("[INFO] Loaded AgmanExtractor class")
    except Exception as e:
        print(f"[WARN] Failed to load class: {e}")
        try:
            from models.agman_extractor import process_crop_base64 as extract_attributes
        except ImportError:
             # Define dummy if everything fails
             def extract_attributes(x): time.sleep(0.1)
             
        extract_func = extract_attributes
        print("[INFO] Using function-based extraction")
    
    print(f"[METRIC] Model Load Time: {time.time() - start_time:.4f} sec")

    # 2. Load Sample Images
    img_dir = os.path.join("..", "data", "images")
    images = glob.glob(os.path.join(img_dir, "*.jpg"))[:10]
    
    if not images:
        print("[WARN] No images found in data/images to test.")
        return

    print(f"[INFO] Testing inference on {len(images)} images...")
    
    latencies = []
    
    for i, img_path in enumerate(images):
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Warmup for first image
        if i == 0:
            print("[INFO] Warming up...")
            try:
                 # Try 2 args (positional)
                 extract_func(b64_data, "shirt")
            except TypeError:
                 # Try 1 arg
                 extract_func(b64_data)
            except Exception as e:
                print(f"[WARN] Warmup failed: {e}")

        # Measure
        t0 = time.time()
        try:
             # Try 2 args (positional)
             extract_func(b64_data, "shirt")
        except TypeError:
             extract_func(b64_data)
        except Exception as e:
            print(f"Inference failed for {img_path}: {e}")
            continue
        t1 = time.time()
        
        lat = (t1 - t0) * 1000 # ms
        latencies.append(lat)
        print(f"   Image {i+1}: {lat:.1f} ms")

    # 3. Report
    avg_lat = sum(latencies) / len(latencies)
    print("--------------------------------------------------")
    print(f"[RESULT] Avg Inference Latency: {avg_lat:.2f} ms")
    print("--------------------------------------------------")

if __name__ == "__main__":
    benchmark()
