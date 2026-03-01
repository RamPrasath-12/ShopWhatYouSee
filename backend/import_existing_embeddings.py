"""
Import existing AGMAN embeddings from .npy files
"""
import numpy as np
import psycopg2
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # ShopWhatYouSee/
AGMAN_DIR = os.path.join(BASE_DIR, "agman_output")

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

def import_embeddings():
    print(f"Checking directory: {AGMAN_DIR}")
    
    try:
        # Load files
        print("Loading numpy files...")
        labels_path = os.path.join(AGMAN_DIR, "labels.npy")
        emb_path = os.path.join(AGMAN_DIR, "agman_refined_embeddings.npy")
        
        if not os.path.exists(labels_path) or not os.path.exists(emb_path):
            print(f"Error: Files not found in {AGMAN_DIR}")
            print(f"Looking for: {labels_path}")
            return

        labels = np.load(labels_path, allow_pickle=True)
        print(f"Labels shape: {labels.shape}")
        
        files = ["agman_refined_embeddings.npy", "refined_embeddings.npy", "base_embeddings.npy"]
        for f in files:
            p = os.path.join(AGMAN_DIR, f)
            if os.path.exists(p):
                try:
                    e = np.load(p)
                    print(f"File: {f}, Shape: {e.shape}")
                    if len(e) == len(labels):
                        print(f"MATCH FOUND! Using {f}")
                        embeddings = e
                        break
                except Exception as ex:
                    print(f"Error reading {f}: {ex}")
        
        if 'embeddings' not in locals():
            print("No matching embedding file found (length mismatch).")
            # Fallback: import partial if any
            if os.path.exists(emb_path):
                print("Importing PARTIAL `agman_refined_embeddings.npy` (first N items)...")
                embeddings = np.load(emb_path)
            else:
                return

        # Connect DB
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        updated = 0
        
        print("Updating database...")
        for i, product_id in enumerate(labels):
            emb_list = embeddings[i].tolist()
            emb_json = json.dumps(emb_list)
            
            # Ensure product_id is string
            pid_str = str(product_id).replace(".jpg", "")
            
            cur.execute("UPDATE products SET embedding = %s WHERE product_id = %s", (emb_json, pid_str))
            
            updated += 1
            if updated % 1000 == 0:
                conn.commit()
                print(f"Progress: {updated}/{len(labels)}")
        
        conn.commit()
        print(f"✅ Success! Updated {updated} products with existing AGMAN embeddings.")
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    import_embeddings()
