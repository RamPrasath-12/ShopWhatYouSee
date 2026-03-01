import faiss
import json
import numpy as np
import sqlite3
import argparse

DB_PATH = "../data/products.db"
INDEX_PATH = "../data/products.faiss"
ID_MAP_PATH = "../data/product_ids.json"

def test_retrieval():
    print("Loading resources...")
    
    # 1. Load FAISS index
    index = faiss.read_index(INDEX_PATH)
    print(f"FAISS Index loaded: {index.ntotal} vectors")
    
    # 2. Load ID map
    with open(ID_MAP_PATH, 'r') as f:
        # Convert keys to int because JSON loads keys as strings
        id_map = {int(k): v for k, v in json.load(f).items()}
    print(f"ID Map loaded: {len(id_map)} entries")
    
    # 3. Get a random product from DB to query with
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT product_id, product_name, embedding FROM products WHERE processing_status='completed' AND embedding IS NOT NULL ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    
    if not row:
        print("Error: No products found in DB.")
        return

    query_pid, query_name, query_emb_json = row
    query_emb = np.array([json.loads(query_emb_json)], dtype=np.float32)
    faiss.normalize_L2(query_emb)
    
    print("\n" + "="*60)
    print(f"QUERY PRODUCT: {query_pid}")
    print(f"Name: {query_name}")
    print("="*60)
    
    # 4. Search
    k = 5
    distances, indices = index.search(query_emb, k)
    
    print(f"\nTop {k} Results:")
    print("-" * 60)
    
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            print(f"{rank+1}. Empty result")
            continue
            
        retrieved_pid = id_map.get(idx, "Unknown")
        
        # Get details from DB
        c.execute("SELECT product_name, yolo_category, primary_color, pattern FROM products WHERE product_id=?", (retrieved_pid,))
        details = c.fetchone()
        
        if details:
            r_name, r_cat, r_color, r_pattern = details
            match_status = "✅ MATCH" if retrieved_pid == query_pid else "Similarity"
            print(f"{rank+1}. [Score: {dist:.4f}] {retrieved_pid} - {r_name[:40]}... ({match_status})")
            print(f"   Category: {r_cat}, Color: {r_color}, Pattern: {r_pattern}")
        else:
            print(f"{rank+1}. [Score: {dist:.4f}] {retrieved_pid} (Details not found in DB)")
            
    conn.close()

if __name__ == "__main__":
    test_retrieval()
