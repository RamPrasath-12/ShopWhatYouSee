import psycopg2
import json
import numpy as np
import faiss
import os
import time
import argparse

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}
INDEX_PATH = "../data/products.faiss"
ID_MAP_PATH = "../data/product_ids.json"
EMBEDDING_DIM = 2048 # ResNet50 dim (was 512, default to 2048 but will auto-detect)

def build_index(index_path=INDEX_PATH, id_map_path=ID_MAP_PATH):
    """
    Build FAISS index from PostgreSQL database
    """
    print(f"Connecting to PostgreSQL...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    c = conn.cursor()
    
    # Check for available embeddings
    c.execute("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")
    total_count = c.fetchone()[0]
    
    if total_count == 0:
        print("Error: No embeddings found in database. Run backfill_embeddings.py first.")
        return
        
    print(f"Found {total_count} embeddings to index.")
    
    # Fetch all embeddings and IDs
    c.execute("SELECT id, product_id, embedding FROM products WHERE embedding IS NOT NULL")
    
    embeddings = []
    id_map = {} # Maps FAISS index (int) -> Product ID
    
    start_time = time.time()
    
    # Chunked fetch
    BATCH_SIZE = 5000
    processed = 0
    
    current_dim = EMBEDDING_DIM
    
    while True:
        rows = c.fetchmany(BATCH_SIZE)
        if not rows:
            break
            
        for row in rows:
            db_id, product_id, emb_bytes = row
            try:
                # Decode BYTEA -> JSON string -> List
                emb_str = bytes(emb_bytes).decode('utf-8') 
                emb = json.loads(emb_str)
                
                # Check dim
                if len(embeddings) == 0:
                    current_dim = len(emb)
                    print(f"Detected embedding dimension: {current_dim}")
                
                if len(emb) != current_dim:
                    # Skip mismatch
                    continue
                    
                embeddings.append(emb)
                
                # Store mapping
                current_idx = len(embeddings) - 1
                id_map[current_idx] = product_id
                
            except Exception as e:
                # print(f"Error parsing embedding for {product_id}: {e}")
                pass
                
        processed += len(rows)
        print(f"Loaded {processed}/{total_count} ({processed/total_count*100:.1f}%)", end='\r')
        
    print(f"\nLoaded {len(embeddings)} valid embeddings in {time.time() - start_time:.2f}s")
    
    if not embeddings:
        print("No valid embeddings loaded.")
        return

    # Convert to numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(embeddings_np)
    
    print(f"Building FAISS index (dim={current_dim})...")
    index = faiss.IndexFlatIP(current_dim)
    index.add(embeddings_np)
    
    print(f"Index built with {index.ntotal} vectors.")
    
    # Save Index
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Save ID mapping
    print(f"Saving ID map to {id_map_path}...")
    with open(id_map_path, 'w') as f:
        json.dump(id_map, f)
        
    print("Done! FAISS index is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from PostgreSQL database")
    parser.add_argument("--out", default=INDEX_PATH, help="Output FAISS index path")
    
    args = parser.parse_args()
    build_index(args.out)
