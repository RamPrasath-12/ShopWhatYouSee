"""
Backfill AGMAN Embeddings & Attributes into PostgreSQL
Iterates over products with missing embeddings and populates them using agman_extractor.
"""
import psycopg2
import base64
import os
import sys
import json
import logging
from tqdm import tqdm
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.agman_extractor import process_crop_base64

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}
IMAGES_DIR = Path("../data/images")

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def backfill():
    conn = get_db()
    cur = conn.cursor()
    
    # Check total to process
    cur.execute("SELECT COUNT(*) FROM products WHERE embedding IS NULL OR base_colour = ''")
    total = cur.fetchone()[0]
    logger.info(f"Found {total} products needing AGMAN processing")
    
    # Fetch batch
    cur.execute("SELECT product_id, yolo_category FROM products WHERE embedding IS NULL OR base_colour = ''")
    rows = cur.fetchall()
    
    success = 0
    failed = 0
    
    for product_id, category in tqdm(rows, desc="Processing"):
        image_path = IMAGES_DIR / f"{product_id}.jpg"
        
        if not image_path.exists():
            continue
            
        try:
            # Read image -> Base64 (needed for extractor API)
            with open(image_path, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                
            # Run AGMAN Extractor
            result = process_crop_base64(b64_data, category)
            
            attrs = result.get("attributes", {})
            embedding = result.get("embedding", [])
            
            # Map attributes to DB columns
            embedding_bytes = psycopg2.Binary(json.dumps(embedding).encode('utf-8')) # Store as JSON string in BYTEA or just JSONB? 
            # Original populate script used BYTEA for raw bytes. 
            # But process_crop_base64 returns list. 
            # FAISS builder expects JSON list. 
            # Let's store as JSON string encoded to bytes for BYTEA column, 
            # OR better: The original schema has BYTEA. 
            # BUT product_retrieval doesn't use it. build_faiss_index uses it.
            # build_faiss_index expects JSON: `emb = json.loads(emb_json)` from fetchone.
            # So if column is BYTEA, we should store bytes of json string.
            
            emb_json_bytes = psycopg2.Binary(json.dumps(embedding).encode('utf-8'))
            
            base_colour = attrs.get("color_hex")
            colour1 = attrs.get("secondary_color_hex")
            pattern = attrs.get("pattern")
            
            # Update DB
            update_sql = """
                UPDATE products 
                SET embedding = %s,
                    base_colour = COALESCE(NULLIF(base_colour, ''), %s),
                    colour1 = %s,
                    pattern = %s,
                    article_attributes = %s
                WHERE product_id = %s
            """
            
            cur.execute(update_sql, (
                emb_json_bytes, 
                base_colour, 
                colour1, 
                pattern, 
                json.dumps(attrs), 
                product_id
            ))
            
            success += 1
            if success % 100 == 0:
                conn.commit()
                
        except Exception as e:
            # logger.error(f"Failed {product_id}: {e}")
            failed += 1
            
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info(f"Complete! Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    backfill()
