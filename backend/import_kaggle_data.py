"""
Import AGMAN attributes and embeddings from Kaggle output files
Run this after downloading the Kaggle outputs
"""
import os
import csv
import json
import psycopg2
import numpy as np

# Files from Kaggle - UPDATE THESE PATHS!
CSV_PATH = "agman_attributes.csv"  # Downloaded from Kaggle
NPZ_PATH = "agman_embeddings.npz"  # Downloaded from Kaggle

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

def ensure_columns(conn):
    """Add AGMAN columns if they don't exist"""
    columns = [
        ("embedding", "TEXT"),
        ("primary_color", "VARCHAR(50)"),
        ("secondary_color", "VARCHAR(50)"),
        ("agman_pattern", "VARCHAR(50)"),
        ("agman_sleeve", "VARCHAR(50)")
    ]
    for col_name, col_type in columns:
        try:
            cur = conn.cursor()
            cur.execute(f"ALTER TABLE products ADD COLUMN {col_name} {col_type}")
            conn.commit()
            print(f"Added column: {col_name}")
            cur.close()
        except:
            conn.rollback()

def import_data():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Ensure columns exist
    ensure_columns(conn)
    
    cur = conn.cursor()
    
    # Load embeddings if file exists
    embeddings = {}
    if os.path.exists(NPZ_PATH):
        print(f"Loading embeddings from {NPZ_PATH}...")
        data = np.load(NPZ_PATH)
        for key in data.files:
            embeddings[key] = data[key].tolist()
        print(f"Loaded {len(embeddings)} embeddings")
    
    # Import CSV
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        return
    
    print(f"Importing from {CSV_PATH}...")
    updated = 0
    errors = 0
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                product_id = row['product_id']
                
                # Get embedding if available
                emb_json = None
                if product_id in embeddings:
                    emb_json = json.dumps(embeddings[product_id])
                
                cur.execute("""
                    UPDATE products SET
                        primary_color = %s,
                        secondary_color = %s,
                        agman_pattern = %s,
                        agman_sleeve = %s,
                        embedding = COALESCE(%s, embedding)
                    WHERE product_id = %s
                """, (
                    row.get('primary_color'),
                    row.get('secondary_color'),
                    row.get('agman_pattern'),
                    row.get('agman_sleeve'),
                    emb_json,
                    product_id
                ))
                updated += 1
                
                if updated % 1000 == 0:
                    conn.commit()
                    print(f"Progress: {updated}...")
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"Error: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✅ Import Complete!")
    print(f"   Updated: {updated}")
    print(f"   Errors: {errors}")

if __name__ == "__main__":
    import_data()
