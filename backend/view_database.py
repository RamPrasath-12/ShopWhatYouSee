"""
View Product Database - Quick inspection tool
"""

import sqlite3
import json
import sys

DB_PATH = "../data/products.db"

def view_stats():
    """Show database statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("=" * 60)
    print("PRODUCT DATABASE STATISTICS")
    print("=" * 60)
    
    # Total products
    c.execute("SELECT COUNT(*) FROM products")
    total = c.fetchone()[0]
    print(f"\nTotal products in database: {total}")
    
    # Completed
    c.execute("SELECT COUNT(*) FROM products WHERE processing_status='completed'")
    completed = c.fetchone()[0]
    print(f"Completed products: {completed}")
    print(f"Progress: {completed/total*100:.1f}%")
    
    # Category distribution
    print("\n" + "-" * 60)
    print("TOP 10 CATEGORIES:")
    print("-" * 60)
    c.execute("""
        SELECT yolo_category, COUNT(*) as count 
        FROM products 
        WHERE processing_status='completed'
        GROUP BY yolo_category 
        ORDER BY count DESC 
        LIMIT 10
    """)
    for cat, count in c.fetchall():
        print(f"  {cat:20s}: {count:5d}")
    
    conn.close()

def view_samples(limit=10):
    """Show sample products with attributes"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("\n" + "=" * 60)
    print(f"SAMPLE PRODUCTS (showing {limit})")
    print("=" * 60)
    
    c.execute(f"""
        SELECT product_id, product_name, brand, yolo_category,
               primary_color, secondary_color, pattern, sleeve,
               LENGTH(embedding) as emb_len
        FROM products 
        WHERE processing_status='completed'
        LIMIT {limit}
    """)
    
    for row in c.fetchall():
        pid, name, brand, cat, p_color, s_color, pattern, sleeve, emb_len = row
        print(f"\n[Product {pid}]")
        print(f"  Name: {name[:50]}")
        print(f"  Brand: {brand}")
        print(f"  Category: {cat}")
        print(f"  Primary Color: {p_color}")
        print(f"  Secondary Color: {s_color}")
        print(f"  Pattern: {pattern}")
        print(f"  Sleeve: {sleeve}")
        print(f"  Embedding: {emb_len} bytes (512D vector)")
    
    conn.close()

def view_embedding_sample():
    """Show a sample embedding vector"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("\n" + "=" * 60)
    print("SAMPLE EMBEDDING VECTOR")
    print("=" * 60)
    
    c.execute("""
        SELECT product_id, product_name, embedding 
        FROM products 
        WHERE embedding IS NOT NULL 
        LIMIT 1
    """)
    
    row = c.fetchone()
    if row:
        pid, name, embedding_json = row
        embedding = json.loads(embedding_json)
        
        print(f"\nProduct: {pid} - {name[:40]}")
        print(f"Embedding dimension: {len(embedding)}D")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Embedding type: {type(embedding)}")
        print(f"Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")
    else:
        print("No embeddings found yet (processing may still be running)")
    
    conn.close()

if __name__ == "__main__":
    try:
        view_stats()
        view_samples(limit=5)
        view_embedding_sample()
        
        print("\n" + "=" * 60)
        print("Database location: ../data/products.db")
        print("=" * 60)
        
    except sqlite3.OperationalError as e:
        print(f"Error: Could not open database. Make sure the build has started.")
        print(f"Database path: {DB_PATH}")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
