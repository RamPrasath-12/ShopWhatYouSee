import psycopg2
import json
import time
import os

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

def check():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Count total vs processed
        cur.execute("SELECT COUNT(*) FROM products")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM products WHERE primary_color IS NOT NULL AND embedding IS NOT NULL")
        processed = cur.fetchone()[0]
        
        # Check integrity of a sample
        cur.execute("""
            SELECT product_id, primary_color, agman_pattern, agman_sleeve, embedding 
            FROM products 
            WHERE primary_color IS NOT NULL 
            ORDER BY RANDOM()
            LIMIT 1
        """)
        row = cur.fetchone()
        
        # Clear screen roughly
        print("\n" * 3)
        print("-" * 50)
        print(f"STATUS REPORT")
        print("-" * 50)
        print(f"Total Products: {total}")
        print(f"Processed:      {processed}")
        print(f"Remaining:      {total - processed}")
        print(f"Progress:       {processed/total*100:.2f}%")
        
        if row:
            pid, color, pat, slv, emb_data = row
            
            # Handle potential memoryview from psycopg2
            if isinstance(emb_data, memoryview):
                emb_str = emb_data.tobytes().decode('utf-8')
            else:
                emb_str = emb_data
                
            try:
                emb = json.loads(emb_str)
                dim = len(emb)
            except:
                print(f"Error parsing JSON for {pid}. Data length: {len(str(emb_str))}")
                dim = 0
            
            print("-" * 50)
            print(f"SAMPLE DATA (ID: {pid})")
            print("-" * 50)
            print(f"Color:      {color}")
            print(f"Pattern:    {pat}")
            print(f"Sleeve:     {slv}")
            # print(f"Embedding:  {dim} dimensions (Target: 512)")
            
            if dim == 512:
                print(f"✅ INTEGRITY CHECK PASSED: Embedding size {dim} (Finetuned AGMAN Model)")
            elif dim == 2048:
                print(f"⚠️ WARNING: Embedding size {dim} (ResNet50 Base - Not Finetuned)")
            else:
                print(f"⚠️ WARNING: Unexpected dimension {dim}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Monitoring... (Ctrl+C to stop)")
    while True:
        check()
        time.sleep(5)
