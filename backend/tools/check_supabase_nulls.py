import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_config import DATABASE_URL
import psycopg2

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

columns_to_sync = [
    'primary_color_hex', 'rating_count', 'color_confidence', 'product_url', 
    'secondary_color_name', 'secondary_color_hex', 'extraction_quality', 'size', 
    'secondary_confidence', 'rating', 'image_path', 'discount_pct', 
    'pattern_confidence', 'sleeve_confidence'
]

print("Checking null rows in Supabase for each column:")
for col in columns_to_sync:
    try:
        cur.execute(f"SELECT COUNT(*) FROM visual_attributes WHERE {col} IS NULL")
        missing = cur.fetchone()[0]
        print(f"{col}: {missing} missing rows")
    except Exception as e:
        print(f"{col}: ERROR - {e}")
        conn.rollback()

try:
    cur.execute("SELECT COUNT(*) FROM ratings")
    print(f"ratings table: {cur.fetchone()[0]} rows")
except Exception as e:
    print(f"ratings table ERROR: {e}")
    conn.rollback()

conn.close()
