"""
FAST PostgreSQL Population - CSV Only (No Embeddings)
Populates products table with all metadata from CSV.
Embeddings can be added as a separate step later.
"""
import psycopg2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL Connection
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

CSV_PATH = Path("../data/product_database.csv")

def create_table(conn):
    """Drop and create products table."""
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE IF EXISTS products CASCADE;
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            product_id INTEGER UNIQUE,
            product_name TEXT,
            brand TEXT,
            price NUMERIC,
            discounted_price NUMERIC,
            discount_percent NUMERIC,
            rating NUMERIC,
            base_colour TEXT,
            colour1 TEXT,
            colour2 TEXT,
            master_category TEXT,
            sub_category TEXT,
            article_type TEXT,
            yolo_category TEXT,
            gender TEXT,
            age_group TEXT,
            usage TEXT,
            season TEXT,
            year INTEGER,
            fashion_type TEXT,
            product_descriptors TEXT,
            article_attributes TEXT,
            style_type TEXT,
            catalog_add_date TEXT,
            landing_page_url TEXT,
            image_url TEXT,
            embedding BYTEA,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX idx_products_category ON products(yolo_category);
        CREATE INDEX idx_products_colour ON products(base_colour);
        CREATE INDEX idx_products_brand ON products(brand);
        CREATE INDEX idx_products_article ON products(article_type);
    """)
    conn.commit()
    logger.info("✅ Table 'products' created!")

def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except:
        return default

def safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except:
        return default

def safe_str(val, max_len=None):
    if pd.isna(val):
        return ''
    s = str(val)
    if max_len:
        return s[:max_len]
    return s

def populate_database():
    """Main population function."""
    logger.info(f"📂 Loading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    logger.info(f"📊 Found {total} products")
    
    logger.info("🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info("✅ Connected!")
    
    create_table(conn)
    
    cur = conn.cursor()
    success = 0
    failed = 0
    
    insert_sql = """
        INSERT INTO products (
            product_id, product_name, brand, price, discounted_price,
            discount_percent, rating, base_colour, colour1, colour2,
            master_category, sub_category, article_type, yolo_category,
            gender, age_group, usage, season, year, fashion_type,
            product_descriptors, article_attributes, style_type,
            catalog_add_date, landing_page_url, image_url
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    batch = []
    batch_size = 500
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="Inserting"):
        try:
            product_id = int(row['product_id'])
            values = (
                product_id,
                safe_str(row.get('product_name', ''), 500),
                safe_str(row.get('brand', '')),
                safe_float(row.get('price')),
                safe_float(row.get('discounted_price')),
                safe_float(row.get('discount_percent')),
                safe_float(row.get('rating')),
                safe_str(row.get('baseColour', '')),
                safe_str(row.get('colour1', '')),
                safe_str(row.get('colour2', '')),
                safe_str(row.get('masterCategory', '')),
                safe_str(row.get('subCategory', '')),
                safe_str(row.get('articleType', '')),
                safe_str(row.get('yolo_category', '')),
                safe_str(row.get('gender', '')),
                safe_str(row.get('ageGroup', '')),
                safe_str(row.get('usage', '')),
                safe_str(row.get('season', '')),
                safe_int(row.get('year')),
                safe_str(row.get('fashionType', '')),
                safe_str(row.get('productDescriptors', ''), 1000),
                safe_str(row.get('articleAttributes', ''), 1000),
                safe_str(row.get('styleType', '')),
                safe_str(row.get('catalogAddDate', '')),
                safe_str(row.get('landingPageUrl', ''), 500),
                f"/static/images/{product_id}.jpg"
            )
            batch.append(values)
            
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                conn.commit()
                success += len(batch)
                batch = []
                
        except Exception as e:
            logger.warning(f"Row {idx} failed: {e}")
            failed += 1
    
    # Insert remaining
    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        success += len(batch)
    
    cur.close()
    conn.close()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ COMPLETE: {success} inserted, {failed} failed")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    populate_database()
