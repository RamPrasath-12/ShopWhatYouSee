"""
Fast PostgreSQL Database Population Script
Populates products table with:
- All metadata from product_database.csv
- Image local links
- AGMAN attributes & embeddings (extracted on-the-fly)
"""
import psycopg2
import pandas as pd
import numpy as np
import torch
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL Connection
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

# Paths
CSV_PATH = Path("../data/product_database.csv")
IMAGES_DIR = Path("../data/images")
AGMAN_WEIGHTS = Path("models/agman_model_best.pth")

# AGMAN Model (lazy load)
agman_model = None
agman_transform = None

def load_agman():
    """Load AGMAN model for embedding extraction."""
    global agman_model, agman_transform
    if agman_model is not None:
        return
    
    from models.agman_model import AGMAN
    from torchvision import transforms
    
    logger.info("Loading AGMAN model...")
    agman_model = AGMAN()
    
    # Load weights with strict=False (ignore classifier)
    checkpoint = torch.load(str(AGMAN_WEIGHTS), map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    agman_model.load_state_dict(state_dict, strict=False)
    agman_model.eval()
    
    if torch.cuda.is_available():
        agman_model = agman_model.cuda()
    
    agman_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logger.info("AGMAN model loaded!")

def get_embedding(image_path: Path) -> bytes:
    """Extract 512-dim embedding from image."""
    global agman_model, agman_transform
    
    if agman_model is None:
        load_agman()
    
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = agman_transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        with torch.no_grad():
            embedding = agman_model.backbone(tensor)
            embedding = agman_model.attention(embedding)
            embedding = embedding.view(embedding.size(0), -1)
            embedding = agman_model.fc(embedding)
            embedding = embedding.cpu().numpy().flatten()
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32).tobytes()
    except Exception as e:
        logger.warning(f"Embedding failed for {image_path}: {e}")
        return np.zeros(512, dtype=np.float32).tobytes()

def create_table(conn):
    """Create products table with all fields."""
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
    """)
    conn.commit()
    logger.info("Table 'products' created!")

def populate_database():
    """Main function to populate PostgreSQL."""
    # Load CSV
    logger.info(f"Loading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    logger.info(f"Found {total} products")
    
    # Connect to PostgreSQL
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Create table
    create_table(conn)
    
    # Insert with embeddings
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
            catalog_add_date, landing_page_url, image_url, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    batch_size = 100
    batch = []
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing"):
        try:
            product_id = int(row['product_id'])
            image_path = IMAGES_DIR / f"{product_id}.jpg"
            
            # Get embedding if image exists
            if image_path.exists():
                embedding = get_embedding(image_path)
            else:
                embedding = np.zeros(512, dtype=np.float32).tobytes()
            
            # Prepare row
            values = (
                product_id,
                str(row.get('product_name', ''))[:500],
                str(row.get('brand', '')),
                float(row['price']) if pd.notna(row.get('price')) else 0,
                float(row['discounted_price']) if pd.notna(row.get('discounted_price')) else 0,
                float(row['discount_percent']) if pd.notna(row.get('discount_percent')) else 0,
                float(row['rating']) if pd.notna(row.get('rating')) else 0,
                str(row.get('baseColour', '')),
                str(row.get('colour1', '')),
                str(row.get('colour2', '')),
                str(row.get('masterCategory', '')),
                str(row.get('subCategory', '')),
                str(row.get('articleType', '')),
                str(row.get('yolo_category', '')),
                str(row.get('gender', '')),
                str(row.get('ageGroup', '')),
                str(row.get('usage', '')),
                str(row.get('season', '')),
                int(row['year']) if pd.notna(row.get('year')) else None,
                str(row.get('fashionType', '')),
                str(row.get('productDescriptors', ''))[:1000],
                str(row.get('articleAttributes', ''))[:1000],
                str(row.get('styleType', '')),
                str(row.get('catalogAddDate', '')),
                str(row.get('landingPageUrl', ''))[:500],
                f"/static/images/{product_id}.jpg",
                psycopg2.Binary(embedding)
            )
            batch.append(values)
            
            # Batch insert
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
