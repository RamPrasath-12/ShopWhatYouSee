"""
Product Database Builder Script

This script processes product images through YOLO detection and AGMAN attribute extraction,
then stores the results in a SQLite database for the product retrieval pipeline.

Pipeline:
1. Load product metadata from CSV
2. For each product:
   - Load image from local path
   - Run YOLO detection to crop fashion items
   - Pass crops to AGMAN extractor for attributes and embeddings
   - Store results in SQLite database
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import logging
from datetime import datetime
from tqdm import tqdm
import argparse

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Make YOLO optional (not needed for product database since products are already cropped)
try:
    from models.yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLODetector = None
    YOLO_AVAILABLE = False

from models.agman_extractor import process_crop_base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Database schema
CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT UNIQUE NOT NULL,
    product_name TEXT,
    brand TEXT,
    price REAL,
    yolo_category TEXT,
    original_colors TEXT,
    image_path TEXT NOT NULL,
    
    -- Extracted attributes from AGMAN
    primary_color TEXT,
    secondary_color TEXT,
    pattern TEXT,
    sleeve TEXT,
    
    -- Embedding stored as JSON array
    embedding TEXT,
    
    -- Metadata
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_confidence REAL,
    processing_status TEXT DEFAULT 'pending'
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_products_category ON products(yolo_category);
CREATE INDEX IF NOT EXISTS idx_products_color ON products(primary_color);
CREATE INDEX IF NOT EXISTS idx_products_pattern ON products(pattern);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
"""


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(CREATE_PRODUCTS_TABLE)
    cursor.executescript(CREATE_INDEX)
    conn.commit()
    logger.info(f"Database initialized at {db_path}")
    return conn


def pil_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def process_single_product(
    row: pd.Series,
    yolo_detector: YOLODetector,
    data_dir: Path,
    conn: sqlite3.Connection
) -> bool:
    """
    Process a single product through the pipeline.
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    product_id = str(row['product_id'])
    
    try:
        # Check if already processed
        cursor = conn.cursor()
        cursor.execute(
            "SELECT processing_status FROM products WHERE product_id = ?",
            (product_id,)
        )
        existing = cursor.fetchone()
        if existing and existing[0] == 'completed':
            return True  # Already processed
        
        # Build image path
        # The CSV has paths like 'dataset_images/10000.jpg' relative to the data dir
        image_rel_path = row.get('image_path', '')
        if not image_rel_path:
            # Fallback: try to construct from product_id
            image_rel_path = f"{product_id}.jpg"
        
        # Handle both 'dataset_images/xxx.jpg' and 'xxx.jpg' formats
        if 'dataset_images' in image_rel_path:
            # Extract just the filename from 'dataset_images/xxx.jpg'
            image_filename = os.path.basename(image_rel_path)
            image_path = data_dir / 'images' / image_filename
        else:
            image_path = data_dir / 'images' / image_rel_path
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return False
        
        # Load image
        pil_img = Image.open(image_path).convert('RGB')
        
        # Prepare data for YOLO
        img_b64 = pil_to_base64(pil_img)
        
        # Run YOLO detection
        yolo_category = row.get('yolo_category', 'unknown')
        
        # For product database, we might skip YOLO detection if we already have the category
        # and the image is already a product shot (not a scene with multiple items)
        # In this case, we process the full image as if it were a crop
        
        # Run AGMAN extraction on the full image
        # (product images are typically already cropped to the item)
        attributes_result = process_crop_base64(img_b64, yolo_category)
        
        if not attributes_result:
            logger.warning(f"Failed to extract attributes for product {product_id}")
            return False
        
        attributes = attributes_result.get('attributes', {})
        embedding = attributes_result.get('embedding', [])
        
        # Convert embedding to JSON string for storage
        embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
        
        # Prepare database record
        record = {
            'product_id': product_id,
            'product_name': row.get('product_name', ''),
            'brand': row.get('brand', ''),
            'price': float(row.get('price', 0)) if pd.notna(row.get('price')) else None,
            'yolo_category': yolo_category,
            'original_colors': str(row.get('colors', '')),
            'image_path': str(image_path),
            'primary_color': attributes.get('color_hex'),
            'secondary_color': attributes.get('secondary_color_hex'),
            'pattern': attributes.get('pattern'),
            'sleeve': attributes.get('sleeve'),
            'embedding': embedding_json,
            'detection_confidence': 1.0,  # Full confidence for product images
            'processing_status': 'completed'
        }
        
        # Insert or update in database
        cursor.execute("""
            INSERT OR REPLACE INTO products 
            (product_id, product_name, brand, price, yolo_category, original_colors,
             image_path, primary_color, secondary_color, pattern, sleeve, 
             embedding, detection_confidence, processing_status, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record['product_id'],
            record['product_name'],
            record['brand'],
            record['price'],
            record['yolo_category'],
            record['original_colors'],
            record['image_path'],
            record['primary_color'],
            record['secondary_color'],
            record['pattern'],
            record['sleeve'],
            record['embedding'],
            record['detection_confidence'],
            record['processing_status'],
            datetime.now().isoformat()
        ))
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing product {product_id}: {e}")
        return False


def build_database(
    csv_path: str,
    db_path: str,
    data_dir: str,
    batch_size: int = 100,
    limit: int = None,
    resume: bool = True
):
    """
    Main function to build the product database.
    
    Args:
        csv_path: Path to the product_database.csv file
        db_path: Path for the output SQLite database
        data_dir: Directory containing the data (with images/ subdirectory)
        batch_size: Number of products to process before committing
        limit: Optional limit on number of products to process
        resume: Whether to skip already-processed products
    """
    logger.info("Starting database build...")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Data directory: {data_dir}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    total_products = len(df)
    logger.info(f"Loaded {total_products} products from CSV")
    
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit} products")
    
    # Initialize database
    conn = init_database(db_path)
    
    # Initialize YOLO detector (for potential future use with scene images)
    yolo_detector = None
    if YOLO_AVAILABLE:
        try:
            yolo_detector = YOLODetector()
            logger.info("YOLO detector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize YOLO detector: {e}")
    else:
        logger.info("YOLO not available - processing as product images directly")
    
    data_dir = Path(data_dir)
    
    # Process products
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing products"):
        try:
            success = process_single_product(row, yolo_detector, data_dir, conn)
            
            if success:
                success_count += 1
            else:
                error_count += 1
            
            # Commit periodically
            if (idx + 1) % batch_size == 0:
                conn.commit()
                logger.info(f"Progress: {idx + 1}/{len(df)} products processed")
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            error_count += 1
    
    # Final commit
    conn.commit()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Database build complete!")
    logger.info(f"Total products: {len(df)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Skipped (already done): {skip_count}")
    
    # Get database stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products WHERE processing_status = 'completed'")
    completed = cursor.fetchone()[0]
    logger.info(f"Total products in database: {completed}")
    
    # Category distribution
    cursor.execute("""
        SELECT yolo_category, COUNT(*) as count 
        FROM products 
        GROUP BY yolo_category 
        ORDER BY count DESC 
        LIMIT 10
    """)
    logger.info("\nTop 10 categories:")
    for cat, count in cursor.fetchall():
        logger.info(f"  {cat}: {count}")
    
    conn.close()
    logger.info(f"Database saved to: {db_path}")


def main():
    parser = argparse.ArgumentParser(description='Build product database with extracted attributes')
    parser.add_argument('--csv', type=str, 
                        default='../data/product_database.csv',
                        help='Path to product CSV file')
    parser.add_argument('--db', type=str,
                        default='../data/products.db',
                        help='Path for output SQLite database')
    parser.add_argument('--data-dir', type=str,
                        default='../data',
                        help='Directory containing image data')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for commits')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of products to process')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = script_dir / csv_path
    
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = script_dir / db_path
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    
    build_database(
        csv_path=str(csv_path),
        db_path=str(db_path),
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        limit=args.limit,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
