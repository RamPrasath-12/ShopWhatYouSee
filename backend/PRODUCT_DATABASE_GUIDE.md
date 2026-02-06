# ============================================================
# PRODUCT DATABASE BUILD GUIDE
# ============================================================

## OVERVIEW
This guide covers building the product database with AGMAN embeddings
for product similarity search.

## FILES CHECKED AND VERIFIED ✅

1. ✅ `backend/build_product_database.py` - Main database builder script
2. ✅ `backend/models/agman_extractor.py` - Processes images → embeddings + attributes
3. ✅ `backend/models/agman_loader.py` - Loads fine-tuned AGMAN model (87.4% Recall@5)
4. ✅ `backend/models/agman_model.py` - AGMAN model architecture
5. ✅ `models/agman_model_best.pth` - Fine-tuned AGMAN weights (20MB)
6. ✅ `data/product_database.csv` - Product metadata (34,215 products)
7. ✅ `data/images/` - Product images (34,209 images)

## PIPELINE OVERVIEW

```
CSV File → Load Product → Load Image → AGMAN Extractor → SQLite DB
                                           ↓
                                    1. ResNet (2048D)
                                    2. AGMAN Model (512D) ← Fine-tuned!
                                    3. Extract Attributes
                                           ↓
                                    Returns:
                                    - embedding: [512D vector]
                                    - attributes:
                                      • primary_color
                                      • secondary_color
                                      • pattern
                                      • sleeve
```

## DATABASE SCHEMA

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT UNIQUE NOT NULL,
    product_name TEXT,
    brand TEXT,
    price REAL,
    yolo_category TEXT,
    
    -- AGMAN Extracted Attributes
    primary_color TEXT,
    secondary_color TEXT,
    pattern TEXT,
    sleeve TEXT,
    
    -- Embedding (512D) stored as JSON
    embedding TEXT,
    
    -- Metadata
    image_path TEXT NOT NULL,
    processed_at TIMESTAMP,
    detection_confidence REAL,
    processing_status TEXT DEFAULT 'pending'
);
```

## STEP-BY-STEP EXECUTION

### Step 1: Activate Virtual Environment
```powershell
cd D:\Final_Year_Project\ShopWhatYouSee\backend
.\venv\Scripts\Activate
```

### Step 2: Test with Small Sample First (RECOMMENDED)
```powershell
# Test with first 100 products
python build_product_database.py --limit 100

# Check the log file
cat build_database.log | Select-Object -Last 20
```

### Step 3: Check Results
```powershell
# Install sqlite3 tool if needed
# Or use Python to query:
python -c "import sqlite3; conn = sqlite3.connect('../data/products.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM products WHERE processing_status = \"completed\"'); print(f'Total products: {cursor.fetchone()[0]}')"
```

### Step 4: Run Full Database Build
```powershell
# Process all ~34k products (this will take time!)
python build_product_database.py

# To resume if interrupted:
python build_product_database.py --resume

# Custom paths:
python build_product_database.py \
    --csv ../data/product_database.csv \
    --db ../data/products.db \
    --data-dir ../data \
    --batch-size 100
```

## COMMAND LINE OPTIONS

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `../data/product_database.csv` | Path to CSV file |
| `--db` | `../data/products.db` | Output database path |
| `--data-dir` | `../data` | Directory with images/ folder |
| `--batch-size` | `100` | Commit every N products |
| `--limit` | `None` | Limit number of products (for testing) |
| `--no-resume` | `False` | Start fresh (ignore existing) |

## EXPECTED PERFORMANCE

- **Processing Speed**: ~5-10 products/second (depends on GPU)
- **Total Time for 34k products**: ~1-2 hours
- **Database Size**: ~500MB (with embeddings)

## WHAT GETS EXTRACTED

For each product image, AGMAN extracts:

1. **Embedding** (512D vector) - For similarity search
2. **Primary Color** - Hex code (e.g., `#FF5733`)
3. **Secondary Color** - Hex code or `null`
4. **Pattern** - Only for fabric items:
   - `solid`, `striped`, `checked`, `patterned`
5. **Sleeve Length** - Only for upper wear:
   - `long`, `half`, `short`, `sleeveless`

## VERIFICATION STEPS

After running, check:

```powershell
# 1. Check total processed
sqlite3 ../data/products.db "SELECT COUNT(*) FROM products WHERE processing_status='completed';"

# 2. Check category distribution
sqlite3 ../data/products.db "SELECT yolo_category, COUNT(*) FROM products GROUP BY yolo_category ORDER BY COUNT(*) DESC LIMIT 10;"

# 3. Sample a product
sqlite3 ../data/products.db "SELECT product_id, product_name, primary_color, pattern, sleeve FROM products LIMIT 5;"

# 4. Check embedding exists
sqlite3 ../data/products.db "SELECT product_id, LENGTH(embedding) as emb_length FROM products WHERE embedding IS NOT NULL LIMIT 5;"
```

## TROUBLESHOOTING

### Issue: "Image not found"
- Check that `data/images/` contains the images
- CSV has paths like `dataset_images/15970.jpg` but images are in `data/images/15970.jpg`
- The script handles this automatically

### Issue: "Failed to extract attributes"
- Check AGMAN model loaded: `models/agman_model_best.pth`
- Ensure PIL can open the image
- Check GPU/CPU availability

### Issue: "Out of memory"
- Reduce `--batch-size` to 50 or less
- Process in chunks with `--limit`

### Issue: Slow processing
- Enable GPU: Check `torch.cuda.is_available()`
- Reduce image count for testing
- Close other GPU-intensive apps

## NEXT STEPS AFTER DATABASE BUILD

1. **Build FAISS Index** - For fast similarity search
2. **Integrate with API** - `/api/search-similar`
3. **Test Retrieval** - Query with test images
4. **Deploy** - Production database

## FILES CREATED

After successful run:
- `../data/products.db` - SQLite database with embeddings
- `build_database.log` - Processing log with errors/warnings

## QUALITY CHECKS

The AGMAN model used has been verified to have:
- ✅ **Recall@5**: 87.4%
- ✅ **Recall@10**: 92.8%
- ✅ **Recall@20**: 95.1%

This means for product similarity:
- 87% of the time, the correct similar product is in top 5 results
- 93% of the time, it's in top 10 results
