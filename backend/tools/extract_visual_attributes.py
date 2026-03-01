"""
Visual Attribute Extraction Pipeline
=====================================
Extracts structured visual attributes from Myntra scraped images using
crop-only preprocessing + AG-MAN, validates, and stores in local PostgreSQL.

Architecture:
  CSV metadata + Image → crop-only preprocessing → AG-MAN extract
    → strict schema mapping → validation → local PostgreSQL
    → (later) Supabase push

Corrections applied:
  1. Embedding as REAL[] (not JSONB)
  2. Rating as REAL, rating_count as INTEGER (parse "2.1k" → 2100)
  3. Indexes on category + product_id
  4. product_id = md5(product_url) for deterministic, re-runnable IDs
  5. Expanded validation: isfinite, hex regex, category whitelist
  6. Skip missing/corrupted images
  7. Batch commit every 100 rows, single DB connection
  8. extraction_quality < 0.3 → skip, < 0.5 → warn
  9. --dry-run mode (extract + validate, no DB insert)

Usage:
  python tools/extract_visual_attributes.py --categories shirts --n 5
  python tools/extract_visual_attributes.py --dry-run --n 10
  python tools/extract_visual_attributes.py --n 0   # all images, all categories
"""

import os
import sys
import io
import re
import csv
import json
import math
import time
import hashlib
import base64
import logging
import argparse
from datetime import datetime

import numpy as np
import psycopg2
from PIL import Image

# ===================================================================
# SETUP
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extractor")

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BACKEND_DIR)

# ===================================================================
# CONFIGURATION
# ===================================================================
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

IMAGE_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase\myntra"
CSV_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase"
BATCH_COMMIT_SIZE = 100
QUIET_MODE = False  # Set via --quiet flag; suppresses AG-MAN verbose prints

# ===================================================================
# CATEGORY DEFINITIONS (STRICT WHITELIST)
# ===================================================================
ALLOWED_CATEGORIES = {
    "shirts", "tshirt", "blazer", "Jacket", "pant", "shorts",
    "skirt", "churidhar", "dhoti",
    "Footwear_sandals", "Footwear_shoes",
    "caps", "glasses",
    "belt", "tie",
    "earrings", "necklace", "watch",
}

# AG-MAN category name mapping (folder → AG-MAN)
FOLDER_TO_AGMAN = {
    "shirts": "shirt", "tshirt": "tshirt", "blazer": "blazer",
    "Jacket": "jacket", "pant": "pant", "shorts": "shorts",
    "skirt": "skirt", "churidhar": "churidhar", "dhoti": "dhoti",
    "Footwear_sandals": "footwear_sandals", "Footwear_shoes": "footwear_shoes",
    "caps": "cap", "glasses": "glass",
    "belt": "belt", "tie": "tie",
    "earrings": "earring", "necklace": "necklace", "watch": "watch",
}

# Upper-wear categories (sleeve extraction applies ONLY to these)
UPPER_WEAR = {"shirts", "tshirt", "blazer", "Jacket"}

# Fabric categories (pattern extraction applies ONLY to these)
FABRIC_CATEGORIES = {
    "shirts", "tshirt", "blazer", "Jacket", "pant", "shorts",
    "skirt", "churidhar", "dhoti",
}

# Crop rules (same as preprocessing_pipeline.py)
CROP_RULES = {
    "shirts": (0.08, 0.08), "tshirt": (0.08, 0.08), "blazer": (0.08, 0.08),
    "Jacket": (0.08, 0.08), "pant": (0.08, 0.08), "shorts": (0.08, 0.08),
    "skirt": (0.08, 0.08), "churidhar": (0.08, 0.08), "dhoti": (0.08, 0.08),
    "Footwear_sandals": (0.18, 0.08), "Footwear_shoes": (0.18, 0.08),
    "caps": (0.05, 0.22), "glasses": (0.05, 0.22),
    "belt": (0.13, 0.13), "tie": (0.13, 0.13),
    "earrings": (0.08, 0.06), "necklace": (0.08, 0.06), "watch": (0.08, 0.06),
}

HEX_REGEX = re.compile(r"^#[A-Fa-f0-9]{6}$")


# ===================================================================
# DATABASE
# ===================================================================

def create_table(conn):
    """Create visual_attributes table with proper types and indexes."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS visual_attributes (
            id                  SERIAL PRIMARY KEY,
            product_id          VARCHAR(100) NOT NULL UNIQUE,
            category            VARCHAR(50) NOT NULL,
            brand               TEXT,
            product_name        TEXT,
            size                TEXT,
            rating              REAL,
            rating_count        INTEGER,
            discounted_price    INTEGER,
            original_price      INTEGER,
            discount_pct        TEXT,
            product_url         TEXT,
            image_url           TEXT,
            image_path          TEXT,
            -- Visual attributes (strict schema)
            embedding           REAL[] NOT NULL,
            primary_color_name  VARCHAR(50),
            primary_color_hex   VARCHAR(10),
            color_confidence    REAL,
            secondary_color_name VARCHAR(50),
            secondary_color_hex VARCHAR(10),
            secondary_confidence REAL,
            pattern_value       VARCHAR(50),
            pattern_confidence  REAL,
            sleeve_value        VARCHAR(50) NOT NULL,
            sleeve_confidence   REAL NOT NULL,
            extraction_quality  REAL NOT NULL,
            created_at          TIMESTAMP DEFAULT NOW()
        );
    """)

    # Indexes for query performance
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_visual_category
        ON visual_attributes(category);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_visual_product_id
        ON visual_attributes(product_id);
    """)

    conn.commit()
    cur.close()
    log.info("Table 'visual_attributes' ready (with indexes)")


def insert_record(conn, record):
    """Insert a validated record into visual_attributes. Returns True on success."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO visual_attributes (
                product_id, category, brand, product_name, size,
                rating, rating_count, discounted_price, original_price,
                discount_pct, product_url, image_url, image_path,
                embedding,
                primary_color_name, primary_color_hex, color_confidence,
                secondary_color_name, secondary_color_hex, secondary_confidence,
                pattern_value, pattern_confidence,
                sleeve_value, sleeve_confidence,
                extraction_quality
            ) VALUES (
                %(product_id)s, %(category)s, %(brand)s, %(product_name)s, %(size)s,
                %(rating)s, %(rating_count)s, %(discounted_price)s, %(original_price)s,
                %(discount_pct)s, %(product_url)s, %(image_url)s, %(image_path)s,
                %(embedding)s,
                %(primary_color_name)s, %(primary_color_hex)s, %(color_confidence)s,
                %(secondary_color_name)s, %(secondary_color_hex)s, %(secondary_confidence)s,
                %(pattern_value)s, %(pattern_confidence)s,
                %(sleeve_value)s, %(sleeve_confidence)s,
                %(extraction_quality)s
            )
            ON CONFLICT (product_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                primary_color_name = EXCLUDED.primary_color_name,
                primary_color_hex = EXCLUDED.primary_color_hex,
                color_confidence = EXCLUDED.color_confidence,
                secondary_color_name = EXCLUDED.secondary_color_name,
                secondary_color_hex = EXCLUDED.secondary_color_hex,
                secondary_confidence = EXCLUDED.secondary_confidence,
                pattern_value = EXCLUDED.pattern_value,
                pattern_confidence = EXCLUDED.pattern_confidence,
                sleeve_value = EXCLUDED.sleeve_value,
                sleeve_confidence = EXCLUDED.sleeve_confidence,
                extraction_quality = EXCLUDED.extraction_quality,
                created_at = NOW()
        """, record)
        return True
    except Exception as e:
        conn.rollback()
        log.error(f"DB insert failed for {record.get('product_id')}: {e}")
        return False
    finally:
        cur.close()


# ===================================================================
# CSV PARSING
# ===================================================================

def parse_price(price_text):
    """Parse price string like 'Rs. 1049' → 1049."""
    if not isinstance(price_text, str):
        return None
    cleaned = re.sub(r"[^\d]", "", price_text)
    return int(cleaned) if cleaned else None


def parse_rating(rating_text):
    """Parse rating string → float. Returns None on failure."""
    if not isinstance(rating_text, str) or not rating_text.strip():
        return None
    try:
        return float(rating_text.strip())
    except ValueError:
        return None


def parse_rating_count(count_text):
    """Parse rating count like '2.1k' → 2100, '146' → 146."""
    if not isinstance(count_text, str) or not count_text.strip():
        return None
    text = count_text.strip().lower()
    try:
        if text.endswith("k"):
            return int(float(text[:-1]) * 1000)
        return int(text)
    except (ValueError, TypeError):
        return None


def generate_product_id(category, product_url, index):
    """
    Generate deterministic product_id from product_url hash.
    Falls back to category_index if URL is missing.
    """
    if isinstance(product_url, str) and product_url.strip():
        url_hash = hashlib.md5(product_url.strip().encode("utf-8")).hexdigest()[:12]
        return f"{category}_{url_hash}"
    # Fallback: use category + index (less ideal but stable per CSV)
    return f"{category}_{index}"


def load_csv_metadata(csv_path, category, n=0):
    """
    Load product metadata from CSV.
    Returns list of dicts with parsed fields.
    n=0 means all rows.
    """
    if not os.path.exists(csv_path):
        log.warning(f"CSV not found: {csv_path}")
        return []

    records = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if n > 0 and i >= n:
                break

            product_url = row.get("product-base href", "")
            product_id = generate_product_id(category, product_url, i)

            records.append({
                "index": i,
                "product_id": product_id,
                "category": category,
                "brand": row.get("product-brand", "").strip() or None,
                "product_name": row.get("product-product", "").strip() or None,
                "size": row.get("product-sizeInventoryPresent", "").strip() or None,
                "rating": parse_rating(row.get("product-ratingsContainer", "")),
                "rating_count": parse_rating_count(row.get("product-ratingsCount", "")),
                "discounted_price": parse_price(row.get("product-discountedPrice", "")),
                "original_price": parse_price(row.get("product-strike", "")),
                "discount_pct": row.get("product-discountPercentage", "").strip() or None,
                "product_url": product_url.strip() or None,
                "image_url": row.get("img-responsive src", "").strip() or None,
            })

    return records


# ===================================================================
# PREPROCESSING (crop-only, from preprocessing_pipeline.py)
# ===================================================================

def crop_image(pil_img, category):
    """Apply category-aware vertical crop. Returns cropped PIL image."""
    crop_top, crop_bottom = CROP_RULES.get(category, (0.0, 0.0))
    w, h = pil_img.size
    y_top = int(h * crop_top)
    y_bot = int(h * (1.0 - crop_bottom))

    # Safety: keep at least 50%
    if y_bot - y_top < h * 0.5:
        y_top = int(h * 0.05)
        y_bot = int(h * 0.95)

    return pil_img.crop((0, y_top, w, y_bot))


def pil_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===================================================================
# ATTRIBUTE EXTRACTION (STRICT SCHEMA)
# ===================================================================

def extract_structured(image_path, category):
    """
    Full extraction: load → crop → AG-MAN → strict schema.

    Returns structured dict or None if image is missing/corrupted.
    """
    from models.agman_extractor import process_crop_base64

    # --- Guard: image must exist ---
    if not os.path.exists(image_path):
        log.warning(f"Image missing: {image_path}")
        return None

    # --- Load image ---
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"Corrupted image: {image_path}: {e}")
        return None

    # --- Crop-only preprocessing ---
    cropped = crop_image(pil_img, category)
    b64 = pil_to_base64(cropped)

    # --- AG-MAN extraction ---
    agman_category = FOLDER_TO_AGMAN.get(category, category.lower())
    try:
        if QUIET_MODE:
            # Suppress AG-MAN's verbose print statements
            # Must use utf-8 encoding on Windows to handle emoji characters
            _old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')
            try:
                result = process_crop_base64(b64, agman_category)
            finally:
                sys.stdout.close()
                sys.stdout = _old_stdout
        else:
            result = process_crop_base64(b64, agman_category)
    except Exception as e:
        log.error(f"AG-MAN failed: {image_path}: {e}")
        return None

    embedding = result.get("embedding")
    attrs = result.get("attributes", {})

    if not embedding:
        log.error(f"No embedding returned: {image_path}")
        return None

    # --- Convert embedding to native Python list of floats ---
    embedding_list = [float(v) for v in embedding]

    # --- Strict color mapping ---
    color_struct = attrs.get("color", {})
    primary_name = None
    primary_hex = None
    color_confidence = 0.0

    if isinstance(color_struct, dict):
        primary_name = color_struct.get("value")
        primary_hex = color_struct.get("hex")
        color_confidence = float(color_struct.get("confidence", 0.0))
    else:
        # Fallback to legacy flat fields
        primary_name = attrs.get("color_name")
        primary_hex = attrs.get("color_hex")
        color_confidence = 0.5  # unknown confidence

    secondary_color = attrs.get("secondary_color")
    sec_name = None
    sec_hex = None
    sec_confidence = 0.0

    if isinstance(secondary_color, dict):
        sec_name = secondary_color.get("value")
        sec_hex = secondary_color.get("hex")
        # Clamp: AG-MAN can produce secondary_confidence > 1.0
        sec_confidence = min(1.0, max(0.0, float(secondary_color.get("confidence", 0.0))))

    # --- Strict pattern mapping ---
    pattern_value = None
    pattern_confidence = 0.0

    if category in FABRIC_CATEGORIES:
        pat_struct = attrs.get("pattern_structured")
        if isinstance(pat_struct, dict):
            pattern_value = pat_struct.get("value")
            pattern_confidence = float(pat_struct.get("confidence", 0.0))
        else:
            # Legacy flat field
            pattern_value = attrs.get("pattern")
            pattern_confidence = 0.5

    # --- Strict sleeve mapping ---
    if category in UPPER_WEAR:
        sleeve_struct = attrs.get("sleeve_structured")
        if isinstance(sleeve_struct, dict):
            sleeve_value = sleeve_struct.get("value") or "unknown"
            sleeve_confidence = float(sleeve_struct.get("confidence", 0.0))
        else:
            sleeve_value = attrs.get("sleeve") or "unknown"
            sleeve_confidence = 0.5
    else:
        # NOT upper-wear: do NOT attempt sleeve detection
        sleeve_value = "not_applicable"
        sleeve_confidence = 1.0

    # --- Extraction quality ---
    extraction_quality = float(attrs.get("extraction_quality", 0.0))

    return {
        "embedding": embedding_list,
        "primary_color_name": primary_name,
        "primary_color_hex": primary_hex,
        "color_confidence": round(color_confidence, 4),
        "secondary_color_name": sec_name,
        "secondary_color_hex": sec_hex,
        "secondary_confidence": round(sec_confidence, 4),
        "pattern_value": pattern_value,
        "pattern_confidence": round(pattern_confidence, 4),
        "sleeve_value": sleeve_value,
        "sleeve_confidence": round(sleeve_confidence, 4),
        "extraction_quality": round(extraction_quality, 4),
    }


# ===================================================================
# VALIDATION
# ===================================================================

def validate_record(record):
    """
    Validate a record before DB insertion.
    Returns (is_valid, error_message).
    """
    pid = record.get("product_id", "?")

    # --- Category whitelist ---
    if record.get("category") not in ALLOWED_CATEGORIES:
        return False, f"[{pid}] Invalid category: {record.get('category')}"

    emb = record.get("embedding")

    # --- Embedding length ---
    if not isinstance(emb, list) or len(emb) != 512:
        return False, f"[{pid}] Embedding length: {len(emb) if isinstance(emb, list) else 'None'}"

    # --- All values finite (no NaN, no Inf) ---
    for i, v in enumerate(emb):
        if not math.isfinite(v):
            return False, f"[{pid}] Embedding[{i}] is not finite: {v}"

    # --- Embedding norm ≈ 1.0 ---
    norm = math.sqrt(sum(v * v for v in emb))
    if not (0.98 <= norm <= 1.02):
        return False, f"[{pid}] Embedding norm out of range: {norm:.6f}"

    # --- Confidence bounds [0.0, 1.0] ---
    for field in ["color_confidence", "secondary_confidence",
                   "pattern_confidence", "sleeve_confidence"]:
        val = record.get(field)
        if val is not None and not (0.0 <= val <= 1.0):
            return False, f"[{pid}] {field} out of bounds: {val}"

    # --- extraction_quality [0.0, 1.0] ---
    eq = record.get("extraction_quality")
    if eq is not None and not (0.0 <= eq <= 1.0):
        return False, f"[{pid}] extraction_quality out of bounds: {eq}"

    # --- Hex color format ---
    for hex_field in ["primary_color_hex", "secondary_color_hex"]:
        val = record.get(hex_field)
        if val is not None and not HEX_REGEX.match(val):
            return False, f"[{pid}] {hex_field} malformed: {val}"

    return True, None


# ===================================================================
# BATCH PROCESSOR
# ===================================================================

def process_category(category, conn, n=0, dry_run=False):
    """
    Process all images in a category:
      1. Load CSV metadata
      2. For each row: extract → validate → insert
      3. Commit every BATCH_COMMIT_SIZE rows
      4. Return stats dict

    Args:
        category: folder name (must be in ALLOWED_CATEGORIES)
        conn: psycopg2 connection (single, kept open)
        n: max images (0 = all)
        dry_run: if True, extract + validate but do NOT insert
    """
    csv_path = os.path.join(CSV_BASE, f"{category}.csv")
    image_dir = os.path.join(IMAGE_BASE, category)

    if not os.path.isdir(image_dir):
        log.warning(f"Image dir missing: {image_dir}")
        return {"category": category, "skipped": True, "reason": "no_image_dir"}

    # Load metadata
    metadata_rows = load_csv_metadata(csv_path, category, n)
    if not metadata_rows:
        log.warning(f"No CSV data for {category}")
        return {"category": category, "skipped": True, "reason": "no_csv"}

    log.info(f"Processing {category}: {len(metadata_rows)} products")

    stats = {
        "category": category,
        "total": len(metadata_rows),
        "extracted": 0,
        "validated": 0,
        "inserted": 0,
        "skipped_missing": 0,
        "skipped_extraction": 0,
        "skipped_validation": 0,
        "skipped_quality": 0,
        "errors": 0,
    }

    insert_counter = 0

    for meta in metadata_rows:
        idx = meta["index"]
        pid = meta["product_id"]
        image_path = os.path.join(image_dir, f"{idx}.jpg")

        # --- Guard: image must exist ---
        if not os.path.exists(image_path):
            stats["skipped_missing"] += 1
            continue

        # --- Extract ---
        try:
            extracted = extract_structured(image_path, category)
        except Exception as e:
            log.error(f"[{pid}] Extraction crash: {e}")
            stats["errors"] += 1
            continue

        if extracted is None:
            stats["skipped_extraction"] += 1
            continue

        stats["extracted"] += 1

        # --- Quality gate ---
        eq = extracted["extraction_quality"]
        if eq < 0.3:
            log.warning(f"[{pid}] SKIP: extraction_quality={eq:.3f} < 0.3")
            stats["skipped_quality"] += 1
            continue
        if eq < 0.5:
            log.warning(f"[{pid}] WARN: extraction_quality={eq:.3f} < 0.5")

        # --- Merge metadata + extracted attributes ---
        record = {
            **meta,
            **extracted,
            "image_path": image_path,
        }

        # --- Validate ---
        is_valid, error_msg = validate_record(record)
        if not is_valid:
            log.error(f"Validation FAIL: {error_msg}")
            stats["skipped_validation"] += 1
            continue

        stats["validated"] += 1

        # --- Log per product ---
        emb_preview = record["embedding"][:5]
        log.info(
            f"  [{pid}] cat={category} eq={eq:.3f} "
            f"emb[0:5]={[round(v,4) for v in emb_preview]} "
            f"{'DRY' if dry_run else 'INSERT'}"
        )

        # --- Insert (unless dry-run) ---
        if not dry_run:
            success = insert_record(conn, record)
            if success:
                stats["inserted"] += 1
                insert_counter += 1

                # Batch commit
                if insert_counter % BATCH_COMMIT_SIZE == 0:
                    conn.commit()
                    log.info(f"  Committed {insert_counter} records")
            else:
                stats["errors"] += 1
        else:
            stats["inserted"] += 1  # count as "would insert" in dry-run

    # Final commit for remaining records
    if not dry_run:
        conn.commit()

    log.info(
        f"  {category} DONE: "
        f"extracted={stats['extracted']}, validated={stats['validated']}, "
        f"inserted={stats['inserted']}, "
        f"skip_missing={stats['skipped_missing']}, "
        f"skip_quality={stats['skipped_quality']}, "
        f"errors={stats['errors']}"
    )

    return stats


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Visual Attribute Extraction Pipeline")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Categories to process (folder names). Default: all 18")
    parser.add_argument("--n", type=int, default=0,
                        help="Max images per category (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract + validate only, do NOT insert into DB")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress AG-MAN verbose per-image logging")
    args = parser.parse_args()

    global QUIET_MODE
    QUIET_MODE = args.quiet

    categories = args.categories or sorted(ALLOWED_CATEGORIES)
    n_per_cat = args.n
    dry_run = args.dry_run

    print("=" * 70)
    print("VISUAL ATTRIBUTE EXTRACTION PIPELINE")
    print(f"  Categories: {len(categories)}")
    print(f"  Images per category: {'all' if n_per_cat == 0 else n_per_cat}")
    print(f"  Mode: {'DRY-RUN (no DB writes)' if dry_run else 'LIVE (writing to DB)'}")
    print(f"  DB: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("=" * 70)

    # Single DB connection for entire run
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        # Create table (idempotent)
        if not dry_run:
            create_table(conn)
        else:
            log.info("DRY-RUN: Skipping table creation")

        all_stats = []
        t0 = time.time()

        for cat in categories:
            if cat not in ALLOWED_CATEGORIES:
                log.warning(f"Unknown category '{cat}', skipping")
                continue

            cat_t0 = time.time()
            stats = process_category(cat, conn, n=n_per_cat, dry_run=dry_run)
            stats["elapsed_s"] = round(time.time() - cat_t0, 1)
            all_stats.append(stats)

        total_elapsed = time.time() - t0

        # --- Summary ---
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)
        totals = {"extracted": 0, "validated": 0, "inserted": 0,
                  "skipped_missing": 0, "skipped_quality": 0, "errors": 0}

        print(f"{'Category':20s} | {'Extract':>7s} | {'Valid':>5s} | {'Insert':>6s} | "
              f"{'Miss':>4s} | {'QSkip':>5s} | {'Err':>3s} | {'Time':>5s}")
        print("-" * 75)

        for s in all_stats:
            if s.get("skipped"):
                print(f"  {s['category']:20s} | SKIPPED: {s.get('reason', '?')}")
                continue
            print(
                f"  {s['category']:20s} | {s['extracted']:7d} | {s['validated']:5d} | "
                f"{s['inserted']:6d} | {s['skipped_missing']:4d} | "
                f"{s['skipped_quality']:5d} | {s['errors']:3d} | "
                f"{s.get('elapsed_s', 0):5.1f}s"
            )
            for k in totals:
                totals[k] += s.get(k, 0)

        print("-" * 75)
        print(
            f"  {'TOTAL':20s} | {totals['extracted']:7d} | {totals['validated']:5d} | "
            f"{totals['inserted']:6d} | {totals['skipped_missing']:4d} | "
            f"{totals['skipped_quality']:5d} | {totals['errors']:3d} | "
            f"{total_elapsed:5.1f}s"
        )

        if not dry_run:
            # Verify row count
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM visual_attributes")
            count = cur.fetchone()[0]
            cur.close()
            print(f"\nDB row count: {count}")

        print(f"\nMode: {'DRY-RUN' if dry_run else 'LIVE'}")
        print("Done!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
