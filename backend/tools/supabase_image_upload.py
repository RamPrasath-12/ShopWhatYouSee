"""
Supabase Image Upload
==============================================
Uploads product images to Supabase Storage as-is (original JPEGs).
Original images are 224x224 JPEG quality=70 (~16KB each), already compact.
Total estimated: ~560 MB — well within Supabase 1GB limit.
Updates visual_attributes.image_url in Supabase with the public URL.

Only uploads images that have a matching record in the database (34,787).

Mapping logic:
  CSV row index → local image file myntra/{category}/{i}.jpg
  product_id = {category}_{md5(product_url)[:12]}
  Match product_id against DB records to filter out unmatched images.

Usage:
  python tools/supabase_image_upload.py                   # full upload
  python tools/supabase_image_upload.py --dry-run         # mapping check only
  python tools/supabase_image_upload.py --limit 10        # 10 per category
  python tools/supabase_image_upload.py --category shirts  # single category
"""

import os
import sys
import csv
import json
import time
import hashlib
import urllib.request
import urllib.error
import psycopg2
import argparse

# ─── Configuration ────────────────────────────────────────────────
SUPABASE_URL = "https://mxjpueufbooxgewqxccm.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im14anB1ZXVmYm9veGdld3F4Y2NtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIs"
    "ImlhdCI6MTc3MDk2NjA1NCwiZXhwIjoyMDg2NTQyMDU0fQ."
    "HcqDaYfBevEqfm_in1y4kFRsP25ZY6hnpZ_hvFOVHqM"
)
SUPABASE_DSN = (
    "postgresql://postgres:ShopWhatYouSee123"
    "@db.mxjpueufbooxgewqxccm.supabase.co:5432/postgres"
    "?connect_timeout=15"
)
BUCKET = "product-images"

IMAGE_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase\myntra"
CSV_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase"

CATEGORIES = [
    "belt", "blazer", "caps", "churidhar", "dhoti", "earrings",
    "Footwear_sandals", "Footwear_shoes", "glasses", "Jacket",
    "necklace", "pant", "shirts", "shorts", "skirt", "tie",
    "tshirt", "watch",
]

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, doubles each retry


# ─── Helpers ──────────────────────────────────────────────────────

def generate_product_id(category, product_url, index):
    """Same logic as extract_visual_attributes.py — deterministic product_id."""
    if isinstance(product_url, str) and product_url.strip():
        url_hash = hashlib.md5(product_url.strip().encode("utf-8")).hexdigest()[:12]
        return f"{category}_{url_hash}"
    return f"{category}_{index}"


def load_csv_mapping(category, limit=0):
    """
    Read category CSV, regenerate product_ids, return list of
    {index, product_id, image_path} for rows that have a local image.
    """
    csv_path = os.path.join(CSV_BASE, f"{category}.csv")
    if not os.path.exists(csv_path):
        print(f"  ⚠ CSV not found: {csv_path}")
        return []

    entries = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit > 0 and i >= limit:
                break
            product_url = row.get("product-base href", "")
            pid = generate_product_id(category, product_url, i)
            img_path = os.path.join(IMAGE_BASE, category, f"{i}.jpg")
            if os.path.exists(img_path):
                entries.append({
                    "index": i,
                    "product_id": pid,
                    "image_path": img_path,
                })
    return entries


def upload_to_storage_with_retry(file_bytes, storage_path):
    """
    Upload bytes to Supabase Storage via REST API with retry logic.
    Returns the public URL on success, or None on failure.
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{storage_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg",
        "x-upsert": "true",
    }

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, data=file_bytes, headers=headers, method="POST")
            resp = urllib.request.urlopen(req, timeout=60)
            if resp.status in (200, 201):
                return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
            return None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "already exists" in body.lower():
                # Try PUT for update
                try:
                    req2 = urllib.request.Request(url, data=file_bytes, headers=headers, method="PUT")
                    resp2 = urllib.request.urlopen(req2, timeout=60)
                    if resp2.status in (200, 201):
                        return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
                except Exception:
                    pass
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            return None

    return None


def ensure_bucket_exists():
    """Create bucket if it doesn't exist."""
    url = f"{SUPABASE_URL}/storage/v1/bucket"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "id": BUCKET,
        "name": BUCKET,
        "public": True,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"  Bucket '{BUCKET}' created")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 409 or "already exists" in body.lower():
            print(f"  Bucket '{BUCKET}' already exists ✓")
        else:
            print(f"  ⚠ Bucket creation issue ({e.code}): {body[:200]}")
    except Exception as e:
        print(f"  ⚠ Bucket check error: {e}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload images to Supabase Storage")
    parser.add_argument("--dry-run", action="store_true", help="Check mapping only, no upload")
    parser.add_argument("--limit", type=int, default=0, help="Max images per category (0=all)")
    parser.add_argument("--category", type=str, default="", help="Single category to process")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPABASE IMAGE UPLOAD — Direct JPEG (Sequential)")
    print("=" * 60)

    # Step 1: Get product_ids + already-uploaded URLs from Supabase DB
    print("\n[1] Fetching product_ids from Supabase...")
    supa_conn = psycopg2.connect(SUPABASE_DSN)
    supa_cur = supa_conn.cursor()
    supa_cur.execute("SELECT product_id, image_url FROM visual_attributes")
    rows = supa_cur.fetchall()
    db_product_ids = set(r[0] for r in rows)
    already_uploaded = set(
        r[0] for r in rows
        if r[1] and "supabase.co/storage" in (r[1] or "")
    )
    supa_cur.close()
    print(f"  DB records: {len(db_product_ids)}")
    print(f"  Already uploaded: {len(already_uploaded)} (will skip)")

    # Step 2: Ensure bucket exists
    if not args.dry_run:
        print("\n[2] Checking storage bucket...")
        ensure_bucket_exists()

    # Step 3: Process categories sequentially
    categories = [args.category] if args.category else CATEGORIES

    total_uploaded = 0
    total_skipped_nodb = 0
    total_skipped_done = 0
    total_failed = 0
    total_bytes = 0
    url_updates = []

    for cat in categories:
        print(f"\n[{cat}] Loading CSV mapping...")
        entries = load_csv_mapping(cat, limit=args.limit)
        print(f"  CSV entries with local images: {len(entries)}")

        # Filter to only those with DB records
        matched = [e for e in entries if e["product_id"] in db_product_ids]
        skipped_nodb = len(entries) - len(matched)
        total_skipped_nodb += skipped_nodb

        # Filter out already-uploaded
        to_upload = [e for e in matched if e["product_id"] not in already_uploaded]
        skipped_done = len(matched) - len(to_upload)
        total_skipped_done += skipped_done

        print(f"  Matched to DB: {len(matched)}  (skipped {skipped_nodb} unmatched)")
        print(f"  To upload: {len(to_upload)}  (skipped {skipped_done} already done)")

        if args.dry_run or len(to_upload) == 0:
            continue

        # Sequential upload with progress
        cat_uploaded = 0
        cat_failed = 0
        t0 = time.time()

        for entry in to_upload:
            pid = entry["product_id"]
            img_path = entry["image_path"]

            try:
                file_size = os.path.getsize(img_path)
                with open(img_path, "rb") as f:
                    file_bytes = f.read()

                storage_path = f"{category}/{pid}.jpg" if False else f"{cat}/{pid}.jpg"
                public_url = upload_to_storage_with_retry(file_bytes, storage_path)

                if public_url:
                    cat_uploaded += 1
                    total_bytes += file_size
                    url_updates.append((pid, public_url))
                else:
                    cat_failed += 1
                    if cat_failed <= 5:
                        print(f"    ✗ {pid}: upload_failed after retries")
            except Exception as e:
                cat_failed += 1
                if cat_failed <= 5:
                    print(f"    ✗ {pid}: {e}")

            done = cat_uploaded + cat_failed
            if done % 200 == 0 and done > 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{cat}] {done}/{len(to_upload)} "
                      f"({cat_uploaded} ok, {cat_failed} fail) "
                      f"[{rate:.1f} img/s]")

        # Final progress for category
        elapsed = time.time() - t0
        rate = (cat_uploaded + cat_failed) / elapsed if elapsed > 0 else 0
        print(f"  [{cat}] DONE: {cat_uploaded}/{len(to_upload)} uploaded, "
              f"{cat_failed} failed [{rate:.1f} img/s, {elapsed:.0f}s]")

        total_uploaded += cat_uploaded
        total_failed += cat_failed

        # Batch update DB after each category (for safety/checkpoint)
        if url_updates:
            print(f"  Updating {len(url_updates)} image_urls in Supabase DB...")
            supa_cur = supa_conn.cursor()
            for pid, purl in url_updates:
                supa_cur.execute(
                    "UPDATE visual_attributes SET image_url = %s WHERE product_id = %s",
                    (purl, pid)
                )
            supa_conn.commit()
            supa_cur.close()
            print(f"  DB updated ✓")
            url_updates = []  # reset for next category

    supa_conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"  DB records:        {len(db_product_ids)}")
    print(f"  Uploaded (new):    {total_uploaded}")
    print(f"  Already done:      {total_skipped_done}")
    print(f"  Failed:            {total_failed}")
    print(f"  Skipped (no DB):   {total_skipped_nodb}")
    if total_bytes > 0:
        total_mb = total_bytes / (1024 * 1024)
        print(f"  Total size:        {total_mb:.1f} MB")
        print(f"  Supabase usage:    ~{total_mb:.1f} MB / 1024 MB")
    print("=" * 60)

    # Save report
    report = {
        "db_records": len(db_product_ids),
        "uploaded": total_uploaded,
        "already_done": total_skipped_done,
        "failed": total_failed,
        "skipped_no_db": total_skipped_nodb,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 1) if total_bytes else 0,
    }
    with open("tools/supabase_image_upload_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: tools/supabase_image_upload_report.json")


if __name__ == "__main__":
    main()
