"""
Memory Audit (Phase 3)
========================
Granular memory measurement of RetrievalService initialization.

Uses:
  - tracemalloc for Python heap tracking
  - sys.getsizeof for individual object sizes
  - psutil for RSS measurement
  - gc.collect() to force cleanup before final measurement

Goal: Confirm final memory ~ numpy + FAISS + metadata arrays.
Target: < 300MB RSS after cleanup.

NO changes to retrieval logic. Measurement only.
"""

import os
import sys
import gc
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_rss_mb():
    """Get current process RSS in MB."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


def sizeof_deep(obj, seen=None):
    """Recursively measure size of a Python object (for lists of lists)."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(sizeof_deep(v, seen) for v in obj.values())
        size += sum(sizeof_deep(k, seen) for k in obj.keys())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            size += sum(sizeof_deep(i, seen) for i in obj)
        except TypeError:
            pass
    return size


def main():
    print("=" * 60)
    print("MEMORY AUDIT")
    print("=" * 60)

    # Start tracemalloc
    tracemalloc.start()

    rss_baseline = get_rss_mb()
    print(f"\n[Stage 0] Baseline")
    print(f"  RSS: {rss_baseline:.1f} MB")
    snap0 = tracemalloc.take_snapshot()

    # ---- Stage 1: Import dependencies ----
    print(f"\n[Stage 1] Importing numpy, faiss, psycopg2...")
    import numpy as np
    import faiss
    import psycopg2

    rss_imports = get_rss_mb()
    print(f"  RSS: {rss_imports:.1f} MB (+{rss_imports - rss_baseline:.1f} MB)")

    # ---- Stage 2 + 3: DB Load + Parse (combined to avoid double memory) ----
    print(f"\n[Stage 2+3] Loading from DB + parsing (streamed, no fetchall)...")
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "shopwhatyousee"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASS", "postgres123@"),
    }

    product_ids = []
    categories = []
    genders = []
    styles = []
    materials = []
    price_buckets = []
    color_families = []
    brands = []
    product_names = []
    image_urls = []
    discounted_prices = []
    original_prices = []
    primary_colors = []
    patterns = []
    sleeves = []
    embeddings = []

    t0 = time.time()
    conn = psycopg2.connect(**DB_CONFIG)
    # Use named cursor for server-side streaming (no fetchall)
    cur = conn.cursor(name="memory_audit_cursor")
    cur.itersize = 2000  # Fetch 2000 rows at a time from server

    cur.execute("""
        SELECT
            product_id, category, gender, style, material,
            price_bucket, color_family, brand, product_name,
            image_url, discounted_price, original_price,
            primary_color_name, pattern_value, sleeve_value,
            embedding
        FROM visual_attributes
        ORDER BY id
    """)

    row_count = 0
    for row in cur:
        (pid, cat, gen, sty, mat, pbk, cfm, brand, pname,
         img_url, dprice, oprice, pcolor, pattern, sleeve, emb) = row

        product_ids.append(pid)
        categories.append(cat)
        genders.append(gen)
        styles.append(sty)
        materials.append(mat)
        price_buckets.append(pbk)
        color_families.append(cfm)
        brands.append(brand)
        product_names.append(pname)
        image_urls.append(img_url)
        discounted_prices.append(dprice)
        original_prices.append(oprice)
        primary_colors.append(pcolor)
        patterns.append(pattern)
        sleeves.append(sleeve)
        embeddings.append(emb)
        row_count += 1

    cur.close()
    conn.close()
    db_time = time.time() - t0

    rss_parsed = get_rss_mb()

    # Measure individual sizes (FAST estimation for embeddings)
    # sizeof_deep on 34K x 512 Python float lists is too slow (17.8M objects)
    # Sample 10 embeddings and extrapolate instead
    if embeddings:
        sample_size = min(10, len(embeddings))
        avg_emb_size = sum(sizeof_deep(embeddings[i]) for i in range(sample_size)) / sample_size
        embeddings_size_mb = (avg_emb_size * len(embeddings) + sys.getsizeof(embeddings)) / (1024 * 1024)
    else:
        embeddings_size_mb = 0

    metadata_lists = [product_ids, categories, genders, styles, materials,
                      price_buckets, color_families, brands, product_names,
                      image_urls, discounted_prices, original_prices,
                      primary_colors, patterns, sleeves]
    metadata_size_mb = sum(sizeof_deep(lst) for lst in metadata_lists) / (1024 * 1024)

    print(f"  Rows: {row_count}")
    print(f"  DB load+parse time: {db_time:.1f}s")
    print(f"  RSS: {rss_parsed:.1f} MB (+{rss_parsed - rss_imports:.1f} MB)")
    print(f"  sizeof(embeddings list): {embeddings_size_mb:.1f} MB")
    print(f"  sizeof(metadata lists): {metadata_size_mb:.1f} MB")

    # ---- Stage 4: Convert to numpy ----
    print(f"\n[Stage 4] Converting to numpy float32...")
    t1 = time.time()
    emb_np = np.array(embeddings, dtype=np.float32)
    np_time = time.time() - t1

    rss_numpy = get_rss_mb()
    numpy_size_mb = emb_np.nbytes / (1024 * 1024)
    theoretical_mb = len(embeddings) * 512 * 4 / (1024 * 1024)

    print(f"  Shape: {emb_np.shape}")
    print(f"  numpy.nbytes: {numpy_size_mb:.1f} MB")
    print(f"  Theoretical (N x 512 x 4): {theoretical_mb:.1f} MB")
    print(f"  Conversion time: {np_time:.3f}s")
    print(f"  RSS: {rss_numpy:.1f} MB (+{rss_numpy - rss_parsed:.1f} MB)")

    # ---- Stage 5: Build FAISS index ----
    print(f"\n[Stage 5] Building FAISS IndexFlatIP(512)...")
    norms = np.linalg.norm(emb_np, axis=1)
    if abs(norms.mean() - 1.0) > 0.01:
        faiss.normalize_L2(emb_np)

    t2 = time.time()
    index = faiss.IndexFlatIP(512)
    index.add(emb_np)
    faiss_time = time.time() - t2

    rss_faiss = get_rss_mb()
    faiss_theoretical_mb = index.ntotal * 512 * 4 / (1024 * 1024)

    print(f"  Index vectors: {index.ntotal}")
    print(f"  FAISS theoretical: {faiss_theoretical_mb:.1f} MB")
    print(f"  Build time: {faiss_time:.3f}s")
    print(f"  RSS: {rss_faiss:.1f} MB (+{rss_faiss - rss_numpy:.1f} MB)")

    # ---- Stage 6: Cleanup intermediate data ----
    print(f"\n[Stage 6] Cleaning up intermediate data...")

    # Delete the embeddings Python list and numpy copy
    # (no raw `rows` to delete since we streamed)
    del embeddings
    # Keep emb_np only if FAISS needs it (IndexFlatIP copies data, so safe to delete)
    del emb_np
    gc.collect()

    rss_clean = get_rss_mb()
    snap_final = tracemalloc.take_snapshot()

    print(f"  RSS after gc.collect(): {rss_clean:.1f} MB")
    print(f"  Freed: {rss_faiss - rss_clean:.1f} MB")

    # ---- tracemalloc top 10 ----
    print(f"\n[tracemalloc] Top 10 memory consumers:")
    stats = snap_final.compare_to(snap0, 'lineno')
    for stat in stats[:10]:
        print(f"  {stat}")

    # ---- Final Summary ----
    print(f"\n{'=' * 60}")
    print(f"MEMORY AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline RSS:        {rss_baseline:7.1f} MB")
    print(f"  After imports:       {rss_imports:7.1f} MB")
    print(f"  After DB+parse:      {rss_parsed:7.1f} MB  (emb={embeddings_size_mb:.1f}MB + meta={metadata_size_mb:.1f}MB)")
    print(f"  After numpy:         {rss_numpy:7.1f} MB  (ndarray={numpy_size_mb:.1f}MB)")
    print(f"  After FAISS:         {rss_faiss:7.1f} MB  (index={faiss_theoretical_mb:.1f}MB)")
    print(f"  After cleanup:       {rss_clean:7.1f} MB")
    print(f"")
    print(f"  Expected minimum: numpy({numpy_size_mb:.1f}) + FAISS({faiss_theoretical_mb:.1f}) + metadata({metadata_size_mb:.1f}) = {numpy_size_mb + faiss_theoretical_mb + metadata_size_mb:.1f} MB")
    print(f"  Actual final RSS:    {rss_clean:.1f} MB")
    target_met = rss_clean < 300
    print(f"  Target <300MB:       {'PASS' if target_met else 'FAIL'}")
    print(f"{'=' * 60}")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
