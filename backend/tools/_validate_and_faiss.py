"""
Steps 3 + 4: Embedding validation + FAISS retrieval sanity test.
"""
import json
import math
import numpy as np
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ===================================================
# STEP 3: Validate embedding distribution
# ===================================================
print("=" * 60)
print("STEP 3: EMBEDDING VALIDATION")
print("=" * 60)

# Row counts per category
cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category ORDER BY category")
print("\nRow counts:")
for row in cur.fetchall():
    print(f"  {row[0]:20s}: {row[1]}")

# Embedding dimensions
cur.execute("SELECT MIN(array_length(embedding,1)), MAX(array_length(embedding,1)) FROM visual_attributes")
dims = cur.fetchone()
print(f"\nEmbedding dims: min={dims[0]}, max={dims[1]}")

# Sample 10 embeddings and check norm manually
cur.execute("SELECT product_id, embedding FROM visual_attributes ORDER BY RANDOM() LIMIT 10")
print("\nNorm check (10 random samples):")
all_norms_ok = True
for pid, emb in cur.fetchall():
    norm = math.sqrt(sum(v*v for v in emb))
    status = "✓" if 0.98 <= norm <= 1.02 else "✗"
    if status == "✗":
        all_norms_ok = False
    print(f"  {pid:30s} norm={norm:.6f} {status}")

print(f"\nAll norms OK: {all_norms_ok}")

# Check for any NaN/Inf
cur.execute("SELECT product_id, embedding FROM visual_attributes")
nan_count = 0
for pid, emb in cur.fetchall():
    for v in emb:
        if not math.isfinite(v):
            nan_count += 1
            print(f"  NaN/Inf found in {pid}!")
            break
print(f"Records with NaN/Inf: {nan_count}")

# ===================================================
# STEP 4: FAISS retrieval sanity test on shirts
# ===================================================
print("\n" + "=" * 60)
print("STEP 4: FAISS RETRIEVAL SANITY TEST (shirts)")
print("=" * 60)

import faiss

# Load all shirt embeddings
cur.execute("""
    SELECT product_id, primary_color_name, pattern_value, sleeve_value, embedding
    FROM visual_attributes
    WHERE category = 'shirts'
""")
rows = cur.fetchall()
print(f"\nLoaded {len(rows)} shirt embeddings")

if len(rows) > 5:
    # Build FAISS index
    product_ids = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    patterns = [r[2] for r in rows]
    sleeves = [r[3] for r in rows]
    embeddings = np.array([list(r[4]) for r in rows], dtype=np.float32)

    # Normalize (should already be, but just in case)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(512)  # Inner product = cosine similarity for normalized vecs
    index.add(embeddings)

    # Pick 3 random query images
    print("\n--- Retrieval Test ---")
    rng = np.random.RandomState(42)
    for trial in range(3):
        q_idx = rng.randint(0, len(rows))
        q_vec = embeddings[q_idx:q_idx+1]

        D, I = index.search(q_vec, 6)  # Top 6 (includes self)

        print(f"\nQuery: {product_ids[q_idx]} | color={colors[q_idx]} | pattern={patterns[q_idx]} | sleeve={sleeves[q_idx]}")
        print(f"  Top-5 nearest:")
        for rank, (dist, idx) in enumerate(zip(D[0][1:], I[0][1:])):  # skip self
            print(f"    #{rank+1} sim={dist:.4f} | {product_ids[idx]} | color={colors[idx]} | pattern={patterns[idx]} | sleeve={sleeves[idx]}")

    # Intra-category average cosine similarity
    n = len(embeddings)
    if n > 1:
        sim_matrix = embeddings @ embeddings.T
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_sim = sim_matrix[mask].mean()
        print(f"\nAvg intra-category cosine similarity (shirts): {avg_sim:.4f}")
        print(f"  {'✓ > 0.3 threshold' if avg_sim > 0.3 else '✗ Below 0.3 — REVIEW NEEDED'}")

cur.close()
conn.close()
print("\nDone!")
