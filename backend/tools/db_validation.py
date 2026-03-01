"""
Phase 1 — Database Validation Script
=====================================
Runs all integrity checks required before Supabase migration.
"""
import os
import sys
import json
import random
import math
import psycopg2

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "shopwhatyousee"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres123@"),
}

# Color family values will be loaded dynamically from DB

def main():
    print("=" * 60)
    print("PHASE 1: DATABASE VALIDATION")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    report = {}

    # ─── Check 1: Total row count ────────────────────────────────
    print("\n[CHECK 1] Total row count")
    cur.execute("SELECT COUNT(*) FROM visual_attributes;")
    total = cur.fetchone()[0]
    print(f"  Total rows: {total}")
    report["total_rows"] = total
    report["row_count_ok"] = total == 34787

    # ─── Check 2: NULL embeddings ────────────────────────────────
    print("\n[CHECK 2] NULL embeddings")
    cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE embedding IS NULL;")
    null_count = cur.fetchone()[0]
    print(f"  NULL embeddings: {null_count}")
    report["null_embeddings"] = null_count
    report["null_ok"] = null_count == 0

    # ─── Check 3: Duplicate product_ids ──────────────────────────
    print("\n[CHECK 3] Duplicate product_ids")
    cur.execute("""
        SELECT product_id, COUNT(*)
        FROM visual_attributes
        GROUP BY product_id
        HAVING COUNT(*) > 1;
    """)
    duplicates = cur.fetchall()
    dup_count = len(duplicates)
    print(f"  Duplicate product_ids: {dup_count}")
    if duplicates:
        for pid, cnt in duplicates[:10]:
            print(f"    {pid}: {cnt} rows")
        if dup_count > 10:
            print(f"    ... and {dup_count - 10} more")
    report["duplicate_product_ids"] = dup_count
    report["duplicates_ok"] = dup_count == 0

    # ─── Check 4: Sample 50 rows — validate embeddings + fields ─
    print("\n[CHECK 4] Random sample validation (50 rows)")
    cur.execute("""
        SELECT product_id, category, color_family, embedding
        FROM visual_attributes
        ORDER BY RANDOM()
        LIMIT 50;
    """)
    samples = cur.fetchall()

    dim_ok = 0
    norm_ok = 0
    cat_ok = 0
    color_ok = 0
    norm_values = []
    dim_issues = []
    norm_issues = []
    cat_issues = []
    color_issues = []

    for pid, cat, color, emb in samples:
        # Embedding dimension
        if emb is not None and len(emb) == 512:
            dim_ok += 1
        else:
            dim_issues.append(f"{pid}: dim={len(emb) if emb else 'NULL'}")

        # Embedding norm ≈ 1.0
        if emb is not None:
            norm = math.sqrt(sum(x*x for x in emb))
            norm_values.append(norm)
            if abs(norm - 1.0) < 0.01:
                norm_ok += 1
            else:
                norm_issues.append(f"{pid}: norm={norm:.6f}")
        else:
            norm_issues.append(f"{pid}: norm=NULL")

        # Category non-null
        if cat is not None and cat.strip():
            cat_ok += 1
        else:
            cat_issues.append(f"{pid}: category=NULL")

        # Color family valid (non-empty string)
        if color is not None and isinstance(color, str) and color.strip():
            color_ok += 1
        else:
            color_issues.append(f"{pid}: color_family={repr(color)}")

    print(f"  Embedding dim=512: {dim_ok}/50")
    if dim_issues:
        for d in dim_issues[:5]:
            print(f"    ISSUE: {d}")

    avg_norm = sum(norm_values) / len(norm_values) if norm_values else 0
    std_norm = (sum((n - avg_norm)**2 for n in norm_values) / len(norm_values))**0.5 if norm_values else 0
    print(f"  Embedding norm≈1.0: {norm_ok}/50 (avg={avg_norm:.6f}, std={std_norm:.6f})")
    if norm_issues:
        for n in norm_issues[:5]:
            print(f"    ISSUE: {n}")

    print(f"  Category non-null: {cat_ok}/50")
    if cat_issues:
        for c in cat_issues[:5]:
            print(f"    ISSUE: {c}")

    print(f"  Color family valid: {color_ok}/50")
    if color_issues:
        for c in color_issues[:5]:
            print(f"    ISSUE: {c}")

    report["sample_size"] = 50
    report["sample_dim_ok"] = dim_ok
    report["sample_norm_ok"] = norm_ok
    report["sample_norm_avg"] = round(avg_norm, 6)
    report["sample_norm_std"] = round(std_norm, 6)
    report["sample_cat_ok"] = cat_ok
    report["sample_color_ok"] = color_ok

    # ─── Check 5: Category distribution ──────────────────────────
    print("\n[CHECK 5] Category distribution")
    cur.execute("""
        SELECT category, COUNT(*) as cnt
        FROM visual_attributes
        GROUP BY category
        ORDER BY cnt DESC;
    """)
    cat_dist = cur.fetchall()
    report["categories"] = {}
    for cat, cnt in cat_dist:
        print(f"  {cat}: {cnt}")
        report["categories"][cat or "NULL"] = cnt

    # ─── Check 6: Schema columns ────────────────────────────────
    print("\n[CHECK 6] Schema validation")
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'visual_attributes'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()
    expected_cols = {
        "product_id", "category", "gender", "style", "material",
        "price_bucket", "color_family", "brand", "product_name",
        "image_url", "discounted_price", "original_price",
        "primary_color_name", "pattern_value", "sleeve_value",
        "embedding"
    }
    actual_cols = {col for col, _ in columns}
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols - {"id", "created_at", "updated_at",
                                           "primary_color_hex", "extraction_quality",
                                           "embedding_model"}
    print(f"  Total columns: {len(columns)}")
    for col, dtype in columns:
        print(f"    {col}: {dtype}")
    print(f"  Missing required columns: {missing or 'None'}")
    print(f"  Extra columns: {extra or 'None'}")
    report["schema_columns"] = len(columns)
    report["schema_missing"] = list(missing)
    report["schema_ok"] = len(missing) == 0

    cur.close()
    conn.close()

    # ─── Final Summary ───────────────────────────────────────────
    all_ok = (
        report["row_count_ok"]
        and report["null_ok"]
        and report["duplicates_ok"]
        and report["sample_dim_ok"] == 50
        and report["sample_norm_ok"] >= 48  # allow 2 marginal
        and report["sample_cat_ok"] == 50
        and report["schema_ok"]
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total rows:        {report['total_rows']} (expected 34787) {'✅' if report['row_count_ok'] else '❌'}")
    print(f"  NULL embeddings:   {report['null_embeddings']} {'✅' if report['null_ok'] else '❌'}")
    print(f"  Duplicate PIDs:    {report['duplicate_product_ids']} {'✅' if report['duplicates_ok'] else '❌'}")
    print(f"  Embedding dim:     {report['sample_dim_ok']}/50 {'✅' if report['sample_dim_ok']==50 else '❌'}")
    print(f"  Embedding norm:    {report['sample_norm_ok']}/50 (avg={report['sample_norm_avg']}) {'✅' if report['sample_norm_ok']>=48 else '❌'}")
    print(f"  Category non-null: {report['sample_cat_ok']}/50 {'✅' if report['sample_cat_ok']==50 else '❌'}")
    print(f"  Color valid:       {report['sample_color_ok']}/50 {'✅' if report['sample_color_ok']==50 else '❌'}")
    print(f"  Schema complete:   {'✅' if report['schema_ok'] else '❌'}")
    print(f"  Categories:        {len(report['categories'])}")
    print("=" * 60)

    if all_ok:
        print("DATASET VALIDATED — READY TO FREEZE")
        print("Dataset Version 1.0 — 34,787 rows (Locked)")
    else:
        print("VALIDATION FAILED — DO NOT PROCEED")
    print("=" * 60)

    # Save report as JSON
    report["all_ok"] = all_ok
    with open("tools/db_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: tools/db_validation_report.json")


if __name__ == "__main__":
    main()
