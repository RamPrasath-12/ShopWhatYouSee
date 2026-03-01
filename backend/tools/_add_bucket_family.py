"""Add price_bucket and color_family, then populate all 34,787 rows."""
import psycopg2
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

# ─── Color Family Mapping (same as retrieval_validation.py) ─────────
COLOR_FAMILIES = {
    "neutral": {"Black", "White", "Grey", "Charcoal", "Silver", "Off White", "Cream", "Multi"},
    "red":     {"Red", "Maroon", "Burgundy", "Wine", "Rust"},
    "pink":    {"Pink", "Hot Pink", "Rose", "Magenta", "Peach", "Coral"},
    "orange":  {"Orange", "Mustard", "Gold"},
    "yellow":  {"Yellow", "Lime Yellow"},
    "green":   {"Green", "Olive", "Khaki", "Lime Green", "Sea Green"},
    "blue":    {"Blue", "Navy Blue", "Sky Blue", "Turquoise", "Teal", "Teal Blue", "Steel Blue"},
    "purple":  {"Purple", "Lavender", "Violet", "Mauve"},
    "brown":   {"Brown", "Coffee Brown", "Chocolate", "Taupe", "Beige", "Tan"},
}

_COLOR_TO_FAMILY = {}
for fam, colors in COLOR_FAMILIES.items():
    for c in colors:
        _COLOR_TO_FAMILY[c] = fam

def get_color_family(color_name):
    if not color_name:
        return None
    return _COLOR_TO_FAMILY.get(color_name.strip().title(), None)

# ─── Price Bucket ───────────────────────────────────────────────────
def get_price_bucket(discounted_price):
    """Derive from discounted_price. INR-based buckets."""
    if discounted_price is None:
        return None
    if discounted_price < 500:
        return "budget"
    elif discounted_price < 1500:
        return "mid"
    elif discounted_price < 5000:
        return "premium"
    else:
        return "luxury"

ALLOWED_BUCKETS = {"budget", "mid", "premium", "luxury"}
ALLOWED_FAMILIES = set(COLOR_FAMILIES.keys())

# ─── Main ───────────────────────────────────────────────────────────
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 1. Add columns
print("Adding columns...")
cur.execute("""
    ALTER TABLE visual_attributes
    ADD COLUMN IF NOT EXISTS price_bucket VARCHAR(20),
    ADD COLUMN IF NOT EXISTS color_family VARCHAR(20);
""")
conn.commit()
print("  Columns added.")

# 2. Load data
cur.execute("SELECT id, discounted_price, primary_color_name FROM visual_attributes ORDER BY id")
rows = cur.fetchall()
total = len(rows)
print(f"  Loaded {total} rows")

# 3. Populate
t0 = time.time()
batch_size = 500
updated = 0
bucket_counts = {}
family_counts = {}

for row_id, price, color in rows:
    bucket = get_price_bucket(price)
    family = get_color_family(color)

    # Enforce allowed values
    if bucket is not None:
        assert bucket in ALLOWED_BUCKETS, f"Bad bucket: {bucket}"
    if family is not None:
        assert family in ALLOWED_FAMILIES, f"Bad family: {family}"

    cur.execute(
        "UPDATE visual_attributes SET price_bucket=%s, color_family=%s WHERE id=%s",
        (bucket, family, row_id)
    )
    updated += 1

    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    family_counts[family] = family_counts.get(family, 0) + 1

    if updated % batch_size == 0:
        conn.commit()
        elapsed = time.time() - t0
        print(f"    ... {updated}/{total} ({updated*100//total}%)  {updated/elapsed:.0f} rows/s")

conn.commit()
elapsed = time.time() - t0
print(f"\n  Done: {updated}/{total} in {elapsed:.1f}s")

# 4. Distributions
print("\n  --- Price Bucket Distribution ---")
for k, v in sorted(bucket_counts.items(), key=lambda x: -x[1]):
    print(f"    {str(k):10s}  {v}")

print("\n  --- Color Family Distribution ---")
for k, v in sorted(family_counts.items(), key=lambda x: -x[1]):
    print(f"    {str(k):10s}  {v}")

# 5. Verify
cur.execute("""
    SELECT
      COUNT(*) FILTER (WHERE price_bucket IS NULL) * 100.0 / COUNT(*),
      COUNT(*) FILTER (WHERE color_family IS NULL) * 100.0 / COUNT(*)
    FROM visual_attributes
""")
r = cur.fetchone()
print(f"\n  NULL rates:")
print(f"    price_bucket: {r[0]:.1f}%")
print(f"    color_family: {r[1]:.1f}%")

cur.close()
conn.close()
print("\n  Complete.")
