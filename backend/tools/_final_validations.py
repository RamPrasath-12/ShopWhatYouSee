"""Final 6 Validations before Supabase deployment."""
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

all_pass = True

def check(label, passed, detail=""):
    global all_pass
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")

# ─── VALIDATION 1: Embedding Integrity ──────────────────────────────
print("=" * 60)
print("VALIDATION 1: EMBEDDING INTEGRITY")
print("=" * 60)

cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE embedding IS NULL")
null_emb = cur.fetchone()[0]
check("NULL embeddings", null_emb == 0, f"NULL count: {null_emb}")

cur.execute("SELECT MIN(array_length(embedding,1)), MAX(array_length(embedding,1)) FROM visual_attributes")
r = cur.fetchone()
check("Dimension consistency", r[0] == 512 and r[1] == 512, f"MIN={r[0]}, MAX={r[1]}")

# ─── VALIDATION 2: Confidence Ranges ────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION 2: CONFIDENCE RANGE SAFETY (0.0 - 1.0)")
print("=" * 60)

for col in ['color_confidence', 'pattern_confidence', 'sleeve_confidence']:
    cur.execute(f"SELECT MIN({col}), MAX({col}) FROM visual_attributes")
    r = cur.fetchone()
    lo, hi = r[0], r[1]
    passed = (lo is None or lo >= 0.0) and (hi is None or hi <= 1.0)
    check(f"{col}", passed, f"range=[{lo}, {hi}]")

# ─── VALIDATION 3: Extraction Quality ───────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION 3: EXTRACTION QUALITY DISTRIBUTION")
print("=" * 60)

cur.execute("SELECT MIN(extraction_quality), MAX(extraction_quality), AVG(extraction_quality) FROM visual_attributes")
r = cur.fetchone()
check("Quality range", r[0] >= 0.5, f"MIN={r[0]:.3f}  MAX={r[1]:.3f}  AVG={r[2]:.3f}")

cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE extraction_quality < 0.5")
low_q = cur.fetchone()[0]
check("Low quality (<0.5)", low_q == 0, f"Count: {low_q}")

# ─── VALIDATION 4: Duplicate Protection ─────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION 4: DUPLICATE PROTECTION")
print("=" * 60)

cur.execute("SELECT product_id, COUNT(*) FROM visual_attributes GROUP BY product_id HAVING COUNT(*) > 1")
dupes = cur.fetchall()
check("Product ID uniqueness", len(dupes) == 0, f"Duplicates: {len(dupes)}")
if dupes:
    for d in dupes[:5]:
        print(f"         dup: {d[0]} (count={d[1]})")

# ─── VALIDATION 5: Price Integrity ──────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION 5: PRICE INTEGRITY")
print("=" * 60)

cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE discounted_price > original_price AND original_price IS NOT NULL AND discounted_price IS NOT NULL")
bad_price = cur.fetchone()[0]
check("Discount < Original", bad_price == 0, f"Violations: {bad_price}")

cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE discounted_price IS NULL")
null_disc = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE original_price IS NULL")
null_orig = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM visual_attributes")
total = cur.fetchone()[0]
print(f"         discounted_price NULL: {null_disc}/{total}")
print(f"         original_price NULL:   {null_orig}/{total}")

# Check types
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns
    WHERE table_name='visual_attributes' AND column_name IN ('discounted_price','original_price')
""")
for r in cur.fetchall():
    check(f"{r[0]} type is numeric", r[1] == 'integer', f"type={r[1]}")

# ─── VALIDATION 6: NULL Profile Audit ───────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION 6: NULL PROFILE AUDIT")
print("=" * 60)

cur.execute("""
    SELECT
      COUNT(*) as total,
      COUNT(*) FILTER (WHERE gender IS NULL) * 100.0 / COUNT(*) AS gender_null,
      COUNT(*) FILTER (WHERE style IS NULL) * 100.0 / COUNT(*) AS style_null,
      COUNT(*) FILTER (WHERE material IS NULL) * 100.0 / COUNT(*) AS material_null,
      COUNT(*) FILTER (WHERE primary_color_name IS NULL) * 100.0 / COUNT(*) AS color_null,
      COUNT(*) FILTER (WHERE pattern_value IS NULL) * 100.0 / COUNT(*) AS pattern_null,
      COUNT(*) FILTER (WHERE sleeve_value IS NULL) * 100.0 / COUNT(*) AS sleeve_null,
      COUNT(*) FILTER (WHERE embedding IS NULL) * 100.0 / COUNT(*) AS emb_null,
      COUNT(*) FILTER (WHERE product_name IS NULL) * 100.0 / COUNT(*) AS name_null,
      COUNT(*) FILTER (WHERE brand IS NULL) * 100.0 / COUNT(*) AS brand_null,
      COUNT(*) FILTER (WHERE image_url IS NULL) * 100.0 / COUNT(*) AS img_null
    FROM visual_attributes
""")
r = cur.fetchone()
total = r[0]
print(f"  Total rows: {total}")
labels = ['gender','style','material','color','pattern','sleeve','embedding','product_name','brand','image_url']
expected_nulls = {'gender': 43, 'style': 72, 'material': 81}
for i, label in enumerate(labels):
    null_pct = r[i+1]
    exp = expected_nulls.get(label)
    exp_str = f"  (expected ~{exp}%)" if exp else ""
    print(f"  {label:20s}  NULL={null_pct:5.1f}%{exp_str}")

# ─── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_pass:
    print("ALL 6 VALIDATIONS PASSED")
else:
    print("SOME VALIDATIONS FAILED — REVIEW ABOVE")
print("=" * 60)

cur.close()
conn.close()
