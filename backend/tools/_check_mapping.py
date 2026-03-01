"""Quick check: product_id format + image_url format in visual_attributes."""
import psycopg2, os

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    database=os.getenv("DB_NAME", "shopwhatyousee"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "postgres123@"),
)
cur = conn.cursor()

# Sample product_ids and image_urls
cur.execute("SELECT product_id, category, image_url FROM visual_attributes ORDER BY product_id LIMIT 10")
print("=== Sample product_id + image_url ===")
for r in cur.fetchall():
    print(f"  pid={r[0]}, cat={r[1]}, img_url={r[2]}")

# Check image_url patterns
cur.execute("SELECT image_url FROM visual_attributes WHERE image_url IS NOT NULL LIMIT 3")
print("\n=== Sample image_urls ===")
for r in cur.fetchall():
    print(f"  {r[0][:120]}")

# Count rows per category
cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category ORDER BY category")
print("\n=== Category counts ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# Check if image_url uses Myntra URLs or local paths
cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE image_url LIKE '%myntra%'")
myntra_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE image_url LIKE '/static%'")
static_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE image_url IS NULL")
null_count = cur.fetchone()[0]
print(f"\n  Myntra URLs: {myntra_count}")
print(f"  /static URLs: {static_count}")
print(f"  NULL: {null_count}")

cur.close()
conn.close()
