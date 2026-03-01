"""Add gender, style, material columns to visual_attributes table."""
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
    ALTER TABLE visual_attributes
    ADD COLUMN IF NOT EXISTS gender VARCHAR(20),
    ADD COLUMN IF NOT EXISTS style VARCHAR(50),
    ADD COLUMN IF NOT EXISTS material VARCHAR(50);
""")

conn.commit()

# Verify
cur.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name='visual_attributes' AND column_name IN ('gender','style','material') "
    "ORDER BY ordinal_position"
)
for r in cur.fetchall():
    print(f"  {r[0]:15s} {r[1]}")

cur.execute("SELECT COUNT(*) FROM visual_attributes")
print(f"\n  Total rows (unchanged): {cur.fetchone()[0]}")

cur.close()
conn.close()
print("\n  Migration complete.")
