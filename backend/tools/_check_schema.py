import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'database': 'shopwhatyousee',
    'user': 'postgres',
    'password': 'postgres123@'
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='products' ORDER BY ordinal_position")
print("EXISTING products TABLE COLUMNS:")
for r in cur.fetchall():
    print(f"  {r[0]:30s} {r[1]}")

print("\nROW COUNT:")
cur.execute("SELECT COUNT(*) FROM products")
print(f"  {cur.fetchone()[0]} rows")

# Check if visual_attributes table exists
cur.execute("SELECT EXISTS(SELECT FROM information_schema.tables WHERE table_name='visual_attributes')")
exists = cur.fetchone()[0]
print(f"\nvisual_attributes table exists: {exists}")

cur.close()
conn.close()
