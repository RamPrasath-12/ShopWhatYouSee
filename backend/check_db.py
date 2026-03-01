"""Quick database check script"""
import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'database': 'shopwhatyousee',
    'user': 'postgres',
    'password': 'postgres123@'
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Check distinct yolo_category values
print("\n=== DISTINCT YOLO CATEGORIES ===")
cur.execute("SELECT DISTINCT yolo_category FROM products WHERE yolo_category IS NOT NULL ORDER BY yolo_category")
for row in cur.fetchall():
    print(f"  {row[0]}")

# Check what T_shirt category returns
print("\n=== SAMPLE T_SHIRT PRODUCTS ===")
cur.execute("""
    SELECT product_id, product_name, yolo_category, article_type, base_colour 
    FROM products 
    WHERE LOWER(yolo_category) = 't_shirt' 
    LIMIT 5
""")
for row in cur.fetchall():
    print(f"  {row}")

# Check if there are any products with 'Top' in article_type
print("\n=== PRODUCTS WITH 'TOP' IN NAME/TYPE ===")
cur.execute("""
    SELECT COUNT(*), article_type 
    FROM products 
    WHERE LOWER(article_type) LIKE '%top%'
    GROUP BY article_type
    LIMIT 10
""")
for row in cur.fetchall():
    print(f"  {row}")

# Check RED + T_shirt products
print("\n=== RED T_SHIRT PRODUCTS ===")
cur.execute("""
    SELECT product_id, product_name, yolo_category, base_colour, primary_color
    FROM products 
    WHERE LOWER(yolo_category) = 't_shirt' 
    AND (LOWER(base_colour) LIKE '%red%' OR LOWER(primary_color) LIKE '%red%')
    LIMIT 10
""")
for row in cur.fetchall():
    print(f"  {row}")

cur.close()
conn.close()
print("\n=== DONE ===")
