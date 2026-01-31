import psycopg2, csv

conn = psycopg2.connect(
    host="localhost",
    database="shopwhatyousee",
    user="postgres",
    password="123456"
)
cur = conn.cursor()

with open("static/products/products.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        cur.execute("""
            INSERT INTO products
            (name, category, gender, color, pattern, style, price, image_url)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            r["name"],
            r["category"],
            r["gender"],
            r["color"],
            r["pattern"],
            r["style"],
            int(r["price"]),
            f"/static/products/images/{r['filename']}"
        ))

conn.commit()
cur.close()
conn.close()
print("Imported products")
