import psycopg2

def get_db():
    return psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="postgres123@"   # change if needed
    )

def search_products(filters):
    query = """
    SELECT id, name, price, image_url
    FROM products
    WHERE 1=1
    """
    params = []

    if filters.get("color"):
        query += " AND color = %s"
        params.append(filters["color"])

    if filters.get("pattern"):
        query += " AND pattern = %s"
        params.append(filters["pattern"])

    if filters.get("style"):
        query += " AND style = %s"
        params.append(filters["style"])

    if filters.get("price_max"):
        query += " AND price <= %s"
        params.append(filters["price_max"])

    query += " LIMIT 20"

    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "name": r[1],
            "price": r[2],
            "image_url": r[3]   # <-- already stored in DB
        })

    return results
