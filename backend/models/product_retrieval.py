import psycopg2

def get_db():
    return psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="postgres123@"
    )
    
def normalize_category(cat):
    if not cat:
        return None
    cat = cat.lower()
    mapping = {
        "short": "shorts",
        "tee": "t-shirt",
        "tshirt": "t-shirt"
    }
    return mapping.get(cat, cat)
    

def search_products(filters):
    category = normalize_category(filters.get("category"))
    if not filters.get("category"):
        return []

    conn = get_db()
    cur = conn.cursor()

    query = """
        SELECT id, name, price, image_url
        FROM products
        WHERE category = %s
    """
    params = [category]


    for k in ["color","pattern","style"]:
        if filters.get(k):
            query += f" AND {k} = %s"
            params.append(filters[k])

    if filters.get("price_max"):
        query += " AND price <= %s"
        params.append(filters["price_max"])

    query += " ORDER BY price ASC LIMIT 20"

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [{
        "id": r[0],
        "name": r[1],
        "price": float(r[2]),
        "image_url": f"http://localhost:5000{r[3]}"

    } for r in rows]
