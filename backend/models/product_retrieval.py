# backend/models/product_retrieval.py

import psycopg2


# --------------------------------------------------
# Database connection
# --------------------------------------------------
def get_db():
    return psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="123456"
    )


# --------------------------------------------------
# Category normalization (YOLO → DB)
# --------------------------------------------------
def normalize_category(cat):
    if not cat:
        return None

    cat = cat.lower()

    mapping = {
        # shirts
        "tee": "t-shirt",
        "tshirt": "t-shirt",

        # bottoms
        "pant": "trousers",
        "pants": "trousers",
        "trouser": "trousers",

        # shorts
        "short": "shorts",

        # jackets / outerwear
        "coat": "jacket",
        "blazer": "jacket"
    }

    return mapping.get(cat, cat)


# ==================================================
# Progressive Retrieval (Review-1) – FIXED
# ==================================================
def search_products(filters):
    """
    Retrieval strategy (Review-1):
    1) Category + color + pattern + style (STRICT)
    2) Relax style
    3) Relax pattern
    4) Relax color
    5) Category-only fallback

    NOTE:
    sleeve_length is a visual attribute (AG-MAN)
    and is NOT queried from DB in Review-1
    """

    category = normalize_category(filters.get("category"))
    if not category:
        return []

    conn = get_db()
    cur = conn.cursor()

    base_query = """
        SELECT id, name, price, image_url
        FROM products
        WHERE category = %s
    """
    base_params = [category]

    def run_query(extra_conditions, extra_params):
        q = base_query + extra_conditions

        if filters.get("price_max"):
            q += " AND price <= %s"
            extra_params.append(filters["price_max"])

        q += " ORDER BY price ASC LIMIT 5"
        cur.execute(q, base_params + extra_params)
        return cur.fetchall()

    results = []

    # -----------------------------
    # Stage 1: Strict match
    # -----------------------------
    conditions = ""
    params = []

    for k in ["color", "pattern", "style"]:
        if filters.get(k):
            conditions += f" AND {k} = %s"
            params.append(filters[k])

    results = run_query(conditions, params)

    # -----------------------------
    # Stage 2: Relax style
    # -----------------------------
    if len(results) < 5:
        conditions = ""
        params = []

        for k in ["color", "pattern"]:
            if filters.get(k):
                conditions += f" AND {k} = %s"
                params.append(filters[k])

        results = run_query(conditions, params)

    # -----------------------------
    # Stage 3: Relax pattern
    # -----------------------------
    if len(results) < 5:
        conditions = ""
        params = []

        if filters.get("color"):
            conditions += " AND color = %s"
            params.append(filters["color"])

        results = run_query(conditions, params)

    # -----------------------------
    # Stage 4: Category-only fallback
    # -----------------------------
    if len(results) < 5:
        results = run_query("", [])

    cur.close()
    conn.close()

    return [
        {
            "id": r[0],
            "name": r[1],
            "price": float(r[2]),
            "image_url": f"http://localhost:5000{r[3]}"
        }
        for r in results
    ]
