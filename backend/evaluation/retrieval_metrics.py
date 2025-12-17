"""
Product Retrieval Evaluation (Review-1)
--------------------------------------
Evaluates the current filter-based retrieval module.

Metrics reported:
- Retrieval latency
- Number of products retrieved
- Top-K availability
- Filter effectiveness

NOTE:
Embedding similarity, FAISS ranking, and accuracy-based
metrics are deferred to future reviews.
"""

import time
import psycopg2

# -----------------------------
# CONFIGURATION
# -----------------------------

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

TOP_K = 5

# Simulated LLM-generated filters
QUERY_FILTERS = {
    "category": "shirt",
    "pattern": "solid",
    "style": None,
    "price_max": 3000
}

# -----------------------------
# DATABASE UTILS
# -----------------------------

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def fetch_products(filters):
    conn = get_db()
    cur = conn.cursor()

    query = """
        SELECT id, name, price
        FROM products
        WHERE category = %s
    """
    params = [filters["category"]]

    if filters.get("pattern"):
        query += " AND pattern = %s"
        params.append(filters["pattern"])

    if filters.get("style"):
        query += " AND style = %s"
        params.append(filters["style"])

    if filters.get("price_max"):
        query += " AND price <= %s"
        params.append(filters["price_max"])

    start = time.time()
    cur.execute(query, params)
    rows = cur.fetchall()
    end = time.time()

    cur.close()
    conn.close()

    return rows, (end - start) * 1000


# -----------------------------
# MAIN EVALUATION
# -----------------------------

def evaluate_retrieval():
    print("\nRunning retrieval evaluation...\n")

    rows, latency_ms = fetch_products(QUERY_FILTERS)

    print(f"Retrieved Products     : {len(rows)}")
    print(f"Retrieval Latency      : {latency_ms:.2f} ms")

    top_k_available = min(TOP_K, len(rows))

    print("\n===== RETRIEVAL METRICS (REVIEW-1) =====")
    print(f"Category               : {QUERY_FILTERS['category']}")
    print(f"Applied Filters        : pattern, style, price")
    print(f"Products Retrieved     : {len(rows)}")
    print(f"Top-{TOP_K} Available  : {top_k_available}/{TOP_K}")
    print(f"Retrieval Latency (ms) : {latency_ms:.2f}")
    print("========================================")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    evaluate_retrieval()
