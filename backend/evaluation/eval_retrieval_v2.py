"""
Section 5 - Retrieval Engine Evaluation
=======================================
Evaluates SQL + cosine hybrid retrieval with Precision@K, Recall@K, mAP, NDCG.
"""

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

K_VALUES = [1, 5, 10, 20]
NUM_QUERIES = 120


def _connect():
    from db_config import get_db_connection
    return get_db_connection()


def _fetch_queries(limit_n):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT product_id, category,
               primary_color_name, pattern_value, sleeve_value,
               embedding
        FROM visual_attributes
        WHERE category IS NOT NULL
          AND embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT %s
        """,
        (limit_n,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _category_counts():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category")
    data = {r[0]: int(r[1]) for r in cur.fetchall()}
    cur.close()
    conn.close()
    return data


def _parse_embedding(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None
    try:
        return list(v)
    except Exception:
        return None


def _dcg(rels, k):
    out = 0.0
    for i, rel in enumerate(rels[:k]):
        out += rel / np.log2(i + 2)
    return out


def _ndcg(rels, k):
    ideal = sorted(rels, reverse=True)
    denom = _dcg(ideal, k)
    return (_dcg(rels, k) / denom) if denom > 0 else 0.0


def _average_precision(rels):
    hit = 0
    precisions = []
    for i, rel in enumerate(rels, start=1):
        if rel:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions)) if precisions else 0.0


def run():
    print("\n" + "=" * 60)
    print("SECTION 5: RETRIEVAL ENGINE EVALUATION")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    try:
        from models.product_retrieval import search_products_v3
    except Exception as e:
        print(f"  ERROR import retrieval: {e}")
        return {}

    queries = _fetch_queries(NUM_QUERIES)
    if not queries:
        print("  ERROR no query products found")
        return {}

    cat_counts = _category_counts()

    p_at_k = defaultdict(list)
    r_at_k = defaultdict(list)
    ndcg_at_k = defaultdict(list)
    ap_values = []
    first_rel_rank = []

    for idx, row in enumerate(queries, start=1):
        pid, category, color, pattern, sleeve, embedding = row
        emb = _parse_embedding(embedding)

        qctx = {
            "category": category,
            "embedding": emb,
            "detected_attributes": {
                "category": category,
                "color": color,
                "pattern": pattern,
                "sleeve": sleeve,
            },
            "user_filters": {"category": category},
            "price_max": None,
            "scene": None,
            "extraction_quality": 1.0,
        }

        try:
            out = search_products_v3(qctx, top_k=max(K_VALUES))
            products = out.get("products", []) if isinstance(out, dict) else []
        except Exception:
            products = []

        rels = [1 if p.get("category") == category else 0 for p in products]

        rank = 0
        for i, r in enumerate(rels, start=1):
            if r == 1:
                rank = i
                break
        first_rel_rank.append(rank if rank > 0 else max(K_VALUES) + 1)

        total_rel = max(1, int(cat_counts.get(category, 1)))
        for k in K_VALUES:
            rel_k = rels[:k]
            hits = sum(rel_k)
            p_at_k[k].append(hits / k)
            r_at_k[k].append(min(1.0, hits / total_rel))
            ndcg_at_k[k].append(_ndcg(rel_k, k))

        ap_values.append(_average_precision(rels))

        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(queries)}")

    metrics = {
        "Precision@1": round(float(np.mean(p_at_k[1])), 4),
        "Precision@5": round(float(np.mean(p_at_k[5])), 4),
        "Precision@10": round(float(np.mean(p_at_k[10])), 4),
        "Precision@20": round(float(np.mean(p_at_k[20])), 4),
        "Recall@1": round(float(np.mean(r_at_k[1])), 4),
        "Recall@5": round(float(np.mean(r_at_k[5])), 4),
        "Recall@10": round(float(np.mean(r_at_k[10])), 4),
        "Recall@20": round(float(np.mean(r_at_k[20])), 4),
        "NDCG@1": round(float(np.mean(ndcg_at_k[1])), 4),
        "NDCG@5": round(float(np.mean(ndcg_at_k[5])), 4),
        "NDCG@10": round(float(np.mean(ndcg_at_k[10])), 4),
        "NDCG@20": round(float(np.mean(ndcg_at_k[20])), 4),
        "mAP": round(float(np.mean(ap_values)), 4),
        "num_queries": len(queries),
    }

    csv_path = os.path.join(TABLES_DIR, "retrieval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k in [
            "Precision@1", "Precision@5", "Precision@10", "Precision@20",
            "Recall@1", "Recall@5", "Recall@10", "Recall@20",
            "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20", "mAP",
        ]:
            w.writerow([k, metrics[k]])
    print(f"  OK {csv_path}")

    p_vals = [metrics[f"Precision@{k}"] for k in K_VALUES]
    r_vals = [metrics[f"Recall@{k}"] for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_VALUES, p_vals, marker="o", color="#4C78A8")
    ax.set_title("Precision@K Curve", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p_path = os.path.join(GRAPHS_DIR, "precision_at_k.png")
    plt.savefig(p_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {p_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_VALUES, r_vals, marker="s", color="#E45756")
    ax.set_title("Recall@K Curve", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    r_path = os.path.join(GRAPHS_DIR, "recall_at_k.png")
    plt.savefig(r_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {r_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(first_rel_rank, bins=min(20, max(K_VALUES) + 1), color="#72B7B2", edgecolor="white")
    ax.set_title("Ranking Distribution (First Relevant Rank)", fontweight="bold")
    ax.set_xlabel("Rank of first relevant item")
    ax.set_ylabel("Query count")
    plt.tight_layout()
    rank_path = os.path.join(GRAPHS_DIR, "ranking_distribution.png")
    plt.savefig(rank_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {rank_path}")

    json_path = os.path.join(METRICS_DIR, "retrieval_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  OK {json_path}")

    return metrics


if __name__ == "__main__":
    run()
