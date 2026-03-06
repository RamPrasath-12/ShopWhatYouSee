"""
compute_metrics.py — Compute REAL performance metrics by querying the live backend.
==================================================================================
Runs actual images from the database through the full pipeline and measures:
  - Top-1, Top-3, Top-5 category accuracy
  - Recall@K
  - MRR (Mean Reciprocal Rank)
  - Per-stage latency (YOLO, AGMAN, Retrieval, Total)
  - LLM filter accuracy (intent extraction)

REQUIRES: Backend running on http://localhost:5000
"""

import os, sys, json, time, base64, urllib.request, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:5000"

# ---- HTTP Helpers ----
def post_json(endpoint, data, timeout=120):
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed = (time.time() - t0) * 1000
        return resp.status, json.loads(resp.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - t0) * 1000
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"error": str(e)}
        return e.code, body, elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return 0, {"error": str(e)}, elapsed


def download_image_b64(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        img_bytes = resp.read()
        return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        return None


def get_test_images(n_per_cat=5):
    """Get n images per category from DB for testing."""
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
    
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = psycopg2.connect(db_url)
    else:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "shopwhatyousee"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASS", "postgres123@"),
        )
    
    cur = conn.cursor()
    # Get all distinct categories
    cur.execute("SELECT DISTINCT category FROM visual_attributes WHERE category IS NOT NULL AND image_url IS NOT NULL")
    categories = [r[0] for r in cur.fetchall() if r[0]]
    
    test_set = {}
    for cat in categories:
        cur.execute(
            "SELECT product_id, category, image_url FROM visual_attributes "
            "WHERE category = %s AND image_url IS NOT NULL AND embedding IS NOT NULL "
            "ORDER BY RANDOM() LIMIT %s",
            (cat, n_per_cat)
        )
        rows = cur.fetchall()
        if rows:
            test_set[cat] = [{"product_id": r[0], "category": r[1], "image_url": r[2]} for r in rows]
    
    cur.close()
    conn.close()
    return test_set


# ========================================
# METRIC 1: RETRIEVAL ACCURACY & LATENCY
# ========================================
def compute_retrieval_metrics(test_images, top_k=10):
    """
    For each test image:
      1. Send to /search-by-image
      2. Check if correct category appears in Top-1, Top-3, Top-5
      3. Compute Recall@K, MRR
      4. Collect per-stage latency
    """
    print("\n" + "=" * 70)
    print("COMPUTING RETRIEVAL METRICS (Real queries to live backend)")
    print("=" * 70)
    
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    total_queries = 0
    skipped = 0
    failed = 0
    
    reciprocal_ranks = []
    recall_at_5_list = []
    recall_at_10_list = []
    
    # Latency collectors
    yolo_times = []
    agman_times = []
    retrieval_times = []  # faiss/sql
    total_times = []
    e2e_times = []  # wall-clock including network
    
    per_category_results = {}
    
    for category, images in test_images.items():
        cat_top1 = 0
        cat_top3 = 0
        cat_top5 = 0
        cat_total = 0
        
        print(f"\n  [{category}] Testing {len(images)} images...")
        
        for i, img_info in enumerate(images):
            pid = img_info["product_id"]
            url = img_info["image_url"]
            
            # Download image
            b64 = download_image_b64(url)
            if not b64:
                skipped += 1
                print(f"    [{i+1}] SKIP: download failed for {url[:60]}...")
                continue
            
            # Query the live backend
            status, body, wall_ms = post_json("/search-by-image", {
                "image": b64,
                "top_k": top_k,
            })
            
            if status != 200:
                failed += 1
                print(f"    [{i+1}] FAIL: HTTP {status}")
                continue
            
            products = body.get("products", [])
            timings = body.get("timings", {})
            detected_cat = body.get("detected_category", "?")
            
            total_queries += 1
            cat_total += 1
            
            # Collect latency
            if "yolo_ms" in timings:
                yolo_times.append(timings["yolo_ms"])
            if "agman_ms" in timings:
                agman_times.append(timings["agman_ms"])
            if "faiss_ms" in timings:
                retrieval_times.append(timings["faiss_ms"])
            if "total_ms" in timings:
                total_times.append(timings["total_ms"])
            e2e_times.append(wall_ms)
            
            # Extract result categories
            result_cats = [p.get("category", "") for p in products]
            
            # Top-1 accuracy
            if len(result_cats) >= 1 and result_cats[0] == category:
                top1_hits += 1
                cat_top1 += 1
            
            # Top-3 accuracy (correct category appears anywhere in top 3)
            if category in result_cats[:3]:
                top3_hits += 1
                cat_top3 += 1
            
            # Top-5 accuracy
            if category in result_cats[:5]:
                top5_hits += 1
                cat_top5 += 1
            
            # MRR: rank of first correct result
            first_rank = None
            for rank, rc in enumerate(result_cats):
                if rc == category:
                    first_rank = rank + 1
                    break
            if first_rank:
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)
            
            # Recall@5 and Recall@10
            # Count how many results in top-K match the expected category
            relevant_in_5 = sum(1 for c in result_cats[:5] if c == category)
            relevant_in_10 = sum(1 for c in result_cats[:10] if c == category)
            # Normalize by min(total_relevant_in_db, K) — but we approximate by K
            recall_at_5_list.append(relevant_in_5 / 5.0)
            recall_at_10_list.append(relevant_in_10 / 10.0)
            
            status_str = "✓" if category in result_cats[:3] else "✗"
            print(f"    [{i+1}] {status_str} YOLO={detected_cat} | "
                  f"Top1={'✓' if result_cats and result_cats[0]==category else '✗'} "
                  f"Top3={'✓' if category in result_cats[:3] else '✗'} "
                  f"Top5={'✓' if category in result_cats[:5] else '✗'} | "
                  f"T={timings.get('total_ms', '?')}ms")
        
        if cat_total > 0:
            per_category_results[category] = {
                "top1": cat_top1 / cat_total,
                "top3": cat_top3 / cat_total,
                "top5": cat_top5 / cat_total,
                "total": cat_total,
            }
    
    # ---- PRINT RESULTS ----
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS RESULTS")
    print("=" * 70)
    
    if total_queries == 0:
        print("ERROR: No queries succeeded!")
        return
    
    top1_acc = top1_hits / total_queries
    top3_acc = top3_hits / total_queries
    top5_acc = top5_hits / total_queries
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0
    r_at_5 = statistics.mean(recall_at_5_list) if recall_at_5_list else 0
    r_at_10 = statistics.mean(recall_at_10_list) if recall_at_10_list else 0
    
    print(f"\n  Total queries:   {total_queries}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ RETRIEVAL ACCURACY                   │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │ Top-1 Accuracy:  {top1_acc:.4f} ({top1_hits}/{total_queries})     │")
    print(f"  │ Top-3 Accuracy:  {top3_acc:.4f} ({top3_hits}/{total_queries})     │")
    print(f"  │ Top-5 Accuracy:  {top5_acc:.4f} ({top5_hits}/{total_queries})     │")
    print(f"  │ MRR:             {mrr:.4f}              │")
    print(f"  │ Recall@5:        {r_at_5:.4f}              │")
    print(f"  │ Recall@10:       {r_at_10:.4f}              │")
    print(f"  └─────────────────────────────────────┘")
    
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ LATENCY (ms)                        │")
    print(f"  ├─────────────────────────────────────┤")
    if yolo_times:
        print(f"  │ YOLO:     avg={statistics.mean(yolo_times):7.1f}  med={statistics.median(yolo_times):7.1f}  max={max(yolo_times):7.1f} │")
    if agman_times:
        print(f"  │ AGMAN:    avg={statistics.mean(agman_times):7.1f}  med={statistics.median(agman_times):7.1f}  max={max(agman_times):7.1f} │")
    if retrieval_times:
        print(f"  │ Retrieval:avg={statistics.mean(retrieval_times):7.1f}  med={statistics.median(retrieval_times):7.1f}  max={max(retrieval_times):7.1f} │")
    if total_times:
        print(f"  │ Pipeline: avg={statistics.mean(total_times):7.1f}  med={statistics.median(total_times):7.1f}  max={max(total_times):7.1f} │")
    if e2e_times:
        print(f"  │ E2E(wall):avg={statistics.mean(e2e_times):7.1f}  med={statistics.median(e2e_times):7.1f}  max={max(e2e_times):7.1f} │")
    print(f"  └─────────────────────────────────────┘")
    
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ PER-CATEGORY TOP-3 ACCURACY          │")
    print(f"  ├─────────────────────────────────────┤")
    for cat, res in sorted(per_category_results.items(), key=lambda x: -x[1]["top3"]):
        print(f"  │ {cat:20s}: {res['top3']:.2%} ({int(res['top3']*res['total'])}/{res['total']}) │")
    print(f"  └─────────────────────────────────────┘")


# ========================================
# METRIC 2: LLM FILTER ACCURACY
# ========================================
def compute_llm_accuracy():
    """
    Test LLM filter generation with known queries.
    Compare output filters against expected ground truth.
    """
    print("\n" + "=" * 70)
    print("COMPUTING LLM FILTER ACCURACY (Intent Extraction)")
    print("=" * 70)
    
    # Ground truth test cases: query -> expected filter fields
    test_cases = [
        {
            "query": "red shirt",
            "category": "shirts",
            "expected": {"category": "shirts", "primary_color_name": "red"},
        },
        {
            "query": "blue tshirt for men",
            "category": "tshirt",
            "expected": {"category": "tshirt", "primary_color_name": "blue", "gender": "Men"},
        },
        {
            "query": "full sleeve black jacket",
            "category": "Jacket",
            "expected": {"category": "Jacket", "sleeve_value": "Full Sleeves", "primary_color_name": "black"},
        },
        {
            "query": "green churidhar",
            "category": "churidhar",
            "expected": {"category": "churidhar", "primary_color_name": "green"},
        },
        {
            "query": "short sleeve white tshirt",
            "category": "tshirt",
            "expected": {"category": "tshirt", "sleeve_value": "Short Sleeves", "primary_color_name": "white"},
        },
        {
            "query": "striped formal shirt",
            "category": "shirts",
            "expected": {"category": "shirts", "pattern_value": "striped"},
        },
        {
            "query": "women blazer",
            "category": "blazer",
            "expected": {"category": "blazer", "gender": "Women"},
        },
        {
            "query": "maroon pant",
            "category": "pant",
            "expected": {"category": "pant", "primary_color_name": "maroon"},
        },
        {
            "query": "sleeveless black top",
            "category": "tshirt",
            "expected": {"sleeve_value": "Sleeveless", "primary_color_name": "black"},
        },
        {
            "query": "show only watches",
            "category": "watch",
            "expected": {"category": "watch"},
        },
    ]
    
    total_fields = 0
    correct_fields = 0
    total_queries = 0
    correct_queries = 0  # exact match on all fields
    
    per_attr_stats = {}  # attribute -> {correct, total}
    
    for tc in test_cases:
        query = tc["query"]
        yolo_cat = tc["category"]
        expected = tc["expected"]
        
        # Call the LLM endpoint
        status, body, elapsed = post_json("/llm", {
            "user_query": query,
            "yolo_category": yolo_cat,
            "is_image_search": False,
        })
        
        if status != 200:
            print(f"  FAIL: '{query}' -> HTTP {status}")
            continue
        
        total_queries += 1
        
        # Extract the LLM's output filters
        llm_filters = body.get("filters", {})
        add_filters = llm_filters.get("add", {})
        
        query_all_correct = True
        
        for field, exp_val in expected.items():
            total_fields += 1
            
            if field not in per_attr_stats:
                per_attr_stats[field] = {"correct": 0, "total": 0}
            per_attr_stats[field]["total"] += 1
            
            # Get the actual value from LLM output
            actual_val = add_filters.get(field, "")
            
            # Normalize for comparison
            exp_lower = exp_val.lower().strip() if isinstance(exp_val, str) else str(exp_val)
            act_lower = actual_val.lower().strip() if isinstance(actual_val, str) else str(actual_val)
            
            matched = (exp_lower == act_lower) or (exp_lower in act_lower) or (act_lower in exp_lower and act_lower)
            
            if matched:
                correct_fields += 1
                per_attr_stats[field]["correct"] += 1
            else:
                query_all_correct = False
                print(f"  MISMATCH: '{query}' -> field '{field}': expected='{exp_val}', got='{actual_val}'")
        
        if query_all_correct:
            correct_queries += 1
            print(f"  ✓ '{query}' -> ALL CORRECT ({elapsed:.0f}ms)")
        else:
            print(f"  ✗ '{query}' -> PARTIAL ({elapsed:.0f}ms)")
    
    # ---- PRINT RESULTS ----
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ LLM FILTER ACCURACY                  │")
    print(f"  ├─────────────────────────────────────┤")
    if total_queries > 0:
        print(f"  │ Intent Accuracy (exact): {correct_queries}/{total_queries} = {correct_queries/total_queries:.2%}    │")
    if total_fields > 0:
        print(f"  │ Field-Level Accuracy:    {correct_fields}/{total_fields} = {correct_fields/total_fields:.2%}    │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │ PER-ATTRIBUTE ACCURACY               │")
    for attr, stats in sorted(per_attr_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  │   {attr:25s}: {stats['correct']}/{stats['total']} = {acc:.2%} │")
    print(f"  └─────────────────────────────────────┘")


# ========================================
# MAIN
# ========================================
def main():
    print("=" * 70)
    print("ShopWhatYouSee — REAL METRICS COMPUTATION")
    print("=" * 70)
    print(f"Backend: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check server
    try:
        req = urllib.request.Request(f"{BASE_URL}/debug/filter-schema")
        resp = urllib.request.urlopen(req, timeout=5)
        print("Server: CONNECTED ✓")
    except Exception as e:
        print(f"FATAL: Cannot connect to backend at {BASE_URL}")
        print(f"Start the server first: python app.py")
        sys.exit(1)
    
    # Load test images (5 per category)
    print("\nLoading test images from database...")
    test_images = get_test_images(n_per_cat=5)
    total = sum(len(v) for v in test_images.values())
    print(f"Got {total} images across {len(test_images)} categories: {list(test_images.keys())}")
    
    # 1. Retrieval metrics + latency
    compute_retrieval_metrics(test_images, top_k=10)
    
    # 2. LLM filter accuracy
    compute_llm_accuracy()
    
    print("\n" + "=" * 70)
    print("DONE — All metrics computed from REAL data")
    print("=" * 70)


if __name__ == "__main__":
    main()
