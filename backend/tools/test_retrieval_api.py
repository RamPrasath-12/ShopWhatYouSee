"""
End-to-End Retrieval API Test
=============================
Tests the retrieval service directly (no HTTP needed).
Validates:
  1. Service initialization
  2. Category-specific retrieval
  3. Filtered queries
  4. Performance benchmarking
"""
import sys
import os
import time
import numpy as np
from collections import Counter

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.retrieval_service import retrieval_service


def test_service_init():
    """Test 1: Service initializes correctly."""
    print("=" * 60)
    print("TEST 1: Service Initialization")
    print("=" * 60)

    t0 = time.time()
    retrieval_service.init()
    init_time = time.time() - t0

    assert retrieval_service.is_ready, "Service not ready!"
    assert retrieval_service.n_vectors == 34787, f"Expected 34787, got {retrieval_service.n_vectors}"
    assert retrieval_service.index is not None, "FAISS index is None!"

    print(f"  PASS: {retrieval_service.n_vectors} vectors loaded in {init_time:.1f}s")
    return True


def test_category_retrieval():
    """Test 2: Search returns same-category products."""
    print("\n" + "=" * 60)
    print("TEST 2: Category-Specific Retrieval")
    print("=" * 60)

    # Pick a known product from each test category
    test_categories = ["shirts", "Footwear_shoes", "watch", "skirt"]
    all_pass = True

    for cat in test_categories:
        # Find first product of this category
        idx = None
        for i, c in enumerate(retrieval_service.categories):
            if c == cat:
                idx = i
                break

        if idx is None:
            print(f"  [{cat}] SKIP — category not found")
            continue

        # Use its embedding as query
        query_emb = retrieval_service.index.reconstruct(idx).tolist()
        pid = retrieval_service.product_ids[idx]

        t0 = time.time()
        results = retrieval_service.search(
            query_embedding=query_emb,
            filters={"category": cat},
            top_k=20,
            exclude_product_id=pid,
        )
        latency_ms = (time.time() - t0) * 1000

        # Check results
        result_cats = [r["category"] for r in results]
        same_cat = sum(1 for c in result_cats if c == cat)
        same_cat_pct = same_cat / len(results) * 100 if results else 0

        avg_sim = np.mean([r["similarity_score"] for r in results]) if results else 0

        passed = same_cat_pct == 100  # With category filter, should be 100%
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"  [{status}] {cat}: {len(results)} results, "
              f"{same_cat_pct:.0f}% same-cat, "
              f"avg_sim={avg_sim:.3f}, "
              f"{latency_ms:.1f}ms")

        # Show top-3
        for r in results[:3]:
            print(f"        {r['product_name'][:50]} | {r['similarity_score']:.3f}")

    return all_pass


def test_filtered_queries():
    """Test 3: Filtered search with graceful relaxation."""
    print("\n" + "=" * 60)
    print("TEST 3: Filtered Queries")
    print("=" * 60)

    # Find a men's shirt product to use as query
    query_idx = None
    for i in range(retrieval_service.n_vectors):
        if (retrieval_service.categories[i] == "shirts" and
                retrieval_service.genders[i] == "men"):
            query_idx = i
            break

    if query_idx is None:
        print("  SKIP — no men's shirt found")
        return True

    query_emb = retrieval_service.index.reconstruct(query_idx).tolist()
    query_pid = retrieval_service.product_ids[query_idx]

    test_cases = [
        {"label": "men shirts", "filters": {"category": "shirts", "gender": "men"}},
        {"label": "men casual shirts", "filters": {"category": "shirts", "gender": "men", "style": "casual"}},
        {"label": "men cotton shirts", "filters": {"category": "shirts", "gender": "men", "material": "cotton"}},
        {"label": "leather belt", "filters": {"category": "belt", "material": "leather"}},
        {"label": "no filter", "filters": {}},
    ]

    all_pass = True
    for tc in test_cases:
        t0 = time.time()
        results = retrieval_service.search(
            query_embedding=query_emb,
            filters=tc["filters"],
            top_k=10,
            exclude_product_id=query_pid,
        )
        latency_ms = (time.time() - t0) * 1000

        cat_dist = Counter(r["category"] for r in results)
        gender_dist = Counter(r["gender"] for r in results if r["gender"])

        passed = len(results) > 0
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {tc['label']}: {len(results)} results, {latency_ms:.1f}ms")
        print(f"        cats: {dict(cat_dist)}")
        if gender_dist:
            print(f"        genders: {dict(gender_dist)}")
        if results:
            avg_sim = np.mean([r["similarity_score"] for r in results])
            print(f"        avg_sim: {avg_sim:.3f}")

    return all_pass


def test_performance():
    """Test 4: FAISS query performance benchmark."""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Benchmark")
    print("=" * 60)

    # Random embedding
    rng = np.random.RandomState(42)
    query = rng.randn(512).astype(np.float32)
    query /= np.linalg.norm(query)

    # Warmup
    retrieval_service.search(query.tolist(), top_k=20)

    # Benchmark: 50 queries
    n_queries = 50
    latencies = []
    for _ in range(n_queries):
        t0 = time.time()
        retrieval_service.search(query.tolist(), top_k=20)
        latencies.append((time.time() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"  Queries: {n_queries}")
    print(f"  Mean:    {latencies.mean():.2f}ms")
    print(f"  Median:  {np.median(latencies):.2f}ms")
    print(f"  P95:     {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:     {np.percentile(latencies, 99):.2f}ms")
    print(f"  Max:     {latencies.max():.2f}ms")

    passed = latencies.mean() < 10  # FAISS query should be < 10ms
    print(f"  {'PASS' if passed else 'FAIL'}: Mean < 10ms target")
    return passed


def test_no_legacy_deps():
    """Test 5: Verify no legacy dependencies remain in search code."""
    print("\n" + "=" * 60)
    print("TEST 5: No Legacy Dependencies")
    print("=" * 60)

    import ast

    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("from models.product_retrieval import search_products", "Legacy V1 import"),
        ("search_products_v2", "Legacy V2 import"),
        (".npy", "NPY file reference"),
    ]

    all_pass = True
    for pattern, label in checks:
        found = pattern in content
        if found:
            all_pass = False
        status = "PASS" if not found else "FAIL"
        print(f"  [{status}] No '{label}' in app.py")

    # Check retrieval_service is used
    has_service = "from services.retrieval_service import retrieval_service" in content
    print(f"  [{'PASS' if has_service else 'FAIL'}] retrieval_service imported")

    has_init = "retrieval_service.init()" in content
    print(f"  [{'PASS' if has_init else 'FAIL'}] retrieval_service.init() at startup")

    has_search_by_image = "/search-by-image" in content
    print(f"  [{'PASS' if has_search_by_image else 'FAIL'}] /search-by-image endpoint exists")

    return all_pass and has_service and has_init and has_search_by_image


# ─── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    results["init"] = test_service_init()
    results["category"] = test_category_retrieval()
    results["filters"] = test_filtered_queries()
    results["performance"] = test_performance()
    results["no_legacy"] = test_no_legacy_deps()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
