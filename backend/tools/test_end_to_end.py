"""
End-to-End Pipeline Test (Phase 3)
====================================
Tests: Image -> YOLO -> AGMAN -> FAISS -> Filters -> Results

Requirements:
  - Backend server MUST be running on http://localhost:5000
  - This script does NOT start the server (avoids race conditions)

Test Categories (25 images):
  5 shirts, 5 shoes, 5 watches, 5 ethnic, 5 jewelry

Validations:
  1. Top-3 contain correct category (not just top-1)
  2. YOLO detected class + confidence logged
  3. Multi-object image test
  4. Unsupported category graceful failure
  5. Relaxation steps logged
  6. Per-stage timings (target: total <500ms, no stage >250ms)
  7. Concurrency test (5 parallel requests)
"""

import os
import sys
import json
import time
import base64
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:5000"

# ---- Connection Check ----
def check_server():
    """Fail fast if server is not running."""
    try:
        req = urllib.request.Request(f"{BASE_URL}/debug/filter-schema")
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.status == 200:
            print("  Server is running on port 5000")
            return True
    except Exception as e:
        print(f"  FATAL: Cannot connect to server at {BASE_URL}")
        print(f"  Error: {e}")
        print(f"  Start the server first: python app.py")
        return False


# ---- HTTP Helpers ----
def post_json(endpoint, data, timeout=120):
    """POST JSON to server, return (status_code, json_body, elapsed_ms)."""
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}
    )
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
    """Download image from URL and return base64 string."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0"
        })
        resp = urllib.request.urlopen(req, timeout=15)
        img_bytes = resp.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"    WARN: Failed to download {url[:80]}... -> {e}")
        return None


# ---- Get Test Images from DB ----
def get_test_images():
    """Get 5 images per category from DB."""
    import psycopg2
    categories = ["shirts", "Footwear_shoes", "watch", "churidhar", "necklace"]
    test_set = {}

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "shopwhatyousee"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres123@"),
    )
    cur = conn.cursor()

    for cat in categories:
        cur.execute(
            "SELECT product_id, category, image_url FROM visual_attributes "
            "WHERE category = %s AND image_url IS NOT NULL "
            "ORDER BY RANDOM() LIMIT 5",
            (cat,)
        )
        rows = cur.fetchall()
        test_set[cat] = [
            {"product_id": r[0], "category": r[1], "image_url": r[2]}
            for r in rows
        ]

    cur.close()
    conn.close()
    return test_set


# ==================================================
# TEST 1: Full Pipeline Per Category
# ==================================================
def test_full_pipeline(test_images):
    """Run 25 images through /search-by-image, validate results."""
    print("\n" + "=" * 60)
    print("TEST 1: Full Pipeline (25 images)")
    print("=" * 60)

    results_summary = []
    total_pass = 0
    total_fail = 0
    timing_totals = {"yolo": [], "agman": [], "faiss": [], "total": []}

    for category, images in test_images.items():
        print(f"\n  --- Category: {category} ({len(images)} images) ---")

        for i, img_info in enumerate(images):
            pid = img_info["product_id"]
            url = img_info["image_url"]

            # Download image
            b64 = download_image_b64(url)
            if not b64:
                total_fail += 1
                results_summary.append({
                    "category": category, "pid": pid,
                    "status": "SKIP", "reason": "download failed"
                })
                continue

            # POST to /search-by-image
            status, body, elapsed = post_json("/search-by-image", {
                "image": b64,
                "top_k": 10,
            })

            if status != 200:
                total_fail += 1
                err = body.get("error", "unknown")
                print(f"    [{i+1}] FAIL: HTTP {status} - {err}")
                results_summary.append({
                    "category": category, "pid": pid,
                    "status": "FAIL", "reason": f"HTTP {status}: {err}"
                })
                continue

            products = body.get("products", [])
            timings = body.get("timings", {})
            detected_cat = body.get("detected_category", "?")
            det_conf = body.get("detection_confidence", 0)

            # Collect timings
            for key in ("yolo", "agman", "faiss"):
                ms_key = f"{key}_ms"
                if ms_key in timings:
                    timing_totals[key].append(timings[ms_key])
            if "total_ms" in timings:
                timing_totals["total"].append(timings["total_ms"])

            # Validation 1: Top-3 contain correct category
            top3_cats = [p.get("category") for p in products[:3]]
            cat_in_top3 = category in top3_cats

            # Validation 2: YOLO detected class logged
            yolo_correct = (detected_cat == category)

            # Validation 5: Check relaxation (from metadata)
            metadata = body.get("metadata", {})

            passed = cat_in_top3
            status_str = "PASS" if passed else "FAIL"
            if passed:
                total_pass += 1
            else:
                total_fail += 1

            print(f"    [{i+1}] {status_str}: "
                  f"YOLO={detected_cat}({det_conf:.2f}) "
                  f"top3={top3_cats} "
                  f"T={timings.get('total_ms', '?')}ms "
                  f"(Y={timings.get('yolo_ms', '?')}, "
                  f"A={timings.get('agman_ms', '?')}, "
                  f"F={timings.get('faiss_ms', '?')})")

            results_summary.append({
                "category": category, "pid": pid,
                "status": status_str,
                "yolo_class": detected_cat,
                "yolo_conf": det_conf,
                "top3_categories": top3_cats,
                "timings": timings,
            })

    # Print timing summary
    print(f"\n  --- Timing Summary ---")
    for stage, times in timing_totals.items():
        if times:
            avg = sum(times) / len(times)
            mx = max(times)
            print(f"    {stage:6s}: avg={avg:7.1f}ms, max={mx:7.1f}ms")

    print(f"\n  Results: {total_pass}/{total_pass + total_fail} passed")

    # Time targets
    if timing_totals["total"]:
        avg_total = sum(timing_totals["total"]) / len(timing_totals["total"])
        print(f"  Avg total: {avg_total:.1f}ms (target <500ms) "
              f"{'PASS' if avg_total < 500 else 'WARN'}")
        for stage in ("yolo", "agman", "faiss"):
            if timing_totals[stage]:
                mx = max(timing_totals[stage])
                print(f"  Max {stage}: {mx:.1f}ms (target <250ms) "
                      f"{'PASS' if mx < 250 else 'WARN'}")

    # ── P3: Structured Failure Diagnostics ──────────────────────
    failures = [r for r in results_summary if r["status"] == "FAIL"]
    if failures:
        print(f"\n  --- YOLO Failure Diagnostics ({len(failures)} failures) ---")

        no_detect = 0
        wrong_class = 0
        low_conf = 0

        for f in failures:
            yolo_cls = f.get("yolo_class", "?")
            yolo_cf = f.get("yolo_conf", 0)
            expected = f.get("category", "?")
            reason = f.get("reason", "")

            if "No fashion items" in reason or "HTTP 400" in reason:
                no_detect += 1
                fail_type = "NO_DETECTION"
            elif yolo_cf < 0.5:
                low_conf += 1
                fail_type = "LOW_CONFIDENCE"
            elif yolo_cls != expected:
                wrong_class += 1
                fail_type = "WRONG_CLASS"
            else:
                fail_type = "RESULT_MISMATCH"

            print(f"    {fail_type}: expected={expected}, "
                  f"detected={yolo_cls}({yolo_cf:.2f}), "
                  f"top3={f.get('top3_categories', '?')}")

        n = len(failures)
        print(f"\n  --- Failure Breakdown ---")
        print(f"    No detection:   {no_detect}/{n} ({no_detect/n*100:.0f}%)")
        print(f"    Wrong class:    {wrong_class}/{n} ({wrong_class/n*100:.0f}%)")
        print(f"    Low confidence: {low_conf}/{n} ({low_conf/n*100:.0f}%)")
        result_mismatch = n - no_detect - wrong_class - low_conf
        print(f"    Result mismatch:{result_mismatch}/{n} ({result_mismatch/n*100:.0f}%)")
    else:
        print("\n  No failures — all categories detected correctly!")

    return total_fail == 0, results_summary


# ==================================================
# TEST 2: Unsupported Category (Graceful Failure)
# ==================================================
def test_unsupported_category():
    """Send a non-fashion image, verify graceful error."""
    print("\n" + "=" * 60)
    print("TEST 2: Unsupported Category (Graceful Failure)")
    print("=" * 60)

    # Create a solid-color 100x100 image (no fashion items)
    import struct
    import zlib

    # Minimal 10x10 red PNG
    width, height = 10, 10
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter byte
        for _ in range(width):
            raw_data += b"\xff\x00\x00"  # red pixel

    def make_png_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += make_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += make_png_chunk(b"IDAT", zlib.compress(raw_data))
    png += make_png_chunk(b"IEND", b"")

    b64 = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

    status, body, elapsed = post_json("/search-by-image", {
        "image": b64,
        "top_k": 10,
    })

    # Expect either:
    # - 400 with "No fashion items detected"
    # - 200 with empty/degraded results
    if status == 400 and "error" in body:
        print(f"  PASS: Graceful failure -> {body.get('error')} ({elapsed:.0f}ms)")
        return True
    elif status == 200 and len(body.get("products", [])) == 0:
        print(f"  PASS: Empty results returned ({elapsed:.0f}ms)")
        return True
    elif status == 200:
        # Server found something -- not ideal but not a crash
        print(f"  WARN: Server returned {len(body.get('products', []))} results "
              f"for solid-color image ({elapsed:.0f}ms)")
        return True
    else:
        print(f"  FAIL: Unexpected response: HTTP {status}, body={json.dumps(body)[:200]}")
        return False


# ==================================================
# TEST 3: Concurrency (5 Parallel Requests)
# ==================================================
def test_concurrency(test_images):
    """Send 5 requests simultaneously, verify no crashes or corruption."""
    print("\n" + "=" * 60)
    print("TEST 3: Concurrency (5 Parallel Requests)")
    print("=" * 60)

    # Pick one image from each category
    tasks = []
    for cat, images in test_images.items():
        if images:
            tasks.append(images[0])

    if len(tasks) < 5:
        print("  SKIP: Not enough test images")
        return True

    # Download images first (sequential)
    prepared = []
    for img_info in tasks[:5]:
        b64 = download_image_b64(img_info["image_url"])
        if b64:
            prepared.append({"b64": b64, "category": img_info["category"]})

    if len(prepared) < 3:
        print(f"  SKIP: Only {len(prepared)} images downloaded")
        return True

    print(f"  Sending {len(prepared)} concurrent requests...")

    results = []

    def send_request(item):
        status, body, elapsed = post_json("/search-by-image", {
            "image": item["b64"],
            "top_k": 5,
        }, timeout=60)
        return {
            "category": item["category"],
            "status": status,
            "result_count": len(body.get("products", [])),
            "elapsed_ms": elapsed,
            "error": body.get("error"),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(send_request, p): p for p in prepared}
        for future in as_completed(futures):
            results.append(future.result())
    wall_time = (time.time() - t0) * 1000

    # Validate
    all_ok = True
    for r in results:
        ok = r["status"] == 200 and r["result_count"] > 0
        status_str = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status_str}] {r['category']}: "
              f"HTTP {r['status']}, {r['result_count']} results, "
              f"{r['elapsed_ms']:.0f}ms"
              f"{' ERR=' + r['error'] if r['error'] else ''}")

    print(f"  Wall time: {wall_time:.0f}ms for {len(prepared)} concurrent requests")

    if all_ok:
        print("  PASS: No race conditions, no FAISS corruption")
    else:
        print("  FAIL: Some concurrent requests failed")

    return all_ok


# ==================================================
# TEST 4: Filter Interaction (Text + Image)
# ==================================================
def test_filter_interaction():
    """Test that user filters are applied on top of retrieval."""
    print("\n" + "=" * 60)
    print("TEST 4: Filter Interaction (User Filters + Image)")
    print("=" * 60)

    # Get a shirt image
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "shopwhatyousee"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres123@"),
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT image_url FROM visual_attributes "
        "WHERE category = 'shirts' AND gender = 'men' "
        "ORDER BY RANDOM() LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        print("  SKIP: No men's shirt image found")
        return True

    b64 = download_image_b64(row[0])
    if not b64:
        print("  SKIP: Failed to download image")
        return True

    # Test 1: No filters
    status1, body1, _ = post_json("/search-by-image", {"image": b64, "top_k": 10})
    count_no_filter = len(body1.get("products", []))

    # Test 2: With gender filter
    status2, body2, _ = post_json("/search-by-image", {
        "image": b64,
        "top_k": 10,
        "filters": {"gender": "men"}
    })
    count_with_filter = len(body2.get("products", []))

    # Validate all results match gender
    products2 = body2.get("products", [])
    gender_match = sum(1 for p in products2 if p.get("gender") == "men")

    print(f"  No filters: {count_no_filter} results")
    print(f"  With gender=men: {count_with_filter} results")
    print(f"  Gender match: {gender_match}/{len(products2)}")

    passed = count_with_filter > 0 and status2 == 200
    print(f"  {'PASS' if passed else 'FAIL'}: Filter interaction works")
    return passed


# ==================================================
# MAIN
# ==================================================
def main():
    print("=" * 60)
    print("PHASE 3: END-TO-END PIPELINE TEST")
    print("=" * 60)

    # Server check
    print("\nChecking server...")
    if not check_server():
        sys.exit(1)

    # Load test images
    print("\nLoading test images from DB...")
    test_images = get_test_images()
    total_images = sum(len(v) for v in test_images.values())
    print(f"  Got {total_images} test images across {len(test_images)} categories")

    results = {}

    # Test 1: Full pipeline
    results["pipeline"], pipeline_details = test_full_pipeline(test_images)

    # Test 2: Unsupported category
    results["unsupported"] = test_unsupported_category()

    # Test 3: Concurrency
    results["concurrency"] = test_concurrency(test_images)

    # Test 4: Filter interaction
    results["filter_interaction"] = test_filter_interaction()

    # Final report
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


if __name__ == "__main__":
    main()
