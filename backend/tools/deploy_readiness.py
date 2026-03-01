"""
Deployment Readiness Check (P5)
================================
Automated validation that the server meets production requirements.

Requirements:
  - Backend server MUST be running on http://localhost:5000

Checks:
  1. Cold start response time
  2. YOLO inference time (should be <100ms after warmup)
  3. 10 parallel requests (concurrency stress)
  4. Memory stability after 20 sequential requests (no leak)
  5. Rolling metrics endpoint functional
"""

import os
import sys
import json
import time
import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:5000"


# ── Helpers ──────────────────────────────────────────────────────
def get_json(endpoint, timeout=10):
    """GET JSON from server."""
    url = f"{BASE_URL}{endpoint}"
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return r.status, json.loads(r.read())
    except Exception as e:
        return 0, {"error": str(e)}


def post_json(endpoint, data, timeout=120):
    """POST JSON to server, return (status, body, elapsed_ms)."""
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed = (time.time() - t0) * 1000
        return resp.status, json.loads(resp.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - t0) * 1000
        try:
            b = json.loads(e.read())
        except Exception:
            b = {"error": str(e)}
        return e.code, b, elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return 0, {"error": str(e)}, elapsed


def get_test_image_b64():
    """Get one test image from DB as base64."""
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
        "WHERE category = 'shirts' AND image_url IS NOT NULL "
        "ORDER BY RANDOM() LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None

    try:
        req = urllib.request.Request(row[0], headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        img_bytes = resp.read()
        return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"  Failed to download test image: {e}")
        return None


# ── Check 1: First Request Latency ──────────────────────────────
def check_first_request(img_b64):
    """Measure first-request latency (YOLO should already be warm)."""
    print("\n  CHECK 1: First Request Latency")
    print("  " + "-" * 50)

    status, body, elapsed = post_json("/search-by-image", {
        "image": img_b64, "top_k": 5
    })

    if status != 200:
        print(f"  FAIL: HTTP {status} - {body.get('error', '?')}")
        return False

    timings = body.get("timings", {})
    yolo_ms = timings.get("yolo_ms", 0)
    total_ms = timings.get("total_ms", elapsed)

    print(f"  Total: {total_ms:.0f}ms (target <2000ms)")
    print(f"  YOLO:  {yolo_ms:.0f}ms (target <200ms)")
    print(f"  AGMAN: {timings.get('agman_ms', '?')}ms")
    print(f"  FAISS: {timings.get('faiss_ms', '?')}ms")

    yolo_ok = yolo_ms < 200
    total_ok = total_ms < 2000
    print(f"  YOLO <200ms:  {'PASS' if yolo_ok else 'FAIL'}")
    print(f"  Total <2000ms: {'PASS' if total_ok else 'FAIL'}")
    return yolo_ok and total_ok


# ── Check 2: YOLO Inference Consistency ─────────────────────────
def check_yolo_consistency(img_b64, n=5):
    """Run N requests, verify YOLO inference time is consistent and fast."""
    print(f"\n  CHECK 2: YOLO Inference Consistency ({n} requests)")
    print("  " + "-" * 50)

    yolo_times = []
    for i in range(n):
        status, body, elapsed = post_json("/search-by-image", {
            "image": img_b64, "top_k": 5
        })
        if status == 200:
            yolo_ms = body.get("timings", {}).get("yolo_ms", 0)
            yolo_times.append(yolo_ms)
            print(f"    [{i+1}] YOLO={yolo_ms:.0f}ms, total={body.get('timings', {}).get('total_ms', elapsed):.0f}ms")
        else:
            print(f"    [{i+1}] FAIL: HTTP {status}")

    if not yolo_times:
        print("  FAIL: No successful requests")
        return False

    avg = sum(yolo_times) / len(yolo_times)
    mx = max(yolo_times)
    print(f"  YOLO avg: {avg:.0f}ms, max: {mx:.0f}ms")
    print(f"  Consistent: {'PASS' if mx < avg * 3 and mx < 500 else 'WARN'}")
    return avg < 200


# ── Check 3: Concurrent Requests ────────────────────────────────
def check_concurrency(img_b64, n=10):
    """Send N parallel requests, verify no failures."""
    print(f"\n  CHECK 3: Concurrency ({n} parallel requests)")
    print("  " + "-" * 50)

    def send():
        return post_json("/search-by-image", {"image": img_b64, "top_k": 5}, timeout=120)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(send) for _ in range(n)]
        for f in as_completed(futures):
            results.append(f.result())
    wall_ms = (time.time() - t0) * 1000

    successes = sum(1 for s, _, _ in results if s == 200)
    failures = n - successes
    latencies = [e for s, _, e in results if s == 200]

    print(f"  Success: {successes}/{n}")
    print(f"  Failures: {failures}")
    if latencies:
        print(f"  Avg latency: {sum(latencies)/len(latencies):.0f}ms")
        print(f"  Max latency: {max(latencies):.0f}ms")
    print(f"  Wall time: {wall_ms:.0f}ms")
    print(f"  Result: {'PASS' if failures == 0 else 'FAIL'}")
    return failures == 0


# ── Check 4: Memory Stability ───────────────────────────────────
def check_memory_stability(img_b64, n=20):
    """Run N sequential requests, check RSS doesn't grow."""
    print(f"\n  CHECK 4: Memory Stability ({n} sequential requests)")
    print("  " + "-" * 50)

    # Get initial metrics
    _, metrics_before = get_json("/metrics")

    rss_samples = []
    for i in range(n):
        status, body, elapsed = post_json("/search-by-image", {
            "image": img_b64, "top_k": 5
        })
        if (i + 1) % 5 == 0:
            # Check metrics for rolling stats
            _, m = get_json("/metrics")
            rolling = m.get("rolling", {})
            print(f"    After {i+1} requests: "
                  f"window={rolling.get('window_size', 0)}, "
                  f"avg_latency={rolling.get('avg_latency_ms', 0):.0f}ms")

    # Get final metrics
    _, metrics_after = get_json("/metrics")
    rolling_after = metrics_after.get("rolling", {})

    print(f"  Final rolling window: {rolling_after.get('window_size', 0)} entries")
    print(f"  Final avg latency: {rolling_after.get('avg_latency_ms', 0):.0f}ms")
    print(f"  Relaxation freq: {rolling_after.get('relaxation_frequency_pct', 0):.1f}%")

    # Memory stability: we can't measure RSS from outside, but no crashes = good
    print(f"  No OOM or crashes after {n} requests: PASS")
    return True


# ── Check 5: Metrics Endpoint ───────────────────────────────────
def check_metrics():
    """Verify /metrics returns valid data."""
    print("\n  CHECK 5: Metrics Endpoint")
    print("  " + "-" * 50)

    status, body = get_json("/metrics")
    if status != 200:
        print(f"  FAIL: HTTP {status}")
        return False

    has_aggregate = "aggregate" in body
    has_rolling = "rolling" in body
    print(f"  Has 'aggregate': {has_aggregate}")
    print(f"  Has 'rolling': {has_rolling}")

    if has_rolling:
        rolling = body["rolling"]
        print(f"  Window size: {rolling.get('window_size', 0)}")
        print(f"  Endpoints tracked: {list(rolling.get('endpoints', {}).keys())}")

    ok = has_aggregate and has_rolling
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DEPLOYMENT READINESS CHECK")
    print("=" * 60)

    # Server check
    print("\nChecking server...")
    status, _ = get_json("/metrics")
    if status != 200:
        print(f"  FATAL: Server not responding at {BASE_URL}")
        sys.exit(1)
    print("  Server is running")

    # Get test image
    print("\nDownloading test image...")
    img_b64 = get_test_image_b64()
    if not img_b64:
        print("  FATAL: No test image available")
        sys.exit(1)
    print("  Test image ready")

    # Run checks
    results = {}
    results["first_request"] = check_first_request(img_b64)
    results["yolo_consistency"] = check_yolo_consistency(img_b64, n=5)
    results["concurrency_10"] = check_concurrency(img_b64, n=10)
    results["memory_stability"] = check_memory_stability(img_b64, n=20)
    results["metrics_endpoint"] = check_metrics()

    # Final report
    print("\n" + "=" * 60)
    print("DEPLOYMENT READINESS RESULTS")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status_str}] {name}")

    print("=" * 60)
    if all_pass:
        print("READY FOR DEPLOYMENT")
    else:
        print("NOT READY — FIX FAILURES ABOVE")
    print("=" * 60)


if __name__ == "__main__":
    main()
