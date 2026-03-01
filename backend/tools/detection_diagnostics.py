"""
Phase 3 — Detection Diagnostics Report
========================================
Run 100 test images through /search-by-image and produce structured
failure analysis with confusion matrix.

Requirements:
  - Backend server MUST be running on http://localhost:5000

Reports:
  - No detection %
  - Wrong class %
  - Low confidence %
  - Result mismatch %
  - Average confidence score
  - Most confused class pairs
  - Structured JSON + summary table
"""
import os
import sys
import json
import time
import base64
import urllib.request
from collections import defaultdict

BASE_URL = "http://localhost:5000"

# Categories to test — pull more per category for 100 total
# 18 categories: aim for ~6 per category = 108 total, cap at 100
IMAGES_PER_CATEGORY = 6


def post_json(endpoint, data, timeout=120):
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


def download_image_b64(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        img_bytes = resp.read()
        return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def get_test_images():
    """Get ~6 images per category from local DB for ~100 total."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "shopwhatyousee"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres123@"),
    )
    cur = conn.cursor()

    # Get all categories
    cur.execute("SELECT DISTINCT category FROM visual_attributes ORDER BY category")
    categories = [r[0] for r in cur.fetchall()]

    test_set = {}
    total = 0
    for cat in categories:
        cur.execute(
            "SELECT product_id, category, image_url FROM visual_attributes "
            "WHERE category = %s AND image_url IS NOT NULL "
            "ORDER BY RANDOM() LIMIT %s",
            (cat, IMAGES_PER_CATEGORY)
        )
        rows = cur.fetchall()
        test_set[cat] = [
            {"product_id": r[0], "category": r[1], "image_url": r[2]}
            for r in rows
        ]
        total += len(rows)

    cur.close()
    conn.close()
    return test_set, total


def main():
    print("=" * 60)
    print("PHASE 3: DETECTION DIAGNOSTICS REPORT")
    print("=" * 60)

    # Check server
    print("\nChecking server...")
    try:
        r = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=5)
        print("  Server is running")
    except Exception:
        print("  FATAL: Server not responding")
        sys.exit(1)

    # Load test images
    print("\nLoading test images...")
    test_images, total_images = get_test_images()
    print(f"  Got {total_images} images across {len(test_images)} categories")

    # Run diagnostics
    results = []
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[expected][detected]
    yolo_confs = []
    no_detect = 0
    wrong_class = 0
    low_conf = 0
    result_mismatch = 0
    total_pass = 0
    total_fail = 0
    skipped = 0
    tested = 0

    for cat, images in test_images.items():
        for i, img_info in enumerate(images):
            if tested >= 100:
                break

            pid = img_info["product_id"]
            b64 = download_image_b64(img_info["image_url"])
            if not b64:
                skipped += 1
                continue

            tested += 1
            status, body, elapsed = post_json("/search-by-image", {
                "image": b64, "top_k": 5
            })

            if status == 400:
                # No detection
                no_detect += 1
                total_fail += 1
                confusion[cat]["NO_DETECTION"] += 1
                results.append({
                    "pid": pid, "expected": cat,
                    "detected": None, "conf": 0,
                    "status": "NO_DETECTION", "latency_ms": round(elapsed, 1)
                })
                print(f"  [{tested}] NO_DETECTION: expected={cat} ({elapsed:.0f}ms)")
                continue

            if status != 200:
                total_fail += 1
                results.append({
                    "pid": pid, "expected": cat,
                    "detected": None, "conf": 0,
                    "status": f"HTTP_{status}", "latency_ms": round(elapsed, 1)
                })
                print(f"  [{tested}] HTTP_{status}: expected={cat} ({elapsed:.0f}ms)")
                continue

            detected_cat = body.get("detected_category", "?")
            det_conf = body.get("detection_confidence", 0)
            products = body.get("products", [])
            top5_cats = [p.get("category") for p in products[:5]]
            yolo_confs.append(det_conf)

            # Update confusion matrix
            confusion[cat][detected_cat] += 1

            # Classify failure type (case-insensitive comparison)
            if det_conf < 0.5:
                fail_type = "LOW_CONFIDENCE"
                low_conf += 1
                total_fail += 1
            elif detected_cat.lower() != cat.lower():
                fail_type = "WRONG_CLASS"
                wrong_class += 1
                total_fail += 1
            elif cat.lower() not in [c.lower() for c in top5_cats if c]:
                fail_type = "RESULT_MISMATCH"
                result_mismatch += 1
                total_fail += 1
            else:
                fail_type = "PASS"
                total_pass += 1

            results.append({
                "pid": pid, "expected": cat,
                "detected": detected_cat, "conf": round(det_conf, 3),
                "top5_categories": top5_cats,
                "status": fail_type, "latency_ms": round(elapsed, 1)
            })

            marker = "✅" if fail_type == "PASS" else "❌"
            print(f"  [{tested}] {marker} {fail_type}: "
                  f"expected={cat}, detected={detected_cat}({det_conf:.2f}) "
                  f"({elapsed:.0f}ms)")

        if tested >= 100:
            break

    # ── Compute metrics ──────────────────────────────────────────
    avg_conf = sum(yolo_confs) / len(yolo_confs) if yolo_confs else 0

    # Find most confused class pairs
    confused_pairs = []
    for expected, detected_map in confusion.items():
        for detected, count in detected_map.items():
            if detected != expected and detected != "NO_DETECTION":
                confused_pairs.append((expected, detected, count))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # ── Build report ─────────────────────────────────────────────
    report = {
        "total_tested": tested,
        "total_skipped": skipped,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "pass_rate_pct": round(total_pass / tested * 100, 1) if tested else 0,
        "no_detection": no_detect,
        "no_detection_pct": round(no_detect / tested * 100, 1) if tested else 0,
        "wrong_class": wrong_class,
        "wrong_class_pct": round(wrong_class / tested * 100, 1) if tested else 0,
        "low_confidence": low_conf,
        "low_confidence_pct": round(low_conf / tested * 100, 1) if tested else 0,
        "result_mismatch": result_mismatch,
        "result_mismatch_pct": round(result_mismatch / tested * 100, 1) if tested else 0,
        "avg_confidence": round(avg_conf, 3),
        "most_confused_pairs": [
            {"expected": e, "detected": d, "count": c}
            for e, d, c in confused_pairs[:10]
        ],
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "detailed_results": results,
    }

    # ── Print summary table ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("DETECTION DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Total tested:       {tested}")
    print(f"  Pass rate:          {report['pass_rate_pct']}%")
    print(f"  No detection:       {no_detect}/{tested} ({report['no_detection_pct']}%)")
    print(f"  Wrong class:        {wrong_class}/{tested} ({report['wrong_class_pct']}%)")
    print(f"  Low confidence:     {low_conf}/{tested} ({report['low_confidence_pct']}%)")
    print(f"  Result mismatch:    {result_mismatch}/{tested} ({report['result_mismatch_pct']}%)")
    print(f"  Avg confidence:     {report['avg_confidence']}")

    if confused_pairs:
        print(f"\n  Most confused class pairs:")
        for e, d, c in confused_pairs[:5]:
            print(f"    {e} → {d}: {c} times")

    print("\n  Confusion Matrix (top rows):")
    print(f"  {'Expected':<20} {'Detected':<20} {'Count'}")
    print(f"  {'-'*20} {'-'*20} {'-'*5}")
    for expected in sorted(confusion.keys()):
        for detected, count in sorted(confusion[expected].items(),
                                       key=lambda x: -x[1]):
            print(f"  {expected:<20} {detected:<20} {count}")

    print("=" * 60)

    # ── Save JSON ────────────────────────────────────────────────
    with open("tools/detection_diagnostics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: tools/detection_diagnostics_report.json")


if __name__ == "__main__":
    main()
