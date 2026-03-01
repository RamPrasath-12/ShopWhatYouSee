"""
test_retrieval_v2.py — Structured validation of the SQL-FIRST retrieval pipeline.

17 test cases across 6 categories + mapping validation:
  MAP — Category Mapping Unit Test
  A   — Pure Similarity (3 tests)
  B   — Color Override (3 tests)
  C   — Sleeve Override (2 tests)
  D   — Category Transform (2 tests)
  E   — Combined Overrides (3 tests)
  F   — Progressive Relaxation (2 tests)

Usage:
  python test_retrieval_v2.py           # run all tests
  python test_retrieval_v2.py A         # run only category A
  python test_retrieval_v2.py MAP D     # run mapping + category D
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from models.product_retrieval import search_products_v2, normalize_category, get_db

# =========================================================================
# Helpers
# =========================================================================

def _get_real_embedding(db_category):
    """Grab a real embedding from the DB for testing."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT embedding FROM visual_attributes
            WHERE LOWER(category) = LOWER(%s) AND embedding IS NOT NULL
            LIMIT 1
        """, [db_category])
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            emb = row[0]
            if isinstance(emb, str):
                emb = json.loads(emb)
            return emb
    except Exception as e:
        print(f"  [WARN] Could not fetch embedding for '{db_category}': {e}")
    return [0.0] * 512


PASS = 0
FAIL = 0
RESULTS = []


def run_test(test_id, description, detected_attrs, user_filters, embedding,
             expected_category=None, expected_color=None, expected_sleeve=None,
             min_pool_size=1, check_no_original_color=None,
             expect_relaxation=False, expect_hard_fields_kept=None):
    """Run one retrieval test and validate the results."""
    global PASS, FAIL

    print(f"\n{'='*70}")
    print(f"  TEST {test_id}: {description}")
    print(f"{'='*70}")
    print(f"  detected_attrs: {detected_attrs}")
    print(f"  user_filters:   {user_filters}")

    # Build query_context as search_products_v2 expects
    query_context = {
        "category": detected_attrs.get("category", ""),
        "embedding": embedding,
        "detected_attributes": detected_attrs,
        "user_filters": user_filters,
    }

    t0 = time.time()
    try:
        result = search_products_v2(query_context, top_k=20)
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1
        RESULTS.append((test_id, "FAIL", f"Exception: {e}"))
        return
    elapsed = time.time() - t0

    products = result.get("products", [])
    metadata = result.get("metadata", {})
    pool_size = metadata.get("pool_size", 0)
    mode = metadata.get("mode", "?")
    relaxation = metadata.get("relaxation_log", [])

    print(f"  Mode: {mode}  |  Pool: {pool_size}  |  Returned: {len(products)}  |  Time: {elapsed:.1f}s")
    if relaxation:
        print(f"  Relaxation steps: {relaxation}")

    errors = []

    # --- Pool size check ---
    if pool_size < min_pool_size:
        errors.append(f"Pool size {pool_size} < expected min {min_pool_size}")

    # --- Category check ---
    if expected_category and products:
        norm_expected = normalize_category(expected_category)
        bad_cats = [p for p in products if p.get("category", "").lower() != norm_expected.lower()]
        if bad_cats:
            actual_cats = set(p.get("category", "").lower() for p in products)
            errors.append(f"Category mismatch: expected '{norm_expected}', got {actual_cats}")

    # --- Color check ---
    if expected_color and products:
        expected_lower = expected_color.lower()
        bad_color = []
        for p in products[:10]:
            pc = (p.get("color") or "").lower()
            cf = (p.get("color_family") or "").lower()
            if expected_lower not in pc and expected_lower not in cf:
                bad_color.append(f"{p.get('product_id')}: color={pc}, family={cf}")
        if len(bad_color) > len(products[:10]) * 0.5:  # >50% wrong = fail
            errors.append(f"Color mismatch: expected '{expected_color}', {len(bad_color)}/10 wrong")

    # --- Sleeve check ---
    if expected_sleeve and products:
        expected_sl = expected_sleeve.lower()
        bad_sleeve = [p for p in products[:10] if expected_sl not in (p.get("sleeve") or "").lower()]
        if len(bad_sleeve) > len(products[:10]) * 0.5:
            errors.append(f"Sleeve mismatch: expected '{expected_sleeve}', {len(bad_sleeve)}/10 wrong")

    # --- No original color leakage ---
    if check_no_original_color and products:
        orig_lower = check_no_original_color.lower()
        leaked = [p for p in products[:5]
                  if orig_lower in (p.get("color") or "").lower()
                  or orig_lower in (p.get("color_family") or "").lower()]
        if len(leaked) > 1:
            errors.append(f"Original color '{check_no_original_color}' leaked into {len(leaked)}/5 results")

    # --- Relaxation check ---
    if expect_relaxation and not relaxation:
        errors.append("Expected relaxation to trigger but it didn't")

    # --- Hard fields never relaxed ---
    if expect_hard_fields_kept and relaxation:
        for field in expect_hard_fields_kept:
            for step in relaxation:
                if field.lower() in step.lower():
                    errors.append(f"Hard field '{field}' was relaxed: {step}")

    # --- Verdict ---
    if errors:
        FAIL += 1
        status = "FAIL"
        print(f"  ❌ FAIL:")
        for e in errors:
            print(f"     • {e}")
    else:
        PASS += 1
        status = "PASS"
        print(f"  ✅ PASS")

    if products:
        top5 = [(p.get("product_id","?")[:8], p.get("category","?"), p.get("color","?"), p.get("sleeve","?")) for p in products[:5]]
        print(f"  Top-5: {top5}")

    RESULTS.append((test_id, status, "; ".join(errors) if errors else "OK"))


# =========================================================================
# TEST DEFINITIONS
# =========================================================================

def test_A_pure_similarity():
    """A — Pure Similarity: detected item, no user query overrides."""

    emb = _get_real_embedding("shirts")
    run_test("A1", "Shirt → no query (pure similarity)",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts"},
             embedding=emb,
             expected_category="shirts", min_pool_size=100)

    emb = _get_real_embedding("tshirt")
    run_test("A2", "Tshirt → no query (pure similarity)",
             detected_attrs={"category": "tshirt", "color_name": "Black", "sleeve": "short", "pattern": "", "gender": ""},
             user_filters={"category": "tshirt"},
             embedding=emb,
             expected_category="tshirt", min_pool_size=1)

    emb = _get_real_embedding("blazer")
    run_test("A3", "Blazer → no query (pure similarity)",
             detected_attrs={"category": "blazer", "color_name": "Black", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "blazer"},
             embedding=emb,
             expected_category="blazer", min_pool_size=1)


def test_B_color_override():
    """B — Color Override: user requests a different color."""
    emb = _get_real_embedding("shirts")

    run_test("B1", "Blue shirt → red override",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "color": "red", "color_family": "red"},
             embedding=emb,
             expected_category="shirts", expected_color="red",
             check_no_original_color="Navy Blue", min_pool_size=1)

    run_test("B2", "Blue shirt → yellow override",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "color": "yellow", "color_family": "yellow"},
             embedding=emb,
             expected_category="shirts", expected_color="yellow",
             check_no_original_color="Navy Blue", min_pool_size=1)

    run_test("B3", "Blue shirt → black override",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "color": "black", "color_family": "black"},
             embedding=emb,
             expected_category="shirts", expected_color="black",
             check_no_original_color="Navy Blue", min_pool_size=1)


def test_C_sleeve_override():
    """C — Sleeve Override: user requests different sleeve length."""
    emb = _get_real_embedding("shirts")

    run_test("C1", "Full sleeve shirt → half sleeve override",
             detected_attrs={"category": "shirts", "color_name": "White", "sleeve": "long", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "sleeve_value": "short"},
             embedding=emb,
             expected_category="shirts", expected_sleeve="short", min_pool_size=1)

    run_test("C2", "Half sleeve shirt → full sleeve override",
             detected_attrs={"category": "shirts", "color_name": "White", "sleeve": "short", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "sleeve_value": "long"},
             embedding=emb,
             expected_category="shirts", expected_sleeve="long", min_pool_size=1)


def test_D_category_transform():
    """D — Category Transform: user asks for an entirely different category."""
    emb = _get_real_embedding("shirts")

    run_test("D1", "Shirt → T-shirt category transform",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "tshirt"},
             embedding=emb,
             expected_category="tshirt", min_pool_size=1)

    run_test("D2", "Shirt → Blazer category transform",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "blazer"},
             embedding=emb,
             expected_category="blazer", min_pool_size=1)


def test_E_combined():
    """E — Combined: category + attribute overrides together."""
    emb = _get_real_embedding("shirts")

    run_test("E1", "Shirt → Red T-shirt (category + color)",
             detected_attrs={"category": "shirts", "color_name": "Navy Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "tshirt", "color": "red", "color_family": "red"},
             embedding=emb,
             expected_category="tshirt", expected_color="red", min_pool_size=1)

    emb_b = _get_real_embedding("blazer")
    run_test("E2", "Blazer → Black half sleeve (color + sleeve)",
             detected_attrs={"category": "blazer", "color_name": "Grey", "sleeve": "long", "pattern": "", "gender": ""},
             user_filters={"category": "blazer", "color": "black", "color_family": "black", "sleeve_value": "short"},
             embedding=emb_b,
             expected_category="blazer", expected_color="black", min_pool_size=1)

    run_test("E3", "Shirt → Red under 1000 (color + price)",
             detected_attrs={"category": "shirts", "color_name": "Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "color": "red", "color_family": "red", "price_bucket": "budget"},
             embedding=emb,
             expected_category="shirts", expected_color="red", min_pool_size=1)


def test_F_relaxation():
    """F — Progressive Relaxation: impossible combo should relax SOFT fields only."""
    emb = _get_real_embedding("shirts")

    run_test("F1", "Shirt → neon green silk under 100 (should relax price/material)",
             detected_attrs={"category": "shirts", "color_name": "Blue", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "shirts", "color": "green", "color_family": "green",
                           "material": "silk", "price_bucket": "budget"},
             embedding=emb,
             expected_category="shirts",
             expect_relaxation=True,
             expect_hard_fields_kept=["category", "color"])

    emb_b = _get_real_embedding("blazer")
    run_test("F2", "Blazer → purple velvet under 500 (should relax)",
             detected_attrs={"category": "blazer", "color_name": "Black", "sleeve": "", "pattern": "", "gender": ""},
             user_filters={"category": "blazer", "color": "purple", "color_family": "purple",
                           "material": "velvet", "price_bucket": "budget"},
             embedding=emb_b,
             expected_category="blazer",
             expect_relaxation=True,
             expect_hard_fields_kept=["category", "color"])


# =========================================================================
# Category mapping unit tests
# =========================================================================

def test_category_mapping():
    """Validate normalize_category returns exact DB values."""
    print(f"\n{'='*70}")
    print(f"  TEST MAP: Category Mapping Validation")
    print(f"{'='*70}")

    global PASS, FAIL
    expected = {
        "shirt": "shirts", "shirts": "shirts",
        "tshirt": "tshirt", "t-shirt": "tshirt", "t_shirt": "tshirt", "tee": "tshirt",
        "blazer": "blazer", "jacket": "Jacket", "Jacket": "Jacket",
        "pant": "pant", "pants": "pant",
        "shorts": "shorts", "cap": "caps", "glasses": "glasses", "glass": "glasses",
        "earring": "earrings", "necklace": "necklace", "watch": "watch", "belt": "belt",
    }

    errors = []
    for input_cat, expected_val in expected.items():
        actual = normalize_category(input_cat)
        if actual != expected_val:
            errors.append(f"  normalize_category('{input_cat}') = '{actual}', expected '{expected_val}'")

    if errors:
        FAIL += 1
        print("  ❌ FAIL:")
        for e in errors:
            print(f"     {e}")
        RESULTS.append(("MAP", "FAIL", f"{len(errors)} mappings wrong"))
    else:
        PASS += 1
        print(f"  ✅ PASS — all {len(expected)} mappings correct")
        RESULTS.append(("MAP", "PASS", "OK"))


# =========================================================================
# MAIN
# =========================================================================

ALL_TESTS = {
    "MAP": test_category_mapping,
    "A": test_A_pure_similarity,
    "B": test_B_color_override,
    "C": test_C_sleeve_override,
    "D": test_D_category_transform,
    "E": test_E_combined,
    "F": test_F_relaxation,
}

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RETRIEVAL V2 — STRUCTURED TEST SUITE")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    args = sys.argv[1:]
    if not args:
        tests_to_run = list(ALL_TESTS.keys())
    else:
        tests_to_run = []
        for a in args:
            a_upper = a.upper()
            if a_upper in ALL_TESTS:
                tests_to_run.append(a_upper)
            else:
                for key in ALL_TESTS:
                    if a_upper.startswith(key):
                        tests_to_run.append(key)
                        break

    # Deduplicate
    seen = set()
    unique = [t for t in tests_to_run if t not in seen and not seen.add(t)]

    t_start = time.time()
    for test_key in unique:
        ALL_TESTS[test_key]()

    total_time = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {PASS + FAIL} tests, {PASS} passed, {FAIL} failed ({total_time:.1f}s)")
    print(f"{'='*70}")
    for tid, status, detail in RESULTS:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {tid:5s} {status:4s}  {detail}")
    print(f"{'='*70}\n")

    sys.exit(1 if FAIL > 0 else 0)
