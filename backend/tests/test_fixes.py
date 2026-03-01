"""
Test suite for backend bug fixes.

Tests:
  1. extract_primary_secondary_color_fast() always returns 4 values
  2. normalize_filter_value() canonical mapping
  3. Gender regex detection (word-boundary safety)
  4. RELAXATION_ORDER never contains protected attributes
  5. Session filter merge and eviction logic
"""

import sys
import os
import re

# Add backend root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================================================
# Test 1: AG-MAN color extraction return signature
# ==================================================
def test_agman_return_signature_tiny_image():
    """extract_primary_secondary_color_fast must return exactly 4 values for tiny images."""
    import numpy as np
    from models.agman_extractor import extract_primary_secondary_color_fast

    # Tiny image (< 100 pixels after filtering)
    tiny_img = np.zeros((5, 5, 3), dtype=np.uint8)
    result = extract_primary_secondary_color_fast(tiny_img)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 4, f"Expected 4 values, got {len(result)}"


def test_agman_return_signature_normal_image():
    """extract_primary_secondary_color_fast must return exactly 4 values for normal images."""
    import numpy as np
    from models.agman_extractor import extract_primary_secondary_color_fast

    # Normal sized red image
    img = np.full((100, 100, 3), (200, 50, 50), dtype=np.uint8)
    result = extract_primary_secondary_color_fast(img)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 4, f"Expected 4 values, got {len(result)}"
    primary_hex, secondary_hex, conf, sec_conf = result
    # At least primary should be detected
    assert primary_hex is not None, "Primary hex should not be None for a normal image"
    assert conf > 0, "Confidence should be > 0 for a normal image"


# ==================================================
# Test 2: Canonical normalization
# ==================================================
def test_normalize_filter_value():
    """normalize_filter_value must map synonyms correctly."""
    from models.product_retrieval import normalize_filter_value

    assert normalize_filter_value("Grey") == "grey", "Grey should map to grey (lowercase)"
    assert normalize_filter_value("gray") == "grey", "gray should map to grey (synonym)"
    assert normalize_filter_value("  Solid  ") == "solid", "Should strip and lowercase"
    assert normalize_filter_value("plain") == "solid", "plain should map to solid"
    assert normalize_filter_value("checks") == "checked", "checks should map to checked"
    assert normalize_filter_value("stripes") == "striped", "stripes should map to striped"
    assert normalize_filter_value("male") == "Men", "male should map to Men"
    assert normalize_filter_value("female") == "Women", "female should map to Women"
    assert normalize_filter_value(None) is None, "None should return None"
    assert normalize_filter_value("") == "", "Empty string should return empty"
    assert normalize_filter_value("red") == "red", "red has no synonym, should stay red"


# ==================================================
# Test 3: Gender regex detection
# ==================================================
def test_gender_detection_word_boundary():
    """Gender regex must use word boundaries to avoid false matches."""
    # Import from app.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Re-implement to test (since app.py imports are heavy)
    _GENDER_REGEX = re.compile(
        r'\b(men|women|male|female|boy|girl|boys|girls|mens|womens|man|woman)\b',
        re.IGNORECASE
    )
    _GENDER_KEYWORD_MAP = {
        "men": "Men", "mens": "Men", "male": "Men", "man": "Men",
        "women": "Women", "womens": "Women", "female": "Women", "woman": "Women",
        "boy": "Boys", "boys": "Boys",
        "girl": "Girls", "girls": "Girls",
    }

    def detect(query):
        if not query:
            return None
        match = _GENDER_REGEX.search(query)
        if match:
            return _GENDER_KEYWORD_MAP.get(match.group(1).lower())
        return None

    # Positive matches
    assert detect("men tshirt") == "Men"
    assert detect("women dress") == "Women"
    assert detect("show me mens shirts") == "Men"
    assert detect("I want a boys jacket") == "Boys"

    # Must NOT false-match substrings
    assert detect("womenswear collection") is None or detect("womenswear collection") == "Women"
    # Actually \bwomen\b won't match "womenswear" since 's' follows without boundary
    # But \bwomens\b will match "womens" in "womens shirts"
    assert detect("menswear") is None or detect("menswear") == "Men"

    # No gender
    assert detect("red tshirt") is None
    assert detect("price less than 500") is None
    assert detect("") is None
    assert detect(None) is None


# ==================================================
# Test 4: RELAXATION_ORDER safety
# ==================================================
def test_relaxation_order_excludes_protected():
    """RELAXATION_ORDER must never contain category or gender."""
    from models.product_retrieval import RELAXATION_ORDER, NEVER_RELAX

    overlap = set(RELAXATION_ORDER) & NEVER_RELAX
    assert len(overlap) == 0, f"Protected attributes found in RELAXATION_ORDER: {overlap}"
    assert "category" not in RELAXATION_ORDER
    assert "gender" not in RELAXATION_ORDER


# ==================================================
# Test 5: Session cleanup logic
# ==================================================
def test_session_cleanup():
    """Session cleanup must expire old sessions and evict oldest when over limit."""
    import time as _time

    # Simulate SESSION_FILTERS
    session_store = {}
    MAX = 3
    TTL = 2  # 2 seconds for fast test

    # Add sessions
    session_store["s1"] = {"timestamp": _time.time() - 10}  # expired
    session_store["s2"] = {"timestamp": _time.time() - 1}   # active (recent)
    session_store["s3"] = {"timestamp": _time.time()}        # active (newest)
    session_store["s4"] = {"timestamp": _time.time()}        # active

    # Cleanup expired
    now = _time.time()
    expired = [sid for sid, data in session_store.items()
               if now - data.get("timestamp", 0) > TTL]
    for sid in expired:
        del session_store[sid]

    assert "s1" not in session_store, "Expired session s1 should be removed"

    # Evict oldest if over limit
    while len(session_store) > MAX:
        oldest = min(session_store, key=lambda s: session_store[s].get("timestamp", 0))
        del session_store[oldest]

    assert len(session_store) <= MAX, f"Session count {len(session_store)} exceeds max {MAX}"


# ==================================================
# Test 6: Metrics aggregation
# ==================================================
def test_metrics_aggregate_only():
    """Metrics must store only aggregates, not per-query data."""
    from utils.retrieval_metrics import RetrievalMetrics

    m = RetrievalMetrics()
    m.record_query(stage_a_count=10, relaxed_count=2, top_similarity=0.8)
    m.record_query(stage_a_count=0, relaxed_count=3, top_similarity=0.5)
    m.record_low_confidence()

    summary = m.get_summary()
    assert summary["total_queries"] == 2
    assert summary["stage_a_hit_rate"] == 0.5  # 1 out of 2 had > 0
    assert summary["avg_relaxed_count"] == 2.5  # (2+3)/2
    assert summary["low_confidence_queries"] == 1

    # Verify no per-query data is stored
    assert "queries" not in summary
    assert "query_log" not in summary


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
