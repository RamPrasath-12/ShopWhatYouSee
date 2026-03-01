"""
Data Consistency Validation Script

Compares DB values (base_colour, agman_pattern, gender) against
SYNONYM_MAP and AG-MAN predicted values to identify mismatches.

Usage:
    python tools/validate_data_consistency.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_db
from models.product_retrieval import SYNONYM_MAP, normalize_filter_value


def validate():
    conn = get_db()
    cur = conn.cursor()

    print("=" * 60)
    print("DATA CONSISTENCY VALIDATION")
    print("=" * 60)

    # 1. Distinct gender values
    cur.execute("SELECT DISTINCT gender, COUNT(*) FROM products GROUP BY gender ORDER BY gender")
    rows = cur.fetchall()
    print("\n--- GENDER VALUES IN DB ---")
    for val, cnt in rows:
        print(f"  '{val}' -> {cnt} products")

    # 2. Distinct base_colour values
    cur.execute("SELECT DISTINCT LOWER(base_colour), COUNT(*) FROM products GROUP BY LOWER(base_colour) ORDER BY COUNT(*) DESC LIMIT 40")
    rows = cur.fetchall()
    print("\n--- TOP 40 COLOR VALUES IN DB (lowercased) ---")
    for val, cnt in rows:
        print(f"  '{val}' -> {cnt} products")

    # 3. Distinct agman_pattern values
    cur.execute("SELECT DISTINCT agman_pattern, COUNT(*) FROM products GROUP BY agman_pattern ORDER BY COUNT(*) DESC")
    rows = cur.fetchall()
    print("\n--- PATTERN VALUES IN DB ---")
    for val, cnt in rows:
        print(f"  '{val}' -> {cnt} products")

    # 4. Check SYNONYM_MAP alignment
    print("\n--- SYNONYM MAP ALIGNMENT CHECK ---")
    color_synonyms = {k: v for k, v in SYNONYM_MAP.items() if k not in ("male", "female", "man", "woman", "boys", "girls",
                                                                           "plain", "none", "checks", "check", "stripes", "stripe", "plaid")}
    cur.execute("SELECT DISTINCT LOWER(base_colour) FROM products")
    db_colors = {r[0] for r in cur.fetchall() if r[0]}

    for synonym, canonical in color_synonyms.items():
        if canonical.lower() not in db_colors:
            print(f"  ⚠️ SYNONYM '{synonym}' -> '{canonical}' — canonical NOT FOUND in DB colors")
        else:
            print(f"  ✅ '{synonym}' -> '{canonical}' — OK")

    pattern_synonyms = {k: v for k, v in SYNONYM_MAP.items() if k in ("plain", "none", "checks", "check", "stripes", "stripe", "plaid")}
    cur.execute("SELECT DISTINCT agman_pattern FROM products WHERE agman_pattern IS NOT NULL")
    db_patterns = {r[0] for r in cur.fetchall() if r[0]}

    for synonym, canonical in pattern_synonyms.items():
        if canonical not in db_patterns:
            print(f"  ⚠️ PATTERN SYNONYM '{synonym}' -> '{canonical}' — canonical NOT FOUND in DB patterns")
        else:
            print(f"  ✅ '{synonym}' -> '{canonical}' — OK")

    # 5. NULL statistics
    cur.execute("SELECT COUNT(*) FROM products WHERE base_colour IS NULL")
    null_colors = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM products WHERE agman_pattern IS NULL")
    null_patterns = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM products WHERE gender IS NULL")
    null_genders = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM products")
    total = cur.fetchone()[0]

    print(f"\n--- NULL STATISTICS ---")
    print(f"  Total products: {total}")
    print(f"  NULL base_colour: {null_colors} ({null_colors/total*100:.1f}%)")
    print(f"  NULL agman_pattern: {null_patterns} ({null_patterns/total*100:.1f}%)")
    print(f"  NULL gender: {null_genders} ({null_genders/total*100:.1f}%)")

    cur.close()
    conn.close()
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    validate()
