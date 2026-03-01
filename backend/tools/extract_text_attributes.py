"""
Textual Attribute Extraction — gender / style / material
=========================================================
Deterministic rule-based extraction from product_name.
No LLM, no guessing, no free-text.

Corrections applied:
  1. Case-insensitive matching (lowercase before extraction)
  2. Gender precedence: boys > girls > kids > women > men > unisex
  3. Style priority: specific → general (running > training > ... > sports)
  4. Material includes textile + metal composition (comment #4)
  5. Category-aware metal rules: gold/silver/copper/brass/steel only for
     jewelry/watch — not for apparel/footwear unless "gold-plated" etc.
  6. Multi-word phrase matching first ("genuine leather", "pure cotton")
  7. NULL rate monitoring vs expected coverage
  8. Strict allowed-value enforcement via assert before DB write
"""

import re
import sys
import time
import argparse
import psycopg2

sys.stdout.reconfigure(encoding='utf-8')

# ─── DB ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

# ─── CONTROLLED VOCABULARIES ────────────────────────────────────────
ALLOWED_GENDERS = {"men", "women", "unisex", "boys", "girls", "kids"}

ALLOWED_STYLES = {
    "casual", "formal", "running", "training", "walking", "yoga",
    "biker", "bomber", "ethnic", "party", "outdoor", "denim",
    "lounge", "sports", "retro",
}

# Material includes textile + metal composition.
# Metal materials (gold, silver, copper, brass, steel) are valid for
# jewelry and watches. For apparel/footwear, only "gold-plated" or
# similar explicit phrases trigger metal materials.
ALLOWED_MATERIALS = {
    "cotton", "polyester", "linen", "denim", "leather", "pu", "silk",
    "wool", "nylon", "viscose", "synthetic", "velvet", "satin",
    "mesh", "suede", "canvas", "fleece", "corduroy", "knit",
    "acrylic", "rayon",
    # Metals (category-aware)
    "copper", "gold", "silver", "brass", "steel",
}

# Categories where metal materials are valid
METAL_CATEGORIES = {"earrings", "necklace", "watch"}

# ─── NORMALIZE ──────────────────────────────────────────────────────
def normalize_text(text):
    """Lowercase, remove special characters except hyphens/apostrophes, normalize spacing."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s\-\']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─── GENDER EXTRACTION ──────────────────────────────────────────────
# Precedence: boys > girls > kids(age) > kids > women > men > unisex
_AGE_PATTERN = re.compile(r"\b\d{1,2}-\d{1,2}y\b", re.IGNORECASE)

def extract_gender(product_name):
    """Extract gender from product name. Returns allowed value or None."""
    text = normalize_text(product_name)
    if not text:
        return None

    # 1. boys / girls first (most specific)
    if re.search(r"\bboys?\b", text):
        return "boys"
    if re.search(r"\bgirls?\b", text):
        return "girls"

    # 2. Kids age pattern: "7-8y", "13-14y" etc.
    if _AGE_PATTERN.search(product_name):  # check original (has digits)
        return "kids"
    if re.search(r"\bkids?\b", text):
        return "kids"

    # 3. women before men (avoid "women" matching "men" substring)
    if re.search(r"\bwomen'?s?\b", text):
        return "women"
    if re.search(r"\bmen'?s?\b", text):
        return "men"

    # 4. unisex last
    if re.search(r"\bunisex\b", text):
        return "unisex"

    return None


# ─── STYLE EXTRACTION ───────────────────────────────────────────────
# Priority: most specific first. First match wins.
STYLE_PRIORITY = [
    ("running",  [r"\brunning\b"]),
    ("training", [r"\btraining\b", r"\bgym\b", r"\bactive\b"]),
    ("walking",  [r"\bwalking\b", r"\btrekking\b", r"\bhiking\b"]),
    ("yoga",     [r"\byoga\b"]),
    ("biker",    [r"\bbiker\b"]),
    ("bomber",   [r"\bbomber\b"]),
    ("ethnic",   [r"\bethnic\b", r"\bprayer\b"]),
    ("party",    [r"\bparty\b", r"\bwedding\b"]),
    ("outdoor",  [r"\boutdoor\b", r"\btravel\b", r"\bbeach\b"]),
    ("denim",    [r"\bdenim\b"]),
    ("formal",   [r"\bformal\b"]),
    ("casual",   [r"\bcasual\b"]),
    ("lounge",   [r"\blounge\b"]),
    ("sports",   [r"\bsports?\b", r"\bsporty\b", r"\bworkout\b", r"\bathleisure\b"]),
    ("retro",    [r"\bretro\b", r"\bvintage\b"]),
]

# Pre-compile
_STYLE_COMPILED = [
    (style, [re.compile(p) for p in patterns])
    for style, patterns in STYLE_PRIORITY
]

def extract_style(product_name):
    """Extract style from product name. First match by priority wins."""
    text = normalize_text(product_name)
    if not text:
        return None
    for style, patterns in _STYLE_COMPILED:
        for pat in patterns:
            if pat.search(text):
                return style
    return None


# ─── MATERIAL EXTRACTION ────────────────────────────────────────────
# Multi-word phrases first (Correction #6), then single words.
# Metal keywords are category-aware (Correction #5).

# Phrase → material (checked first)
MATERIAL_PHRASES = [
    ("genuine leather",  "leather"),
    ("pure cotton",      "cotton"),
    ("stainless steel",  "steel"),
    ("gold-plated",      "gold"),
    ("gold plated",      "gold"),
    ("silver-plated",    "silver"),
    ("silver plated",    "silver"),
    ("rhodium-plated",   "silver"),
    ("rhodium plated",   "silver"),
    ("faux leather",     "pu"),
    ("vegan leather",    "pu"),
]

# Single-word → material (non-metal)
MATERIAL_KEYWORDS_TEXTILE = [
    (r"\bcotton\b",     "cotton"),
    (r"\bpolyester\b",  "polyester"),
    (r"\blinen\b",      "linen"),
    (r"\bdenim\b",      "denim"),
    (r"\bleather\b",    "leather"),
    (r"\b(?:pu)\b",     "pu"),
    (r"\bsilk\b",       "silk"),
    (r"\bsatin\b",      "silk"),
    (r"\bwool\b",       "wool"),
    (r"\bnylon\b",      "synthetic"),
    (r"\bviscose\b",    "rayon"),
    (r"\brayon\b",      "rayon"),
    (r"\bsynthetic\b",  "synthetic"),
    (r"\bvelvet\b",     "velvet"),
    (r"\bmesh\b",       "mesh"),
    (r"\bsuede\b",      "suede"),
    (r"\bcanvas\b",     "canvas"),
    (r"\bfleece\b",     "fleece"),
    (r"\bcorduroy\b",   "corduroy"),
    (r"\bknit(?:ted)?\b", "knit"),
    (r"\bjersey\b",     "knit"),
    (r"\bacrylic\b",    "acrylic"),
    (r"\blycra\b",      "synthetic"),
    (r"\bspandex\b",    "synthetic"),
]

# Single-word metal keywords (only for jewelry/watch categories)
MATERIAL_KEYWORDS_METAL = [
    (r"\bcopper\b",     "copper"),
    (r"\bgold\b",       "gold"),
    (r"\bsilver\b",     "silver"),
    (r"\bbrass\b",      "brass"),
    (r"\bsteel\b",      "steel"),
    (r"\brhodium\b",    "silver"),
]

_TEXTILE_COMPILED = [(re.compile(p), m) for p, m in MATERIAL_KEYWORDS_TEXTILE]
_METAL_COMPILED = [(re.compile(p), m) for p, m in MATERIAL_KEYWORDS_METAL]


def extract_material(product_name, category):
    """Extract material. Category-aware: metals only for jewelry/watch."""
    text = normalize_text(product_name)
    if not text:
        return None

    is_metal_category = category in METAL_CATEGORIES

    # 1. Multi-word phrases first
    for phrase, mat in MATERIAL_PHRASES:
        if phrase in text:
            # Metal phrase results are allowed even in non-metal categories
            # (e.g. "gold-plated sneakers" is legitimate)
            return mat

    # 2. Single-word textile matches
    for pat, mat in _TEXTILE_COMPILED:
        if pat.search(text):
            return mat

    # 3. Single-word metal matches (only for metal categories)
    if is_metal_category:
        for pat, mat in _METAL_COMPILED:
            if pat.search(text):
                return mat

    return None


# ─── VALIDATION ─────────────────────────────────────────────────────
def validate_text_attributes(gender, style, material):
    """Strict allowed-value enforcement (Correction #8)."""
    if gender is not None:
        assert gender in ALLOWED_GENDERS, f"Invalid gender: {gender!r}"
    if style is not None:
        assert style in ALLOWED_STYLES, f"Invalid style: {style!r}"
    if material is not None:
        assert material in ALLOWED_MATERIALS, f"Invalid material: {material!r}"
    return True


# ─── MAIN PIPELINE ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Textual Attribute Extraction")
    parser.add_argument("--n", type=int, default=0,
                        help="Max rows to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract + print, do NOT update DB")
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N random sample rows after extraction")
    args = parser.parse_args()

    n_limit = args.n
    dry_run = args.dry_run
    sample_n = args.sample

    print("=" * 70)
    print("TEXTUAL ATTRIBUTE EXTRACTION PIPELINE")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'LIVE (writing to DB)'}")
    print(f"  Rows: {'all' if n_limit == 0 else n_limit}")
    print("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Load rows
    query = "SELECT id, product_name, category FROM visual_attributes ORDER BY id"
    if n_limit > 0:
        query += f" LIMIT {n_limit}"
    cur.execute(query)
    rows = cur.fetchall()

    print(f"\n  Loaded {len(rows)} rows")

    # Stats
    total = len(rows)
    gender_count = 0
    style_count = 0
    material_count = 0
    errors = 0
    batch_size = 500
    updated = 0

    t0 = time.time()

    for i, (row_id, product_name, category) in enumerate(rows):
        try:
            gender = extract_gender(product_name)
            style = extract_style(product_name)
            material = extract_material(product_name, category)

            validate_text_attributes(gender, style, material)

            if gender: gender_count += 1
            if style: style_count += 1
            if material: material_count += 1

            if not dry_run:
                cur.execute(
                    "UPDATE visual_attributes SET gender=%s, style=%s, material=%s WHERE id=%s",
                    (gender, style, material, row_id)
                )
                updated += 1

                if updated % batch_size == 0:
                    conn.commit()
                    elapsed = time.time() - t0
                    rate = updated / elapsed if elapsed > 0 else 0
                    print(f"    ... {updated}/{total} ({updated*100//total}%) "
                          f"  {rate:.0f} rows/s")

        except AssertionError as e:
            errors += 1
            print(f"  VALIDATION FAIL row {row_id}: {e}")
        except Exception as e:
            errors += 1
            print(f"  ❌ ERROR row {row_id}: {e}")

    # Final commit
    if not dry_run:
        conn.commit()

    elapsed = time.time() - t0

    # ─── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Total rows:     {total}")
    print(f"  Updated:        {updated}")
    print(f"  Errors:         {errors}")
    print(f"  Time:           {elapsed:.1f}s")

    gender_null_pct = (1 - gender_count / total) * 100 if total > 0 else 0
    style_null_pct = (1 - style_count / total) * 100 if total > 0 else 0
    material_null_pct = (1 - material_count / total) * 100 if total > 0 else 0

    print(f"\n  Coverage:")
    print(f"    Gender:   {gender_count}/{total} ({100-gender_null_pct:.1f}%)  NULL={gender_null_pct:.1f}%  (expected ~43%)")
    print(f"    Style:    {style_count}/{total} ({100-style_null_pct:.1f}%)  NULL={style_null_pct:.1f}%  (expected ~72%)")
    print(f"    Material: {material_count}/{total} ({100-material_null_pct:.1f}%)  NULL={material_null_pct:.1f}%  (expected ~81%)")

    # NULL rate deviation check (Correction #7)
    print(f"\n  NULL rate deviation from expected:")
    deviations = [
        ("Gender",   gender_null_pct,   43.0),
        ("Style",    style_null_pct,    72.0),
        ("Material", material_null_pct, 81.0),
    ]
    for name, actual, expected in deviations:
        delta = abs(actual - expected)
        status = "OK" if delta < 15 else "INVESTIGATE"
        print(f"    {name:10s}  actual={actual:.1f}%  expected~{expected:.0f}%  delta={delta:.1f}%  {status}")

    # ─── Sample Rows ────────────────────────────────────────────────
    if sample_n > 0 and not dry_run:
        print(f"\n  --- {sample_n} Random Samples ---")
        cur.execute(
            f"SELECT product_name, category, gender, style, material "
            f"FROM visual_attributes ORDER BY RANDOM() LIMIT {sample_n}"
        )
        for r in cur.fetchall():
            print(f"    Name:     {r[0]}")
            print(f"    Category: {r[1]}  |  gender={r[2]}  style={r[3]}  material={r[4]}")
            print()

    # ─── Distribution ───────────────────────────────────────────────
    if not dry_run:
        print("\n  --- Gender Distribution ---")
        cur.execute("SELECT gender, COUNT(*) FROM visual_attributes GROUP BY gender ORDER BY COUNT(*) DESC")
        for r in cur.fetchall():
            print(f"    {str(r[0]):15s}  {r[1]}")

        print("\n  --- Style Distribution ---")
        cur.execute("SELECT style, COUNT(*) FROM visual_attributes GROUP BY style ORDER BY COUNT(*) DESC")
        for r in cur.fetchall():
            print(f"    {str(r[0]):15s}  {r[1]}")

        print("\n  --- Material Distribution ---")
        cur.execute("SELECT material, COUNT(*) FROM visual_attributes GROUP BY material ORDER BY COUNT(*) DESC")
        for r in cur.fetchall():
            print(f"    {str(r[0]):15s}  {r[1]}")

    cur.close()
    conn.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
