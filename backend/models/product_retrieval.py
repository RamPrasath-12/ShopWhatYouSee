"""
Product Retrieval Module - Production v3 (Fixed)
====================================================
Fixes applied:
1. normalize_category() now maps ALL DB-exact values correctly (glasses plural, caps, earrings, etc.)
2. _detect_overrides() uses final normalized category to prevent glass vs glasses mismatch
3. Category fuzzy fallback uses ILIKE / IN with all known variations
4. Relaxation phases execute correctly even when rows==0 at tier 5
"""

import psycopg2
import os
import faiss
import json
import gc
import numpy as np
import time as _time

from db_config import DB_CONFIG

# =============================================================================
# FAISS CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'products.faiss')
ID_MAP_PATH = os.path.join(DATA_DIR, 'product_ids.json')

faiss_index = None
id_map = None

def load_faiss():
    global faiss_index, id_map
    if faiss_index is not None:
        return True
    
    if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        try:
            faiss_index = faiss.read_index(INDEX_PATH)
            with open(ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
            id_map = {int(k): v for k, v in id_map.items()}
            print(f"[FAISS] Loaded index with {faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"[FAISS] Failed to load: {e}")
            return False
    return False

def get_db():
    return psycopg2.connect(**DB_CONFIG)

# =============================================================================
# CATEGORY NORMALIZATION
# ✅ FIX: All values must match EXACT DB category strings (verified from DB list)
# DB categories: Footwear_sandals, Footwear_shoes, Jacket, belt, blazer, caps,
#                churidhar, dhoti, earrings, glasses, necklace, pant, shirts,
#                shorts, skirt, tie, tshirt, watch
# =============================================================================

def normalize_category(cat):
    """
    Map all YOLO/AGMAN/user category inputs to exact DB values.
    ✅ FIX: DB stores 'glasses' (plural), 'caps' (plural), 'earrings' (plural)
            These must all map correctly.
    """
    if not cat:
        return None
    cat = cat.lower().strip().replace("-", "_").replace(" ", "_")

    mapping = {
        # Upper wear
        "shirt": "shirts",
        "shirts": "shirts",
        "tshirt": "tshirt",
        "t_shirt": "tshirt",
        "tee": "tshirt",
        "blouse": "blouse",
        "blazer": "blazer",
        "jacket": "Jacket",       # ✅ DB stores "Jacket" with capital J
        "coat": "Jacket",

        # Lower wear
        "pant": "pant",
        "pants": "pant",
        "jeans": "pant",
        "shorts": "shorts",
        "short": "shorts",
        "skirt": "skirt",
        "leggings": "leggings",

        # Traditional
        "churidhar": "churidhar",
        "dhoti": "dhoti",
        "saree": "saree",
        "shawl": "shawl",

        # Footwear — ✅ DB stores with capital F and underscore
        "footwear_shoes": "Footwear_shoes",
        "shoes": "Footwear_shoes",
        "shoe": "Footwear_shoes",
        "footwear_sandals": "Footwear_sandals",
        "sandals": "Footwear_sandals",
        "sandal": "Footwear_sandals",
        "footwear_heels": "Footwear_sandals",  # map heels to sandals (closest)
        "heels": "Footwear_sandals",
        "heel": "Footwear_sandals",

        # Accessories
        "bag": "bag",
        "bags": "bag",
        "purse": "purse",
        "bangle": "bangle",
        "bangles": "bangle",
        "bracelet": "bracelet",

        # ✅ FIX: DB stores "caps" (plural)
        "cap": "caps",
        "caps": "caps",
        "hat": "caps",
        "hats": "caps",

        "belt": "belt",

        "tie": "tie",

        # ✅ FIX: DB stores "earrings" (plural)
        "earring": "earrings",
        "earrings": "earrings",

        "necklace": "necklace",
        "necklaces": "necklace",

        "ring": "ring",
        "rings": "ring",

        # ✅ FIX: DB stores "watch" (singular) — note trailing dot in DB list is a typo
        "watch": "watch",
        "watches": "watch",

        # ✅ CRITICAL FIX: DB stores "glasses" (plural)
        # AGMAN converts "glasses" → "glass" — we must catch both
        "glass": "glasses",
        "glasses": "glasses",
        "sunglass": "glasses",
        "sunglasses": "glasses",
        "eyewear": "glasses",
        "spectacles": "glasses",

        "hairclip": "hairclip",
        "hair_clip": "hairclip",
    }

    result = mapping.get(cat)
    if result:
        return result

    # Fallback: return as-is (do not lowercase here — DB may be case-sensitive)
    return cat


# =============================================================================
# KNOWN DB CATEGORIES (for fuzzy fallback)
# ✅ FIX: Exact strings as they appear in DB
# =============================================================================

DB_CATEGORIES = [
    "Footwear_sandals", "Footwear_shoes", "Jacket", "belt", "blazer",
    "caps", "churidhar", "dhoti", "earrings", "glasses", "necklace",
    "pant", "shirts", "shorts", "skirt", "tie", "tshirt", "watch",
    # extras that may exist
    "blouse", "saree", "shawl", "leggings", "bag", "purse",
    "bangle", "bracelet", "ring", "hairclip",
]


def get_category_variations(category):
    """
    Get all possible DB-valid variations of a category.
    ✅ FIX: Always return the correct DB plural/singular form,
            not just a naive string manipulation.
    """
    if not category:
        return []

    cat_lower = category.lower().strip()

    # Explicit variation map — these are all DB-valid values
    variation_map = {
        "glass": ["glasses"],
        "glasses": ["glasses"],
        "sunglass": ["glasses"],
        "sunglasses": ["glasses"],
        "eyewear": ["glasses"],
        "cap": ["caps"],
        "caps": ["caps"],
        "hat": ["caps"],
        "earring": ["earrings"],
        "earrings": ["earrings"],
        "jacket": ["Jacket"],
        "coat": ["Jacket"],
        "shoe": ["Footwear_shoes"],
        "shoes": ["Footwear_shoes"],
        "footwear_shoes": ["Footwear_shoes"],
        "sandal": ["Footwear_sandals"],
        "sandals": ["Footwear_sandals"],
        "footwear_sandals": ["Footwear_sandals"],
        "shirt": ["shirts"],
        "shirts": ["shirts"],
        "pant": ["pant"],
        "pants": ["pant"],
        "jeans": ["pant"],
        "watch": ["watch"],
        "watches": ["watch"],
        "necklace": ["necklace"],
        "necklaces": ["necklace"],
    }

    variations = variation_map.get(cat_lower, [category])

    # Also include any exact DB match (case-insensitive)
    for db_cat in DB_CATEGORIES:
        if db_cat.lower() == cat_lower and db_cat not in variations:
            variations.append(db_cat)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            result.append(v)

    return result


# =============================================================================
# SYNONYM MAP
# =============================================================================

SYNONYM_MAP = {
    "gray": "grey",
    "grey melange": "grey",
    "off white": "off white",
    "navy blue": "navy blue",
    "plain": "solid",
    "checks": "checked",
    "stripes": "striped",
    "plaid": "checked",
    "full sleeve": "long sleeves",
    "full sleeves": "long sleeves",
    "long sleeve": "long sleeves",
    "half sleeve": "short sleeves",
    "half sleeves": "short sleeves",
    "short sleeve": "short sleeves",
    "3/4 sleeve": "three-quarter sleeves",
    "three quarter": "three-quarter sleeves",
    "male": "Men",
    "female": "Women",
    "man": "Men",
    "woman": "Women",
}

def normalize_filter_value(value):
    if not value or not isinstance(value, str):
        return value
    normalized = value.lower().strip()
    return SYNONYM_MAP.get(normalized, normalized)

# =============================================================================
# COLOR FAMILIES
# =============================================================================

COLOR_FAMILIES = {
    "black": "neutral", "white": "neutral", "grey": "neutral", "gray": "neutral",
    "charcoal": "neutral", "silver": "neutral", "off white": "neutral", "cream": "neutral",
    "multi": "neutral", "dark grey": "neutral", "light grey": "neutral",
    "grey melange": "neutral", "steel": "neutral", "ivory": "neutral",
    "nude": "neutral", "champagne": "neutral", "beige": "neutral", "transparent": "neutral",
    "red": "red", "maroon": "red", "burgundy": "red", "wine": "red",
    "crimson": "red", "rust": "red", "dark red": "red", "bright red": "red",
    "pink": "pink", "hot pink": "pink", "magenta": "pink", "peach": "pink",
    "coral": "pink", "rose": "pink", "fuchsia": "pink", "light pink": "pink",
    "orange": "orange", "mustard": "orange", "gold": "orange", "burnt orange": "orange",
    "yellow": "yellow", "light yellow": "yellow", "golden yellow": "yellow",
    "green": "green", "olive": "green", "lime": "green", "khaki": "green",
    "sea green": "green", "lime green": "green", "fluorescent green": "green",
    "dark green": "green", "mint": "green", "forest green": "green",
    "blue": "blue", "navy blue": "blue", "navy": "blue", "sky blue": "blue",
    "steel blue": "blue", "teal": "blue", "turquoise blue": "blue", "turquoise": "blue",
    "dark blue": "blue", "light blue": "blue", "royal blue": "blue", "cobalt": "blue", "cyan": "blue",
    "purple": "purple", "lavender": "purple", "violet": "purple", "mauve": "purple", "plum": "purple",
    "brown": "brown", "tan": "brown", "chocolate": "brown", "taupe": "brown",
    "coffee": "brown", "coffee brown": "brown", "camel brown": "brown",
    "bronze": "brown", "copper": "brown", "rose gold": "brown",
}

COLOR_SIMILARITY_TIERS = {
    "teal": {
        "tier_1": ["turquoise", "cyan"],
        "tier_2": ["blue", "sky blue", "steel blue"],
        "tier_3": ["navy blue", "navy", "dark blue"],
        "tier_4": ["green"],
    },
    "navy blue": {
        "tier_1": ["navy", "dark blue"],
        "tier_2": ["blue", "royal blue"],
        "tier_3": ["steel blue", "teal"],
        "tier_4": ["black"],
    },
    "sky blue": {
        "tier_1": ["light blue", "blue"],
        "tier_2": ["turquoise", "cyan"],
        "tier_3": ["teal", "steel blue"],
        "tier_4": ["white"],
    },
    "blue": {
        "tier_1": ["royal blue", "cobalt"],
        "tier_2": ["sky blue", "light blue"],
        "tier_3": ["navy blue", "teal"],
        "tier_4": ["purple"],
    },
    "red": {
        "tier_1": ["bright red", "crimson"],
        "tier_2": ["maroon", "burgundy"],
        "tier_3": ["wine", "dark red"],
        "tier_4": ["pink"],
    },
    "maroon": {
        "tier_1": ["burgundy", "wine"],
        "tier_2": ["dark red", "red"],
        "tier_3": ["brown"],
        "tier_4": ["black"],
    },
    "olive": {
        "tier_1": ["khaki", "green"],
        "tier_2": ["forest green", "dark green"],
        "tier_3": ["brown"],
        "tier_4": ["teal"],
    },
    "green": {
        "tier_1": ["lime green", "mint"],
        "tier_2": ["olive", "forest green"],
        "tier_3": ["teal", "khaki"],
        "tier_4": ["blue"],
    },
    "grey": {
        "tier_1": ["gray", "light grey", "dark grey"],
        "tier_2": ["charcoal", "silver"],
        "tier_3": ["black", "white"],
        "tier_4": ["beige"],
    },
    "black": {
        "tier_1": ["charcoal", "dark grey"],
        "tier_2": ["grey"],
        "tier_3": ["navy blue"],
        "tier_4": ["maroon"],
    },
    "white": {
        "tier_1": ["off white", "cream", "ivory"],
        "tier_2": ["beige", "nude"],
        "tier_3": ["light grey", "silver"],
        "tier_4": ["champagne"],
    },
    "pink": {
        "tier_1": ["light pink", "hot pink"],
        "tier_2": ["coral", "peach"],
        "tier_3": ["magenta", "fuchsia"],
        "tier_4": ["red"],
    },
    "brown": {
        "tier_1": ["tan", "camel brown"],
        "tier_2": ["chocolate", "coffee"],
        "tier_3": ["bronze", "copper"],
        "tier_4": ["maroon", "olive"],
    },
}

SCENE_TO_STYLE = {
    "office": "formal",
    "park": "casual",
    "gymnasium": "sports",
    "beach": "casual",
    "nightclub": "party",
    "bamboo_forest": "casual",
}

def get_color_fallback_sequence(target_color):
    target_lower = target_color.lower().strip()
    sequence = [target_lower]
    family = COLOR_FAMILIES.get(target_lower, target_lower)
    if target_lower in COLOR_SIMILARITY_TIERS:
        tiers = COLOR_SIMILARITY_TIERS[target_lower]
        for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            if tier in tiers:
                sequence.extend(tiers[tier])
    family_colors = [c for c, f in COLOR_FAMILIES.items()
                     if f == family and c not in sequence]
    sequence.extend(family_colors)
    return sequence

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

MODE_WEIGHTS = {
    "PURE_SIMILARITY":          {"visual": 0.85, "override": 0.00, "preserved": 0.15},
    "ATTRIBUTE_OVERRIDE":       {"visual": 0.50, "override": 0.40, "preserved": 0.10},
    "COLOR_OVERRIDE":           {"visual": 0.20, "override": 0.65, "preserved": 0.15},
    "CATEGORY_TRANSFORM":       {"visual": 0.60, "override": 0.20, "preserved": 0.20},
    "COMBINED":                 {"visual": 0.45, "override": 0.35, "preserved": 0.20},
    "COLOR_CATEGORY_OVERRIDE":  {"visual": 0.25, "override": 0.60, "preserved": 0.15},
}

ATTRIBUTE_QUERY_WEIGHTS = {
    "visual": 0.20,
    "override": 0.60,
    "preserved": 0.20,
}

# =============================================================================
# QUERY INTENT CLASSIFICATION
# =============================================================================

def classify_query_intent(detected_attributes, user_filters):
    has_attributes = bool(
        user_filters.get("sleeve") or user_filters.get("pattern") or
        user_filters.get("fit") or user_filters.get("material") or
        user_filters.get("price_bucket")
    )
    color_val = user_filters.get("color")
    has_user_color = False
    if color_val:
        if isinstance(color_val, dict):
            has_user_color = color_val.get("source") in ("llm", "query", "user")
        else:
            has_user_color = True
    user_cat = user_filters.get("category")
    detected_cat = detected_attributes.get("category")
    category_changed = (
        user_cat and detected_cat and
        normalize_category(user_cat) != normalize_category(detected_cat)
    )
    if has_attributes and has_user_color:
        return "ATTRIBUTE_FIRST"
    elif has_user_color:
        return "ATTRIBUTE_FIRST"
    elif category_changed:
        return "HYBRID"
    elif has_attributes:
        return "ATTRIBUTE_FIRST"
    else:
        return "VISUAL_FIRST"

# =============================================================================
# OVERRIDE DETECTION
# ✅ FIX: Normalize category through normalize_category() before storing in overrides
# =============================================================================

def _detect_overrides(detected_attributes, user_filters):
    overrides = {}
    preserved = {}
    category_changed = False

    det_category = detected_attributes.get("category", "")
    det_color = detected_attributes.get("color") or detected_attributes.get("color_name", "")
    det_sleeve = detected_attributes.get("sleeve") or detected_attributes.get("sleeve_length", "")
    det_pattern = detected_attributes.get("pattern", "")
    det_gender = detected_attributes.get("gender", "")

    usr_category = user_filters.get("category", "")
    usr_color = user_filters.get("color") or user_filters.get("primary_color_name", "")
    usr_sleeve = user_filters.get("sleeve") or user_filters.get("sleeve_value", "")
    usr_pattern = user_filters.get("pattern") or user_filters.get("pattern_value", "")
    usr_gender = user_filters.get("gender", "")
    usr_material = user_filters.get("material", "")
    usr_style = user_filters.get("style", "")
    usr_price = user_filters.get("price_bucket", "")
    usr_fit = user_filters.get("fit", "")

    # ✅ FIX: Always normalize category through the mapping table
    if usr_category:
        norm_usr = normalize_category(usr_category)    # e.g. "glass" → "glasses"
        norm_det = normalize_category(det_category) if det_category else ""
        if norm_usr and norm_det and norm_usr.lower() != norm_det.lower():
            overrides["category"] = norm_usr
            category_changed = True
        elif norm_usr:
            overrides["category"] = norm_usr            # ✅ Always store normalized value

    if usr_color:
        overrides["color"] = normalize_filter_value(usr_color)
    elif det_color:
        preserved["color"] = normalize_filter_value(det_color)

    if usr_sleeve:
        overrides["sleeve"] = normalize_filter_value(usr_sleeve)
    elif det_sleeve:
        preserved["sleeve"] = normalize_filter_value(det_sleeve)

    if usr_pattern:
        overrides["pattern"] = normalize_filter_value(usr_pattern)
    elif det_pattern:
        preserved["pattern"] = normalize_filter_value(det_pattern)

    if usr_gender:
        overrides["gender"] = normalize_filter_value(usr_gender)
    elif det_gender:
        preserved["gender"] = normalize_filter_value(det_gender)

    if usr_material:
        overrides["material"] = normalize_filter_value(usr_material)
    if usr_style:
        overrides["style"] = normalize_filter_value(usr_style)
    if usr_price:
        overrides["price_bucket"] = usr_price
    if usr_fit:
        overrides["fit"] = normalize_filter_value(usr_fit)

    if category_changed:
        preserved.pop("category", None)

    has_color_override = "color" in overrides
    has_attr_overrides = any(
        k in overrides for k in ("sleeve", "pattern", "material", "style", "price_bucket", "fit")
    )

    if category_changed and has_color_override:
        mode = "COLOR_CATEGORY_OVERRIDE"
    elif has_color_override and has_attr_overrides:
        mode = "COMBINED"
    elif has_color_override:
        mode = "COLOR_OVERRIDE"
    elif category_changed and has_attr_overrides:
        mode = "COMBINED"
    elif category_changed:
        mode = "CATEGORY_TRANSFORM"
    elif has_attr_overrides:
        mode = "ATTRIBUTE_OVERRIDE"
    else:
        mode = "PURE_SIMILARITY"

    return overrides, preserved, mode

# =============================================================================
# COLOR NORMALIZATION FOR SQL
# =============================================================================

def _normalize_color_for_sql(color_value):
    if not color_value:
        return None, None
    color_lower = color_value.lower().strip()
    family = COLOR_FAMILIES.get(color_lower, color_lower)
    try:
        from utils.color_utils import normalize_color_name
        primary = normalize_color_name(color_value)
    except:
        primary = color_value
    return primary, family

# =============================================================================
# SQL BUILDER WITH COLOR TIER SUPPORT
# ✅ FIX: Use ILIKE for case-insensitive category matching against DB
# =============================================================================

def _build_candidate_sql_with_color_tiers(
    overrides,
    preserved,
    effective_category,
    detected_gender,
    price_max=None,
    skip_overrides=None,
    color_tier=0
):
    conditions = []
    params = []
    skip = skip_overrides or set()

    # ✅ FIX: Use ILIKE for category so "glasses" matches regardless of case stored in DB
    if effective_category:
        conditions.append("category ILIKE %s")
        params.append(effective_category)

    gender_val = overrides.get("gender") or detected_gender
    if gender_val and "gender" not in skip:
        gender_norm = SYNONYM_MAP.get(gender_val.lower(), gender_val)
        conditions.append("LOWER(gender) = LOWER(%s)")
        params.append(gender_norm)

    if price_max is not None and "price_max" not in skip:
        try:
            conditions.append("COALESCE(discounted_price, original_price, 99999) <= %s")
            params.append(float(price_max))
        except (ValueError, TypeError):
            pass

    # COLOR WITH TIER FALLBACK
    color_val = overrides.get("color") or preserved.get("color")

    if color_val and "color" not in skip and color_tier < 5:
        color_sequence = get_color_fallback_sequence(color_val)

        if color_tier == 0:
            search_colors = [color_sequence[0]]
        elif color_tier == 1:
            search_colors = color_sequence[:min(4, len(color_sequence))]
        elif color_tier == 2:
            search_colors = color_sequence[:min(7, len(color_sequence))]
        elif color_tier == 3:
            search_colors = color_sequence[:min(10, len(color_sequence))]
        else:  # tier 4
            family = COLOR_FAMILIES.get(color_val.lower(), color_val.lower())
            search_colors = [c for c, f in COLOR_FAMILIES.items() if f == family]

        color_placeholders = ",".join(["%s"] * len(search_colors))
        conditions.append(
            f"(LOWER(COALESCE(scraped_color, primary_color_name, '')) IN ({color_placeholders}) "
            f"OR LOWER(color_family) IN ({color_placeholders}))"
        )
        params.extend(search_colors)
        params.extend(search_colors)

    # OTHER ATTRIBUTES
    for field, value in overrides.items():
        if field in skip or field in ("category", "gender", "color"):
            continue
        if not value:
            continue
        if field == "sleeve":
            conditions.append("LOWER(COALESCE(scraped_sleeve, sleeve_value, '')) LIKE LOWER(%s)")
            params.append(f"%{value}%")
        elif field == "pattern":
            conditions.append("LOWER(COALESCE(scraped_pattern, pattern_value, '')) = LOWER(%s)")
            params.append(value)
        elif field == "material":
            conditions.append("LOWER(COALESCE(scraped_material, material, '')) = LOWER(%s)")
            params.append(value)
        elif field == "style":
            conditions.append("LOWER(COALESCE(style, '')) = LOWER(%s)")
            params.append(value)
        elif field == "fit":
            conditions.append("LOWER(COALESCE(scraped_fit, '')) = LOWER(%s)")
            params.append(value)
        elif field == "price_bucket" and "price_bucket" not in skip:
            conditions.append("LOWER(COALESCE(price_bucket, '')) = LOWER(%s)")
            params.append(value)

    where = " AND ".join(conditions) if conditions else "TRUE"
    return where, params

# =============================================================================
# PROGRESSIVE RELAXATION WITH COLOR TIERS + CATEGORY FALLBACK
# ✅ FIX: Phase 2 (category fuzzy) now always runs when rows < min_results
#         regardless of which tier was reached
# =============================================================================

def fetch_candidates_with_progressive_color_relaxation(
    cur,
    overrides,
    preserved,
    effective_category,
    detected_gender,
    price_max=None,
    min_results=5
):
    relaxation_log = []
    skip_overrides = set()
    color_tier = 0

    sql_template = """
        SELECT product_id, category, gender, style,
               COALESCE(scraped_material, material) as material,
               price_bucket, color_family, brand, product_name,
               image_url, discounted_price, original_price,
               COALESCE(scraped_color, primary_color_name) as eff_color,
               COALESCE(scraped_pattern, pattern_value) as eff_pattern,
               COALESCE(scraped_sleeve, sleeve_value) as eff_sleeve,
               embedding, product_url,
               scraped_fit, scraped_neckline
        FROM visual_attributes
        WHERE {where}
          AND embedding IS NOT NULL
    """

    # =========================================================================
    # PHASE 1: EXACT CATEGORY + COLOR TIERS (0-5)
    # =========================================================================

    # Tier 0: Exact color
    print(f"[ColorRelax] Tier 0: Exact color match for category='{effective_category}'")
    where, params = _build_candidate_sql_with_color_tiers(
        overrides, preserved, effective_category, detected_gender,
        price_max, skip_overrides, color_tier=0
    )
    cur.execute(sql_template.format(where=where), params)
    rows = cur.fetchall()
    print(f"[ColorRelax] Tier 0: {len(rows)} results")
    if len(rows) >= min_results:
        return rows, relaxation_log, 0

    # Relax non-color attributes progressively
    relax_order = ["price_bucket", "material", "pattern", "sleeve", "fit"]
    for attr in relax_order:
        if attr not in overrides and attr != "price_bucket":
            continue
        if attr == "price_bucket" and price_max is None:
            continue
        skip_overrides.add(attr)
        relaxation_log.append(f"Relaxed: {attr}")
        print(f"[ColorRelax] Relaxing {attr}")
        where, params = _build_candidate_sql_with_color_tiers(
            overrides, preserved, effective_category, detected_gender,
            None if attr == "price_bucket" else price_max,
            skip_overrides, color_tier=0
        )
        cur.execute(sql_template.format(where=where), params)
        rows = cur.fetchall()
        print(f"[ColorRelax] After {attr}: {len(rows)} results")
        if len(rows) >= min_results:
            return rows, relaxation_log, 0

    # Color tier relaxation (1-4)
    for tier in range(1, 5):
        tier_names = ["exact", "tier_1", "tier_2", "tier_3", "tier_4"]
        relaxation_log.append(f"Color: {tier_names[tier]}")
        print(f"[ColorRelax] Tier {tier}: {tier_names[tier]}")
        where, params = _build_candidate_sql_with_color_tiers(
            overrides, preserved, effective_category, detected_gender,
            None, skip_overrides, color_tier=tier
        )
        cur.execute(sql_template.format(where=where), params)
        rows = cur.fetchall()
        print(f"[ColorRelax] Tier {tier}: {len(rows)} results")
        if len(rows) >= min_results:
            return rows, relaxation_log, tier

    # Tier 5: No color filter (but still exact category)
    print(f"[ColorRelax] Tier 5: No color filter")
    relaxation_log.append("Color: removed")
    where, params = _build_candidate_sql_with_color_tiers(
        overrides, preserved, effective_category, detected_gender,
        None, skip_overrides, color_tier=5
    )
    cur.execute(sql_template.format(where=where), params)
    rows = cur.fetchall()
    print(f"[ColorRelax] Tier 5: {len(rows)} results")
    if len(rows) >= min_results:
        return rows, relaxation_log, 5

    # =========================================================================
    # PHASE 2: CATEGORY FUZZY MATCHING
    # ✅ FIX: Runs whenever rows < min_results after all color tiers exhausted.
    #         Uses get_category_variations() which returns DB-exact values.
    #         Uses ILIKE for resilient case-insensitive matching.
    # =========================================================================

    if len(rows) < min_results and effective_category:
        print(f"[CategoryFallback] '{effective_category}' yielded {len(rows)}, trying variations...")
        category_variations = get_category_variations(effective_category)
        print(f"[CategoryFallback] Variations to try: {category_variations}")

        for variant in category_variations:
            # Skip if we already tried this exact value (case-insensitive)
            if variant.lower() == effective_category.lower():
                continue

            print(f"[CategoryFallback] Trying category variant: '{variant}'")
            relaxation_log.append(f"Category: {effective_category} → {variant}")

            # Build WHERE with variant, no color filter
            conditions = ["category ILIKE %s"]
            var_params = [variant]

            gender_val = overrides.get("gender") or detected_gender
            if gender_val and "gender" not in skip_overrides:
                gender_norm = SYNONYM_MAP.get(gender_val.lower(), gender_val)
                conditions.append("LOWER(gender) = LOWER(%s)")
                var_params.append(gender_norm)

            where = " AND ".join(conditions) + " AND embedding IS NOT NULL"
            cur.execute(
                f"SELECT product_id, category, gender, style, "
                f"COALESCE(scraped_material, material) as material, "
                f"price_bucket, color_family, brand, product_name, "
                f"image_url, discounted_price, original_price, "
                f"COALESCE(scraped_color, primary_color_name) as eff_color, "
                f"COALESCE(scraped_pattern, pattern_value) as eff_pattern, "
                f"COALESCE(scraped_sleeve, sleeve_value) as eff_sleeve, "
                f"embedding, product_url, scraped_fit, scraped_neckline "
                f"FROM visual_attributes WHERE {where}",
                var_params
            )
            rows = cur.fetchall()
            print(f"[CategoryFallback] Variant '{variant}': {len(rows)} results")

            if len(rows) >= min_results:
                return rows, relaxation_log, 5

    # =========================================================================
    # PHASE 3: CATEGORY FAMILY FALLBACK
    # =========================================================================

    if len(rows) < min_results and effective_category:
        print(f"[CategoryFamily] Trying category family fallback...")

        category_families = {
            "accessories": ["glasses", "caps", "belt", "tie", "watch",
                            "bag", "purse", "necklace", "earrings", "ring",
                            "bangle", "bracelet", "hairclip"],
            "footwear": ["Footwear_shoes", "Footwear_sandals"],
            "tops": ["shirts", "tshirt", "blouse", "blazer", "Jacket"],
            "bottoms": ["pant", "shorts", "skirt", "leggings"],
        }

        target_family = None
        for family, members in category_families.items():
            if effective_category in members or effective_category.lower() in [m.lower() for m in members]:
                target_family = family
                break

        if target_family:
            print(f"[CategoryFamily] '{effective_category}' → family '{target_family}'")
            relaxation_log.append(f"Category family: {target_family}")

            family_members = category_families[target_family]
            category_placeholders = ",".join(["%s"] * len(family_members))

            conditions = [f"category ILIKE ANY(ARRAY[{','.join(['%s']*len(family_members))}])"]
            fam_params = family_members[:]

            gender_val = overrides.get("gender") or detected_gender
            if gender_val and "gender" not in skip_overrides:
                gender_norm = SYNONYM_MAP.get(gender_val.lower(), gender_val)
                conditions.append("LOWER(gender) = LOWER(%s)")
                fam_params.append(gender_norm)

            where = " AND ".join(conditions) + " AND embedding IS NOT NULL"
            cur.execute(
                f"SELECT product_id, category, gender, style, "
                f"COALESCE(scraped_material, material) as material, "
                f"price_bucket, color_family, brand, product_name, "
                f"image_url, discounted_price, original_price, "
                f"COALESCE(scraped_color, primary_color_name) as eff_color, "
                f"COALESCE(scraped_pattern, pattern_value) as eff_pattern, "
                f"COALESCE(scraped_sleeve, sleeve_value) as eff_sleeve, "
                f"embedding, product_url, scraped_fit, scraped_neckline "
                f"FROM visual_attributes WHERE {where}",
                fam_params
            )
            rows = cur.fetchall()
            print(f"[CategoryFamily] Family '{target_family}': {len(rows)} results")

            if len(rows) >= min_results:
                return rows, relaxation_log, 5

    # =========================================================================
    # PHASE 4: ABSOLUTE LAST RESORT — gender only, LIMIT 20
    # =========================================================================

    if len(rows) < min_results:
        print(f"[FinalFallback] Removing category filter entirely...")
        relaxation_log.append("Category: removed")

        conditions = []
        f_params = []

        gender_val = overrides.get("gender") or detected_gender
        if gender_val and "gender" not in skip_overrides:
            gender_norm = SYNONYM_MAP.get(gender_val.lower(), gender_val)
            conditions.append("LOWER(gender) = LOWER(%s)")
            f_params.append(gender_norm)

        where = (" AND ".join(conditions) + " AND embedding IS NOT NULL") if conditions else "embedding IS NOT NULL"
        cur.execute(
            f"SELECT product_id, category, gender, style, "
            f"COALESCE(scraped_material, material) as material, "
            f"price_bucket, color_family, brand, product_name, "
            f"image_url, discounted_price, original_price, "
            f"COALESCE(scraped_color, primary_color_name) as eff_color, "
            f"COALESCE(scraped_pattern, pattern_value) as eff_pattern, "
            f"COALESCE(scraped_sleeve, sleeve_value) as eff_sleeve, "
            f"embedding, product_url, scraped_fit, scraped_neckline "
            f"FROM visual_attributes WHERE {where} LIMIT 20",
            f_params
        )
        rows = cur.fetchall()
        print(f"[FinalFallback] No category filter: {len(rows)} results")

    return rows, relaxation_log, 5

# =============================================================================
# MAIN RETRIEVAL FUNCTION
# =============================================================================

def search_products_v3(query_context, top_k=10):
    """
    Production retrieval with all fixes applied.
    """
    t_start = _time.time()

    query_embedding = query_context.get("embedding")
    detected_attrs = query_context.get("detected_attributes", {})
    user_filters = query_context.get("user_filters", {})
    price_max = query_context.get("price_max")
    scene_label = query_context.get("scene")
    extraction_quality = query_context.get("extraction_quality", 1.0)

    # ✅ FIX: Normalize category at the entry point using the mapping table
    raw_cat = (query_context.get("category") or
               user_filters.get("category") or
               detected_attrs.get("category"))
    effective_category = normalize_category(raw_cat) if raw_cat else None

    print(f"\n{'='*70}")
    print(f"[RetrievalV3] 🚀 PRODUCTION RETRIEVAL")
    print(f"{'='*70}")
    print(f"[RetrievalV3] Raw category input: {raw_cat} → normalized: {effective_category}")
    print(f"[RetrievalV3] Detected: {detected_attrs}")
    print(f"[RetrievalV3] User filters: {user_filters}")
    print(f"[RetrievalV3] Price max: {price_max}")

    if not effective_category:
        print("[RetrievalV3] ❌ No category")
        return {"products": [], "metadata": {"error": "no_category"}}

    intent = classify_query_intent(detected_attrs, user_filters)
    print(f"[RetrievalV3] Intent: {intent}")

    overrides, preserved, mode = _detect_overrides(detected_attrs, user_filters)

    # Drop price_bucket if numeric price_max provided
    if price_max is not None and "price_bucket" in overrides:
        print(f"[RetrievalV3] Dropping price_bucket (using price_max={price_max})")
        del overrides["price_bucket"]

    # ✅ FIX: Use normalized effective_category from overrides if present,
    #         but it is already normalized by _detect_overrides → normalize_category()
    if "category" in overrides and overrides["category"]:
        effective_category = overrides["category"]

    # Select weights
    if intent == "ATTRIBUTE_FIRST":
        weights = ATTRIBUTE_QUERY_WEIGHTS.copy()
    else:
        weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["PURE_SIMILARITY"]).copy()

    if extraction_quality < 0.5:
        scale = max(0.2, extraction_quality)
        orig_pres = weights.get("preserved", 0.15)
        weights["preserved"] = orig_pres * scale
        weights["visual"] += (orig_pres - weights["preserved"])

    detected_gender = preserved.get("gender") or detected_attrs.get("gender", "")

    print(f"[RetrievalV3] Mode: {mode}")
    print(f"[RetrievalV3] Overrides: {overrides}")
    print(f"[RetrievalV3] Preserved: {preserved}")
    print(f"[RetrievalV3] Weights: {weights}")
    print(f"[RetrievalV3] Effective category (final): {effective_category}")

    conn = get_db()
    cur = conn.cursor()

    rows, relaxation_log, final_color_tier = fetch_candidates_with_progressive_color_relaxation(
        cur, overrides, preserved, effective_category, detected_gender,
        price_max, min_results=5
    )

    print(f"[RetrievalV3] Pool: {len(rows)} (color tier: {final_color_tier})")

    if len(rows) == 0:
        print("[RetrievalV3] ❌ No candidates after all relaxation phases")
        cur.close()
        conn.close()
        return {"products": [], "metadata": {"error": "no_candidates"}}

    # =========================================================================
    # COSINE SIMILARITY
    # =========================================================================
    candidates = []
    embeddings = []

    for r in rows:
        prod = {
            "product_id": r[0],
            "category": r[1] or "",
            "gender": r[2] or "",
            "style": r[3] or "",
            "material": r[4] or "",
            "price_bucket": r[5] or "",
            "color_family": r[6] or "",
            "brand": r[7] or "",
            "name": r[8] or "Product",
            "image_url": r[9] if r[9] and str(r[9]).startswith('http') else f"/images/{r[0]}.jpg",
            "price": float(r[10]) if r[10] else (float(r[11]) if r[11] else 0.0),
            "color": r[12] or r[6] or "",
            "pattern": r[13] or "",
            "sleeve": r[14] or "",
            "product_url": r[16] or "",
            "fit": r[17] or "",
            "neckline": r[18] or "",
        }
        candidates.append(prod)

        emb = r[15]
        if emb is not None:
            if isinstance(emb, str):
                emb = json.loads(emb)
            embeddings.append(emb)
        else:
            embeddings.append([0.0] * 512)

    visual_scores = np.zeros(len(candidates), dtype=np.float32)

    if query_embedding and len(embeddings) > 0 and intent != "ATTRIBUTE_FIRST":
        gc.collect()
        q_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        try:
            BATCH_SIZE = 500
            for start_idx in range(0, len(embeddings), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(embeddings))
                batch = np.array(embeddings[start_idx:end_idx], dtype=np.float32)
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch = batch / norms
                visual_scores[start_idx:end_idx] = (batch @ q_vec.T).flatten()
                del batch, norms
        except MemoryError:
            print("[RetrievalV3] ⚠️ MemoryError in cosine similarity")
            visual_scores = np.zeros(len(candidates), dtype=np.float32)

        del embeddings
        gc.collect()

    sim_time = (_time.time() - t_start) * 1000
    print(f"[RetrievalV3] Similarity computed: {sim_time:.1f}ms")

    # =========================================================================
    # SCORING
    # =========================================================================
    w_visual = weights.get("visual", 0.40)
    w_override = weights.get("override", 0.40)
    w_preserved = weights.get("preserved", 0.20)

    override_fields = [k for k in overrides if k not in ("category", "gender")]
    preserved_fields = [k for k in preserved if k not in ("category", "gender")]

    total_overrides = len(override_fields)
    total_preserved = len(preserved_fields)

    if total_overrides == 0:
        w_override = 0
    if total_preserved == 0:
        w_preserved = 0

    total_weight = w_visual + w_override + w_preserved
    if total_weight > 0 and total_weight != 1.0:
        w_visual /= total_weight
        w_override /= total_weight
        w_preserved /= total_weight

    scored_candidates = []

    for i, prod in enumerate(candidates):
        v_score = float(visual_scores[i])

        o_score = 0.0
        color_match_level = 0

        if total_overrides > 0:
            matched_score = 0.0
            for field in override_fields:
                override_val = overrides[field].lower().strip()
                field_score = 0.0

                if field == "color":
                    prod_primary = (prod.get("color") or "").lower().strip()
                    prod_family = (prod.get("color_family") or "").lower().strip()
                    _, override_family = _normalize_color_for_sql(overrides[field])
                    if override_val == prod_primary or override_val in prod_primary:
                        field_score = 1.0
                        color_match_level = 2
                    elif override_family and override_family == prod_family:
                        field_score = 0.75
                        color_match_level = 1
                    elif override_family in prod_primary or prod_family in override_val:
                        field_score = 0.60
                        color_match_level = 1

                elif field == "sleeve":
                    prod_sleeve = (prod.get("sleeve") or "").lower()
                    if override_val == prod_sleeve:
                        field_score = 1.0
                    elif override_val in prod_sleeve or prod_sleeve in override_val:
                        field_score = 0.5

                elif field == "pattern":
                    prod_pattern = (prod.get("pattern") or "").lower()
                    if override_val == prod_pattern:
                        field_score = 1.0
                    elif override_val in prod_pattern:
                        field_score = 0.5

                matched_score += field_score
            o_score = matched_score / total_overrides

        p_score = 0.0
        if total_preserved > 0:
            weighted_matched = 0.0
            weighted_total = 0.0
            for field in preserved_fields:
                pres_val = preserved[field].lower() if preserved[field] else ""
                field_matched = False
                field_weight = 1.5 if field == "color" else 1.0
                if field == "color":
                    prod_color = (prod.get("color") or "").lower()
                    prod_family = (prod.get("color_family") or "").lower()
                    if pres_val == prod_color or pres_val in prod_color:
                        field_matched = True
                    else:
                        pres_family = COLOR_FAMILIES.get(pres_val, pres_val)
                        if pres_family == prod_family:
                            field_matched = True
                weighted_total += field_weight
                if field_matched:
                    weighted_matched += field_weight
            p_score = weighted_matched / weighted_total if weighted_total > 0 else 0.0

        final_score = (w_visual * v_score) + (w_override * o_score) + (w_preserved * p_score)

        prod["similarity_score"] = round(v_score, 4)
        prod["override_score"] = round(o_score, 4)
        prod["preserved_score"] = round(p_score, 4)
        prod["final_score"] = round(final_score, 4)
        prod["match_meta"] = {
            "visual_similarity": round(v_score, 4),
            "color_match": color_match_level > 0,
            "final_score": round(final_score, 4),
        }
        scored_candidates.append(prod)

    scored_candidates.sort(key=lambda x: (-x["final_score"], x["product_id"]))
    results = scored_candidates[:top_k]

    pool_size = len(scored_candidates)
    for rank_idx, prod in enumerate(results):
        prod["match_meta"]["rank_position"] = rank_idx + 1
        prod["match_meta"]["total_candidates"] = pool_size

    total_ms = (_time.time() - t_start) * 1000
    top5_scores = [r["final_score"] for r in results[:5]]

    print(f"[RetrievalV3] ✅ {len(results)} / {pool_size}")
    print(f"[RetrievalV3] Top-5: {top5_scores}")
    print(f"[RetrievalV3] Relaxation: {relaxation_log}")
    print(f"[RetrievalV3] Time: {total_ms:.1f}ms")
    print(f"{'='*70}\n")

    cur.close()
    conn.close()

    return {
        "products": results,
        "metadata": {
            "mode": mode,
            "intent": intent,
            "weights": {
                "visual": round(w_visual, 2),
                "override": round(w_override, 2),
                "preserved": round(w_preserved, 2)
            },
            "pool_size": pool_size,
            "result_count": len(results),
            "relaxation_log": relaxation_log,
            "color_tier": final_color_tier,
            "search_time_ms": round(total_ms, 1),
        }
    }


def search_products_v2(query_context, top_k=10):
    """Legacy redirect."""
    print("[DEPRECATED] search_products_v2 → v3")
    return search_products_v3(query_context, top_k)