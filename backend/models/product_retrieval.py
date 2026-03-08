"""
Product Retrieval Module - Hybrid (PostgreSQL + FAISS)
Queries the products table for matching products based on filters.
Supports vector similarity search if embedding is provided.
"""
import psycopg2
import os
import faiss
import json
import gc
import numpy as np

# Centralized DB config (Supabase in production, local fallback)
from db_config import DB_CONFIG

# FAISS paths - data is at project_root/data, not backend/data
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # ShopWhatYouSee/
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')  # ShopWhatYouSee/data/
INDEX_PATH = os.path.join(DATA_DIR, 'products.faiss')
ID_MAP_PATH = os.path.join(DATA_DIR, 'product_ids.json')

print(f"[ProductRetrieval] FAISS will load from: {INDEX_PATH}")

# Global FAISS index cache
faiss_index = None
id_map = None

# --------------------------------------------------
# Load FAISS (Lazy Loading)
# --------------------------------------------------
def load_faiss():
    global faiss_index, id_map
    if faiss_index is not None:
        return True

    print(f"[FAISS] Checking paths...")
    print(f"[FAISS] INDEX_PATH: {INDEX_PATH} (exists: {os.path.exists(INDEX_PATH)})")
    print(f"[FAISS] ID_MAP_PATH: {ID_MAP_PATH} (exists: {os.path.exists(ID_MAP_PATH)})")

    if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        try:
            print(f"[FAISS] Loading index from {INDEX_PATH}...")
            faiss_index = faiss.read_index(INDEX_PATH)
            with open(ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
            # Convert string keys to int
            id_map = {int(k): v for k, v in id_map.items()}
            print(f"[FAISS] [OK] Index loaded with {faiss_index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"[FAISS] [ERR] Failed to load index: {e}")
            faiss_index = None
            return False
    else:
        print("[FAISS] ❌ Index files not found. Skipping vector search.")
        return False

def load_index():
    """Helper for evaluation scripts to access the index directly"""
    load_faiss()
    if id_map:
        return faiss_index, list(id_map.keys()), id_map
    return None, [], {}

# --------------------------------------------------
# Database connection
# --------------------------------------------------
def get_db():
    return psycopg2.connect(**DB_CONFIG)

# --------------------------------------------------
# Category normalization
# --------------------------------------------------
def normalize_category(cat):
    """Map user/YOLO category names to exact DB visual_attributes.category values."""
    if not cat: return None
    cat = cat.lower().strip().replace("-", "_")  # t-shirt -> t_shirt
    # =================================================================
    # VALUES MUST MATCH visual_attributes.category EXACTLY
    # DB categories: shirts, tshirt, blazer, Jacket, pant, shorts,
    #   skirt, churidhar, dhoti, Footwear_sandals, Footwear_shoes,
    #   caps, glasses, belt, tie, earrings, necklace, watch
    # =================================================================
    mapping = {
        # Upper wear
        "shirt": "shirts", "shirts": "shirts",
        "tee": "tshirt", "tshirt": "tshirt", "t_shirt": "tshirt",
        "blazer": "blazer",
        "coat": "Jacket", "jacket": "Jacket",
        # Lower wear
        "pant": "pant", "pants": "pant", "jeans": "pant", "trousers": "pant",
        "short": "shorts", "shorts": "shorts",
        "skirt": "skirt",
        # Traditional
        "churidhar": "churidhar", "dhoti": "dhoti",
        "kurta": "churidhar", "kurti": "churidhar",
        "saree": "churidhar",
        # Footwear
        "shoe": "Footwear_shoes", "shoes": "Footwear_shoes",
        "sandal": "Footwear_sandals", "sandals": "Footwear_sandals",
        # Accessories
        "cap": "caps", "hat": "caps", "caps": "caps",
        "glass": "glasses", "sunglass": "glasses", "glasses": "glasses",
        "sunglasses": "glasses", "eyewear": "glasses",
        "belt": "belt", "tie": "tie",
        "earring": "earrings", "earrings": "earrings",
        "necklace": "necklace", "ring": "necklace",
        "watch": "watch",
        # Other
        "bag": "belt", "purse": "belt", "handbag": "belt",
        "dress": "shirts", "dresses": "shirts",
        "sweater": "tshirt", "hoodie": "tshirt", "sweatshirt": "tshirt",
        "leggings": "pant",
    }
    result = mapping.get(cat)
    if result:
        return result
    # Fallback: try the raw value (LOWER comparison in SQL handles case)
    return cat

# ==================================================
# PRODUCTION-GRADE RETRIEVAL CONFIGURATION
# ==================================================

# Scoring weights — VISUAL-FIRST queries (default)
SCORING_WEIGHTS = {
    "embedding": 0.40,   # Visual similarity (PRIMARY - style anchor)
    "color": 0.25,       # Color match (important for fashion)
    "attributes": 0.20,  # Pattern, sleeve, fit match
    "scene": 0.10,       # Scene relevance
    "relaxation": -0.05  # Penalty for relaxed matches
}

# Scoring weights — ATTRIBUTE-FIRST queries (e.g. "give full sleeve")
ATTRIBUTE_QUERY_WEIGHTS = {
    "embedding": 0.20,   # Secondary for attribute queries
    "color": 0.15,       # Lower priority
    "attributes": 0.60,  # ← PRIMARY: attribute match MUST dominate
    "scene": 0.00,       # Irrelevant for attribute queries
    "relaxation": -0.05
}

# Relaxation order: ONLY price and color can be relaxed.
# Sleeve, pattern, fit are NEVER relaxed (user explicitly asked for them).
RELAXATION_ORDER = ["price_max", "color"]

# Attributes that must NEVER be relaxed — programmatic enforcement
NEVER_RELAX = {"category", "gender", "sleeve", "pattern", "fit"}
assert not (set(RELAXATION_ORDER) & NEVER_RELAX), (
    f"RELAXATION_ORDER contains protected attributes: {set(RELAXATION_ORDER) & NEVER_RELAX}"
)


def classify_query_intent(hard_constraints, soft_preferences):
    """
    Classify what the user wants to prioritize.
    
    Returns:
        "ATTRIBUTE_FIRST" - user wants sleeve/pattern/fit/color (bypass FAISS, DB-first)
        "VISUAL_FIRST"    - no specific attributes requested (use FAISS pipeline)
        "HYBRID"          - user wants both visual similarity AND attributes
    """
    all_constraints = {}
    all_constraints.update(hard_constraints)
    all_constraints.update(soft_preferences)
    
    has_attributes = bool(
        all_constraints.get("sleeve") or
        all_constraints.get("pattern") or
        all_constraints.get("fit")
    )
    
    # Check if color is user-requested (from LLM) vs visual detection
    color_constraint = all_constraints.get("color")
    has_user_color = False
    if color_constraint:
        if isinstance(color_constraint, dict):
            # User-requested color (from LLM query) should skip FAISS
            has_user_color = color_constraint.get("source") in ("llm", "query", "user")
        else:
            # Plain string value = assume user-requested
            has_user_color = True
    
    if has_attributes or has_user_color:
        return "ATTRIBUTE_FIRST"
    elif color_constraint and not has_user_color:
        # Color from visual detection only — use FAISS
        return "VISUAL_FIRST"
    else:
        return "VISUAL_FIRST"  # Default: visual similarity

# --------------------------------------------------
# Canonical Synonym Map
# Maps common variations to DB-canonical values.
# DB stores: gender as 'Men','Women','Boys','Girls','Unisex'
# DB stores: base_colour as title-case ('Grey','Red', etc.)
# DB stores: agman_pattern as lowercase ('solid','striped', etc.)
# --------------------------------------------------
SYNONYM_MAP = {
    # Color synonyms → DB canonical (title-case base_colour)
    "gray": "grey",
    "grey melange": "grey",
    "off white": "off white",
    "navy blue": "navy blue",
    "sky blue": "blue",
    # Pattern synonyms → DB canonical (lowercase agman_pattern)
    "plain": "solid",
    "none": "solid",
    "checks": "checked",
    "check": "checked",
    "stripes": "striped",
    "stripe": "striped",
    "plaid": "checked",
    # Sleeve synonyms → DB canonical
    "full sleeve": "long sleeves",
    "full sleeves": "long sleeves",
    "long sleeve": "long sleeves",
    "half sleeve": "short sleeves",
    "half sleeves": "short sleeves",
    "3/4 sleeve": "three-quarter sleeves",
    "three quarter": "three-quarter sleeves",
    # Gender synonyms → DB canonical
    "male": "Men",
    "female": "Women",
    "man": "Men",
    "woman": "Women",
    "boys": "Boys",
    "girls": "Girls",
}


def normalize_filter_value(value):
    """Normalize a filter value: lowercase, strip, apply synonym map."""
    if not value or not isinstance(value, str):
        return value
    normalized = value.lower().strip()
    return SYNONYM_MAP.get(normalized, normalized)


# Color families for partial matching
# MUST match the DB color_family values from _add_bucket_family.py:
#   neutral, red, pink, orange, yellow, green, blue, purple, brown
COLOR_FAMILIES = {
    # Neutral family (black, white, grey, silver, cream, off white, charcoal)
    "black": "neutral", "white": "neutral", "grey": "neutral",
    "charcoal": "neutral", "silver": "neutral", "off white": "neutral",
    "cream": "neutral", "multi": "neutral", "dark grey": "neutral",
    "light grey": "neutral", "grey melange": "neutral", "steel": "neutral",
    "ivory": "neutral", "nude": "neutral", "champagne": "neutral",
    "transparent": "neutral",
    # Red family
    "maroon": "red", "burgundy": "red", "wine": "red",
    "crimson": "red", "rust": "red",
    # Pink family
    "hot pink": "pink", "magenta": "pink", "peach": "pink",
    "coral": "pink", "rose": "pink", "fuchsia": "pink",
    # Orange family (includes gold, mustard per DB)
    "mustard": "orange", "gold": "orange",
    # Yellow family
    # (Yellow is its own family in DB)
    # Green family
    "olive": "green", "lime": "green", "khaki": "green",
    "sea green": "green", "lime green": "green",
    "fluorescent green": "green",
    # Blue family (includes teal, turquoise, navy per DB)
    "navy blue": "blue", "navy": "blue", "sky blue": "blue",
    "steel blue": "blue", "teal": "blue", "turquoise blue": "blue",
    # Purple family
    "lavender": "purple", "violet": "purple", "mauve": "purple",
    # Brown family (includes beige, tan, coffee, taupe per DB)
    "tan": "brown", "chocolate": "brown", "taupe": "brown",
    "coffee": "brown", "coffee brown": "brown", "camel brown": "brown",
    "bronze": "brown", "copper": "brown", "rose gold": "brown",
}

# Scene context → suggested style (Places365 labels → product style)
SCENE_TO_STYLE = {
    # Formal / Office
    "office": "formal", "office_building": "formal", "conference_room": "formal",
    "boardroom": "formal", "reception": "formal", "lobby": "formal",
    # Casual / Outdoor
    "park": "casual", "garden": "casual", "playground": "casual",
    "picnic_area": "casual", "campsite": "casual", "forest_road": "casual",
    "bamboo_forest": "casual", "field": "casual", "meadow": "casual",
    # Sports / Active
    "gymnasium": "sports", "athletic_field": "sports", "stadium": "sports",
    "swimming_pool": "sports", "track": "sports", "yoga_studio": "sports",
    # Beach / Resort
    "beach": "casual", "coast": "casual", "ocean": "casual",
    "swimming_pool": "casual", "resort": "casual",
    # Party / Night
    "nightclub": "party", "bar": "party", "ballroom": "party",
    "discotheque": "party", "banquet_hall": "party",
    # Street
    "street": "casual", "shopping_mall": "casual", "market": "casual",
    "downtown": "casual", "alley": "casual",
}

# Minimum results before relaxation is needed
MIN_RESULTS = 5

# ==================================================
# Hybrid Retrieval
# ==================================================
def search_products(filters, top_k=50):
    from utils.color_utils import hex_to_color_name, normalize_color_name
    
    category = normalize_category(filters.get("category"))
    # Prioritize LLM-derived color name over AGMAN hex
    color_raw = filters.get("color_name") or filters.get("color")
    price_max = filters.get("price_max")
    query_embedding = filters.get("embedding") # List of floats
    
    # Convert hex color to color name for better matching
    if color_raw and color_raw.startswith("#"):
        color = hex_to_color_name(color_raw)
        print(f"[Search] Color: {color_raw} -> {color}")
    else:
        color = normalize_color_name(color_raw) if color_raw else None

    conn = get_db()
    cur = conn.cursor()
    
    # --------------------------------------------------
    # 1. VECTOR SEARCH (if embedding provided)
    # --------------------------------------------------
    vector_ids = []
    vector_scores = {}  # CRITICAL FIX: Initialize to prevent NameError if FAISS skipped
    print(f"\n[Search] Category: {category}, Color: {color}, Has embedding: {query_embedding is not None}")
    
    if query_embedding:
        print(f"[Search] Embedding provided ({len(query_embedding)} dims). Attempting FAISS search...")
        faiss_loaded = load_faiss()
        if faiss_loaded and faiss_index and id_map:
            try:
                # Convert to float32 numpy array
                xq = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(xq)
                
                # Search top 50 matches
                D, I = faiss_index.search(xq, 50)
                
                # Log detailed FAISS scores
                print(f"[Search] FAISS returned {len(I[0])} indices.")
                print(f"[Search] Top 5 Scores (Distances): {D[0][:5]}")
                print(f"[Search] Top 5 Indices: {I[0][:5]}")
                
                # Check for bad scores (typically > 300-400 for L2 with 512 dims means poor match)
                if D[0][0] > 1000:
                    print(f"[Search] [WARN] Nearest neighbor distance is high ({D[0][0]}), matches may be irrelevant.")
                
                # Map FAISS IDs back to Product IDs (int) with scores
                vector_scores = {}  # product_id -> similarity_score
                for i, idx in enumerate(I[0]):
                    if idx != -1 and idx in id_map:
                        product_id = int(id_map[idx])
                        vector_ids.append(product_id)
                        # Store similarity score (lower distance = higher similarity)
                        # Convert to 0-1 scale (invert distance)
                        vector_scores[product_id] = float(1.0 / (1.0 + D[0][i]))
                
                print(f"[Search] [OK] Vector search found {len(vector_ids)} candidate products")
                if vector_ids:
                    print(f"[Search] Top vector IDs: {vector_ids[:5]}")
            except Exception as e:
                print(f"[Search] [ERR] Vector search failed: {e}")
        else:
            print("[Search] [WARN] FAISS not available, falling back to text search")
    else:
        print("[Search] [WARN] No embedding provided, using text search only")

    results = []
    seen_ids = set()

    # --------------------------------------------------
    # Helper - Run SQL Query
    # --------------------------------------------------
    def run_query(base_query, params):
        q = f"""
            SELECT id, product_id, product_name, price, brand, base_colour, image_url
            FROM visual_attributes
            WHERE {base_query}
        """
        # Price Filter
        if price_max:
            try:
                q += f" AND price <= {float(price_max)}"
            except: pass
            
        # Vector Filter (if we have vector results, prioritize them)
        if vector_ids:
            # We want products THAT ARE IN vector_ids AND match category/color
            # But PostgreSQL doesn't have list inputs easily without ANY()
            # Construct a safe list string
            ids_str = ",".join(str(vid) for vid in vector_ids)
            q += f" AND product_id IN ({ids_str})"
            
            # Order by explicit ordering of vector_ids (nearest first)
            # This is complex in standard SQL without joining a values table.
            # Simplified: Just filter. We will re-sort in Python if needed.
        
        q += " LIMIT 20"
        
        cur.execute(q, params)
        rows = cur.fetchall()
        
        # If we used vector_ids, sort rows by the order in vector_ids
        if vector_ids:
            row_map = {r[1]: r for r in rows} # product_id -> row
            sorted_rows = []
            for vid in vector_ids:
                if vid in row_map:
                    sorted_rows.append(row_map[vid])
            return sorted_rows
        
        return rows

    def add_results(rows):
        for r in rows:
            pid = r[0]
            if pid not in seen_ids and len(results) < 5:
                seen_ids.add(pid)
                product_dict = {
                    "id": r[0],
                    "product_id": r[1],
                    "name": r[2] or "Product",
                    "price": float(r[3]) if r[3] else 0.0,
                    "brand": r[4] or "",
                    "color": r[5] or "",
                    # Use correct image path: if DB has http URL use it, otherwise use local path
                    "image_url": r[6] if r[6] and r[6].startswith('http') else f"/images/{r[1]}.jpg"
                }

                # CRITICAL: Check if image exists on disk (if local)
                if "localhost" in product_dict["image_url"]:
                    local_path = os.path.join(r"D:\Final_Year_Project\ShopWhatYouSee\data\images", f"{r[1]}.jpg")
                    if not os.path.exists(local_path):
                        # print(f"[Search] Skipping {r[1]} - Image missing: {local_path}")
                        continue

                # Add similarity score if available from FAISS
                if r[1] in vector_scores:
                    product_dict["similarity_score"] = vector_scores[r[1]]
                else:
                    product_dict["similarity_score"] = 0.0
                results.append(product_dict)

    # --------------------------------------------------
    # Stage 1: Vector Search + Category + COLOR Filter
    # CRITICAL: Must filter by color here to respect LLM's color_name
    # --------------------------------------------------
    if category and vector_ids:
        print(f"[Search] Stage 1: Checking {len(vector_ids)} vector candidates with filters: category={category}, color={color}")
        
        if color:
            # Filter by BOTH category AND color (LLM filter respected!)
            rows = run_query(
                "LOWER(yolo_category) = LOWER(%s) AND (LOWER(base_colour) LIKE LOWER(%s) OR LOWER(COALESCE(primary_color, '')) LIKE LOWER(%s))",
                [category, f"%{color}%", f"%{color}%"]
            )
        else:
            # No color filter specified, just category
            rows = run_query(
                "LOWER(yolo_category) = LOWER(%s)", 
                [category]
            )
        add_results(rows)
        print(f"[Search] Stage 1 results: {len(results)}")

    # --------------------------------------------------
    # Stage 2: Fallback Category + Color (without vector IDs)
    # --------------------------------------------------
    # Only run if we don't have enough results from Stage 1
    if len(results) < 5 and category and color:
        print(f"[Search] Stage 2: Text-only search for {category} + {color}...")
        # Remove vector_ids constraint for broader search
        q = f"""
            SELECT id, product_id, product_name, price, brand, base_colour, image_url
            FROM visual_attributes
            WHERE LOWER(yolo_category) = LOWER(%s) 
            AND (LOWER(base_colour) LIKE LOWER(%s) OR LOWER(COALESCE(primary_color, '')) LIKE LOWER(%s))
            LIMIT 20
        """
        cur.execute(q, [category, f"%{color}%", f"%{color}%"])
        extra_rows = cur.fetchall()
        for r in extra_rows:
            if r[0] not in seen_ids and len(results) < 5:
                seen_ids.add(r[0])
                results.append({
                    "id": r[0],
                    "product_id": r[1],
                    "name": r[2] or "Product",
                    "price": float(r[3]) if r[3] else 0.0,
                    "brand": r[4] or "",
                    "color": r[5] or "",
                    "image_url": r[6] if r[6] and r[6].startswith('http') else f"/images/{r[1]}.jpg",
                    "similarity_score": 0.5  # Lower score for text-only matches
                })
        print(f"[Search] Stage 2 results total: {len(results)}")

    # --------------------------------------------------
    # CRITICAL FALLBACK CONTROLS
    # --------------------------------------------------
    # If we found matches in Stage 1 or 2 (Vector or Color), DO NOT fill with random items
    # unless we have very few results (e.g., < 2)
    has_specific_matches = len(results) > 0

    # --------------------------------------------------
    # Stage 3: Category Only (Generic Fallback)
    # --------------------------------------------------
    # Only run if we have essentially NO results from specific searches
    if not has_specific_matches and len(results) < 5 and category:
        print(f"[Search] Stage 3: Generic category fallback (no specific matches found)...")
        rows = run_query(
            "LOWER(yolo_category) = LOWER(%s)",
            [category]
        )
        add_results(rows)
        
    # --------------------------------------------------
    # Stage 4: Fuzzy Fallback
    # --------------------------------------------------
    if not has_specific_matches and len(results) < 5 and category:
         print(f"[Search] Stage 4: Fuzzy fallback...")
         rows = run_query(
            "LOWER(yolo_category) LIKE LOWER(%s)",
            [f"%{category}%"]
        )
         add_results(rows)

    print(f"[Search] Final results count: {len(results)}")
    
    # --------------------------------------------------
    # PROGRESSIVE FILTER RELAXATION (Ensure min 5 results)
    # Priority order: price_max → pattern → sleeve → COLOR LAST
    # --------------------------------------------------
    MIN_RESULTS = 5
    relaxation_order = ['price_max', 'pattern', 'sleeve', 'color']  # COLOR LAST
    
    if len(results) < MIN_RESULTS and category:
        print(f"\n[Search] ⚠️ Only {len(results)} results, need {MIN_RESULTS}. Relaxing filters...")
        
        relaxed_filters = filters.copy()
        
        for attr in relaxation_order:
            if attr in relaxed_filters and relaxed_filters.get(attr):
                print(f"[Search] 🔓 Relaxing filter: {attr} = {relaxed_filters[attr]}")
                del relaxed_filters[attr]
                
                # Re-run search with relaxed filters
                # Simple fallback - just get more products from category
                try:
                    q = """
                        SELECT id, product_id, product_name, price, brand, base_colour, image_url
                        FROM visual_attributes
                        WHERE LOWER(yolo_category) = LOWER(%s)
                        LIMIT 20
                    """
                    cur.execute(q, [category])
                    extra_rows = cur.fetchall()
                    
                    for r in extra_rows:
                        if r[0] not in seen_ids and len(results) < 5:
                            seen_ids.add(r[0])
                            results.append({
                                "id": r[0],
                                "product_id": r[1],
                                "name": r[2] or "Fashion Item",
                                "price": float(r[3]) if r[3] else 999.0,
                                "brand": r[4] or "Unknown",
                                "color": r[5] or "N/A",
                                "image_url": r[6] or f"/images/{r[1]}.jpg",
                                "similarity_score": 0.3,  # Lower score for relaxed matches
                                "relaxed_match": True
                            })
                    
                    print(f"[Search] ➕ After relaxing {attr}: {len(results)} results")
                    
                    if len(results) >= MIN_RESULTS:
                        print(f"[Search] ✅ Reached {MIN_RESULTS} results, stopping relaxation")
                        break
                        
                except Exception as e:
                    print(f"[Search] Relaxation error: {e}")
    
    
    # --------------------------------------------------
    # COLOR RE-RANKING (User Request: Same/Close color first)
    # --------------------------------------------------
    # Map specific shades to broader families
    COLOR_FAMILIES = {
        "maroon": "red", "burgundy": "red", "wine": "red", "crimson": "red", "rust": "red",
        "navy blue": "blue", "sky blue": "blue", "steel blue": "blue",
        "off white": "white", "cream": "white", "ivory": "white", "beige": "white", "nude": "white",
        "charcoal": "black", "dark grey": "grey", "light grey": "grey", "silver": "grey",
        "olive": "green", "lime": "green", "khaki": "green", "sea green": "green",
        "mustard": "yellow", "gold": "yellow",
        "hot pink": "pink", "magenta": "pink", "peach": "pink", "coral": "pink",
        "tan": "brown", "chocolate": "brown", "taupe": "brown", "coffee": "brown",
        "lavender": "purple", "violet": "purple", "mauve": "purple"
    }

    if color_raw:
        target_c = color_raw.lower().strip()
        target_family = COLOR_FAMILIES.get(target_c, target_c)
        
        print(f"[Search] 🎨 Re-ranking results for color: {target_c} (Family: {target_family})")
        
        for res in results:
            p_color = (res.get("color") or "").lower().strip()
            p_family = COLOR_FAMILIES.get(p_color, p_color)
            
            current_score = res.get("similarity_score", 0)
            
            # 1. Exact Match (Huge Boost)
            if p_color == target_c:
                res["similarity_score"] = current_score + 0.5
                res["color_match"] = "exact"
                
            # 2. Family Match (Strong Boost) - e.g. "Red" matches "Maroon"
            elif p_family == target_family:
                res["similarity_score"] = current_score + 0.35
                res["color_match"] = "family"

            # 3. Partial Literal Match (Medium Boost) - e.g. "Dark Blue" matches "Blue"
            elif target_c in p_color or p_color in target_c:
                res["similarity_score"] = current_score + 0.2
                res["color_match"] = "partial"
                
            # 4. No match (No boost)

    # Sort by similarity score (descending)
    results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    print(f"[Search] ✅ Returning {len(results)} products (sorted by similarity)")
    
    cur.close()
    conn.close()
    return results


# ==================================================
# RETRIEVAL ARCHITECTURE v2 — SQL-First + Cosine Ranking
# ==================================================
#
# Core Principle: SQL builds candidate pool → numpy cosine similarity ranks.
#                 FAISS never restricts the candidate pool.
#
# Modes:
#   PURE_SIMILARITY     — no user overrides, rank by visual similarity
#   ATTRIBUTE_OVERRIDE  — user overrides color/sleeve/etc, same category
#   CATEGORY_TRANSFORM  — user requests different category
#   COMBINED            — category + attribute overrides
#
# Tables: visual_attributes (Supabase/PostgreSQL)
# ==================================================

# Dynamic scoring weights per mode
MODE_WEIGHTS = {
    "PURE_SIMILARITY":    {"visual": 0.85, "override": 0.00, "preserved": 0.15},
    "ATTRIBUTE_OVERRIDE": {"visual": 0.50, "override": 0.40, "preserved": 0.10},
    "CATEGORY_TRANSFORM": {"visual": 0.60, "override": 0.20, "preserved": 0.20},
    "COMBINED":           {"visual": 0.45, "override": 0.35, "preserved": 0.20},
}

# Override strictness defaults
# HARD = never relax, SOFT = can be relaxed if pool empty
HARD_OVERRIDE_FIELDS = {"category", "gender"}
SOFT_OVERRIDE_FIELDS = {"price_bucket", "color", "material", "style", "sleeve", "pattern"}

# Relaxation order for SOFT overrides only
RELAX_ORDER = ["price_bucket", "material", "style", "pattern", "sleeve", "color"]


def _detect_overrides(detected_attributes, user_filters):
    """
    Compare AG-MAN detected attributes vs LLM user filters.
    
    Returns:
        overrides: dict of {field: value} — user explicitly changed these
        preserved: dict of {field: value} — keep from detection, scoring only
        mode: one of PURE_SIMILARITY, ATTRIBUTE_OVERRIDE, CATEGORY_TRANSFORM, COMBINED
    """
    overrides = {}
    preserved = {}
    category_changed = False
    
    # Extract detected values
    det_category = detected_attributes.get("category", "")
    det_color = detected_attributes.get("color") or detected_attributes.get("color_name", "")
    det_sleeve = detected_attributes.get("sleeve") or detected_attributes.get("sleeve_length", "")
    det_pattern = detected_attributes.get("pattern", "")
    det_gender = detected_attributes.get("gender", "")
    
    # Extract user values (from LLM filters)
    usr_category = user_filters.get("category", "")
    usr_color = user_filters.get("color") or user_filters.get("primary_color_name") or user_filters.get("color_family", "")
    usr_sleeve = user_filters.get("sleeve") or user_filters.get("sleeve_value", "")
    usr_pattern = user_filters.get("pattern") or user_filters.get("pattern_value", "")
    usr_gender = user_filters.get("gender", "")
    usr_material = user_filters.get("material", "")
    usr_style = user_filters.get("style", "")
    usr_price = user_filters.get("price_bucket", "")
    
    # --- Category override ---
    if usr_category:
        norm_usr = normalize_category(usr_category)
        norm_det = normalize_category(det_category) if det_category else ""
        if norm_usr and norm_det and norm_usr.lower() != norm_det.lower():
            overrides["category"] = norm_usr
            category_changed = True
        elif norm_usr:
            # Same category or no detected → still set it
            overrides["category"] = norm_usr
    
    # --- Color override ---
    if usr_color:
        overrides["color"] = normalize_filter_value(usr_color)
    elif det_color:
        preserved["color"] = normalize_filter_value(det_color)
    
    # --- Sleeve override ---
    if usr_sleeve:
        overrides["sleeve"] = normalize_filter_value(usr_sleeve)
    elif det_sleeve:
        preserved["sleeve"] = normalize_filter_value(det_sleeve)
    
    # --- Pattern override ---
    if usr_pattern:
        overrides["pattern"] = normalize_filter_value(usr_pattern)
    elif det_pattern:
        preserved["pattern"] = normalize_filter_value(det_pattern)
    
    # --- Gender override ---
    if usr_gender:
        overrides["gender"] = normalize_filter_value(usr_gender)
    elif det_gender:
        preserved["gender"] = normalize_filter_value(det_gender)
    
    # --- Soft overrides (material, style, price) ---
    if usr_material:
        overrides["material"] = normalize_filter_value(usr_material)
    if usr_style:
        overrides["style"] = normalize_filter_value(usr_style)
    if usr_price:
        overrides["price_bucket"] = usr_price
    
    # --- In CATEGORY_TRANSFORM: remove category from preserved ---
    if category_changed:
        preserved.pop("category", None)
    
    # --- Determine mode ---
    has_attr_overrides = any(
        k in overrides for k in ("color", "sleeve", "pattern", "material", "style", "price_bucket")
    )
    
    if category_changed and has_attr_overrides:
        mode = "COMBINED"
    elif category_changed:
        mode = "CATEGORY_TRANSFORM"
    elif has_attr_overrides:
        mode = "ATTRIBUTE_OVERRIDE"
    else:
        mode = "PURE_SIMILARITY"
    
    return overrides, preserved, mode


def _normalize_color_for_sql(color_value):
    """
    Normalize a user color to both primary_color_name and color_family
    for SQL matching.
    
    Returns:
        (primary_name, family_name) — e.g. ("Maroon", "red") or ("Red", "red")
    """
    if not color_value:
        return None, None
    
    color_lower = color_value.lower().strip()
    
    # Map to color family
    family = COLOR_FAMILIES.get(color_lower, color_lower)
    
    # Map to canonical primary name (title-case for DB matching)
    from utils.color_utils import normalize_color_name
    primary = normalize_color_name(color_value)
    
    return primary, family


def _build_candidate_sql(overrides, preserved, effective_category, detected_gender, skip_overrides=None):
    """
    Build SQL WHERE clause for candidate pool.
    
    Rules:
      - Only explicit overrides become SQL constraints
      - Preserved attributes NEVER in SQL (scoring only)
      - Gender preserved as constraint if not overridden
    
    Args:
        overrides: dict of user overrides
        preserved: dict of preserved attrs (NOT used in SQL)
        effective_category: normalized category for pool
        detected_gender: gender to preserve if not overridden
        skip_overrides: set of override keys to skip (for relaxation)
    
    Returns:
        (where_clause, params_list)
    """
    conditions = []
    params = []
    skip = skip_overrides or set()
    
    # --- Category: always in SQL ---
    if effective_category:
        conditions.append("LOWER(category) = LOWER(%s)")
        params.append(effective_category)
    
    # --- Gender: preserved if not overridden ---
    gender_val = overrides.get("gender") or detected_gender
    if gender_val and "gender" not in skip:
        gender_norm = SYNONYM_MAP.get(gender_val.lower().strip(), gender_val)
        conditions.append("LOWER(gender) = LOWER(%s)")
        params.append(gender_norm)
    
    # --- Override attributes in SQL ---
    for field, value in overrides.items():
        if field in skip or field in ("category", "gender"):
            continue
        if not value:
            continue
            
        if field == "color":
            primary, family = _normalize_color_for_sql(value)
            if primary and family:
                conditions.append(
                    "(LOWER(COALESCE(scraped_color, primary_color_name, '')) = LOWER(%s) "
                    "OR LOWER(color_family) = LOWER(%s) "
                    "OR LOWER(COALESCE(scraped_color, primary_color_name, '')) LIKE LOWER(%s))"
                )
                params.extend([primary, family, f"%{family}%"])
            elif primary:
                conditions.append("LOWER(COALESCE(scraped_color, primary_color_name, '')) = LOWER(%s)")
                params.append(primary)
        elif field == "sleeve":
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
        elif field == "price_bucket":
            conditions.append("LOWER(COALESCE(price_bucket, '')) = LOWER(%s)")
            params.append(value)
    
    where = " AND ".join(conditions) if conditions else "TRUE"
    return where, params


def search_products_v2(query_context, top_k=10):
    """
    SQL-First Retrieval Architecture v2.
    
    1. Detect overrides (compare detected vs user attributes)
    2. Build SQL candidate pool (only overrides in WHERE, never preserved)
    3. Compute vectorized numpy cosine similarity
    4. Score with dynamic weights per mode
    5. Progressive relaxation if pool empty
    
    Args:
        query_context: {
            "category": str,
            "embedding": list,
            "detected_attributes": {category, color_name, sleeve, pattern, gender},
            "user_filters": {category, color, sleeve, pattern, ...},
            "hard_constraints": {...},  (legacy, converted to user_filters)
            "soft_preferences": {...},  (legacy, converted to user_filters)
        }
    
    Returns:
        {"products": [...], "metadata": {...}}
    """
    import time as _time
    t_start = _time.time()
    
    # =========================================================
    # EXTRACT INPUTS
    # =========================================================
    query_embedding = query_context.get("embedding")
    detected_attrs = query_context.get("detected_attributes", {})
    user_filters = query_context.get("user_filters", {})
    
    # Legacy compat: if hard_constraints/soft_preferences present, merge into user_filters
    hc = query_context.get("hard_constraints", {})
    sp = query_context.get("soft_preferences", {})
    if (hc or sp) and not user_filters:
        user_filters = {}
        for key, val in {**hc, **sp}.items():
            if isinstance(val, dict):
                user_filters[key] = val.get("value", "")
            else:
                user_filters[key] = val
    
    # Category resolution
    raw_cat = query_context.get("category") or user_filters.get("category") or detected_attrs.get("category")
    effective_category = normalize_category(raw_cat) if raw_cat else None
    
    # Price cap (numeric, e.g. 500 for "under 500")
    price_max = query_context.get("price_max")
    
    # Ensure detected_attrs has category
    if not detected_attrs.get("category") and query_context.get("category"):
        detected_attrs["category"] = query_context["category"]
    
    print(f"\n{'='*60}")
    print(f"[RetrievalV2] 🚀 SQL-FIRST RETRIEVAL")
    print(f"{'='*60}")
    print(f"[RetrievalV2] Category: {effective_category}")
    print(f"[RetrievalV2] Detected attrs: {detected_attrs}")
    print(f"[RetrievalV2] User filters: {user_filters}")
    print(f"[RetrievalV2] Has embedding: {query_embedding is not None}")
    
    # Scene context (from Places365)
    scene_label = query_context.get("scene")
    scene_style = None
    if scene_label:
        scene_style = SCENE_TO_STYLE.get(scene_label)
        print(f"[RetrievalV2] 🎬 Scene: {scene_label}" + (f" → style hint: {scene_style}" if scene_style else ""))
    
    if not effective_category:
        print(f"[RetrievalV2] ❌ No category — returning empty")
        return {"products": [], "metadata": {"error": "no_category"}}
    
    # =========================================================
    # STEP 0: DETECT OVERRIDES
    # =========================================================
    overrides, preserved, mode = _detect_overrides(detected_attrs, user_filters)
    
    # CRITICAL: If numeric price_max is present, DROP price_bucket from overrides
    # to avoid SQL conflict (bucket "mid"=500-1499 + price<=500 excludes "budget" items <500)
    if price_max is not None and "price_bucket" in overrides:
        print(f"[RetrievalV2] 💰 Dropping price_bucket override ('{overrides['price_bucket']}') — using numeric price_max={price_max} instead")
        del overrides["price_bucket"]
    
    # CRITICAL: If user overrode category, USE the override for SQL
    if "category" in overrides and overrides["category"]:
        effective_category = overrides["category"]
    
    weights = MODE_WEIGHTS[mode].copy()  # copy so we can scale

    # ── Extraction quality scaling (Problem 3) ──
    # If AGMAN extraction quality is low, reduce trust in preserved attrs
    eq = query_context.get("extraction_quality", 1.0)
    if eq is not None and eq < 0.5:
        scale = max(0.2, eq)  # floor at 0.2 to avoid zeroing out
        original_preserved = weights["preserved"]
        weights["preserved"] = round(original_preserved * scale, 4)
        weights["visual"] = round(weights["visual"] + (original_preserved - weights["preserved"]), 4)
        print(f"[RetrievalV2] ⚠️ Low extraction_quality={eq:.2f}, "
              f"preserved weight {original_preserved:.3f} → {weights['preserved']:.3f}")
    
    # Detected gender for SQL constraint (preserved if not overridden)
    detected_gender = preserved.get("gender") or detected_attrs.get("gender", "")
    
    print(f"[RetrievalV2] Mode: {mode}")
    print(f"[RetrievalV2] Overrides: {overrides}")
    print(f"[RetrievalV2] Preserved (scoring only): {preserved}")
    print(f"[RetrievalV2] Weights: {weights}")
    print(f"[RetrievalV2] Detected gender: {detected_gender}")
    
    # =========================================================
    # STEP 1: SQL CANDIDATE POOL (Rebuild from scratch)
    # =========================================================
    conn = get_db()
    cur = conn.cursor()
    relaxation_log = []
    skip_overrides = set()
    
    def fetch_candidates(skip_set):
        """Build SQL, fetch candidates + embeddings. Rebuild from scratch each time."""
        where_clause, params = _build_candidate_sql(
            overrides, preserved, effective_category, detected_gender,
            skip_overrides=skip_set
        )
        
        # Add numeric price cap if provided (e.g. "less than 500")
        if price_max is not None:
            try:
                price_val = float(price_max)
                where_clause += " AND COALESCE(discounted_price, original_price, 99999) <= %s"
                params.append(price_val)
                print(f"[RetrievalV2] 💰 Price cap applied: <= {price_val}")
            except (ValueError, TypeError):
                print(f"[RetrievalV2] ⚠️ Invalid price_max: {price_max}, ignoring")
        
        sql = f"""
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
            WHERE {where_clause}
              AND embedding IS NOT NULL
        """
        
        print(f"[RetrievalV2] SQL WHERE: {where_clause}")
        print(f"[RetrievalV2] SQL params: {params}")
        
        cur.execute(sql, params)
        rows = cur.fetchall()
        print(f"[RetrievalV2] SQL pool size: {len(rows)}")
        return rows
    
    # First attempt
    rows = fetch_candidates(skip_overrides)
    
    # Progressive relaxation if pool too small (< 5 candidates)
    MIN_POOL = 5
    if len(rows) < MIN_POOL:
        # Relaxation order: relax least-important filters first
        relax_order = ["pattern", "sleeve", "material", "style", "price_bucket", "color"]
        hard_fields = {"category"}  # Never relax category
        for relax_field in relax_order:
            if relax_field not in overrides:
                continue
            if relax_field in hard_fields:
                continue
            
            skip_overrides.add(relax_field)
            relaxation_log.append(f"Relaxed: {relax_field}")
            print(f"[RetrievalV2] ⚠️ Relaxing: {relax_field} (pool was {len(rows)})")
            
            rows = fetch_candidates(skip_overrides)
            if len(rows) >= MIN_POOL:
                print(f"[RetrievalV2] ✅ Found {len(rows)} after relaxing {relax_field}")
                break
    
    if len(rows) == 0:
        print(f"[RetrievalV2] ❌ No candidates after all relaxation — returning empty")
        cur.close()
        conn.close()
        return {
            "products": [],
            "metadata": {
                "mode": mode,
                "pool_size": 0,
                "relaxation_log": relaxation_log,
                "error": "no_candidates"
            }
        }
    
    # =========================================================
    # STEP 2: COSINE SIMILARITY (Vectorized Numpy)
    # =========================================================
    # Parse candidates
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
            "color": r[12] or r[6] or "",  # COALESCE(scraped_color, primary_color_name) or color_family
            "pattern": r[13] or "",        # COALESCE(scraped_pattern, pattern_value)
            "sleeve": r[14] or "",         # COALESCE(scraped_sleeve, sleeve_value)
            "product_url": r[16] or "",
            "fit": r[17] or "",            # scraped_fit
            "neckline": r[18] or "",       # scraped_neckline
        }
        candidates.append(prod)
        
        # Parse embedding
        emb = r[15]
        if emb is not None:
            if isinstance(emb, str):
                emb = json.loads(emb)
            embeddings.append(emb)
        else:
            embeddings.append([0.0] * 512)
    
    # Compute cosine similarity (batched to limit peak memory)
    visual_scores = np.zeros(len(candidates), dtype=np.float32)
    if query_embedding and len(embeddings) > 0:
        # Free unreferenced memory before heavy allocation
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
                del batch, norms  # free immediately
        except MemoryError:
            print("[RetrievalV2] ⚠️ MemoryError during cosine sim — falling back to zero scores")
            visual_scores = np.zeros(len(candidates), dtype=np.float32)
        
        # Free the raw embeddings list now that we're done
        del embeddings
        gc.collect()
    
    sim_time = (_time.time() - t_start) * 1000
    print(f"[RetrievalV2] Cosine similarity computed for {len(candidates)} candidates in {sim_time:.1f}ms")
    
    # =========================================================
    # STEP 3: DYNAMIC SCORING
    # =========================================================
    w_visual = weights["visual"]
    w_override = weights["override"]
    w_preserved = weights["preserved"]
    
    # Count override and preserved fields for scoring
    override_fields = [k for k in overrides if k not in ("category", "gender") and k not in skip_overrides]
    preserved_fields = [k for k in preserved if k not in ("category", "gender")]
    
    total_overrides = len(override_fields)
    total_preserved = len(preserved_fields)
    
    # Zero-division guard: if no overrides/preserved → weight = 0
    if total_overrides == 0:
        w_override = 0
    if total_preserved == 0:
        w_preserved = 0
    
    # Redistribute weights if zeroed out
    total_weight = w_visual + w_override + w_preserved
    if total_weight > 0 and total_weight != 1.0:
        w_visual /= total_weight
        w_override /= total_weight
        w_preserved /= total_weight
    
    scored_candidates = []
    for i, prod in enumerate(candidates):
        v_score = float(visual_scores[i])
        
        # ── Per-attribute match tracking (for explainability) ──
        color_match_level = 0  # 0: none, 1: family, 2: exact
        pattern_matched = False
        sleeve_matched = False
        
        # Override score: how well does this product match user constraints?
        # EXACT match = 1.0 per field, FAMILY/FUZZY match = 0.5
        o_score = 0.0
        override_match_details = {}
        if total_overrides > 0:
            matched_score = 0.0
            for field in override_fields:
                override_val = overrides[field].lower().strip()
                field_score = 0.0  # 0 = no match, 0.5 = family, 1.0 = exact
                
                if field == "color":
                    # Check both primary_color_name and color_family
                    prod_primary = (prod.get("color") or "").lower().strip()
                    prod_family = (prod.get("color_family") or "").lower().strip()
                    _, override_family = _normalize_color_for_sql(overrides[field])
                    
                    # Exact match on primary color name
                    if override_val == prod_primary:
                        field_score = 1.0
                        color_match_level = max(color_match_level, 2)
                    elif override_val in prod_primary:
                        field_score = 0.9
                        color_match_level = max(color_match_level, 2)
                    # Family match (e.g. "green" family includes olive, teal)
                    elif override_family and (override_family == prod_family or override_family in prod_primary):
                        field_score = 0.05  # Severe penalty for family-only match
                        color_match_level = max(color_match_level, 1)
                    elif override_val == prod_family or override_val in prod_family:
                        field_score = 0.05  # Severe penalty for family-only match
                        color_match_level = max(color_match_level, 1)
                elif field == "sleeve":
                    prod_sleeve = (prod.get("sleeve") or "").lower().strip()
                    if override_val == prod_sleeve:
                        field_score = 1.0
                    elif override_val in prod_sleeve or prod_sleeve in override_val:
                        field_score = 0.5
                    sleeve_matched = field_score > 0
                elif field == "pattern":
                    prod_pattern = (prod.get("pattern") or "").lower().strip()
                    if override_val == prod_pattern:
                        field_score = 1.0
                    elif override_val in prod_pattern or prod_pattern in override_val:
                        field_score = 0.5
                    pattern_matched = field_score > 0
                elif field == "material":
                    if override_val == (prod.get("material") or "").lower():
                        field_score = 1.0
                elif field == "style":
                    if override_val in (prod.get("style") or "").lower():
                        field_score = 1.0
                
                matched_score += field_score
                override_match_details[field] = field_score > 0
            
            o_score = matched_score / total_overrides
        
        # Preserved score: how many preserved attrs match this product?
        # Color gets a 1.5x multiplier so it's the strongest preserved signal
        p_score = 0.0
        if total_preserved > 0:
            weighted_matched = 0.0
            weighted_total = 0.0
            for field in preserved_fields:
                pres_val = preserved[field].lower().strip() if preserved[field] else ""
                field_matched = False
                field_weight = 1.5 if field == "color" else 1.0  # color priority boost
                
                if field == "color":
                    prod_color = (prod.get("color") or "").lower()
                    prod_family = (prod.get("color_family") or "").lower()
                    # Check exact match
                    if pres_val == prod_color or pres_val in prod_color or pres_val == prod_family:
                        field_matched = True
                        color_match_level = max(color_match_level, 2)
                    else:
                        # Also check color family match (e.g. navy blue → blue)
                        pres_family = COLOR_FAMILIES.get(pres_val, pres_val)
                        if pres_family == prod_family or pres_family in prod_color:
                            field_matched = True
                            color_match_level = max(color_match_level, 1)
                elif field == "sleeve":
                    if pres_val == (prod.get("sleeve") or "").lower():
                        field_matched = True
                    if not sleeve_matched:
                        sleeve_matched = field_matched
                elif field == "pattern":
                    if pres_val == (prod.get("pattern") or "").lower():
                        field_matched = True
                    if not pattern_matched:
                        pattern_matched = field_matched
                
                weighted_total += field_weight
                if field_matched:
                    weighted_matched += field_weight
            
            p_score = weighted_matched / weighted_total if weighted_total > 0 else 0.0
        
        # Final weighted score
        final_score = (w_visual * v_score) + (w_override * o_score) + (w_preserved * p_score)
        
        # ── COLOR BOOST HACK ──
        # Provide a hard absolute boost to final score if color matches exactly/family
        # This fixes "white dress outranking correctly colored teal dress" by overriding FAISS bias.
        color_boost_applied = 0.0
        if "color" in override_match_details and override_match_details["color"]:
            # It matched from explicit user override -> boost it massively
            color_boost_applied = 0.40 if color_match_level == 2 else 0.15
            final_score += color_boost_applied
        elif color_match_level > 0 and o_score == 0:
            # It matched from preserved attributes -> boost it
            color_boost_applied = 0.35 if color_match_level == 2 else 0.10
            final_score += color_boost_applied
        
        # ── SCENE STYLE BOOST ──
        # If scene context suggests a style (e.g., bamboo_forest → casual),
        # give a soft boost to products matching that style
        scene_boost = 0.0
        if scene_style and "style" not in overrides:
            prod_style = (prod.get("style") or "").lower().strip()
            if scene_style.lower() in prod_style:
                scene_boost = 0.15
                final_score += scene_boost
        
        prod["similarity_score"] = round(v_score, 4)
        prod["override_score"] = round(o_score, 4)
        prod["preserved_score"] = round(p_score, 4)
        prod["final_score"] = round(final_score, 4)
        
        # ── Explanation metadata (grounded, deterministic) ──
        prod["match_meta"] = {
            "visual_similarity": round(v_score, 4),
            "color_match": color_match_level > 0,
            "pattern_match": pattern_matched,
            "sleeve_match": sleeve_matched,
            "color_score": round(o_score if "color" in override_match_details else (p_score if "color" in preserved else 0.0), 4),
            "pattern_score": round(o_score if "pattern" in override_match_details else (p_score if "pattern" in preserved else 0.0), 4),
            "sleeve_score": round(o_score if "sleeve" in override_match_details else (p_score if "sleeve" in preserved else 0.0), 4),
            "override_applied": len(override_fields) > 0,
            "override_details": override_match_details,
            "relaxed_constraints": [r.replace("Relaxed: ", "") for r in relaxation_log] if relaxation_log else [],
            "visual_weight": round(w_visual, 3),
            "override_weight": round(w_override, 3),
            "preserved_weight": round(w_preserved, 3),
            "final_score": round(final_score, 4),
        }
        
        scored_candidates.append(prod)
    
    # Sort by final_score descending, then product_id for determinism
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["product_id"]))
    
    # Take top K
    results = scored_candidates[:top_k]
    
    # ── Annotate rank position + percentile for explainability ──
    pool_size = len(scored_candidates)
    for rank_idx, prod in enumerate(results):
        prod["match_meta"]["rank_position"] = rank_idx + 1
        prod["match_meta"]["total_candidates"] = pool_size
        prod["match_meta"]["visual_percentile"] = round(
            100.0 * (1.0 - rank_idx / max(pool_size, 1)), 1
        )
    
    # =========================================================
    # LOGGING
    # =========================================================
    total_ms = (_time.time() - t_start) * 1000
    
    # Top-5 breakdown
    top5_categories = [r["category"] for r in results[:5]]
    top5_scores = [r["final_score"] for r in results[:5]]
    
    print(f"[RetrievalV2] ✅ Returning {len(results)} / {len(candidates)} candidates")
    print(f"[RetrievalV2] Top-5 categories: {top5_categories}")
    print(f"[RetrievalV2] Top-5 scores: {top5_scores}")
    print(f"[RetrievalV2] Relaxation: {relaxation_log}")
    print(f"[RetrievalV2] Total time: {total_ms:.1f}ms")
    print(f"{'='*60}")
    
    cur.close()
    conn.close()
    
    return {
        "products": results,
        "metadata": {
            "mode": mode,
            "weights": {"visual": round(w_visual, 2), "override": round(w_override, 2), "preserved": round(w_preserved, 2)},
            "overrides": overrides,
            "preserved_attrs": preserved,
            "pool_size": len(candidates),
            "result_count": len(results),
            "relaxation_log": relaxation_log,
            "top_5_categories": top5_categories,
            "search_time_ms": round(total_ms, 1),
        }
    }


