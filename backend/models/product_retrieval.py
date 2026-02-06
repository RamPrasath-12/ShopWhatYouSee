"""
Product Retrieval Module - Hybrid (PostgreSQL + FAISS)
Queries the products table for matching products based on filters.
Supports vector similarity search if embedding is provided.
"""
import psycopg2
import os
import faiss
import json
import numpy as np

# PostgreSQL Connection Config
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

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
    if not cat: return None
    cat = cat.lower().strip().replace("-", "_")  # t-shirt -> t_shirt
    mapping = {
        "pant": "Pant", "pants": "Pant", "jeans": "Pant", "trousers": "Pant",
        "short": "Shorts", "shorts": "Shorts",
        "tee": "T_shirt", "tshirt": "T_shirt", "t_shirt": "T_shirt", "t-shirt": "T_shirt",
        "shirt": "Shirt",
        "coat": "Jacket", "jacket": "Jacket", "blazer": "Blazer",
        "skirt": "Skirt", "leggings": "Leggings", "saree": "Saree",
        "shoe": "Footwear_shoes", "shoes": "Footwear_shoes",
        "sandal": "Footwear_sandals", "sandals": "Footwear_sandals",
        "heel": "Footwear_heels", "heels": "Footwear_heels",
        "watch": "Watch", "bag": "Bag", "purse": "Purse", "handbag": "Purse",
        "belt": "Belt", "cap": "Cap", "hat": "Cap",
        "earring": "Earring", "necklace": "Necklace", "ring": "Ring",
        "sunglass": "Glasses", "glasses": "Glasses"
    }
    return mapping.get(cat, cat.capitalize())

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
            FROM products
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
            if pid not in seen_ids and len(results) < 10:
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
    # Stage 1: Vector Search + Category Filter
    # --------------------------------------------------
    if category and vector_ids:
        print(f"[Search] Stage 1: Checking {len(vector_ids)} vector candidates...")
        rows = run_query(
            "LOWER(yolo_category) = LOWER(%s)", 
            [category]
        )
        add_results(rows)
        print(f"[Search] Stage 1 results: {len(results)}")

    # --------------------------------------------------
    # Stage 2: Category + Color (strict match)
    # --------------------------------------------------
    # Only run if we don't have enough results from vector search
    if len(results) < 10 and category and color:
        print(f"[Search] Stage 2: Text search for {category} + {color}...")
        rows = run_query(
            "LOWER(yolo_category) = LOWER(%s) AND (LOWER(base_colour) LIKE LOWER(%s) OR LOWER(COALESCE(primary_color, '')) LIKE LOWER(%s))",
            [category, f"%{color}%", f"%{color}%"]
        )
        add_results(rows)
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
                        FROM products
                        WHERE LOWER(yolo_category) = LOWER(%s)
                        LIMIT 20
                    """
                    cur.execute(q, [category])
                    extra_rows = cur.fetchall()
                    
                    for r in extra_rows:
                        if r[0] not in seen_ids and len(results) < 15:
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
        "navy blue": "blue", "sky blue": "blue", "teal": "blue", "turquoise": "blue", "steel blue": "blue",
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
