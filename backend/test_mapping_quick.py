"""Quick category mapping test — no heavy imports."""

def normalize_category(cat):
    if not cat: return None
    cat = cat.lower().strip().replace("-", "_")
    mapping = {
        "shirt": "shirts", "shirts": "shirts",
        "tee": "tshirt", "tshirt": "tshirt", "t_shirt": "tshirt",
        "blazer": "blazer",
        "coat": "Jacket", "jacket": "Jacket",
        "pant": "pant", "pants": "pant", "jeans": "pant", "trousers": "pant",
        "short": "shorts", "shorts": "shorts",
        "skirt": "skirt",
        "churidhar": "churidhar", "dhoti": "dhoti",
        "kurta": "churidhar", "kurti": "churidhar",
        "saree": "churidhar",
        "shoe": "Footwear_shoes", "shoes": "Footwear_shoes",
        "sandal": "Footwear_sandals", "sandals": "Footwear_sandals",
        "cap": "caps", "hat": "caps", "caps": "caps",
        "glass": "glasses", "sunglass": "glasses", "glasses": "glasses",
        "sunglasses": "glasses", "eyewear": "glasses",
        "belt": "belt", "tie": "tie",
        "earring": "earrings", "earrings": "earrings",
        "necklace": "necklace", "ring": "necklace",
        "watch": "watch",
        "bag": "belt", "purse": "belt", "handbag": "belt",
        "dress": "shirts", "dresses": "shirts",
        "sweater": "tshirt", "hoodie": "tshirt", "sweatshirt": "tshirt",
        "leggings": "pant",
    }
    result = mapping.get(cat)
    if result:
        return result
    return cat

# DB categories from extract_visual_attributes.py ALLOWED_CATEGORIES
DB_CATEGORIES = {
    "shirts", "tshirt", "blazer", "Jacket", "pant", "shorts",
    "skirt", "churidhar", "dhoti",
    "Footwear_sandals", "Footwear_shoes",
    "caps", "glasses", "belt", "tie", "earrings", "necklace", "watch",
}

# Test cases: input -> expected DB category
TESTS = {
    # YOLO detection outputs
    "shirts": "shirts",
    "tshirt": "tshirt",
    "blazer": "blazer",
    "Jacket": "Jacket",
    "pant": "pant",
    "shorts": "shorts",
    "glasses": "glasses",
    "caps": "caps",
    # User/LLM inputs
    "shirt": "shirts",
    "t-shirt": "tshirt",
    "t_shirt": "tshirt",
    "tee": "tshirt",
    "jacket": "Jacket",
    "pants": "pant",
    "jeans": "pant",
    "cap": "caps",
    "hat": "caps",
    "glass": "glasses",
    "sunglass": "glasses",
    "sunglasses": "glasses",
    "earring": "earrings",
    "earrings": "earrings",
    "necklace": "necklace",
    "watch": "watch",
    "belt": "belt",
    "tie": "tie",
    "shoe": "Footwear_shoes",
    "shoes": "Footwear_shoes",
    "sandal": "Footwear_sandals",
    "sandals": "Footwear_sandals",
    "skirt": "skirt",
}

if __name__ == "__main__":
    fails = 0
    for input_cat, expected in TESTS.items():
        actual = normalize_category(input_cat)
        ok = actual == expected
        in_db = actual.lower() in {c.lower() for c in DB_CATEGORIES}
        if not ok:
            print(f"  FAIL: normalize_category('{input_cat}') = '{actual}', expected '{expected}'")
            fails += 1
        elif not in_db:
            print(f"  WARN: '{actual}' not in DB_CATEGORIES (from input '{input_cat}')")
        
    total = len(TESTS)
    passed = total - fails
    print(f"\nResult: {passed}/{total} PASS, {fails} FAIL")
    if fails == 0:
        print("ALL CATEGORY MAPPINGS CORRECT")
    
    # Also verify every mapped value is a real DB category
    print("\nDB Coverage Check:")
    all_mapped_values = set(normalize_category(k) for k in TESTS.keys())
    missing_from_db = {v for v in all_mapped_values if v.lower() not in {c.lower() for c in DB_CATEGORIES}}
    if missing_from_db:
        print(f"  WARNING: These mapped values don't exist in DB: {missing_from_db}")
    else:
        print(f"  All {len(all_mapped_values)} mapped values exist in DB")
