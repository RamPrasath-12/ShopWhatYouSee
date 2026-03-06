"""
Myntra Product Attribute Scraper
================================
Scrapes product attributes from Myntra product pages and updates
the local PostgreSQL visual_attributes table.

Extracts: base_colour, shade, sleeve, pattern, fit, material, neckline, etc.
from the pdpData JSON embedded in each product page.

Usage:
    python tools/scrape_myntra_attributes.py              # Scrape all products
    python tools/scrape_myntra_attributes.py --limit 100  # Scrape first 100
    python tools/scrape_myntra_attributes.py --category shirts  # Scrape only shirts
    python tools/scrape_myntra_attributes.py --dry-run    # Preview without updating DB
"""

import psycopg2
import requests
import json
import time
import sys
import argparse
import re
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests (be polite to Myntra)
BATCH_SIZE = 50      # commit to DB every N products
MAX_RETRIES = 2      # retry failed requests


# ─────────────────────────────────────────────────────────────
# JSON EXTRACTION (brace-balanced parser)
# ─────────────────────────────────────────────────────────────
def extract_json_object(text, start_key):
    """Extract a complete JSON object after a key using brace balancing."""
    idx = text.find(f'"{start_key}"')
    if idx == -1:
        return None
    colon_idx = text.find(':', idx + len(start_key) + 2)
    if colon_idx == -1:
        return None
    brace_idx = text.find('{', colon_idx)
    if brace_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    for i in range(brace_idx, min(brace_idx + 500000, len(text))):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_idx:i+1])
                except json.JSONDecodeError:
                    return None
    return None


# ─────────────────────────────────────────────────────────────
# ATTRIBUTE EXTRACTION FROM pdpData
# ─────────────────────────────────────────────────────────────

# Normalize color names to match DB color_family values
COLOR_FAMILY_MAP = {
    "red": "red", "maroon": "red", "burgundy": "red", "wine": "red",
    "crimson": "red", "rust": "red", "brick": "red", "cherry": "red",
    "blue": "blue", "navy blue": "blue", "navy": "blue", "sky blue": "blue",
    "steel blue": "blue", "royal blue": "blue", "cobalt": "blue",
    "teal": "teal", "turquoise blue": "teal", "turquoise": "teal",
    "green": "green", "olive": "green", "lime": "green", "khaki": "green",
    "sea green": "green", "mint": "green", "fluorescent green": "green",
    "sage green": "green", "dark green": "green",
    "yellow": "yellow", "mustard": "yellow", "gold": "yellow", "lime yellow": "yellow",
    "orange": "orange", "coral": "orange", "peach": "orange",
    "pink": "pink", "hot pink": "pink", "magenta": "pink", "rose": "pink",
    "mauve": "pink", "fuchsia": "pink", "blush": "pink",
    "purple": "purple", "lavender": "purple", "violet": "purple", "plum": "purple",
    "brown": "brown", "tan": "brown", "coffee brown": "brown", "chocolate": "brown",
    "taupe": "brown", "camel": "brown", "mushroom brown": "brown",
    "black": "neutral", "white": "neutral", "grey": "neutral", "gray": "neutral",
    "off white": "neutral", "cream": "neutral", "beige": "neutral",
    "charcoal": "neutral", "silver": "neutral", "nude": "neutral",
    "multi": "multi", "multicoloured": "multi",
}

# Normalize sleeve values to match DB
SLEEVE_MAP = {
    "long sleeves": "long sleeves",
    "short sleeves": "short sleeves",
    "three-quarter sleeves": "three-quarter sleeves",
    "3/4 sleeves": "three-quarter sleeves",
    "sleeveless": "sleeveless",
    "cap sleeves": "short sleeves",
    "roll-up sleeves": "long sleeves",
    "regular sleeves": None,  # this is sleeve styling, not length
}

# Normalize pattern values
PATTERN_MAP = {
    "solid": "solid",
    "striped": "striped",
    "checked": "checked",
    "printed": "printed",
    "floral": "printed",
    "abstract": "printed",
    "ethnic motifs": "printed",
    "geometric": "printed",
    "polka dots": "printed",
    "graphic": "printed",
    "animal": "printed",
    "typography": "printed",
    "camouflage": "printed",
    "embroidered": "embroidered",
    "woven": "woven",
    "self-design": "self-design",
    "colourblocked": "colourblocked",
    "dyed": "solid",
    "ombre": "printed",
    "paisley": "printed",
    "tribal": "printed",
    "quirky": "printed",
}


def extract_attributes(pdp):
    """
    Extract structured attributes from Myntra pdpData.
    
    Returns dict with keys matching DB columns:
        primary_color_name, color_family, pattern_value, sleeve_value,
        material, style, gender, shade, fit, neckline
    """
    result = {}
    
    # ── BASE COLOUR (most reliable source of truth) ──
    base_colour = pdp.get("baseColour")
    if base_colour:
        result["primary_color_name"] = base_colour.strip()
        # Map to color family
        bc_lower = base_colour.lower().strip()
        result["color_family"] = COLOR_FAMILY_MAP.get(bc_lower, bc_lower)
    
    # ── ARTICLE ATTRIBUTES (rich structured data) ──
    article_attrs = pdp.get("articleAttributes", {})
    if not isinstance(article_attrs, dict):
        article_attrs = {}
    
    # Build a lowercase-key lookup for flexible matching
    attrs_lower = {k.lower().strip(): v for k, v in article_attrs.items()}
    
    # Shade
    shade = attrs_lower.get("shade")
    if shade and shade != "NA":
        result["shade"] = shade.strip()
    
    # Sleeve Length
    sleeve_raw = attrs_lower.get("sleeve length") or attrs_lower.get("sleeves")
    if sleeve_raw and sleeve_raw != "NA":
        sleeve_norm = SLEEVE_MAP.get(sleeve_raw.lower().strip(), sleeve_raw.lower().strip())
        if sleeve_norm:
            result["sleeve_value"] = sleeve_norm
    
    # Pattern (try multiple keys)
    pattern_raw = (attrs_lower.get("print or pattern types") or 
                   attrs_lower.get("pattern") or
                   attrs_lower.get("top pattern") or
                   attrs_lower.get("pattern type"))
    if pattern_raw and pattern_raw != "NA":
        pattern_norm = PATTERN_MAP.get(pattern_raw.lower().strip(), pattern_raw.lower().strip())
        result["pattern_value"] = pattern_norm
    
    # Fit
    fit_raw = attrs_lower.get("fit") or attrs_lower.get("brand fit name")
    if fit_raw and fit_raw != "NA":
        result["fit"] = fit_raw.strip()
    
    # Material/Fabric
    material_raw = (attrs_lower.get("fabric") or 
                    attrs_lower.get("materials") or 
                    attrs_lower.get("material") or
                    attrs_lower.get("top fabric"))
    if material_raw and material_raw != "NA":
        result["material"] = material_raw.strip()
    
    # Style/Occasion
    occasion = attrs_lower.get("occasions") or attrs_lower.get("occasion")
    if occasion and occasion != "NA":
        result["style"] = occasion.strip()
    
    # Neckline
    neck = attrs_lower.get("neck") or attrs_lower.get("neckline") or attrs_lower.get("neck type")
    if neck and neck != "NA":
        result["neckline"] = neck.strip()
    
    # Closure
    closure = attrs_lower.get("closure")
    if closure and closure != "NA":
        result["closure"] = closure.strip()
    
    # Waist Rise (for pants/jeans)
    rise = attrs_lower.get("waist rise") or attrs_lower.get("rise")
    if rise and rise != "NA":
        result["waist_rise"] = rise.strip()
    
    # Length
    length = attrs_lower.get("length") or attrs_lower.get("top length")
    if length and length != "NA":
        result["length"] = length.strip()
    
    # Fade (for denim)
    fade = attrs_lower.get("fade")
    if fade and fade != "NA":
        result["fade"] = fade.strip()
    
    # Store full article_attributes JSON for future use
    result["scraped_attributes_json"] = json.dumps(article_attrs, ensure_ascii=False)
    
    # Extract raw Product Details description text
    desc_text = ""
    descriptors = pdp.get("descriptors", {})
    if isinstance(descriptors, dict):
        desc = descriptors.get("description", {})
        if isinstance(desc, dict):
            desc_text = desc.get("value", "")
            
    if not desc_text:
        prod_det = pdp.get("productDetails", [])
        if isinstance(prod_det, list):
            for item in prod_det:
                if isinstance(item, dict) and "description" in item:
                    # preserve html spacing loosely
                    desc_text += item.get("description", "").replace("<br>", "\n").replace("<p>", "").replace("</p>", "\n") + "\n"
                    
    if desc_text:
        result["description"] = desc_text.strip()
    
    return result


def scrape_product(url, session):
    """Scrape a single Myntra product page and extract attributes."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                # Rate limited - wait and retry
                wait = 5 * (attempt + 1)
                print(f"    ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            
            pdp = extract_json_object(r.text, "pdpData")
            if not pdp:
                return None, "No pdpData found"
            
            attrs = extract_attributes(pdp)
            return attrs, None
            
        except requests.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return None, "Timeout"
        except Exception as e:
            return None, str(e)
    
    return None, "Max retries exceeded"


# ─────────────────────────────────────────────────────────────
# DATABASE UPDATE
# ─────────────────────────────────────────────────────────────
def ensure_columns_exist(conn):
    """Add new columns if they don't exist yet."""
    cur = conn.cursor()
    new_columns = {
        "scraped_color": "VARCHAR(100)",
        "scraped_shade": "VARCHAR(100)",
        "scraped_sleeve": "VARCHAR(100)",
        "scraped_pattern": "VARCHAR(100)",
        "scraped_fit": "VARCHAR(100)",
        "scraped_material": "VARCHAR(200)",
        "scraped_neckline": "VARCHAR(100)",
        "scraped_attributes_json": "TEXT",
        "scraped_description": "TEXT",
        "scraped_at": "TIMESTAMP",
    }
    
    for col, dtype in new_columns.items():
        try:
            cur.execute(f"ALTER TABLE visual_attributes ADD COLUMN IF NOT EXISTS {col} {dtype}")
        except Exception as e:
            print(f"  Column {col}: {e}")
    
    conn.commit()
    cur.close()
    print(f"[DB] Ensured {len(new_columns)} scraped columns exist")


def update_product(cur, product_id, attrs):
    """Update a single product's attributes in the database."""
    # Update BOTH scraped_* columns AND the primary display columns
    updates = []
    params = []
    
    # Primary color (overwrite the AGMAN-extracted one)
    if attrs.get("primary_color_name"):
        updates.append("primary_color_name = %s")
        params.append(attrs["primary_color_name"])
        updates.append("scraped_color = %s")
        params.append(attrs["primary_color_name"])
    
    if attrs.get("color_family"):
        updates.append("color_family = %s")
        params.append(attrs["color_family"])
    
    if attrs.get("sleeve_value"):
        updates.append("sleeve_value = %s")
        params.append(attrs["sleeve_value"])
        updates.append("scraped_sleeve = %s")
        params.append(attrs["sleeve_value"])
    
    if attrs.get("pattern_value"):
        updates.append("pattern_value = %s")
        params.append(attrs["pattern_value"])
        updates.append("scraped_pattern = %s")
        params.append(attrs["pattern_value"])
    
    if attrs.get("material"):
        updates.append("material = %s")
        params.append(attrs["material"])
        updates.append("scraped_material = %s")
        params.append(attrs["material"])
    
    if attrs.get("style"):
        updates.append("style = %s")
        params.append(attrs["style"])
    
    if attrs.get("shade"):
        updates.append("scraped_shade = %s")
        params.append(attrs["shade"])
    
    if attrs.get("fit"):
        updates.append("scraped_fit = %s")
        params.append(attrs["fit"])
    
    if attrs.get("neckline"):
        updates.append("scraped_neckline = %s")
        params.append(attrs["neckline"])
    
    if attrs.get("scraped_attributes_json"):
        updates.append("scraped_attributes_json = %s")
        params.append(attrs["scraped_attributes_json"])
        
    if attrs.get("description"):
        updates.append("scraped_description = %s")
        params.append(attrs["description"])
    
    # Timestamp
    updates.append("scraped_at = %s")
    params.append(datetime.now())
    
    if not updates:
        return False
    
    params.append(product_id)
    sql = f"UPDATE visual_attributes SET {', '.join(updates)} WHERE product_id = %s"
    cur.execute(sql, params)
    return True


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scrape Myntra product attributes")
    parser.add_argument("--limit", type=int, default=None, help="Max products to scrape")
    parser.add_argument("--category", type=str, default=None, help="Only scrape this category")
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB updates")
    parser.add_argument("--offset", type=int, default=0, help="Start from this offset")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests")
    args = parser.parse_args()
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Ensure columns exist
    if not args.dry_run:
        ensure_columns_exist(conn)
    
    # Fetch products to scrape
    cur = conn.cursor()
    query = """
        SELECT product_id, product_url, category, product_name, primary_color_name
        FROM visual_attributes
        WHERE product_url IS NOT NULL
          AND product_url LIKE 'https://www.myntra.com%'
    """
    params = []
    
    # Filter: skip already scraped
    if not args.dry_run:
        query += " AND scraped_at IS NULL"
    
    if args.category:
        query += " AND LOWER(category) = LOWER(%s)"
        params.append(args.category)
    
    query += " ORDER BY id"
    
    if args.offset > 0:
        query += f" OFFSET {args.offset}"
    if args.limit:
        query += f" LIMIT {args.limit}"
    
    if params:
        cur.execute(query, params)
    else:
        cur.execute(query)
    products = cur.fetchall()
    cur.close()
    
    total = len(products)
    print(f"\n{'='*60}")
    print(f"  Myntra Product Attribute Scraper")
    print(f"  Products to scrape: {total}")
    print(f"  Category filter: {args.category or 'ALL'}")
    print(f"  Delay: {args.delay}s per request")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")
    
    if total == 0:
        print("No products to scrape.")
        conn.close()
        return
    
    # Stats
    success = 0
    failed = 0
    skipped = 0
    
    session = requests.Session()
    write_cur = conn.cursor()
    
    start_time = time.time()
    
    for i, (pid, url, cat, name, old_color) in enumerate(products):
        # Progress
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1)
        eta = (total - i - 1) / max(rate, 0.01)
        
        print(f"[{i+1}/{total}] {cat} | {name[:40]:40s} | Old color: {old_color or 'N/A':15s}", end="")
        
        attrs, error = scrape_product(url, session)
        
        if error:
            print(f" ❌ {error}")
            failed += 1
            time.sleep(args.delay)
            continue
        
        if not attrs:
            print(f" ⏭ No attributes")
            skipped += 1
            time.sleep(args.delay)
            continue
        
        new_color = attrs.get("primary_color_name", "?")
        new_sleeve = attrs.get("sleeve_value", "-")
        new_pattern = attrs.get("pattern_value", "-")
        print(f" ✅ Color: {new_color:15s} Sleeve: {new_sleeve:20s} Pattern: {new_pattern}")
        
        if not args.dry_run:
            try:
                update_product(write_cur, pid, attrs)
                success += 1
                
                # Batch commit
                if success % BATCH_SIZE == 0:
                    conn.commit()
                    print(f"  💾 Committed {success} updates ({eta:.0f}s remaining, {rate:.1f}/s)")
            except Exception as e:
                print(f"  ⚠️ DB error: {e}")
                conn.rollback()
                failed += 1
        else:
            success += 1
        
        time.sleep(args.delay)
    
    # Final commit
    if not args.dry_run:
        conn.commit()
    
    write_cur.close()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SCRAPING COMPLETE")
    print(f"  Total: {total} | Success: {success} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Rate: {total/max(elapsed,1):.1f} products/sec")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
