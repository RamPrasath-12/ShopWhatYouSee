"""
generate_products_unsplash.py
-----------------------------------
Downloads N fashion images from Unsplash and creates:
 - backend/static/products/<filename>.jpg
 - products.csv (CSV with attributes)
 - products.sql (SQL INSERT statements for PostgreSQL)

Usage:
  set UNSPLASH_ACCESS_KEY in environment (or create a .env file with UNSPLASH_ACCESS_KEY=...)
  python tools/generate_products_unsplash.py --count 1000

Notes:
 - This script tries to keep requests polite (sleep between calls).
 - Check Unsplash API guidelines for usage/attribution.
"""

import os, sys, time, math, random, csv, argparse, json
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # read .env if present

UNSPLASH_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
if not UNSPLASH_KEY:
    print("ERROR: set environment variable UNSPLASH_ACCESS_KEY (see README).")
    sys.exit(1)

# ---- Config ----
OUT_DIR = Path("backend/static/products")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("backend/static/products/products.csv")
SQL_PATH = Path("backend/static/products/products.sql")

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
HEADERS = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}

# seed search queries chosen to match mixed men/women/outerwear/dresses/ethnic
SEARCH_QUERIES = [
    "formal shirt men", "formal shirt women", "red dress", "summer dress",
    "casual t-shirt", "hoodie", "denim jacket", "leather jacket", "sweatshirt",
    "blazer", "suit jacket", "business suit", "party dress", "jeans",
    "kurta men", "kurta women", "saree", "ethnic wear", "trousers",
    "shorts men", "shorts women", "skirt", "polo shirt", "coat",
    "windbreaker", "sports jersey", "activewear", "blouse", "tank top",
    "cardigan", "puffer jacket", "vest", "maxi dress", "formal blouse",
    "office outfit women", "streetwear outfit", "casual outfit men", "festival outfit",
    "swimwear", "beachwear", "outerwear", "formal shoes", "sneakers"
]

# helper small list for pattern/sample assignment
PATTERNS = ["solid", "striped", "checked", "patterned"]
SLEEVES = ["short", "long", "three_quarter"]
STYLES = ["casual", "formal", "party", "sports", "ethnic"]

# number of photos per search page (Unsplash default 10; we'll use 20)
PER_PAGE = 20

# Sleep between API calls (seconds). Increase if you hit rate limits.
SLEEP_BETWEEN_CALLS = 0.6

# fallback price distribution (INR)
def sample_price():
    # prices mostly between 299 and 3499
    return round(random.choice([
        random.uniform(299, 799),
        random.uniform(800, 1499),
        random.uniform(1500, 3499)
    ]), 2)


# ----- Helpers -----
def fetch_unsplash(query, page=1, per_page=20):
    params = {"query": query, "page": page, "per_page": per_page, "orientation":"portrait"}
    r = requests.get(UNSPLASH_SEARCH_URL, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def download_image(url):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    return r.content


def save_image_bytes(content, out_path):
    # ensure it's a valid jpg
    img = Image.open(BytesIO(content)).convert("RGB")
    img.save(out_path, format="JPEG", quality=85)


def dominant_color_hex(pil_img, resize=80):
    # quantize method (fast) — reduce colors via adaptive palette then find most common
    img = pil_img.copy()
    img.thumbnail((resize, resize))
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=8)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    if not color_counts:
        return "#000000"
    dominant_index = color_counts[0][1]
    dominant_rgb = palette[dominant_index*3:dominant_index*3+3]
    return '#%02x%02x%02x' % tuple(dominant_rgb)


def hex_to_simple_color(hexcode):
    # very simple mapping (keeps it fast)
    if not hexcode or len(hexcode) < 7:
        return None
    r = int(hexcode[1:3], 16)
    g = int(hexcode[3:5], 16)
    b = int(hexcode[5:7], 16)
    # choose top channel
    if r > g and r > b:
        return "red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    # check for near-white or gray
    avg = (r+g+b)/3
    if avg > 220:
        return "white"
    if avg < 40:
        return "black"
    return "brown"


def make_name_from_query(q):
    # create readable product name
    words = [w for w in q.split() if w not in ("men","women","outfit","outfits","photo","photos")]
    if not words:
        words = ["Fashion Item"]
    base = " ".join(words[:3]).title()
    suffix = random.choice(["Collection", "Series", "Edition", "Piece"])
    return f"{base} {suffix}"


# ----- Main generation -----
def generate(count=1000, out_dir=OUT_DIR):
    csv_rows = []
    used_image_ids = set()
    page_counters = {q:1 for q in SEARCH_QUERIES}

    pbar = tqdm(total=count, desc="Downloading images")

    attempts = 0
    while len(csv_rows) < count and attempts < count*5:
        attempts += 1
        # pick random query to diversify dataset
        q = random.choice(SEARCH_QUERIES)
        page = page_counters.get(q, 1)

        try:
            data = fetch_unsplash(q, page=page, per_page=PER_PAGE)
        except Exception as e:
            print("API error for query", q, ":", e)
            time.sleep(2)
            continue

        results = data.get("results", [])
        if not results:
            page_counters[q] = page + 1
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        for res in results:
            if len(csv_rows) >= count:
                break

            img_id = res.get("id")
            if img_id in used_image_ids:
                continue
            used_image_ids.add(img_id)

            # pick a reasonable download url (regular)
            urls = res.get("urls", {})
            img_url = urls.get("regular") or urls.get("small") or urls.get("raw")
            if not img_url:
                continue

            # create filename
            filename = f"{img_id}.jpg"
            out_path = out_dir / filename

            # download image bytes
            try:
                content = download_image(img_url)
            except Exception as e:
                # skip if network fails
                # print("Download fail", e)
                continue

            try:
                # save as jpeg after validation
                save_image_bytes(content, out_path)
            except Exception as e:
                # skip if invalid image
                # print("Save fail", e)
                continue

            # open PIL image for color detection
            try:
                pil = Image.open(out_path).convert("RGB")
            except:
                out_path.unlink(missing_ok=True)
                continue

            # derive attributes
            hexcol = dominant_color_hex(pil)
            color = hex_to_simple_color(hexcol)

            # try to derive category from search query words
            clothing_word = None
            for w in ("dress","shirt","t-shirt","tshirt","hoodie","sweatshirt","jacket","jeans","kurta","saree","skirt","shorts","blouse","coat","suit","trousers","polo"):
                if w in q:
                    clothing_word = w
                    break

            # pattern probability: most are solid
            pattern = random.choices(PATTERNS, weights=[80,8,6,6], k=1)[0]

            # sleeve: if clothing_word suggests short/long
            if clothing_word in ("t-shirt","tshirt","shorts","tank","polo"):
                sleeve = random.choice(["short", "short", "three_quarter"])
            elif clothing_word in ("dress","skirt"):
                sleeve = random.choice(["short","three_quarter","long"])
            else:
                sleeve = random.choices(SLEEVES, weights=[40,40,20])[0]

            # style inference from query
            style = None
            if any(k in q for k in ("formal", "suit", "blazer", "office")):
                style = "formal"
            elif any(k in q for k in ("party","festival","evening","dress")):
                style = "party"
            elif any(k in q for k in ("sport","jersey","active","sneaker")):
                style = "sports"
            elif any(k in q for k in ("kurta","saree","ethnic")):
                style = "ethnic"
            else:
                style = random.choice(["casual", "formal", "party"])  # mix

            price = sample_price()
            name = make_name_from_query(q)

            # image URL path that backend will serve (Flask route: /images/<filename>)
            image_url = f"http://localhost:5000/images/{filename}"

            # append CSV row
            csv_rows.append({
                "id": img_id,
                "name": name,
                "color": color,
                "pattern": pattern,
                "sleeve_length": sleeve,
                "style": style,
                "price": price,
                "image_url": image_url,
                "filename": filename
            })

            pbar.update(1)
            # exit early if reached
            if len(csv_rows) >= count:
                break

            # polite sleep between downloads
            time.sleep(SLEEP_BETWEEN_CALLS)

        # move to next page for this query next time
        page_counters[q] = page + 1

    pbar.close()
    return csv_rows


def write_outputs(rows):
    # write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","name","color","pattern","sleeve_length","style","price","image_url","filename"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write SQL INSERT file (Postgres)
    with open(SQL_PATH, "w", encoding="utf-8") as f:
        f.write("-- products.sql generated by generate_products_unsplash.py\n")
        f.write("BEGIN;\n")
        for i, r in enumerate(rows, start=1):
            # make sure single quotes in name are escaped
            name = r["name"].replace("'", "''")
            color = (r["color"] or "").replace("'", "''")
            pattern = (r["pattern"] or "").replace("'", "''")
            sleeve = (r["sleeve_length"] or "").replace("'", "''")
            style = (r["style"] or "").replace("'", "''")
            price = float(r["price"])
            image_url = (r["image_url"] or "").replace("'", "''")
            sql = f"INSERT INTO products (name, color, pattern, sleeve_length, style, price, image_url) VALUES ('{name}', '{color}', '{pattern}', '{sleeve}', '{style}', {price}, '{image_url}');\n"
            f.write(sql)
        f.write("COMMIT;\n")

    print("Wrote CSV:", CSV_PATH)
    print("Wrote SQL:", SQL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000, help="Number of products to generate")
    args = parser.parse_args()

    print("Starting generation. Count:", args.count)
    rows = generate(count=args.count)
    write_outputs(rows)
    print("Done: generated", len(rows), "products.")
