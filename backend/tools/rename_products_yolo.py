# backend/tools/rename_products_yolo.py
import os
import csv
import torch
from ultralytics import YOLO

INPUT_CSV = "static/products/products_final.csv"
OUTPUT_CSV = "static/products/products_named.csv"
OUTPUT_SQL = "static/products/products_named.sql"
IMG_DIR = "static/products/images"

# ------------------------------
# YOLO MODEL (clothing detection)
# ------------------------------
print("Loading YOLO model...")
model = YOLO("yolov8s-clothes.pt")   # <--- use your trained or downloaded model
print("YOLO loaded.")

# Clothing label mapping (simplify names)
CLOTHES_MAP = {
    "t-shirt": "T-Shirt",
    "shirt": "Shirt",
    "dress": "Dress",
    "jeans": "Jeans",
    "shorts": "Shorts",
    "jacket": "Jacket",
    "coat": "Coat",
    "kurta": "Kurta",
    "sweatshirt": "Sweatshirt",
    "hoodie": "Hoodie",
    "saree": "Saree",
    "skirt": "Skirt",
    "top": "Top",
    "blazer": "Blazer",
    "suit": "Suit"
}

def detect_clothing(image_path):
    try:
        results = model(image_path, verbose=False)
        if len(results[0].boxes) == 0:
            return "Clothing"
        cls_id = int(results[0].boxes[0].cls)
        cls_name = results[0].names[cls_id].lower()

        return CLOTHES_MAP.get(cls_name, "Clothing")

    except Exception:
        return "Clothing"


def build_name(attributes, clothing_type):
    color = attributes["color"]
    style = attributes["style"] or "Casual"

    color_cap = color.capitalize() if color else ""

    return f"{color_cap} {style.capitalize()} {clothing_type}"


# -----------------------------------
# PROCESS CSV
# -----------------------------------
print("Reading product metadata...")

rows = []
with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} products")


new_rows = []

for r in rows:
    img_file = os.path.join(IMG_DIR, r["image_filename"])

    # ---- YOLO clothing detection ----
    clothing_type = detect_clothing(img_file)

    # ---- Beautiful product name ----
    attributes = {
        "color": r["color"],
        "style": r["style"]
    }

    name = build_name(attributes, clothing_type)

    r["name"] = name
    r["clothing_type"] = clothing_type
    new_rows.append(r)

print("Saving renamed product data...")

# Write new CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=new_rows[0].keys())
    writer.writeheader()
    writer.writerows(new_rows)

# Write SQL
with open(OUTPUT_SQL, "w", encoding="utf-8") as f:
    for r in new_rows:
        f.write(
            f"INSERT INTO products (name, color, pattern, sleeve_length, style, price, image_url) "
            f"VALUES ('{r['name']}', '{r['color']}', '{r['pattern']}', '{r['sleeve_length']}', "
            f"'{r['style']}', {r['price']}, '/static/products/images/{r['image_filename']}');\n"
        )

print("\n🎉 DONE!")
print(f"Created:\n - {OUTPUT_CSV}\n - {OUTPUT_SQL}")
