import os
import json
import csv
import base64
import requests
from tqdm import tqdm

# change if needed
BACKEND_URL = "http://localhost:5000/extract-attributes"

IMAGES_DIR = "static/products"
OUT_CSV = "static/products/products_final.csv"
OUT_SQL = "static/products/products_final.sql"

def encode_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')

def infer_style(pattern, sleeve_length):
    # simple classifier (you can improve later)
    if sleeve_length == "long":
        return "formal"
    if pattern == "solid":
        return "casual"
    return "party"

def main():
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
    rows = []

    print("Processing", len(images), "images...\n")

    for filename in tqdm(images):
        filepath = os.path.join(IMAGES_DIR, filename)
        img_b64 = encode_image(filepath)

        # send to AGMAN backend API
        try:
            r = requests.post(BACKEND_URL, json={"image": img_b64}).json()
        except:
            print("❌ ERROR processing", filename)
            continue

        attr = r.get("attributes", {})

        color = attr.get("color_hex")
        pattern = attr.get("pattern")
        sleeve = attr.get("sleeve_length")
        style = infer_style(pattern, sleeve)

        price = 300 + (abs(hash(filename)) % 2500)

        rows.append([
            filename,              # image filename
            f"Product {filename}", # name
            color,
            pattern,
            sleeve,
            style,
            price
        ])

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "name", "color", "pattern", "sleeve_length", "style", "price"])
        writer.writerows(rows)

    # Write SQL
    with open(OUT_SQL, "w") as f:
        f.write("TRUNCATE products RESTART IDENTITY;\n")
        for row in rows:
            filename, name, color, pattern, sleeve, style, price = row
            f.write(
                f"INSERT INTO products (name, color, pattern, sleeve_length, style, price, image_url) "
                f"VALUES ('{name}', '{color}', '{pattern}', '{sleeve}', '{style}', {price}, "
                f"'http://localhost:5000/products/{filename}');\n"
            )

    print("\nDONE! New metadata saved.")
    print("CSV:", OUT_CSV)
    print("SQL:", OUT_SQL)

if __name__ == "__main__":
    main()
