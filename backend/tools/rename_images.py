import pandas as pd
from pathlib import Path

CSV = "static/products/products_final.csv"
OUT_CSV = "static/products/products_named.csv"
OUT_SQL = "static/products/products_named.sql"
IMG_DIR = Path("static/products/images")

df = pd.read_csv(CSV)

rows = []

for i, r in df.iterrows():
    old_name = r["image"]        # <---- FIXED
    color = r["color"].replace("#", "")
    pattern = r["pattern"]
    sleeve = r["sleeve_length"]

    new_name = f"{color}_{pattern}_{sleeve}_{i}.jpg"

    old_path = IMG_DIR / old_name
    new_path = IMG_DIR / new_name

    if old_path.exists():
        old_path.rename(new_path)

    r["image"] = new_name
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# SQL export
with open(OUT_SQL, "w") as f:
    for _, r in df.iterrows():
        f.write(
            f"INSERT INTO products (image, name, color, pattern, sleeve_length, style, price) "
            f"VALUES ('{r['image']}', '{r['name']}', '{r['color']}', "
            f"'{r['pattern']}', '{r['sleeve_length']}', '{r['style']}', {r['price']});\n"
        )

print("DONE: renamed & exported")
