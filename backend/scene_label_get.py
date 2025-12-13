import json
import re

txt_file = "D:/Downloads/categories_places365.txt"   # your downloaded file
json_file = "models/places365_classes.json"

classes = []

with open(txt_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Example: "/a/airfield 0" OR "/b/bakery/shop 20"
        label_raw, idx = line.rsplit(" ", 1)

        # Remove ANY "/x/" prefix (a, b, c, d, ...)
        label = re.sub(r"^/[a-z]/", "", label_raw)

        classes.append(label)

with open(json_file, "w") as f:
    json.dump(classes, f, indent=2)

print("Saved", len(classes), "classes.")
