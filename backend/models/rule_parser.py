import re

def extract_numeric_filters(text: str):
    filters = {}

    if not text:
        return filters

    price_match = re.search(r"(under|below|less than)\s*(\d+)", text.lower())
    if price_match:
        filters["price_max"] = int(price_match.group(2))

    return filters
