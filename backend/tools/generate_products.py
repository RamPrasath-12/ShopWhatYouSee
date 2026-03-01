import random
import psycopg2

colors = ["red", "blue", "black", "white", "green", "yellow", "brown", "pink", "grey"]
patterns = ["solid", "striped", "checked", "patterned"]
sleeves = ["short", "long", "three_quarter"]
styles = ["casual", "formal", "party", "sports", "ethnic"]
brands = ["Nike", "Adidas", "Zara", "H&M", "Puma"]

def random_price():
    return round(random.uniform(250, 2500), 2)

def random_name(style, color):
    base = ["Shirt", "T-Shirt", "Kurta", "Jacket", "Sweatshirt", "Hoodie"]
    return f"{color.title()} {style.title()} {random.choice(base)}"

def generate_items(count=500):
    items = []

    for _ in range(count):
        color = random.choice(colors)
        pattern = random.choice(patterns)
        sleeve = random.choice(sleeves)
        style = random.choice(styles)
        brand = random.choice(brands)
        price = random_price()

        name = random_name(style, color)

        # placeholder image (later replace with real images)
        image_url = f"https://via.placeholder.com/300x400.png?text={color}+{style}"

        items.append((name, color, pattern, sleeve, style, price, image_url, style, brand))

    return items


def insert_to_db(items):
    conn = psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="postgres123@"
    )
    cur = conn.cursor()

    query = """
    INSERT INTO products (name, color, pattern, sleeve_length, style, price, image_url, category, brand)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    for item in items:
        cur.execute(query, item)

    conn.commit()
    cur.close()
    conn.close()
    print("Inserted", len(items), "products.")


if __name__ == "__main__":
    items = generate_items(800)
    insert_to_db(items)
