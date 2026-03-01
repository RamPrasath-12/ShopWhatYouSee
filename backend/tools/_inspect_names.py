"""Sample all unique keywords from product_name for vocabulary design."""
import psycopg2
import sys
import re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("SELECT product_name FROM visual_attributes")
names = [r[0] for r in cur.fetchall() if r[0]]

# Count all words
word_counter = Counter()
for name in names:
    words = re.findall(r'[a-zA-Z]+', name.lower())
    for w in words:
        word_counter[w] += 1

# Print top 100 most common words
print("TOP 100 WORDS:")
for word, count in word_counter.most_common(100):
    print(f"  {word:20s} {count:6d}")

# Gender-related words
print("\nGENDER-RELATED WORD FREQUENCIES:")
for word in ['men', "men's", 'women', "women's", 'unisex', 'boys', 'girls', 'kids', 'boy', 'girl']:
    found = sum(1 for n in names if re.search(r'\b' + word + r'\b', n.lower()))
    print(f"  {word:15s} {found}")

# Style-related words
print("\nSTYLE-RELATED WORD FREQUENCIES:")
for word in ['casual', 'formal', 'sports', 'running', 'walking', 'party', 'ethnic', 'traditional',
             'denim', 'workwear', 'streetwear', 'athleisure', 'training', 'workout', 'wedding',
             'outdoor', 'comfort', 'fashion', 'slim', 'regular', 'relaxed', 'biker', 'bomber',
             'cargo', 'classic', 'daily', 'office', 'trendy', 'sporty', 'retro', 'vintage',
             'lounge', 'active', 'travel', 'trekking', 'hiking', 'beach', 'gym', 'yoga',
             'prayer', 'printed', 'solid', 'striped', 'checked', 'embroidered', 'geometric']:
    found = sum(1 for n in names if re.search(r'\b' + word + r'\b', n.lower()))
    if found > 0:
        print(f"  {word:20s} {found}")

# Material-related words
print("\nMATERIAL-RELATED WORD FREQUENCIES:")
for word in ['cotton', 'polyester', 'linen', 'denim', 'leather', 'pu', 'silk', 'wool',
             'nylon', 'viscose', 'synthetic', 'velvet', 'satin', 'chiffon', 'georgette',
             'crepe', 'rayon', 'canvas', 'suede', 'rubber', 'mesh', 'knit', 'knitted',
             'fleece', 'jersey', 'twill', 'copper', 'brass', 'steel', 'gold', 'silver',
             'rhodium', 'acrylic', 'jute', 'cork', 'eva', 'foam', 'lycra', 'spandex',
             'terry', 'chambray', 'corduroy', 'oxford', 'poplin']:
    found = sum(1 for n in names if re.search(r'\b' + word + r'\b', n.lower()))
    if found > 0:
        print(f"  {word:20s} {found}")

cur.close()
conn.close()
