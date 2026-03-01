"""Category × Color Family cross-tab audit."""
import psycopg2, sys
sys.stdout.reconfigure(encoding='utf-8')

conn = psycopg2.connect(host='localhost', database='shopwhatyousee', user='postgres', password='postgres123@')
cur = conn.cursor()

cur.execute(
    "SELECT category, color_family, COUNT(*) FROM visual_attributes "
    "GROUP BY category, color_family ORDER BY category, COUNT(*) DESC"
)
rows = cur.fetchall()

# Group by category
from collections import defaultdict
cat_data = defaultdict(list)
cat_totals = defaultdict(int)
for cat, fam, cnt in rows:
    cat_data[cat].append((fam, cnt))
    cat_totals[cat] += cnt

for cat in sorted(cat_data.keys()):
    total = cat_totals[cat]
    print(f"\n[{cat}] ({total} products)")
    for fam, cnt in cat_data[cat]:
        pct = cnt * 100 / total
        bar = "#" * int(pct / 2)
        flag = " <<<" if pct > 90 else ""
        print(f"  {str(fam):10s}  {cnt:5d}  ({pct:5.1f}%)  {bar}{flag}")

cur.close()
conn.close()
