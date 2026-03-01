import psycopg2

conn = psycopg2.connect(host='localhost', database='shopwhatyousee', user='postgres', password='postgres123@')
cur = conn.cursor()
cur.execute("""
    SELECT product_id, category, primary_color_name, primary_color_hex,
           color_confidence, pattern_value, sleeve_value, sleeve_confidence,
           extraction_quality, array_length(embedding, 1) as emb_len,
           brand, discounted_price, rating
    FROM visual_attributes
    ORDER BY category, product_id
""")
cols = [d[0] for d in cur.description]
for r in cur.fetchall():
    d = dict(zip(cols, r))
    print(f"  {d['product_id']:30s} | cat={d['category']:15s} | color={str(d['primary_color_name']):12s} "
          f"| pattern={str(d['pattern_value']):10s} | sleeve={str(d['sleeve_value']):15s} "
          f"| eq={d['extraction_quality']:.3f} | emb_len={d['emb_len']} "
          f"| brand={str(d['brand'])[:15]:15s} | price={d['discounted_price']}")

print(f"\nTotal rows: {cur.rowcount}")
cur.close()
conn.close()
