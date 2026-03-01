import psycopg2
conn = psycopg2.connect(host='localhost', database='shopwhatyousee', user='postgres', password='postgres123@')
cur = conn.cursor()

# Check embedding dimensions
cur.execute('SELECT COUNT(*), MIN(array_length(embedding,1)), MAX(array_length(embedding,1)) FROM visual_attributes')
r = cur.fetchone()
print(f'Embedding check: count={r[0]}, min_dim={r[1]}, max_dim={r[2]}')

# Check NULL embeddings
cur.execute('SELECT COUNT(*) FROM visual_attributes WHERE embedding IS NULL')
print(f'NULL embeddings: {cur.fetchone()[0]}')

# Check quality range
cur.execute('SELECT MIN(extraction_quality), AVG(extraction_quality), MAX(extraction_quality) FROM visual_attributes')
eq = cur.fetchone()
print(f'Quality: min={eq[0]:.3f}, avg={eq[1]:.3f}, max={eq[2]:.3f}')

# Check sleeve for non-upper-wear categories
NON_UPPER = ('belt','caps','earrings','Footwear_sandals','Footwear_shoes','glasses','necklace','watch')
cur.execute("SELECT DISTINCT category, sleeve_value FROM visual_attributes WHERE category IN %s ORDER BY category", (NON_UPPER,))
print('\nNon-upper-wear sleeve values:')
for r in cur.fetchall():
    print(f'  {r[0]:25s}: {r[1]}')

# Check sleeve for upper-wear categories
UPPER = ('blazer','churidhar','Jacket','shirts','tshirt')
cur.execute("SELECT category, sleeve_value, COUNT(*) FROM visual_attributes WHERE category IN %s GROUP BY category, sleeve_value ORDER BY category, sleeve_value", (UPPER,))
print('\nUpper-wear sleeve distribution:')
for r in cur.fetchall():
    print(f'  {r[0]:25s}: sleeve={r[1]:20s} count={r[2]:>5d}')

cur.close()
conn.close()
