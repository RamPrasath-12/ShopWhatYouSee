import psycopg2
conn = psycopg2.connect(host='localhost', database='shopwhatyousee', user='postgres', password='postgres123@')
cur = conn.cursor()

# Check partial progress
cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category ORDER BY category")
print("Current data (partial from crashed run):")
total = 0
for row in cur.fetchall():
    print(f"  {row[0]:20s}: {row[1]}")
    total += row[1]
print(f"  TOTAL: {total}")

# Truncate for fresh start
cur.execute('TRUNCATE TABLE visual_attributes RESTART IDENTITY')
conn.commit()
print("\nTable truncated for fresh run")
cur.close()
conn.close()
