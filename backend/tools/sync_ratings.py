"""
Sync 'ratings' table from local DB to Supabase.
Creates the table if it doesn't exist and copies all rows.
"""
import psycopg2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DATABASE_URL

LOCAL_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

def main():
    print("Connecting to local DB...")
    local = psycopg2.connect(**LOCAL_CONFIG)
    print("Connecting to Supabase...")
    supa = psycopg2.connect(DATABASE_URL)

    local_cur = local.cursor()
    supa_cur = supa.cursor()

    # Get local schema for the ratings table
    print("Reading local 'ratings' schema...")
    local_cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'ratings'
        ORDER BY ordinal_position
    """)
    columns = local_cur.fetchall()
    
    if not columns:
        print("Table 'ratings' not found in local DB!")
        return

    # Create table in Supabase
    print("Creating 'ratings' table in Supabase (if not exists)...")
    col_defs = []
    for col, dtype, max_len in columns:
        if dtype == 'character varying':
            sql_type = f"VARCHAR({max_len})" if max_len else "TEXT"
        elif dtype == 'text':
            sql_type = "TEXT"
        elif dtype == 'integer':
            sql_type = "INTEGER"
        elif dtype == 'double precision':
            sql_type = "DOUBLE PRECISION"
        elif dtype == 'numeric':
            sql_type = "NUMERIC"
        elif dtype == 'timestamp without time zone':
            sql_type = "TIMESTAMP"
        else:
            sql_type = "TEXT"
        col_defs.append(f"{col} {sql_type}")
    
    create_sql = f"CREATE TABLE IF NOT EXISTS ratings ({', '.join(col_defs)})"
    supa_cur.execute(create_sql)
    supa.commit()
    print("Table ready.")

    # Get all columns names
    col_names = [c[0] for c in columns]
    col_names_str = ", ".join(col_names)
    placeholders = ", ".join(["%s"] * len(col_names))

    # Copy data
    print("Fetching local data...")
    local_cur.execute(f"SELECT {col_names_str} FROM ratings")
    rows = local_cur.fetchall()
    print(f"Found {len(rows)} rows to copy.")

    if rows:
        print("Copying to Supabase (batching)...")
        # Clear existing data just in case to avoid duplicates if run multiple times without constraints
        supa_cur.execute("TRUNCATE TABLE ratings")
        
        insert_sql = f"INSERT INTO ratings ({col_names_str}) VALUES ({placeholders})"
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            supa_cur.executemany(insert_sql, batch)
            supa.commit()
            print(f"  Inserted {min(i+batch_size, len(rows))}/{len(rows)} rows")

    print("\n✅ Ratings table sync complete!")
    local.close(); supa.close()

if __name__ == "__main__":
    main()
