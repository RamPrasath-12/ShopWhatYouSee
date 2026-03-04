"""
Compare local PostgreSQL vs Supabase visual_attributes table.
Adds missing columns to Supabase and copies data.
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

def get_columns(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'visual_attributes'
        ORDER BY ordinal_position
    """)
    cols = cur.fetchall()
    cur.close()
    return cols

def main():
    print("Connecting to local DB...")
    local = psycopg2.connect(**LOCAL_CONFIG)
    print("Connecting to Supabase...")
    supa = psycopg2.connect(DATABASE_URL)

    local_cols = get_columns(local)
    supa_cols = get_columns(supa)

    local_dict = {c[0]: (c[1], c[2]) for c in local_cols}
    supa_dict = {c[0]: (c[1], c[2]) for c in supa_cols}

    missing = set(local_dict.keys()) - set(supa_dict.keys())

    print(f"\n{'='*60}")
    print(f"Local columns: {len(local_cols)}")
    print(f"Supabase columns: {len(supa_cols)}")
    print(f"Missing in Supabase: {missing}")
    print(f"{'='*60}")

    if not missing:
        print("No missing columns!")
        local.close(); supa.close()
        return

    # Add missing columns
    supa_cur = supa.cursor()
    for col in missing:
        dtype, max_len = local_dict[col]
        # Map data types
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
        elif dtype == 'ARRAY':
            sql_type = "TEXT"  # fallback
        else:
            sql_type = "TEXT"

        print(f"  Adding column: {col} ({sql_type})")
        supa_cur.execute(f"ALTER TABLE visual_attributes ADD COLUMN IF NOT EXISTS {col} {sql_type}")

    supa.commit()
    print("Columns added. Now copying data...")

    # Copy data for missing columns
    local_cur = local.cursor()
    for col in missing:
        print(f"  Copying {col}...")
        # Get all product_id → value pairs from local
        local_cur.execute(f"SELECT product_id, {col} FROM visual_attributes WHERE {col} IS NOT NULL")
        rows = local_cur.fetchall()
        print(f"    {len(rows)} rows with data")

        if rows:
            # Batch update in Supabase
            batch_size = 100
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                for pid, val in batch:
                    supa_cur.execute(
                        f"UPDATE visual_attributes SET {col} = %s WHERE product_id = %s",
                        (val, pid)
                    )
                supa.commit()
                print(f"    Updated {min(i+batch_size, len(rows))}/{len(rows)}")

    local_cur.close(); supa_cur.close()
    local.close(); supa.close()
    print(f"\n{'='*60}")
    print("✅ Sync complete!")

if __name__ == "__main__":
    main()
