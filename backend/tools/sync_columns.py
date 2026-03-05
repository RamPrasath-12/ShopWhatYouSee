"""
Compare local PostgreSQL vs Supabase visual_attributes table.
Adds missing columns to Supabase and copies data using FAST bulk updates.
"""
import psycopg2
from psycopg2.extras import execute_values
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

    # Only process columns that don't have full data or are missing
    # Let's just process all 14 previously missing columns to be safe since it was interrupted
    columns_to_sync = [
        'primary_color_hex', 'rating_count', 'color_confidence', 'product_url', 
        'secondary_color_name', 'secondary_color_hex', 'extraction_quality', 'size', 
        'secondary_confidence', 'rating', 'image_path', 'discount_pct', 
        'pattern_confidence', 'sleeve_confidence'
    ]

    supa_cur = supa.cursor()

    print("Checking for missing columns...")
    for col in columns_to_sync:
        if col not in supa_dict:
            dtype, max_len = local_dict[col]
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
            else:
                sql_type = "TEXT"
            print(f"  Adding column: {col} ({sql_type})")
            supa_cur.execute(f"ALTER TABLE visual_attributes ADD COLUMN IF NOT EXISTS {col} {sql_type}")
    supa.commit()

    print("\nStarting FAST data copy...")
    local_cur = local.cursor()

    for col in columns_to_sync:
        print(f"  Copying {col}...")
        
        # We only want to update rows in Supabase where this col is NULL
        # to save time on columns we already successfully synced
        supa_cur.execute(f"SELECT COUNT(*) FROM visual_attributes WHERE {col} IS NULL")
        missing_count = supa_cur.fetchone()[0]
        
        if missing_count == 0:
            print(f"    ✓ {col} is already fully populated.")
            continue
            
        local_cur.execute(f"SELECT product_id, {col} FROM visual_attributes WHERE {col} IS NOT NULL")
        rows = local_cur.fetchall()
        print(f"    Found {len(rows)} local rows. Updating...")

        if rows:
            # Using execute_values for bulk update (100x faster)
            update_query = f"""
                UPDATE visual_attributes 
                SET {col} = data.new_val 
                FROM (VALUES %s) AS data(pid, new_val) 
                WHERE visual_attributes.product_id = data.pid
                  AND visual_attributes.{col} IS NULL
            """
            # Need to explicitly typecast the values based on column type
            dtype = local_dict[col][0]
            cast_type = "::text"
            if dtype == 'integer':
                cast_type = "::integer"
            
            update_query = f"""
                UPDATE visual_attributes 
                SET {col} = CAST(data.new_val AS {dtype})
                FROM (VALUES %s) AS data(pid, new_val) 
                WHERE visual_attributes.product_id = data.pid
                  AND visual_attributes.{col} IS NULL
            """
            
            # Simple string casting query
            update_query = f"""
                UPDATE visual_attributes 
                SET {col} = CAST(data.new_val AS {dtype})
                FROM (VALUES %s) AS data(pid, new_val) 
                WHERE visual_attributes.product_id::text = data.pid::text
                  AND visual_attributes.{col} IS NULL
            """

            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                execute_values(supa_cur, update_query, batch, template=None, page_size=1000)
                supa.commit()
                print(f"    Bulk updated {min(i+batch_size, len(rows))}/{len(rows)}")

    local_cur.close(); supa_cur.close()
    local.close(); supa.close()
    print(f"\n{'='*60}")
    print("✅ FAST Sync complete!")

if __name__ == "__main__":
    main()
