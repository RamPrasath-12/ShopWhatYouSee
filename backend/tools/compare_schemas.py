"""
Compare all tables and their columns between local PostgreSQL and Supabase.
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

def get_schema(conn):
    cur = conn.cursor()
    # Get all tables in public schema
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = [r[0] for r in cur.fetchall()]
    
    schema = {}
    for t in tables:
        cur.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{t}'
        """)
        schema[t] = {r[0]: r[1] for r in cur.fetchall()}
    cur.close()
    return schema

def main():
    print("Connecting to local DB...")
    local = psycopg2.connect(**LOCAL_CONFIG)
    print("Connecting to Supabase...")
    supa = psycopg2.connect(DATABASE_URL)

    local_schema = get_schema(local)
    supa_schema = get_schema(supa)

    print(f"\n{'='*50}")
    print("TABLE COMPARISON")
    print(f"{'='*50}")
    
    local_tables = set(local_schema.keys())
    supa_tables = set(supa_schema.keys())
    
    missing_tables = local_tables - supa_tables
    extra_tables = supa_tables - local_tables
    
    print(f"Tables missing in Supabase: {missing_tables if missing_tables else 'None'}")
    print(f"Extra tables in Supabase: {extra_tables if extra_tables else 'None'}")
    print(f"Matching tables: {local_tables.intersection(supa_tables)}")
    
    print(f"\n{'='*50}")
    print("COLUMN COMPARISON (for matching tables)")
    print(f"{'='*50}")
    
    for table in local_tables.intersection(supa_tables):
        l_cols = set(local_schema[table].keys())
        s_cols = set(supa_schema[table].keys())
        
        missing = l_cols - s_cols
        if missing:
            print(f"❌ Table '{table}' is missing columns in Supabase: {missing}")
        else:
            print(f"✅ Table '{table}' has all columns locally present")

    local.close(); supa.close()

if __name__ == "__main__":
    main()
