import psycopg2
import sys

LOCAL_URL = 'postgresql://postgres:postgres123%40@localhost:5432/shopwhatyousee'
SUPA_URL  = 'postgresql://postgres:ShopWhatYouSee123@db.mxjpueufbooxgewqxccm.supabase.co:5432/postgres'

def get_schema(url):
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        
        # Get tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public'
        """)
        tables = [r[0] for r in cur.fetchall()]
        
        schema = {}
        row_counts = {}
        for table in tables:
            # Columns
            cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table}'")
            schema[table] = {r[0]: r[1] for r in cur.fetchall()}
            
            # Row counts
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            row_counts[table] = cur.fetchone()[0]
            
        conn.close()
        return schema, row_counts
    except Exception as e:
        print(f'Error connecting to database: {e}')
        return None, None

def main():
    print('Fetching Local DB Schema...')
    local_schema, local_counts = get_schema(LOCAL_URL)

    print('Fetching Supabase DB Schema...')
    supa_schema, supa_counts = get_schema(SUPA_URL)

    if not local_schema or not supa_schema:
        print('Failed to get schema(s)')
        sys.exit(1)

    print('\n=== TABLE COMPARISON ===')
    all_tables = set(local_schema.keys()) | set(supa_schema.keys())
    for table in sorted(list(all_tables)):
        print(f'\nTable: {table}')
        in_local = table in local_schema
        in_supa = table in supa_schema
        
        print(f'  In Local: {in_local} (Rows: {local_counts.get(table, 0)})')
        print(f'  In Supaba: {in_supa} (Rows: {supa_counts.get(table, 0)})')
        
        if in_local and in_supa:
            local_cols = set(local_schema[table].keys())
            supa_cols = set(supa_schema[table].keys())
            
            if local_cols != supa_cols:
                missing_in_supa = local_cols - supa_cols
                extra_in_supa = supa_cols - local_cols
                if missing_in_supa: print(f'  [!] Missing in Supabase: {missing_in_supa}')
                if extra_in_supa: print(f'  [!] Extra in Supabase: {extra_in_supa}')
            else:
                print('  Columns MATCH perfectly.')

if __name__ == "__main__":
    main()
