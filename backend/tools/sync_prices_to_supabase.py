import psycopg2
import concurrent.futures

LOCAL_DSN = "postgresql://postgres:postgres123%40@localhost:5432/shopwhatyousee"
SUPABASE_DSN = "postgresql://postgres:ShopWhatYouSee123@db.mxjpueufbooxgewqxccm.supabase.co:5432/postgres"

def main():
    print("Connecting to local DB...")
    local_conn = psycopg2.connect(LOCAL_DSN)
    local_cur = local_conn.cursor()
    
    # Get all products from local DB to copy their prices
    local_cur.execute("SELECT product_id, discounted_price, original_price FROM visual_attributes WHERE discounted_price > 0 OR original_price > 0")
    rows = local_cur.fetchall()
    local_cur.close()
    local_conn.close()
    
    print(f"Fetched {len(rows)} products with pricing from local DB.")
    
    print("Connecting to Supabase DB...")
    supa_conn = psycopg2.connect(SUPABASE_DSN)
    
    success = 0
    failed = 0
    
    def update_row(row):
        pid, disc, orig = row
        try:
            # Create a new cursor for each thread
            with psycopg2.connect(SUPABASE_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE visual_attributes 
                        SET discounted_price = %s, original_price = %s 
                        WHERE product_id = %s
                    """, (disc, orig, pid))
                conn.commit()
            return True
        except Exception as e:
            print(f"Failed to update {pid}: {e}")
            return False

    print("Updating Supabase over multiple threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(update_row, row): row for row in rows}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result():
                success += 1
            else:
                failed += 1
                
            if (i + 1) % 100 == 0 or (i + 1) == len(rows):
                print(f"  [{i+1}/{len(rows)}] Updated: {success} | Failed: {failed}")

    print("Finished syncing prices to Supabase!")

if __name__ == "__main__":
    main()
