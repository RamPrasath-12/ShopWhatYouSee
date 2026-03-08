"""
Fix products with price=0 by scraping the correct price from Myntra product pages.
Uses the existing product_url field to fetch prices.
"""
import psycopg2
import time
import re
import urllib.request
import json
import concurrent.futures

DSN = "postgresql://postgres:postgres123%40@localhost:5432/shopwhatyousee"

def scrape_price_from_myntra(product_url):
    """Extract price from Myntra product page's embedded pdpData JSON."""
    try:
        req = urllib.request.Request(product_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        
        # Extract schema.org JSON-LD price
        # Looks like: "price" : "370",
        match = re.search(r'"price"\s*:\s*"([0-9\.]+)"', html)
        if match:
            price = float(match.group(1))
            if price > 0:
                # We only got 1 price, so we just return it for both discounted and original
                return price, price
                
        return None, None
    except Exception as e:
        print(f"Error fetching {product_url}: {e}")
        return None, None

def main():
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    
    # Get all products with price = 0
    cur.execute("""
        SELECT product_id, product_url, category 
        FROM visual_attributes 
        WHERE COALESCE(discounted_price, 0) = 0 
          AND COALESCE(original_price, 0) = 0
          AND product_url IS NOT NULL
          AND product_url LIKE '%myntra%'
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} products with price=0")
    
    fixed = 0
    failed = 0
    
    def process_row(row):
        pid, url, cat = row
        disc, orig = scrape_price_from_myntra(url)
        return pid, disc, orig
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            pid, disc, orig = future.result()
            
            if disc and disc > 0:
                cur.execute("""
                    UPDATE visual_attributes 
                    SET discounted_price = %s, original_price = %s 
                    WHERE product_id = %s
                """, (disc, orig, pid))
                conn.commit()
                fixed += 1
            else:
                failed += 1
                
            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                print(f"  [{i+1}/{len(rows)}] Fixed: {fixed} | Failed: {failed}")
    
    print(f"\n✅ Done! Fixed: {fixed} | Failed: {failed} | Total: {len(rows)}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
