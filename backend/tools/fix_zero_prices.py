"""
Fix products with price=0 by scraping the correct price from Myntra product pages.
Uses the existing product_url field to fetch prices.
"""
import psycopg2
import time
import re
import urllib.request
import json

DSN = "postgresql://postgres:postgres123%40@localhost:5432/shopwhatyousee"

def scrape_price_from_myntra(product_url):
    """Extract price from Myntra product page's embedded pdpData JSON."""
    try:
        req = urllib.request.Request(product_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        
        # Extract pdpData JSON
        match = re.search(r'pdpData\s*=\s*(\{.*?\});\s*</', html, re.DOTALL)
        if not match:
            return None, None
        
        data = json.loads(match.group(1))
        price_info = data.get("price", {})
        
        discounted = price_info.get("discounted", 0)
        mrp = price_info.get("mrp", 0)
        
        if discounted > 0 or mrp > 0:
            return discounted or mrp, mrp or discounted
        
        return None, None
    except Exception as e:
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
    
    for i, (pid, url, cat) in enumerate(rows):
        disc, orig = scrape_price_from_myntra(url)
        
        if disc and disc > 0:
            cur.execute("""
                UPDATE visual_attributes 
                SET discounted_price = %s, original_price = %s 
                WHERE product_id = %s
            """, (disc, orig, pid))
            conn.commit()
            fixed += 1
            
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] Fixed: {fixed} | Failed: {failed}")
        else:
            failed += 1
        
        time.sleep(0.2)  # Be polite to Myntra
    
    print(f"\n✅ Done! Fixed: {fixed} | Failed: {failed} | Total: {len(rows)}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
