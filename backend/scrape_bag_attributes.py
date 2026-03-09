"""
Backfill Scraped Attributes for Bags
=====================================
Reads `product_url` from DB for bags where `scraped_attributes_json` is NULL.
Scrapes Myntra product page for detailed attributes (Material, Size & Fit, etc.).
Updates the database.
"""

import os
import sys
import json
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client

# ── Paths ──
BACKEND_DIR = r"d:\Final_Year_Project\ShopWhatYouSee\backend"
sys.path.insert(0, BACKEND_DIR)
from db_config import get_db_connection

# ── Supabase Config ──
from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, '.env'))

HEADERS = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    'Accept-Language': "en-US,en;q=0.9",
}

def scrape_myntra_product_page(url):
    """Scrape product details from Myntra URL."""
    if not url or not str(url).startswith('http'):
        return None, None, None
        
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Myntra loads data dynamically via window.__myx = {...}
        # But we can try to find standard attributes in the HTML first
        soup = BeautifulSoup(response.text, 'html.parser')
        
        material = None
        desc_html = ""
        attributes_json = {}
        
        # Find the script tag containing the initial state
        script_tag = None
        for script in soup.find_all('script'):
            if script.string and 'window.__myx' in script.string:
                script_tag = script.string
                break
                
        if script_tag:
            try:
                # Extract JSON from window.__myx = { ... };
                json_str = script_tag.split('window.__myx = ')[1].split(';\n</script>')[0]
                if json_str.endswith(';'):
                    json_str = json_str[:-1]
                
                data = json.loads(json_str)
                pdp_data = data.get('pdpData', {})
                
                # 1. Attributes
                article_attrs = pdp_data.get('articleAttributes', {})
                for key, value in article_attrs.items():
                    attributes_json[key] = value
                    
                    if 'material' in key.lower():
                        material = value
                
                # 2. Description
                descriptors = pdp_data.get('productDescriptors', {})
                desc_obj = descriptors.get('description', {})
                if desc_obj and desc_obj.get('value'):
                    desc_html = desc_obj.get('value')
                
                # Try to find material specifically if not in attributes
                if not material:
                    materials_list = descriptors.get('materials_care_desc', {})
                    if materials_list and materials_list.get('value'):
                        # Often comes as HTML list
                        m_soup = BeautifulSoup(materials_list.get('value'), 'html.parser')
                        material = m_soup.get_text(strip=True)
                        
            except Exception as e:
                print(f"Error parsing JSON from Myntra script: {e}")
                
        return json.dumps(attributes_json) if attributes_json else None, material, desc_html
        
    except requests.exceptions.RequestException as e:
        # print(f"Request failed for {url}: {e}")
        return None, None, None

def process_product(product):
    pid, url = product
    attr_json, material, desc = scrape_myntra_product_page(url)
    
    if attr_json or material or desc:
        return (pid, attr_json, material, desc, True)
    return (pid, None, None, None, False)

def main():
    print("=" * 60)
    print("  🛍️ BAG ATTRIBUTE SCRAPING")
    print("=" * 60)

    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get bags without scraped data
    cur.execute('''
        SELECT product_id, product_url 
        FROM visual_attributes 
        WHERE category='bag' 
          AND scraped_attributes_json IS NULL
          AND product_url IS NOT NULL
    ''')
    
    products = cur.fetchall()
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(products)} bags missing scraped data.")
    
    if not products:
        print("All bags already have scraped data. Exiting.")
        cur.close()
        conn.close()
        return

    success_count = 0
    fail_count = 0
    
    # Process in batches to update DB incrementally
    BATCH_SIZE = 50
    MAX_WORKERS = 10
    
    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i+BATCH_SIZE]
        results = []
        
        print(f"[{time.strftime('%H:%M:%S')}] Processing batch {i//BATCH_SIZE + 1} ({i} to {i+len(batch)})...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(process_product, p): p for p in batch}
            for future in as_completed(future_to_url):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Thread exception: {e}")
                    
        # Update DB for this batch
        update_count = 0
        for pid, attr_json, material, desc, success in results:
            if success:
                try:
                    cur.execute('''
                        UPDATE visual_attributes 
                        SET scraped_attributes_json = %s,
                            scraped_material = %s,
                            scraped_description = %s,
                            scraped_at = NOW()
                        WHERE product_id = %s
                    ''', (attr_json, material, desc, pid))
                    update_count += 1
                except Exception as e:
                    print(f"DB update error for {pid}: {e}")
            else:
                fail_count += 1
                
        conn.commit()
        success_count += update_count
        print(f"  → Batch {i//BATCH_SIZE + 1} done. Updated {update_count}/{len(batch)}. Total success: {success_count}")
        
        # Polite delay between batches
        time.sleep(2)

    cur.close()
    conn.close()
    
    print("=" * 60)
    print(f"✅ Scraping finished. Re-scraped {success_count} products. Failed {fail_count}.")
    print("=" * 60)

if __name__ == "__main__":
    main()
