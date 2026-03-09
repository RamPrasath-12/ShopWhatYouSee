"""
Playwright-based Scraper for Bag Attributes
=============================================
Uses a real browser engine to bypass basic bot protection.
Extracts the window.__myx JSON payload.
"""

import os
import sys
import json
import time
import asyncio
from playwright.async_api import async_playwright
from db_config import get_db_connection

# ── Paths ──
BACKEND_DIR = r"d:\Final_Year_Project\ShopWhatYouSee\backend"
sys.path.insert(0, BACKEND_DIR)

async def scrape_batch(bags, browser):
    results = []
    # Use a single context but separate pages for speed
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 720}
    )
    
    # Process sequentially in browser but isolated to avoid memory leaks
    for pid, url in bags:
        page = await context.new_page()
        try:
            # Block heavy resources
            await page.route("**/*.{png,jpg,jpeg,webp,gif,css,woff,woff2,ttf,svg}", lambda route: route.abort())
            
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            # Extract window.__myx via JS execution
            myx_data = await page.evaluate("() => window.__myx")
            
            if myx_data and 'pdpData' in myx_data:
                pdp_data = myx_data['pdpData']
                
                # 1. Attributes
                attributes_json = {}
                material = None
                article_attrs = pdp_data.get('articleAttributes', {})
                for key, value in article_attrs.items():
                    attributes_json[key] = value
                    if 'material' in key.lower():
                        material = value
                        
                # 2. Description
                desc_html = ""
                descriptors = pdp_data.get('productDescriptors', {})
                desc_obj = descriptors.get('description', {})
                if desc_obj and desc_obj.get('value'):
                    desc_html = desc_obj.get('value')
                
                # Try to find material if missing
                if not material:
                    materials_list = descriptors.get('materials_care_desc', {})
                    if materials_list and materials_list.get('value'):
                        from bs4 import BeautifulSoup
                        m_soup = BeautifulSoup(materials_list.get('value'), 'html.parser')
                        material = m_soup.get_text(strip=True)
                        
                results.append((pid, json.dumps(attributes_json), material, desc_html, True))
            else:
                results.append((pid, None, None, None, False))
                print(f"  [Warn] No __myx data for {pid}")
                
        except Exception as e:
            print(f"  [Error] Failed {pid}: {str(e)[:100]}")
            results.append((pid, None, None, None, False))
        finally:
            await page.close()
            
    await context.close()
    return results

async def main_async():
    print("=" * 60)
    print("  🛍️ BAG ATTRIBUTE SCRAPING (PLAYWRIGHT)")
    print("=" * 60)

    conn = get_db_connection()
    cur = conn.cursor()
    
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
        print("Done.")
        return

    success_count = 0
    fail_count = 0
    BATCH_SIZE = 20
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for i in range(0, len(products), BATCH_SIZE):
            batch = products[i:i+BATCH_SIZE]
            print(f"[{time.strftime('%H:%M:%S')}] Processing batch {i//BATCH_SIZE + 1} ({i} to {i+len(batch)})...")
            
            results = await scrape_batch(batch, browser)
            
            # Update DB
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
            print(f"  → Updated {update_count}/{len(batch)}. Total success: {success_count}. Fails: {fail_count}")
            
        await browser.close()
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    asyncio.run(main_async())
