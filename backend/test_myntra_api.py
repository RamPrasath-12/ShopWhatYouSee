import sys
import requests
import json
import re

url = "https://www.myntra.com/bags/caprese/caprese-olive-green-textured-baguette-shoulder-bag/29324080/buy"

# Extract product ID
match = re.search(r'/([0-9]+)/buy', url)
if match:
    pid = match.group(1)
    
    api_url = f"https://www.myntra.com/gateway/v2/product/{pid}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'x-myntra-app-version': '541.1',
        'x-myntra-channel': 'web',
        'x-myntra-app-name': 'myntra-web'
    }
    
    print(f"Fetching {api_url}...")
    try:
        r = requests.get(api_url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            article_attrs = data.get('style', {}).get('articleAttributes', {})
            desc = data.get('style', {}).get('productDescriptors', {}).get('description', {}).get('value', '')
            
            material = article_attrs.get('Material', None)
            if not material:
                m_care = data.get('style', {}).get('productDescriptors', {}).get('materials_care_desc', {}).get('value', '')
                import re as regex
                # try to strip html
                m_clean = regex.sub('<[^<]+?>', '', m_care)
                material = m_clean
            
            print(f"Material: {material}")
            print(f"Attributes: {json.dumps(article_attrs)}")
            print(f"Description: {desc[:100]}...")
        else:
            print(f"Failed with {r.status_code}")
    except Exception as e:
        print(e)
else:
    print("Invalid URL format")
