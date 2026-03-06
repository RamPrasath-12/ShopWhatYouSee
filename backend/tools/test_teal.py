import sys
import os
import json
import requests

BASE_URL = "http://localhost:5000"

def test_teal_dress():
    print("Testing query: 'teal color dress'")
    try:
        req = requests.post(f"{BASE_URL}/llm", json={
            "user_query": "teal color dress",
            "yolo_category": "", # text search
            "is_image_search": False
        })
        llm_res = req.json()
        print("LLM extracted filters:", json.dumps(llm_res, indent=2))
        
        # Now search
        filters = llm_res.get("add", {})
        search_req = requests.post(f"{BASE_URL}/search", json={
            "filters": filters
        })
        search_res = search_req.json()
        
        print(f"\nSearch returned {len(search_res.get('products', []))} products.")
        for p in search_res.get('products', [])[:5]:
            print(f" - {p.get('name')} | Cat: {p.get('category')} | Color: {p.get('color')} | Sim: {p.get('similarity_score')}")
            
        print("\nMetadata:", json.dumps(search_res.get('metadata', {}), indent=2))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_teal_dress()
