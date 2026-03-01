import requests
import json
import traceback

# Configuration
BASE_URL = "http://localhost:5000"

def test_llm_logic():
    print("🧪 Testing LLM Logic for Filters & Price...")
    
    # Mock data simulating a request from frontend
    payload = {
        "item": {
            "category": "t_shirt",
            "color_hex": "#000000",
            "pattern": "solid",
            "sleeve_length": "short"
        },
        "scene": "outdoor",
        "user_query": "I want a casual t-shirt under 1500 rupees",
        "session_history": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/llm", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ API Response Received:")
            print(json.dumps(data, indent=2))
            
            filters = data.get("filters", {})
            price = filters.get("price_max")
            
            # Verification Checks
            checks = []
            
            # 1. Price Check
            if price == 1500:
                checks.append("✅ Price extracted correctly (1500)")
            else:
                checks.append(f"❌ Price extraction failed. Expected 1500, got {price}")
                
            # 2. Category Check
            if filters.get("category") == "t_shirt":
                checks.append("✅ Category preserved")
            else:
                checks.append("❌ Category mismatch")
                
            # 3. Source Check
            print(f"ℹ️  Source: {data.get('llm_source')}")
            
            print("\n" + "\n".join(checks))
            
        else:
            print(f"❌ API Failed with status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_logic()
