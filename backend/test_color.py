"""
Quick test for color filter with Groq
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_color():
    print("🧪 Testing RED color filter...")
    
    payload = {
        "item": {
            "category": "t_shirt",
            "color_hex": "#000000",
            "pattern": "solid",
            "sleeve_length": "short"
        },
        "scene": "outdoor",
        "user_query": "I want a red t-shirt",  # Testing RED
        "session_history": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/llm", json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ API Response:")
            print(json.dumps(data, indent=2))
            
            filters = data.get("filters", {})
            if "color" in filters:
                print(f"\n✅ Color filter found: {filters['color']}")
                if "red" in str(filters["color"]).lower():
                    print("✅ RED color extracted correctly!")
                else:
                    print(f"❌ Expected 'red' but got: {filters['color']}")
            else:
                print("\n❌ No color filter in response")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_color()
