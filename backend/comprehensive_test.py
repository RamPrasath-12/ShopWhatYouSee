"""
COMPREHENSIVE SYSTEM DIAGNOSTIC
Tests entire workflow: Detection → LLM → Search → Results → Insights
"""
import requests
import json
import sys

BASE_URL = "http://localhost:5000"
TIMEOUT = 10

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_1_backend_alive():
    print_header("TEST 1: Backend Alive")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print(f"✅ Backend responding: HTTP {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        return False

def test_2_llm_reasoning():
    print_header("TEST 2: LLM Reasoning (Groq)")
    
    payload = {
        "item": {
            "category": "t_shirt",
            "color_hex": "#FF0000",
            "pattern": "solid",
            "sleeve_length": "short"
        },
        "scene": "outdoor",
        "user_query": "I want a red t-shirt under 1500",
        "session_history": []
    }
    
    try:
        print("📤 Sending request...")
        response = requests.post(f"{BASE_URL}/llm", json=payload, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ LLM Response received")
            print(f"   Source: {data.get('llm_source', 'unknown')}")
            print(f"   Confidence: {data.get('confidence', 0)}")
            
            filters = data.get('filters', {})
            print(f"   Filters: {json.dumps(filters, indent=6)}")
            
            # Check if red color was extracted
            color = filters.get('color') or filters.get('color_name')
            if color and 'red' in str(color).lower():
                print(f"   ✅ Color 'red' extracted correctly")
            else:
                print(f"   ⚠️ Color might be wrong: {color}")
            
            # Check if price_max was extracted
            if 'price_max' in filters or 'max_price' in filters:
                print(f"   ✅ Price filter extracted")
            
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def test_3_product_search():
    print_header("TEST 3: Product Search with FAISS")
    
    payload = {
        "category": "t_shirt",
        "filters": {
            "color": "red",
            "price_max": 1500
        }
    }
    
    try:
        print("📤 Searching products...")
        response = requests.post(f"{BASE_URL}/search", json=payload, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            
            print(f"✅ Search successful")
            print(f"   Found: {len(products)} products")
            
            if products:
                first = products[0]
                print(f"   First product:")
                print(f"      ID: {first.get('id')}")
                print(f"      Name: {first.get('name', 'N/A')}")
                print(f"      Price: ₹{first.get('price', 'N/A')}")
                print(f"      Similarity Score: {first.get('similarity_score', 'N/A')}")
                
                if 'similarity_score' in first:
                    print(f"   ✅ FAISS similarity scores present")
                else:
                    print(f"   ⚠️ No similarity scores")
            else:
                print(f"   ⚠️ No products found (database might be empty)")
            
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_4_rating_system():
    print_header("TEST 4: Rating System")
    
    rating_payload = {
        "product_id": "test_123",
        "rating": 5,
        "query": "red t-shirt under 1500",
        "filters": {"color": "red", "price_max": 1500},
        "llm_source": "groq",
        "session_id": "test_session",
        "iteration_count": 1
    }
    
    try:
        print("📤 Submitting rating...")
        response = requests.post(f"{BASE_URL}/rating", json=rating_payload, timeout=TIMEOUT)
        
        if response.status_code == 200:
            print(f"✅ Rating saved successfully")
            return True
        else:
            print(f"❌ Rating failed: HTTP {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Rating test failed: {e}")
        return False

def test_5_insights():
    print_header("TEST 5: Insights Generation")
    
    try:
        print("📤 Fetching insights...")
        response = requests.get(f"{BASE_URL}/insights", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Insights generated")
            print(f"   Total Ratings: {data.get('total_ratings', 0)}")
            print(f"   Avg Rating: {data.get('average_rating', 0)}")
            
            llm_perf = data.get('insights', {}).get('llm_performance', {})
            if llm_perf:
                print(f"   LLM Performance:")
                for llm, stats in llm_perf.items():
                    print(f"      {llm}: {stats.get('avg_rating', 0):.2f} ({stats.get('count', 0)} ratings)")
            
            return True
        else:
            print(f"❌ Insights failed: HTTP {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Insights test failed: {e}")
        return False

def main():
    print("🔬 COMPREHENSIVE SYSTEM DIAGNOSTIC")
    print("Testing: Backend → LLM → Search → Ratings → Insights")
    print("="*60)
    
    results = {
        "Backend Alive": test_1_backend_alive(),
        "LLM Reasoning": test_2_llm_reasoning(),
        "Product Search": test_3_product_search(),
        "Rating System": test_4_rating_system(),
        "Insights": test_5_insights()
    }
    
    print_header("FINAL RESULTS")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"  SCORE: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print(f"  🎉 ALL SYSTEMS OPERATIONAL!")
    else:
        print(f"  ⚠️  Some systems need attention")
    
    print(f"{'='*60}\n")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
