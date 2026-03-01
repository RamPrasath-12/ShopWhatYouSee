
from utils.color_utils import hex_to_color_name

test_cases = [
    # Specific User Report Case
    ("#48574c", "Green"),   # Was mapping to Coffee Brown, should be Green (H=136, S=0.17, V=87)
    
    # Other Dark/Muted Greens
    ("#2f4f4f", "Teal"),    # Dark Slate Gray -> Teal/Green family
    ("#556b2f", "Olive"),   # Dark Olive Green
    
    # Validation of previous fixes
    ("#022136", "Navy Blue"), 
    ("#1a0000", "Maroon"),    
]

def run_tests():
    print(f"Testing {len(test_cases)} colors...")
    passed = 0
    for hex_val, expected in test_cases:
        result = hex_to_color_name(hex_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} {hex_val}: Expected {expected}, Got {result}")
        if result == expected:
            passed += 1
            
    print(f"\nPassed {passed}/{len(test_cases)} tests.")

if __name__ == "__main__":
    run_tests()
