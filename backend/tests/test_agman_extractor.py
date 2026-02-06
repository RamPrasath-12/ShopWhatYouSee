"""
Test script for AG-MAN Extractor v3
Tests attribute extraction across all categories
"""
import sys
import os
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agman_extractor import process_crop_base64

def test_with_image(image_path, category):
    """Test attribute extraction with a local image file"""
    print(f"\n{'='*50}")
    print(f"Testing: {category.upper()}")
    print(f"Image: {image_path}")
    print('='*50)
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    b64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode()
    
    # Process
    try:
        result = process_crop_base64(b64, category)
        attrs = result['attributes']
        
        print(f"  Color (Primary):   {attrs['color_hex']}")
        print(f"  Color (Secondary): {attrs['secondary_color_hex']}")
        print(f"  Pattern:           {attrs['pattern']}")
        print(f"  Sleeve:            {attrs['sleeve']}")
        print(f"  Embedding length:  {len(result['embedding'])}")
        
        return result
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Test image path (use existing test.png in backend folder)
    test_image = "test.png"
    
    if not os.path.exists(test_image):
        print(f"ERROR: Test image not found at {test_image}")
        print("Please provide a test image for validation.")
        return
    
    # Test various categories
    categories_to_test = [
        # Upper wear (should have sleeve)
        "shirt",
        "tshirt", 
        "blazer",
        "jacket",
        
        # Lower wear (should have pattern, no sleeve)
        "pant",
        "shorts",
        "skirt",
        
        # Full body
        "saree",
        
        # Accessories (color only)
        "watch",
        "bag",
        
        # Footwear
        "footwear_shoes"
    ]
    
    print("\n" + "="*60)
    print("AG-MAN EXTRACTOR v3 - ATTRIBUTE EXTRACTION TEST")
    print("="*60)
    
    results = {}
    for cat in categories_to_test:
        result = test_with_image(test_image, cat)
        results[cat] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for cat, res in results.items():
        if res:
            attrs = res['attributes']
            status = "✓"
            details = f"color={attrs['color_hex']}"
            if attrs['sleeve']:
                details += f", sleeve={attrs['sleeve']}"
            if attrs['pattern']:
                details += f", pattern={attrs['pattern']}"
        else:
            status = "✗"
            details = "FAILED"
        print(f"  {status} {cat}: {details}")

if __name__ == "__main__":
    main()
