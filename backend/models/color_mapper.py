"""
Color Mapper for ShopWhatYouSee

Converts hex/RGB colors to human-readable color names using LAB color space.
Uses 50+ color categories to avoid "all dark colors = black" problem.

Usage:
    from color_mapper import hex_to_color_name, rgb_to_color_name
    
    color_name = hex_to_color_name("#1e3a8a")  # → "navy"
    color_name = rgb_to_color_name(30, 58, 138)  # → "navy"
"""

import numpy as np
from typing import Tuple, Optional
import colorsys

# LAB color space matching for accurate color perception
# Each color has: (L, a, b) center and tolerance radius

COLOR_DEFINITIONS = {
    # ═══════════════════════════════════════════════════════════════
    # REDS (10 variants)
    # ═══════════════════════════════════════════════════════════════
    "red": {"lab": (53, 80, 67), "tol": 25, "hex_examples": ["#FF0000", "#E53935"]},
    "crimson": {"lab": (47, 70, 40), "tol": 20, "hex_examples": ["#DC143C"]},
    "maroon": {"lab": (25, 40, 25), "tol": 20, "hex_examples": ["#800000", "#6B1624"]},
    "burgundy": {"lab": (30, 45, 20), "tol": 18, "hex_examples": ["#722F37", "#800020"]},
    "coral": {"lab": (67, 45, 40), "tol": 20, "hex_examples": ["#FF7F50", "#FF6B6B"]},
    "salmon": {"lab": (70, 35, 28), "tol": 20, "hex_examples": ["#FA8072", "#E9967A"]},
    "rose": {"lab": (75, 30, 10), "tol": 18, "hex_examples": ["#FF007F", "#E75480"]},
    "wine": {"lab": (28, 38, 18), "tol": 18, "hex_examples": ["#722F37", "#4E1609"]},
    "rust": {"lab": (45, 40, 50), "tol": 20, "hex_examples": ["#B7410E", "#A45A2A"]},
    "cherry": {"lab": (40, 60, 35), "tol": 18, "hex_examples": ["#DE3163", "#9B111E"]},
    
    # ═══════════════════════════════════════════════════════════════
    # PINKS (6 variants)
    # ═══════════════════════════════════════════════════════════════
    "pink": {"lab": (80, 28, 5), "tol": 22, "hex_examples": ["#FFC0CB", "#FFB6C1"]},
    "hot pink": {"lab": (60, 75, -5), "tol": 22, "hex_examples": ["#FF69B4", "#FF1493"]},
    "blush": {"lab": (85, 18, 10), "tol": 18, "hex_examples": ["#DE5D83", "#E8B4B8"]},
    "magenta": {"lab": (55, 85, -25), "tol": 22, "hex_examples": ["#FF00FF", "#CA1F7B"]},
    "fuchsia": {"lab": (50, 80, -20), "tol": 22, "hex_examples": ["#FF00FF", "#C154C1"]},
    "dusty pink": {"lab": (70, 20, 8), "tol": 18, "hex_examples": ["#D4A5A5", "#DCAE96"]},
    
    # ═══════════════════════════════════════════════════════════════
    # ORANGES (5 variants)
    # ═══════════════════════════════════════════════════════════════
    "orange": {"lab": (70, 45, 70), "tol": 25, "hex_examples": ["#FFA500", "#FF8C00"]},
    "peach": {"lab": (82, 18, 30), "tol": 20, "hex_examples": ["#FFDAB9", "#FFCBA4"]},
    "tangerine": {"lab": (68, 52, 68), "tol": 20, "hex_examples": ["#FF9966", "#ED7014"]},
    "burnt orange": {"lab": (52, 48, 58), "tol": 20, "hex_examples": ["#CC5500", "#BF5700"]},
    "apricot": {"lab": (80, 22, 38), "tol": 18, "hex_examples": ["#FBCEB1", "#FFCBA4"]},
    
    # ═══════════════════════════════════════════════════════════════
    # YELLOWS (6 variants)
    # ═══════════════════════════════════════════════════════════════
    "yellow": {"lab": (95, -10, 90), "tol": 25, "hex_examples": ["#FFFF00", "#FFD700"]},
    "gold": {"lab": (80, 8, 75), "tol": 22, "hex_examples": ["#FFD700", "#DAA520"]},
    "mustard": {"lab": (72, 8, 65), "tol": 22, "hex_examples": ["#FFDB58", "#E1AD01"]},
    "lemon": {"lab": (97, -18, 90), "tol": 20, "hex_examples": ["#FFFACD", "#FFF44F"]},
    "cream": {"lab": (95, 2, 18), "tol": 18, "hex_examples": ["#FFFDD0", "#F5F5DC"]},
    "amber": {"lab": (78, 15, 78), "tol": 20, "hex_examples": ["#FFBF00", "#FF7E00"]},
    
    # ═══════════════════════════════════════════════════════════════
    # GREENS (10 variants)
    # ═══════════════════════════════════════════════════════════════
    "green": {"lab": (50, -55, 45), "tol": 25, "hex_examples": ["#008000", "#228B22"]},
    "olive": {"lab": (55, -12, 45), "tol": 22, "hex_examples": ["#808000", "#6B8E23"]},
    "forest green": {"lab": (35, -35, 28), "tol": 20, "hex_examples": ["#228B22", "#0B6623"]},
    "mint": {"lab": (88, -32, 12), "tol": 22, "hex_examples": ["#98FF98", "#3EB489"]},
    "sage": {"lab": (68, -18, 22), "tol": 20, "hex_examples": ["#9DC183", "#77815C"]},
    "teal": {"lab": (52, -32, -12), "tol": 22, "hex_examples": ["#008080", "#367588"]},
    "emerald": {"lab": (55, -58, 28), "tol": 22, "hex_examples": ["#50C878", "#046307"]},
    "lime": {"lab": (88, -58, 82), "tol": 22, "hex_examples": ["#00FF00", "#32CD32"]},
    "khaki": {"lab": (78, -5, 35), "tol": 20, "hex_examples": ["#F0E68C", "#BDB76B"]},
    "sea green": {"lab": (60, -40, 15), "tol": 20, "hex_examples": ["#2E8B57", "#20B2AA"]},
    
    # ═══════════════════════════════════════════════════════════════
    # BLUES (12 variants) - CRITICAL for fashion
    # ═══════════════════════════════════════════════════════════════
    "blue": {"lab": (45, 15, -55), "tol": 25, "hex_examples": ["#0000FF", "#4169E1"]},
    "navy": {"lab": (18, 12, -38), "tol": 18, "hex_examples": ["#000080", "#1E3A8A"]},
    "royal blue": {"lab": (38, 28, -62), "tol": 22, "hex_examples": ["#4169E1", "#002366"]},
    "sky blue": {"lab": (82, -8, -22), "tol": 22, "hex_examples": ["#87CEEB", "#00BFFF"]},
    "baby blue": {"lab": (88, -5, -15), "tol": 20, "hex_examples": ["#89CFF0", "#A1CAF1"]},
    "cobalt": {"lab": (42, 22, -65), "tol": 20, "hex_examples": ["#0047AB", "#0050A0"]},
    "turquoise": {"lab": (72, -38, -18), "tol": 22, "hex_examples": ["#40E0D0", "#00CED1"]},
    "aqua": {"lab": (82, -42, -8), "tol": 22, "hex_examples": ["#00FFFF", "#7FFFD4"]},
    "indigo": {"lab": (28, 35, -55), "tol": 20, "hex_examples": ["#4B0082", "#3F00FF"]},
    "denim": {"lab": (48, 8, -28), "tol": 22, "hex_examples": ["#1560BD", "#6F8FAF"]},
    "powder blue": {"lab": (85, -8, -12), "tol": 18, "hex_examples": ["#B0E0E6", "#B6D0E2"]},
    "steel blue": {"lab": (55, -5, -25), "tol": 20, "hex_examples": ["#4682B4", "#5F9EA0"]},
    
    # ═══════════════════════════════════════════════════════════════
    # PURPLES (8 variants)
    # ═══════════════════════════════════════════════════════════════
    "purple": {"lab": (38, 55, -48), "tol": 25, "hex_examples": ["#800080", "#9932CC"]},
    "lavender": {"lab": (78, 18, -18), "tol": 20, "hex_examples": ["#E6E6FA", "#B57EDC"]},
    "violet": {"lab": (48, 60, -52), "tol": 22, "hex_examples": ["#8B00FF", "#EE82EE"]},
    "plum": {"lab": (38, 42, -28), "tol": 20, "hex_examples": ["#DDA0DD", "#8E4585"]},
    "mauve": {"lab": (62, 28, -12), "tol": 20, "hex_examples": ["#E0B0FF", "#876289"]},
    "lilac": {"lab": (80, 20, -15), "tol": 18, "hex_examples": ["#C8A2C8", "#B666D2"]},
    "grape": {"lab": (32, 45, -40), "tol": 20, "hex_examples": ["#6F2DA8", "#4A0E4E"]},
    "orchid": {"lab": (65, 45, -30), "tol": 20, "hex_examples": ["#DA70D6", "#AF69EE"]},
    
    # ═══════════════════════════════════════════════════════════════
    # BROWNS (8 variants) - Important for leather, bags, shoes
    # ═══════════════════════════════════════════════════════════════
    "brown": {"lab": (42, 22, 35), "tol": 25, "hex_examples": ["#A52A2A", "#8B4513"]},
    "tan": {"lab": (72, 15, 32), "tol": 22, "hex_examples": ["#D2B48C", "#C19A6B"]},
    "beige": {"lab": (88, 5, 18), "tol": 20, "hex_examples": ["#F5F5DC", "#E8D4A2"]},
    "chocolate": {"lab": (32, 28, 28), "tol": 20, "hex_examples": ["#7B3F00", "#3D1C02"]},
    "camel": {"lab": (68, 18, 40), "tol": 22, "hex_examples": ["#C19A6B", "#A67B5B"]},
    "coffee": {"lab": (38, 20, 25), "tol": 20, "hex_examples": ["#6F4E37", "#4B3621"]},
    "taupe": {"lab": (55, 5, 12), "tol": 20, "hex_examples": ["#483C32", "#8B8589"]},
    "chestnut": {"lab": (40, 35, 35), "tol": 20, "hex_examples": ["#954535", "#CD5C5C"]},
    
    # ═══════════════════════════════════════════════════════════════
    # NEUTRALS (8 variants) - CRITICAL: Prevent dark=black problem
    # ═══════════════════════════════════════════════════════════════
    "black": {"lab": (8, 0, 0), "tol": 12, "hex_examples": ["#000000", "#0A0A0A"]},
    "charcoal": {"lab": (28, 0, 0), "tol": 15, "hex_examples": ["#36454F", "#2C3539"]},
    "dark grey": {"lab": (40, 0, 0), "tol": 15, "hex_examples": ["#555555", "#696969"]},
    "grey": {"lab": (55, 0, 0), "tol": 18, "hex_examples": ["#808080", "#A9A9A9"]},
    "light grey": {"lab": (75, 0, 0), "tol": 15, "hex_examples": ["#C0C0C0", "#D3D3D3"]},
    "silver": {"lab": (78, 0, -3), "tol": 15, "hex_examples": ["#C0C0C0", "#AAA9AD"]},
    "white": {"lab": (98, 0, 0), "tol": 10, "hex_examples": ["#FFFFFF", "#FAFAFA"]},
    "off-white": {"lab": (94, 2, 8), "tol": 12, "hex_examples": ["#FAF9F6", "#F5F5DC"]},
    "ivory": {"lab": (96, 0, 10), "tol": 12, "hex_examples": ["#FFFFF0", "#FAEBD7"]},
    
    # ═══════════════════════════════════════════════════════════════
    # METALLICS (3 variants)
    # ═══════════════════════════════════════════════════════════════
    "gold metallic": {"lab": (78, 10, 68), "tol": 20, "hex_examples": ["#D4AF37", "#CFB53B"]},
    "silver metallic": {"lab": (80, 0, -5), "tol": 18, "hex_examples": ["#C0C0C0", "#AAA9AD"]},
    "bronze": {"lab": (58, 20, 45), "tol": 20, "hex_examples": ["#CD7F32", "#B08D57"]},
    
    # ═══════════════════════════════════════════════════════════════
    # MULTI-COLORS (special cases)
    # ═══════════════════════════════════════════════════════════════
    "multi": {"lab": None, "tol": 0, "hex_examples": []},  # Handled separately
}


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space for perceptual matching."""
    # Normalize RGB to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Convert to linear RGB
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = linearize(r), linearize(g), linearize(b)
    
    # Convert to XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize for D65 white point
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883
    
    # Convert to LAB
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    
    L = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))
    
    return (L, a, b_val)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def color_distance_lab(lab1: Tuple[float, float, float], 
                        lab2: Tuple[float, float, float]) -> float:
    """Calculate Delta E (CIE76) distance between two LAB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """
    Convert RGB color to human-readable color name.
    Uses LAB color space for perceptual accuracy.
    
    Returns the closest color name from 50+ categories.
    """
    input_lab = rgb_to_lab(r, g, b)
    
    best_match = "unknown"
    best_distance = float('inf')
    
    for color_name, color_def in COLOR_DEFINITIONS.items():
        if color_def["lab"] is None:
            continue
            
        center_lab = color_def["lab"]
        tolerance = color_def["tol"]
        
        distance = color_distance_lab(input_lab, center_lab)
        
        # Must be within tolerance to be considered
        if distance <= tolerance and distance < best_distance:
            best_distance = distance
            best_match = color_name
    
    # If no match within tolerance, find closest overall
    if best_match == "unknown":
        for color_name, color_def in COLOR_DEFINITIONS.items():
            if color_def["lab"] is None:
                continue
            distance = color_distance_lab(input_lab, color_def["lab"])
            if distance < best_distance:
                best_distance = distance
                best_match = color_name
    
    return best_match


def hex_to_color_name(hex_color: str) -> str:
    """
    Convert hex color to human-readable color name.
    
    Examples:
        hex_to_color_name("#1e3a8a") → "navy"
        hex_to_color_name("#FF0000") → "red"
        hex_to_color_name("#2C3539") → "charcoal"  # Not black!
    """
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_color_name(r, g, b)


def get_color_family(color_name: str) -> str:
    """Get the color family (e.g., 'navy' → 'blue')."""
    families = {
        "red": ["red", "crimson", "maroon", "burgundy", "coral", "salmon", "rose", "wine", "rust", "cherry"],
        "pink": ["pink", "hot pink", "blush", "magenta", "fuchsia", "dusty pink"],
        "orange": ["orange", "peach", "tangerine", "burnt orange", "apricot"],
        "yellow": ["yellow", "gold", "mustard", "lemon", "cream", "amber"],
        "green": ["green", "olive", "forest green", "mint", "sage", "teal", "emerald", "lime", "khaki", "sea green"],
        "blue": ["blue", "navy", "royal blue", "sky blue", "baby blue", "cobalt", "turquoise", "aqua", "indigo", "denim", "powder blue", "steel blue"],
        "purple": ["purple", "lavender", "violet", "plum", "mauve", "lilac", "grape", "orchid"],
        "brown": ["brown", "tan", "beige", "chocolate", "camel", "coffee", "taupe", "chestnut"],
        "neutral": ["black", "charcoal", "dark grey", "grey", "light grey", "silver", "white", "off-white", "ivory"],
        "metallic": ["gold metallic", "silver metallic", "bronze"],
    }
    
    for family, members in families.items():
        if color_name.lower() in members:
            return family
    return "unknown"


def get_lighter_shade(color_name: str) -> Optional[str]:
    """Get a lighter shade of the given color."""
    shade_map = {
        "navy": "blue",
        "blue": "sky blue",
        "sky blue": "baby blue",
        "maroon": "burgundy",
        "burgundy": "red",
        "forest green": "green",
        "green": "mint",
        "chocolate": "brown",
        "brown": "tan",
        "tan": "beige",
        "black": "charcoal",
        "charcoal": "dark grey",
        "dark grey": "grey",
        "grey": "light grey",
        "purple": "lavender",
        "indigo": "purple",
    }
    return shade_map.get(color_name.lower())


def get_darker_shade(color_name: str) -> Optional[str]:
    """Get a darker shade of the given color."""
    shade_map = {
        "baby blue": "sky blue",
        "sky blue": "blue",
        "blue": "navy",
        "red": "burgundy",
        "burgundy": "maroon",
        "mint": "green",
        "green": "forest green",
        "beige": "tan",
        "tan": "brown",
        "brown": "chocolate",
        "light grey": "grey",
        "grey": "dark grey",
        "dark grey": "charcoal",
        "charcoal": "black",
        "lavender": "purple",
        "purple": "indigo",
    }
    return shade_map.get(color_name.lower())


# Export all color names for validation
ALL_COLOR_NAMES = list(COLOR_DEFINITIONS.keys())


if __name__ == "__main__":
    # Test the color mapper
    test_colors = [
        ("#FF0000", "red"),
        ("#000000", "black"),
        ("#1e3a8a", "navy"),
        ("#2C3539", "charcoal"),  # Should NOT be black!
        ("#36454F", "charcoal"),
        ("#555555", "dark grey"),
        ("#808080", "grey"),
        ("#FFFFFF", "white"),
        ("#FFC0CB", "pink"),
        ("#800080", "purple"),
        ("#008000", "green"),
        ("#FFA500", "orange"),
        ("#FFD700", "gold"),
        ("#8B4513", "brown"),
        ("#D2B48C", "tan"),
    ]
    
    print("Testing Color Mapper (50+ colors)")
    print("=" * 50)
    
    correct = 0
    for hex_val, expected in test_colors:
        result = hex_to_color_name(hex_val)
        status = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"{status} {hex_val} → {result} (expected: {expected})")
    
    print(f"\nAccuracy: {correct}/{len(test_colors)} ({100*correct/len(test_colors):.0f}%)")
    print(f"Total colors defined: {len(COLOR_DEFINITIONS)}")
