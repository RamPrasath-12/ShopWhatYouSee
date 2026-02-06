"""
Improved Hex to Color Name Conversion
With comprehensive color matching based on RGB distance
"""

# Comprehensive color name database with RGB center and tolerance
COLOR_DATABASE = {
    # Neutrals
    "Black": {"rgb": (20, 20, 20), "tolerance": 40},
    "White": {"rgb": (245, 245, 245), "tolerance": 20},
    "Grey": {"rgb": (128, 128, 128), "tolerance": 50},
    "Charcoal": {"rgb": (54, 54, 54), "tolerance": 25},
    "Silver": {"rgb": (192, 192, 192), "tolerance": 30},
    "Off White": {"rgb": (235, 230, 220), "tolerance": 25},
    "Cream": {"rgb": (255, 253, 208), "tolerance": 30},
    "Beige": {"rgb": (200, 180, 150), "tolerance": 40},
    "Tan": {"rgb": (210, 180, 140), "tolerance": 35},
    
    # Reds
    "Red": {"rgb": (220, 40, 40), "tolerance": 50},
    "Maroon": {"rgb": (128, 0, 0), "tolerance": 40},
    "Burgundy": {"rgb": (128, 0, 32), "tolerance": 35},
    "Wine": {"rgb": (114, 47, 55), "tolerance": 35},
    "Coral": {"rgb": (255, 127, 80), "tolerance": 40},
    "Rust": {"rgb": (183, 65, 14), "tolerance": 40},
    
    # Pinks
    "Pink": {"rgb": (255, 192, 203), "tolerance": 40},
    "Hot Pink": {"rgb": (255, 105, 180), "tolerance": 40},
    "Rose": {"rgb": (255, 0, 127), "tolerance": 45},
    "Magenta": {"rgb": (255, 0, 255), "tolerance": 45},
    "Peach": {"rgb": (255, 218, 185), "tolerance": 35},
    
    # Oranges
    "Orange": {"rgb": (255, 140, 0), "tolerance": 45},
    "Mustard": {"rgb": (225, 173, 1), "tolerance": 40},
    "Gold": {"rgb": (255, 215, 0), "tolerance": 40},
    
    # Yellows
    "Yellow": {"rgb": (255, 255, 0), "tolerance": 50},
    "Lime Yellow": {"rgb": (220, 255, 0), "tolerance": 40},
    
    # Greens
    "Green": {"rgb": (0, 128, 0), "tolerance": 50},
    "Olive": {"rgb": (128, 128, 0), "tolerance": 40},
    "Khaki": {"rgb": (189, 183, 107), "tolerance": 40},
    "Lime Green": {"rgb": (50, 205, 50), "tolerance": 45},
    "Teal": {"rgb": (0, 128, 128), "tolerance": 45},
    "Sea Green": {"rgb": (46, 139, 87), "tolerance": 40},
    
    # Blues
    "Blue": {"rgb": (0, 0, 255), "tolerance": 60},
    "Navy Blue": {"rgb": (0, 0, 128), "tolerance": 45},
    "Sky Blue": {"rgb": (135, 206, 235), "tolerance": 40},
    "Turquoise": {"rgb": (64, 224, 208), "tolerance": 45},
    "Teal Blue": {"rgb": (54, 117, 136), "tolerance": 40},
    "Steel Blue": {"rgb": (70, 130, 180), "tolerance": 40},
    
    # Purples
    "Purple": {"rgb": (128, 0, 128), "tolerance": 50},
    "Lavender": {"rgb": (230, 190, 230), "tolerance": 40},
    "Violet": {"rgb": (148, 0, 211), "tolerance": 45},
    "Mauve": {"rgb": (224, 176, 255), "tolerance": 40},
    
    # Browns
    "Brown": {"rgb": (139, 69, 19), "tolerance": 50},
    "Coffee Brown": {"rgb": (111, 78, 55), "tolerance": 40},
    "Chocolate": {"rgb": (123, 63, 0), "tolerance": 40},
    "Taupe": {"rgb": (72, 60, 50), "tolerance": 35},
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (128, 128, 128)  # Default grey

def color_distance(rgb1, rgb2):
    """Calculate weighted color distance (humans are more sensitive to green)"""
    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]
    # Weighted Euclidean distance
    return ((r_diff * 0.30)**2 + (g_diff * 0.59)**2 + (b_diff * 0.11)**2) ** 0.5

def hex_to_color_name(hex_color):
    """
    Convert hex color to the closest color name.
    Returns a human-readable color name for database matching.
    """
    if not hex_color or len(hex_color) < 4:
        return "Multi"
    
    rgb = hex_to_rgb(hex_color)
    
    best_match = "Multi"
    best_distance = float('inf')
    
    for name, data in COLOR_DATABASE.items():
        dist = color_distance(rgb, data["rgb"])
        if dist < data["tolerance"] and dist < best_distance:
            best_distance = dist
            best_match = name
    
    return best_match

# Quick mappings for common categories
SIMPLE_COLOR_MAP = {
    "black": "Black",
    "white": "White", 
    "grey": "Grey",
    "gray": "Grey",
    "red": "Red",
    "blue": "Blue",
    "navy": "Navy Blue",
    "green": "Green",
    "yellow": "Yellow",
    "orange": "Orange",
    "pink": "Pink",
    "purple": "Purple",
    "brown": "Brown",
    "beige": "Beige",
    "cream": "Cream",
    "olive": "Olive",
    "maroon": "Maroon",
    "teal": "Teal",
    "coral": "Coral",
    "khaki": "Khaki",
    "gold": "Gold",
    "silver": "Silver",
    "tan": "Tan",
    "rust": "Rust",
    "burgundy": "Burgundy",
    "lavender": "Lavender",
    "turquoise": "Turquoise Blue",
    "mustard": "Mustard",
    "peach": "Peach",
    "charcoal": "Charcoal",
    "nude": "Beige",
    "multi": "Multi",
}

def normalize_color_name(color):
    """Normalize a color name to match DB values"""
    if not color:
        return None
    color = color.lower().strip()
    return SIMPLE_COLOR_MAP.get(color, color.title())
