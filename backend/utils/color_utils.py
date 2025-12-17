def hex_to_color_name(hex_color):
    hex_color = hex_color.lower()

    mapping = {
        "black": ["#000", "#111", "#1a", "#2b"],
        "white": ["#fff", "#eee", "#f3"],
        "red": ["#a0", "#b0", "#c0", "#d0", "#e0"],
        "blue": ["#00", "#01", "#02", "#03"],
        "green": ["#0a", "#0b", "#0c"],
        "brown": ["#38", "#4a", "#5a"],
        "grey": ["#7a", "#8a", "#9a"],
        "pink": ["#d8", "#e3"],
        "purple": ["#38", "#48", "#58"],
    }

    for name, prefixes in mapping.items():
        for p in prefixes:
            if hex_color.startswith(p):
                return name

    return None
