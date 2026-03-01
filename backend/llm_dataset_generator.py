"""
LLM Training Dataset Generator for ShopWhatYouSee

Generates 3,500 training samples for Phi-3 Mini fine-tuning.
Each sample includes:
- Input context (vision, scene, session history, query)
- Output with REASONING + filters + confidence

Usage:
    python llm_dataset_generator.py
    
Output: llm_training_data.jsonl
"""

import json
import random
from typing import Dict, List, Optional, Any
from itertools import product

# ═══════════════════════════════════════════════════════════════════════════
# ATTRIBUTE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "shirt", "tshirt", "blouse", "blazer", "jacket", "shawl",
    "pant", "shorts", "skirt", "leggings",
    "churidhar", "dhoti", "saree",
    "bag", "purse", "belt", "cap", "earring", "glass", "hairclip",
    "necklace", "ring", "tie", "watch", "bangle", "bracelet",
    "footwear"
]

COLORS = [
    "red", "maroon", "burgundy", "coral", "pink", "hot pink",
    "orange", "peach", "yellow", "gold", "mustard", "cream",
    "green", "olive", "mint", "teal", "emerald",
    "blue", "navy", "royal blue", "sky blue", "turquoise",
    "purple", "lavender", "violet", "plum",
    "brown", "tan", "beige", "chocolate", "camel",
    "black", "charcoal", "grey", "white", "off-white"
]

PATTERNS = ["solid", "striped", "checked", "floral", "printed", "geometric"]

SLEEVES = ["sleeveless", "short", "three-quarter", "full"]

FORMALITY = ["casual", "semi-formal", "formal"]

OCCASIONS = ["wedding", "party", "office", "casual outing", "beach", "gym", "date night"]

SCENES = ["wedding_hall", "office", "beach", "gym", "restaurant", "outdoor", "home", "mall"]

FITS = ["slim", "regular", "loose", "oversized"]

PRICE_TIERS = ["budget", "mid-range", "premium"]

# Attribute applicability rules
ATTRIBUTE_RULES = {
    # Upper wear - has sleeve, pattern, fit
    "shirt": {"sleeve": True, "pattern": True, "fit": True},
    "tshirt": {"sleeve": True, "pattern": True, "fit": True},
    "blouse": {"sleeve": True, "pattern": True, "fit": True},
    "blazer": {"sleeve": True, "pattern": True, "fit": True},
    "jacket": {"sleeve": True, "pattern": True, "fit": True},
    "shawl": {"sleeve": False, "pattern": True, "fit": False},
    
    # Lower wear - no sleeve
    "pant": {"sleeve": False, "pattern": True, "fit": True},
    "shorts": {"sleeve": False, "pattern": True, "fit": True},
    "skirt": {"sleeve": False, "pattern": True, "fit": True},
    "leggings": {"sleeve": False, "pattern": False, "fit": True},
    
    # Traditional - special rules
    "churidhar": {"sleeve": True, "pattern": True, "fit": True},
    "dhoti": {"sleeve": False, "pattern": True, "fit": False},
    "saree": {"sleeve": False, "pattern": True, "fit": False},
    
    # Accessories - no sleeve, pattern, fit
    "bag": {"sleeve": False, "pattern": True, "fit": False},
    "purse": {"sleeve": False, "pattern": True, "fit": False},
    "belt": {"sleeve": False, "pattern": False, "fit": False},
    "cap": {"sleeve": False, "pattern": True, "fit": False},
    "earring": {"sleeve": False, "pattern": False, "fit": False},
    "glass": {"sleeve": False, "pattern": False, "fit": False},
    "hairclip": {"sleeve": False, "pattern": False, "fit": False},
    "necklace": {"sleeve": False, "pattern": False, "fit": False},
    "ring": {"sleeve": False, "pattern": False, "fit": False},
    "tie": {"sleeve": False, "pattern": True, "fit": False},
    "watch": {"sleeve": False, "pattern": False, "fit": False},
    "bangle": {"sleeve": False, "pattern": False, "fit": False},
    "bracelet": {"sleeve": False, "pattern": False, "fit": False},
    "footwear": {"sleeve": False, "pattern": False, "fit": False},
}


# ═══════════════════════════════════════════════════════════════════════════
# QUERY TEMPLATES FOR EACH TYPE
# ═══════════════════════════════════════════════════════════════════════════

# TYPE 1: Exact Preservation
PRESERVE_QUERIES = [
    "show similar ones",
    "find similar items",
    "more like this",
    "same type please",
    "similar products",
    "find matching ones",
    "show me the same",
    "I want this exact style",
    "find identical ones",
    "more of the same"
]

# TYPE 2: Color Change
COLOR_CHANGE_QUERIES = [
    "same but in {color}",
    "I want this in {color}",
    "show me {color} version",
    "{color} color please",
    "make it {color}",
    "change to {color}",
    "same style but {color}",
    "can I get this in {color}?",
    "{color} one instead",
    "find {color} alternative"
]

# TYPE 3: Color Refinement
LIGHTER_QUERIES = [
    "make it lighter",
    "lighter shade please",
    "something lighter",
    "not so dark",
    "brighter version",
    "pastel version"
]

DARKER_QUERIES = [
    "make it darker",
    "darker shade please",
    "something darker",
    "deeper color",
    "richer shade"
]

# TYPE 4: Category Change
CATEGORY_CHANGE_QUERIES = [
    "same but as a {category}",
    "I want a {category} instead",
    "convert to {category}",
    "make it a {category}",
    "{category} version",
    "same color but {category}"
]

# TYPE 5: Attribute Addition
ATTRIBUTE_ADD_QUERIES = [
    "{attribute_value} please",
    "make it {attribute_value}",
    "I want {attribute_value}",
    "add {attribute_value}",
    "should be {attribute_value}"
]

# TYPE 6: Attribute Removal
ATTRIBUTE_REMOVE_QUERIES = [
    "{attribute} doesn't matter",
    "any {attribute} is fine",
    "ignore {attribute}",
    "no preference for {attribute}",
    "don't care about {attribute}"
]

# TYPE 7: Negation
NEGATION_QUERIES = [
    "not {value}",
    "no {value}",
    "anything but {value}",
    "avoid {value}",
    "nothing {value}"
]

# TYPE 8: Scene-Driven (no user text)
SCENE_INFERENCE = {
    "wedding_hall": {"formality": "formal", "occasion": "wedding"},
    "office": {"formality": "formal", "occasion": "office"},
    "beach": {"formality": "casual", "occasion": "beach"},
    "gym": {"formality": "casual", "occasion": "gym"},
    "restaurant": {"formality": "semi-formal", "occasion": "date night"},
    "party": {"formality": "semi-formal", "occasion": "party"},
}

# TYPE 9: Multi-turn Queries
MULTI_TURN_FOLLOWUPS = [
    "in {color}",
    "{sleeve} sleeve",
    "more {formality}",
    "for {occasion}",
    "{fit} fit",
    "actually {pattern}",
    "change to {category}"
]

# TYPE 10: Conflict Resolution
CONFLICT_QUERIES = [
    ("wedding_hall", "casual please"),
    ("office", "something casual"),
    ("beach", "formal attire"),
    ("gym", "elegant version"),
]

# TYPE 11: Price Tier
PRICE_QUERIES = [
    "budget option",
    "affordable version",
    "premium quality",
    "luxury version",
    "mid-range please",
    "under 500",
    "expensive one"
]


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def generate_random_attributes(category: str) -> Dict:
    """Generate random attributes based on category rules."""
    rules = ATTRIBUTE_RULES.get(category, {})
    
    attrs = {
        "color": random.choice(COLORS),
        "pattern": random.choice(PATTERNS) if rules.get("pattern", False) else None,
        "sleeve": random.choice(SLEEVES) if rules.get("sleeve", False) else None,
        "fit": random.choice(FITS) if rules.get("fit", False) else None,
    }
    
    return {k: v for k, v in attrs.items() if v is not None}


def generate_preserve_sample() -> Dict:
    """TYPE 1: Exact preservation - keep all attributes."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    query = random.choice(PRESERVE_QUERIES)
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": "User wants similar items without changes",
                "changes": [],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Clear preservation request"
            },
            "filters": {
                "category": category,
                **attrs
            },
            "confidence": round(random.uniform(0.85, 0.95), 2)
        }
    }


def generate_color_change_sample() -> Dict:
    """TYPE 2: Color change - override color only."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    original_color = attrs["color"]
    new_color = random.choice([c for c in COLORS if c != original_color])
    scene = random.choice(SCENES)
    query = random.choice(COLOR_CHANGE_QUERIES).format(color=new_color)
    
    new_attrs = attrs.copy()
    new_attrs["color"] = new_color
    
    preserved = [k for k in attrs.keys() if k != "color"]
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants to change color from {original_color} to {new_color}",
                "changes": [f"color: {original_color} → {new_color}"],
                "preserved": preserved + ["category"],
                "confidence_factors": "Explicit color request in query"
            },
            "filters": {
                "category": category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.88, 0.95), 2)
        }
    }


def generate_color_refinement_sample() -> Dict:
    """TYPE 3: Lighter/darker shade."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    is_lighter = random.choice([True, False])
    query = random.choice(LIGHTER_QUERIES if is_lighter else DARKER_QUERIES)
    
    # Map to lighter/darker
    shade_map_lighter = {"navy": "blue", "maroon": "red", "forest green": "green", "chocolate": "brown", "charcoal": "grey"}
    shade_map_darker = {"blue": "navy", "red": "maroon", "green": "forest green", "brown": "chocolate", "grey": "charcoal"}
    
    original_color = attrs["color"]
    if is_lighter:
        new_color = shade_map_lighter.get(original_color, original_color)
    else:
        new_color = shade_map_darker.get(original_color, original_color)
    
    new_attrs = attrs.copy()
    new_attrs["color"] = new_color
    
    direction = "lighter" if is_lighter else "darker"
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants a {direction} shade of the current color",
                "changes": [f"color: {original_color} → {new_color} ({direction})"],
                "preserved": ["category", "pattern", "sleeve"],
                "confidence_factors": f"Clear {direction} shade request"
            },
            "filters": {
                "category": category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.80, 0.90), 2)
        }
    }


def generate_category_change_sample() -> Dict:
    """TYPE 4: Category override."""
    old_category = random.choice(CATEGORIES)
    new_category = random.choice([c for c in CATEGORIES if c != old_category])
    
    old_attrs = generate_random_attributes(old_category)
    new_rules = ATTRIBUTE_RULES.get(new_category, {})
    
    # Filter attributes by new category's rules
    new_attrs = {"color": old_attrs["color"]}
    for attr in ["pattern", "sleeve", "fit"]:
        if new_rules.get(attr, False) and attr in old_attrs:
            new_attrs[attr] = old_attrs[attr]
    
    scene = random.choice(SCENES)
    query = random.choice(CATEGORY_CHANGE_QUERIES).format(category=new_category)
    
    return {
        "input": {
            "category": old_category,
            "attributes": old_attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants to change category from {old_category} to {new_category}",
                "changes": [f"category: {old_category} → {new_category}"],
                "preserved": ["color"] + [k for k in new_attrs.keys() if k != "color"],
                "confidence_factors": "Explicit category change request"
            },
            "filters": {
                "category": new_category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.85, 0.95), 2)
        }
    }


def generate_attribute_addition_sample() -> Dict:
    """TYPE 5: Adding new attribute."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("sleeve", False)])
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    # Pick attribute to add
    attr_type = random.choice(["formality", "occasion", "fit", "pattern"])
    
    if attr_type == "formality":
        value = random.choice(FORMALITY)
        query = f"{value} style please"
    elif attr_type == "occasion":
        value = random.choice(OCCASIONS)
        query = f"for {value}"
    elif attr_type == "fit":
        value = random.choice(FITS)
        query = f"{value} fit"
    else:
        value = random.choice(PATTERNS)
        query = f"{value} pattern"
    
    new_attrs = attrs.copy()
    new_attrs[attr_type] = value
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants to add {attr_type} constraint: {value}",
                "changes": [f"{attr_type}: null → {value}"],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": f"Explicit {attr_type} specification"
            },
            "filters": {
                "category": category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.85, 0.92), 2)
        }
    }


def generate_attribute_removal_sample() -> Dict:
    """TYPE 6: Attribute deletion."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("pattern", False)])
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    # Pick attribute to remove
    removable = [k for k in attrs.keys() if k != "color"]
    if not removable:
        removable = ["pattern"]
    
    attr_to_remove = random.choice(removable)
    query = f"{attr_to_remove} doesn't matter"
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants to remove {attr_to_remove} constraint",
                "changes": [f"{attr_to_remove}: {attrs.get(attr_to_remove)} → ignored"],
                "preserved": [k for k in attrs.keys() if k != attr_to_remove],
                "confidence_factors": "User explicitly wants to ignore attribute"
            },
            "filters": {
                "category": category,
                "color": attrs["color"],
                **{k: v for k, v in attrs.items() if k != attr_to_remove and k != "color"}
            },
            "ignored_attributes": [attr_to_remove],
            "confidence": round(random.uniform(0.88, 0.95), 2)
        }
    }


def generate_negation_sample() -> Dict:
    """TYPE 7: Negation queries."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    # Pick attribute to negate
    if random.choice([True, False]) and "pattern" in attrs:
        negated = attrs["pattern"]
        query = f"not {negated}"
        reasoning_text = f"User wants to exclude {negated} pattern"
    else:
        negated = attrs["color"]
        query = f"anything but {negated}"
        reasoning_text = f"User wants to exclude {negated} color"
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": reasoning_text,
                "changes": [f"excluded: {negated}"],
                "preserved": ["category"],
                "confidence_factors": "Clear negation in query"
            },
            "filters": {
                "category": category,
            },
            "excluded": [negated],
            "confidence": round(random.uniform(0.80, 0.90), 2)
        }
    }


def generate_scene_inference_sample() -> Dict:
    """TYPE 8: Scene-driven inference (no/minimal user query)."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("sleeve", False)])
    attrs = generate_random_attributes(category)
    scene = random.choice(list(SCENE_INFERENCE.keys()))
    scene_attrs = SCENE_INFERENCE[scene]
    
    query = random.choice(["", "something suitable", "what goes well here", "match the occasion"])
    
    new_attrs = attrs.copy()
    new_attrs.update(scene_attrs)
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": "User wants items suitable for the detected scene",
                "scene_influence": f"Scene {scene} suggests {scene_attrs}",
                "changes": [f"{k}: inferred from scene → {v}" for k, v in scene_attrs.items()],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Scene context used for inference"
            },
            "filters": {
                "category": category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.75, 0.85), 2)
        }
    }


def generate_multi_turn_sample() -> Dict:
    """TYPE 9: Multi-turn with session history."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("sleeve", False)])
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    # First turn
    turn1_query = random.choice(PRESERVE_QUERIES)
    turn1_filters = {"category": category, **attrs}
    
    # Second turn - modify something
    modification = random.choice(["color", "formality", "sleeve"])
    
    if modification == "color":
        new_color = random.choice([c for c in COLORS if c != attrs["color"]])
        turn2_query = f"in {new_color}"
        turn2_filters = turn1_filters.copy()
        turn2_filters["color"] = new_color
        change = f"color: {attrs['color']} → {new_color}"
    elif modification == "formality":
        formality = random.choice(FORMALITY)
        turn2_query = f"more {formality}"
        turn2_filters = turn1_filters.copy()
        turn2_filters["formality"] = formality
        change = f"formality: null → {formality}"
    else:
        new_sleeve = random.choice([s for s in SLEEVES if s != attrs.get("sleeve", "")])
        turn2_query = f"{new_sleeve} sleeve"
        turn2_filters = turn1_filters.copy()
        turn2_filters["sleeve"] = new_sleeve
        change = f"sleeve: {attrs.get('sleeve', 'null')} → {new_sleeve}"
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [
                {"turn": 1, "query": turn1_query, "filters": turn1_filters}
            ],
            "user_query": turn2_query
        },
        "output": {
            "reasoning": {
                "user_intent": "User is refining previous request",
                "session_context": "Building on turn 1 filters",
                "changes": [change],
                "preserved": [k for k in turn1_filters.keys() if k not in [modification, "category"]],
                "confidence_factors": "Multi-turn refinement"
            },
            "filters": turn2_filters,
            "confidence": round(random.uniform(0.85, 0.92), 2)
        }
    }


def generate_conflict_sample() -> Dict:
    """TYPE 10: Scene vs user query conflict."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("sleeve", False)])
    attrs = generate_random_attributes(category)
    
    scene, user_override = random.choice(CONFLICT_QUERIES)
    scene_attrs = SCENE_INFERENCE.get(scene, {})
    
    # User query overrides scene
    if "casual" in user_override:
        final_formality = "casual"
    elif "formal" in user_override:
        final_formality = "formal"
    else:
        final_formality = "semi-formal"
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": user_override
        },
        "output": {
            "reasoning": {
                "user_intent": "User query conflicts with scene context",
                "scene_influence": f"Scene {scene} suggests {scene_attrs.get('formality', 'unknown')}",
                "conflict_resolution": "User query takes priority over scene",
                "changes": [f"formality: {scene_attrs.get('formality', 'null')} → {final_formality} (user override)"],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "User explicitly overrides scene inference"
            },
            "filters": {
                "category": category,
                **attrs,
                "formality": final_formality
            },
            "confidence": round(random.uniform(0.80, 0.88), 2)
        }
    }


def generate_low_confidence_sample() -> Dict:
    """TYPE 11: Ambiguous query - low confidence."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    vague_queries = [
        "something nice",
        "whatever looks good",
        "you decide",
        "surprise me",
        "anything works",
        "not sure what I want"
    ]
    query = random.choice(vague_queries)
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": "User query is ambiguous - no clear direction",
                "changes": [],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Vague query, cannot infer specific preferences"
            },
            "filters": {
                "category": category,
                **attrs
            },
            "confidence": round(random.uniform(0.45, 0.60), 2)
        }
    }


def generate_accessory_filter_sample() -> Dict:
    """TYPE 12: Accessory with invalid attribute request."""
    accessories = ["ring", "watch", "earring", "necklace", "bangle", "bracelet", "glass"]
    category = random.choice(accessories)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    invalid_queries = [
        "full sleeve",
        "short sleeve",
        "slim fit",
        "striped pattern"
    ]
    query = random.choice(invalid_queries)
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User requests attribute not applicable to {category}",
                "invalid_request": f"'{query}' is not applicable for {category}",
                "changes": [],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Attribute not applicable - request ignored"
            },
            "filters": {
                "category": category,
                **attrs
            },
            "ignored_attributes": ["sleeve" if "sleeve" in query else "fit" if "fit" in query else "pattern"],
            "confidence": round(random.uniform(0.70, 0.80), 2)
        }
    }


def generate_price_tier_sample() -> Dict:
    """TYPE 13: Price tier inference."""
    category = random.choice(CATEGORIES)
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    price_queries = {
        "budget option": "budget",
        "affordable version": "budget",
        "under 500": "budget",
        "mid-range please": "mid-range",
        "around 1000": "mid-range",
        "premium quality": "premium",
        "luxury version": "premium",
        "expensive one": "premium"
    }
    
    query, tier = random.choice(list(price_queries.items()))
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants {tier} price range",
                "changes": [f"price_tier: null → {tier}"],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Price preference in query"
            },
            "filters": {
                "category": category,
                **attrs,
                "price_tier": tier
            },
            "confidence": round(random.uniform(0.85, 0.92), 2)
        }
    }


def generate_style_inference_sample() -> Dict:
    """TYPE 14: Style/occasion inference."""
    category = random.choice([c for c in CATEGORIES if ATTRIBUTE_RULES.get(c, {}).get("sleeve", False)])
    attrs = generate_random_attributes(category)
    scene = random.choice(SCENES)
    
    style_queries = [
        ("for a date", {"occasion": "date night", "formality": "semi-formal"}),
        ("office wear", {"occasion": "office", "formality": "formal"}),
        ("party look", {"occasion": "party", "formality": "semi-formal"}),
        ("casual outing", {"occasion": "casual outing", "formality": "casual"}),
        ("wedding guest", {"occasion": "wedding", "formality": "formal"}),
        ("gym workout", {"occasion": "gym", "formality": "casual"}),
    ]
    
    query, style_attrs = random.choice(style_queries)
    
    new_attrs = attrs.copy()
    new_attrs.update(style_attrs)
    
    return {
        "input": {
            "category": category,
            "attributes": attrs,
            "scene": scene,
            "session_history": [],
            "user_query": query
        },
        "output": {
            "reasoning": {
                "user_intent": f"User wants items for {style_attrs.get('occasion', 'unspecified')}",
                "changes": [f"{k}: null → {v}" for k, v in style_attrs.items()],
                "preserved": list(attrs.keys()) + ["category"],
                "confidence_factors": "Occasion/style mentioned in query"
            },
            "filters": {
                "category": category,
                **new_attrs
            },
            "confidence": round(random.uniform(0.85, 0.92), 2)
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(total_samples: int = 3500) -> List[Dict]:
    """Generate complete training dataset."""
    
    # Distribution of sample types
    distribution = {
        "preserve": 300,
        "color_change": 350,
        "color_refinement": 200,
        "category_change": 300,
        "attribute_addition": 300,
        "attribute_removal": 200,
        "negation": 200,
        "scene_inference": 250,
        "multi_turn": 500,
        "conflict": 200,
        "low_confidence": 200,
        "accessory_filter": 150,
        "price_tier": 150,
        "style_inference": 250,
    }
    
    generators = {
        "preserve": generate_preserve_sample,
        "color_change": generate_color_change_sample,
        "color_refinement": generate_color_refinement_sample,
        "category_change": generate_category_change_sample,
        "attribute_addition": generate_attribute_addition_sample,
        "attribute_removal": generate_attribute_removal_sample,
        "negation": generate_negation_sample,
        "scene_inference": generate_scene_inference_sample,
        "multi_turn": generate_multi_turn_sample,
        "conflict": generate_conflict_sample,
        "low_confidence": generate_low_confidence_sample,
        "accessory_filter": generate_accessory_filter_sample,
        "price_tier": generate_price_tier_sample,
        "style_inference": generate_style_inference_sample,
    }
    
    dataset = []
    
    for sample_type, count in distribution.items():
        print(f"Generating {count} {sample_type} samples...")
        generator = generators[sample_type]
        for _ in range(count):
            try:
                sample = generator()
                sample["type"] = sample_type
                dataset.append(sample)
            except Exception as e:
                print(f"  Error in {sample_type}: {e}")
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """Save dataset to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Training Dataset Generator")
    print("=" * 60)
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate dataset
    dataset = generate_dataset(3500)
    
    # Save to file
    output_path = "llm_training_data.jsonl"
    save_dataset(dataset, output_path)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    type_counts = {}
    for sample in dataset:
        t = sample.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example Sample")
    print("=" * 60)
    print(json.dumps(dataset[0], indent=2))
