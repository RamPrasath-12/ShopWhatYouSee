"""
Explanation Engine — Two-Section Explanations
================================================
Generates deterministic, human-readable explanations.
Every reason references a REAL scoring signal. No hallucination.

Two generators:
  1. generate_visual_explanation() — for Similar Items section
  2. generate_preference_explanation() — for Recommended For You section
  3. generate_relaxation_explanation() — when filters are broadened
"""

from typing import Dict, Any, Optional, List


# ═════════════════════════════════════════════════
# SECTION 1: VISUAL EXPLANATIONS
# ═════════════════════════════════════════════════

def generate_visual_explanation(
    product: Dict,
    visual_score: float,
    detected_category: str,
) -> str:
    """
    Explain why a product appears in the 'Similar Items' section.
    Based purely on visual similarity signals.
    References specific product attributes for transparency.
    """
    # Helper to clean up strings
    def clean(val):
        if not val or val.lower() in ("na", "not applicable", "not_applicable", "unbranded", "unknown", "none"):
            return ""
        return val.strip()
        
    color_val = clean(product.get("color"))
    pattern_val = clean(product.get("pattern"))
    sleeve_val = clean(product.get("sleeve"))
    brand_val = clean(product.get("brand"))
    
    # 1. Build the descriptive noun phrase: "A [sleeve] [color] [pattern] [category] by [brand]"
    desc_parts = []
    
    # Add sleeve
    if sleeve_val:
        # e.g. "long" -> "long-sleeve" (skip if they already included "sleeve")
        sleeve_norm = sleeve_val.lower()
        if "sleeve" not in sleeve_norm:
            desc_parts.append(f"{sleeve_val.capitalize()}-sleeve")
        else:
            desc_parts.append(sleeve_val.capitalize())
            
    # Add color
    if color_val:
        desc_parts.append(color_val.lower())
        
    # Add pattern (don't repeat solid too often unless it's the only thing)
    if pattern_val and pattern_val.lower() != "solid":
        desc_parts.append(pattern_val.lower())
    elif pattern_val and pattern_val.lower() == "solid" and not color_val:
        desc_parts.append("solid")
        
    # Add category
    if desc_parts:
        desc = " ".join(desc_parts) + f" {detected_category}"
    else:
        desc = detected_category.capitalize()
        
    # Add brand
    if brand_val:
        desc += f" by {brand_val}"
        
    # Prefix with 'A ' or 'An '
    vowels = ('a', 'e', 'i', 'o', 'u')
    prefix = "An " if desc.lower().startswith(vowels) else "A "
    noun_phrase = prefix + desc
    
    # 2. Build the similarity context
    match_pct = int(visual_score * 100)
    
    if visual_score >= 0.85:
        sim_phrase = f"highly similar to your detected item ({match_pct}% visual match)."
    elif visual_score >= 0.70:
        sim_phrase = f"closely resembling your search ({match_pct}% visual match)."
    elif visual_score >= 0.50:
        sim_phrase = f"sharing a strong visual similarity ({match_pct}% match)."
    elif visual_score >= 0.30:
        sim_phrase = f"that shares some visual features ({match_pct}% match) with your item."
    else:
        sim_phrase = f"related to your search ({match_pct}% match)."

    return f"{noun_phrase} {sim_phrase}"


# ═════════════════════════════════════════════════
# SECTION 2: PREFERENCE EXPLANATIONS
# ═════════════════════════════════════════════════

def generate_preference_explanation(
    product: Dict,
    pref_signals: Dict[str, float],
    matched_labels: List[str],
    preferences: Dict,
) -> str:
    """
    Explain why a product appears in the 'Recommended For You' section.
    Uses actual continuous scoring signals and matched labels.
    """
    if not matched_labels:
        # No preference matches — explain based on visual similarity
        visual_score = product.get("visual_score", product.get("similarity_score", 0))
        match_pct = int(visual_score * 100)
        return f"Suggested based on {match_pct}% visual similarity to the detected item."

    # Build rich explanation from actual match signals
    intro_parts = []
    detail_parts = []

    # Count strong matches (signal >= 0.7)
    strong_matches = sum(1 for v in pref_signals.values() if v >= 0.7)

    if strong_matches >= 3:
        intro_parts.append("Strongly matches your preferences")
    elif strong_matches >= 2:
        intro_parts.append("Matches several of your preferences")
    elif strong_matches >= 1:
        intro_parts.append("Matches some of your preferences")
    else:
        intro_parts.append("Partially aligned with your preferences")

    # Add specific match details from matched_labels
    for label in matched_labels:
        detail_parts.append(label)

    # Also mention what didn't match for transparency
    misses = []
    if pref_signals.get("color", 1.0) == 0.0 and preferences.get("preferred_colors"):
        pref_color_str = ", ".join(preferences["preferred_colors"][:2])
        prod_color = product.get("color", "unknown")
        misses.append(f"color is {prod_color} (you prefer {pref_color_str})")
    if pref_signals.get("budget", 1.0) == 0.0:
        prod_price = int(product.get("price", 0))
        budget_max = preferences.get("budget_max", 0)
        if budget_max and prod_price > budget_max:
            misses.append(f"₹{prod_price} exceeds your ₹{budget_max} budget")

    explanation = intro_parts[0]
    if detail_parts:
        explanation += ": " + ", ".join(detail_parts)
    explanation += "."

    if misses:
        explanation += " Note: " + ", ".join(misses) + "."

    return explanation


# ═════════════════════════════════════════════════
# GENERIC EXPLANATION (backward-compatible)
# ═════════════════════════════════════════════════

def generate_explanation(
    product: Dict,
    score_signals: Dict[str, Any],
    preferences: Optional[Dict] = None,
    behavioral_profile: Optional[Dict] = None,
) -> str:
    """
    Legacy explanation generator — used when endpoint doesn't use
    the two-section split. Kept for backward compatibility.
    """
    reasons = []

    visual = score_signals.get("visual_score", 0)
    if visual >= 0.8:
        reasons.append("Highly similar to the item you searched for")
    elif visual >= 0.6:
        reasons.append("Visually similar to the searched item")
    elif visual >= 0.4:
        reasons.append("Somewhat similar to what you're looking for")

    prod_color = product.get("color", "")

    if score_signals.get("pref_budget"):
        price = product.get("price", 0)
        reasons.append(f"Within your budget (₹{int(price)})")

    if score_signals.get("pref_color"):
        reasons.append(f"Matches your preferred color ({prod_color})")

    if score_signals.get("pref_gender"):
        reasons.append("Matches your gender preference")

    if score_signals.get("pref_style"):
        reasons.append(f"Matches your preferred style ({product.get('style', '')})")

    if not reasons:
        return "Recommended based on visual similarity to your search."

    return ". ".join(reasons) + "."


# ═════════════════════════════════════════════════
# RELAXATION EXPLANATION
# ═════════════════════════════════════════════════

def generate_relaxation_explanation(
    relaxation_log: List[str],
    user_filters: Dict[str, Any],
    result_count: int,
) -> Optional[str]:
    """
    When the system relaxes constraints to find results,
    explain what was broadened transparently.
    """
    if not relaxation_log:
        return None

    # What user wanted
    wanted_parts = []
    if user_filters.get("color"):
        wanted_parts.append(f"{user_filters['color']}")
    if user_filters.get("category"):
        wanted_parts.append(f"{user_filters['category']}")
    if user_filters.get("price_max"):
        wanted_parts.append(f"under ₹{user_filters['price_max']}")
    if user_filters.get("pattern"):
        wanted_parts.append(f"{user_filters['pattern']} pattern")
    if user_filters.get("sleeve"):
        wanted_parts.append(f"{user_filters['sleeve']} sleeve")

    wanted_str = " ".join(wanted_parts) if wanted_parts else "your exact request"

    # What was relaxed
    relaxed_parts = []
    for step in relaxation_log:
        step_lower = step.lower()
        if "color" in step_lower:
            relaxed_parts.append("expanded to similar colors")
        elif "category" in step_lower:
            relaxed_parts.append("broadened the category")
        elif "removed" in step_lower:
            relaxed_parts.append("removed some filters")
        elif "family" in step_lower:
            relaxed_parts.append("looked in related categories")

    if not relaxed_parts:
        return None

    relaxed_str = " and ".join(relaxed_parts)

    return (
        f"We couldn't find exact matches for {wanted_str}. "
        f"Instead, we {relaxed_str} to show you {result_count} relevant items."
    )


def generate_no_results_explanation(user_filters: Dict[str, Any]) -> str:
    """When no results found at all."""
    parts = []
    if user_filters.get("color"):
        parts.append(f"color: {user_filters['color']}")
    if user_filters.get("category"):
        parts.append(f"category: {user_filters['category']}")
    if user_filters.get("price_max"):
        parts.append(f"max price: ₹{user_filters['price_max']}")
    if user_filters.get("pattern"):
        parts.append(f"pattern: {user_filters['pattern']}")

    filter_desc = ", ".join(parts) if parts else "your search criteria"

    return (
        f"No products found matching {filter_desc}. "
        "Try broadening your search — for example, remove the color filter or increase the budget."
    )
