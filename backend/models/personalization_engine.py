"""
Personalization Engine — Two-Section Recommendations
======================================================
Produces TWO independent recommendation lists:

Section 1 — Similar Items:
    Score = visual_similarity (cosine from AG-MAN embeddings)
    Same category as detected item, ranked by appearance.

Section 2 — Recommended For You:
    Score = 0.6 × preference_match + 0.4 × visual_similarity
    Same OR related categories, filtered by user preferences.

All scoring is CONTINUOUS (not binary) to produce varied rankings.
No LLM hallucination. No randomness. Fully deterministic.
"""

from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────
# RELATED CATEGORIES (real DB categories only)
# Used only for Section 2 (Personalized) to widen pool
# ─────────────────────────────────────────────────
RELATED_CATEGORIES = {
    "Jacket": ["blazer"],
    "blazer": ["Jacket"]
}


# ═════════════════════════════════════════════════
# SECTION 1: VISUAL RECOMMENDATIONS
# ═════════════════════════════════════════════════

def get_visual_recommendations(
    candidates: List[Dict],
    detected_category: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Return top-K products ranked purely by visual similarity.
    Only includes products from the SAME category as the detected item.

    Args:
        candidates: Products from search_products_v3(), each has 'similarity_score'.
        detected_category: The YOLO-detected category (e.g., "Jacket").
        top_k: Number of results.

    Returns:
        List of product dicts with visual_score and explanation added.
    """
    from models.explanation_engine import generate_visual_explanation

    det_cat = (detected_category or "").lower().strip()

    # Filter to same category
    same_cat = [
        p for p in candidates
        if (p.get("category") or "").lower().strip() == det_cat
    ]

    # Sort by raw visual similarity (cosine score), descending
    # similarity_score is the per-product cosine similarity from search_products_v3
    same_cat.sort(key=lambda x: -(x.get("similarity_score", 0)))

    results = []
    for p in same_cat[:top_k]:
        prod = dict(p)
        visual_score = prod.get("similarity_score", 0)
        prod["visual_score"] = round(visual_score, 4)
        prod["explanation"] = generate_visual_explanation(
            prod, visual_score, detected_category
        )
        prod["section"] = "similar"
        results.append(prod)

    return results


# ═════════════════════════════════════════════════
# SECTION 2: PERSONALIZED RECOMMENDATIONS
# ═════════════════════════════════════════════════

# ─── Color family grouping for partial matching ───
COLOR_GROUPS = {
    "red": "warm", "maroon": "warm", "wine": "warm", "crimson": "warm",
    "orange": "warm", "rust": "warm", "coral": "warm",
    "blue": "cool", "navy": "cool", "navy blue": "cool", "teal": "cool",
    "turquoise": "cool", "cyan": "cool",
    "green": "natural", "olive": "natural", "khaki": "natural", "lime": "natural",
    "pink": "soft", "peach": "soft", "lavender": "soft", "mauve": "soft",
    "purple": "rich", "violet": "rich", "plum": "rich", "magenta": "rich",
    "brown": "earth", "beige": "earth", "cream": "earth", "tan": "earth",
    "black": "neutral", "grey": "neutral", "gray": "neutral", "white": "neutral",
    "silver": "neutral", "charcoal": "neutral",
    "gold": "metallic", "bronze": "metallic", "copper": "metallic",
    "yellow": "bright", "mustard": "bright", "lemon": "bright",
}


def _color_match_score(prod_color: str, pref_colors: list) -> float:
    """
    Continuous color matching:
      1.0 = exact match
      0.7 = same color family / substring match
      0.3 = same color group (warm/cool/neutral)
      0.0 = no match
    """
    if not prod_color or not pref_colors:
        return 0.0
    prod_lower = prod_color.lower().strip()

    for pc in pref_colors:
        pc_lower = pc.lower().strip()
        if pc_lower == prod_lower:
            return 1.0

    # Substring / partial
    for pc in pref_colors:
        pc_lower = pc.lower().strip()
        if pc_lower in prod_lower or prod_lower in pc_lower:
            return 0.7

    # Same color group
    prod_group = COLOR_GROUPS.get(prod_lower)
    if prod_group:
        for pc in pref_colors:
            pref_group = COLOR_GROUPS.get(pc.lower().strip())
            if pref_group and pref_group == prod_group:
                return 0.3

    return 0.0


def _budget_match_score(prod_price: float, budget_min: float, budget_max: float) -> float:
    """
    Continuous budget scoring:
      1.0 = within budget range
      0.5-0.9 = slightly outside (within 20% overflow)
      0.0 = way outside budget
    """
    if budget_max <= 0 or budget_max >= 99999:
        return 0.5  # no budget set → neutral score

    if budget_min <= prod_price <= budget_max:
        # Within budget — score higher for products closer to budget_min (better deal)
        range_size = budget_max - budget_min
        if range_size > 0:
            position = (prod_price - budget_min) / range_size
            return 1.0 - (position * 0.2)  # 1.0 at min, 0.8 at max
        return 1.0

    # Slightly over budget (within 20%)
    overflow = budget_max * 1.2
    if prod_price <= overflow:
        overshoot = (prod_price - budget_max) / (overflow - budget_max)
        return max(0.3, 0.7 - (overshoot * 0.4))

    return 0.0  # way over budget


def compute_preference_match(product: Dict, preferences: Dict) -> Dict[str, Any]:
    """
    Compute how well a product matches user preferences.
    CONTINUOUS scoring — not binary. Each signal returns 0.0 to 1.0.

    Returns: {
        score: float (weighted average),
        signals: { gender: float, budget: float, color: float, style: float, brand: float },
        matched_labels: [str]  # human-readable match reasons
    }

    Category is NOT checked here — it's dynamic from YOLO/LLM.
    """
    if not preferences:
        return {"score": 0.0, "signals": {}, "matched_labels": []}

    signals = {}
    weights = {}
    matched_labels = []

    # 1. Gender (weight: 1.0)
    pref_gender = (preferences.get("gender") or "").lower().strip()
    prod_gender = (product.get("gender") or "").lower().strip()
    if pref_gender and pref_gender != "unisex":
        weights["gender"] = 1.0
        if pref_gender == prod_gender:
            signals["gender"] = 1.0
            matched_labels.append(f"gender: {prod_gender}")
        elif prod_gender == "unisex" or not prod_gender:
            signals["gender"] = 0.8
            matched_labels.append("gender: unisex (compatible)")
        else:
            signals["gender"] = 0.0

    # 2. Budget (weight: 1.5 — important signal)
    budget_min = preferences.get("budget_min", 0) or 0
    budget_max = preferences.get("budget_max", 99999) or 99999
    prod_price = product.get("price", 0) or 0
    if budget_max < 99999:
        weights["budget"] = 1.5
        b_score = _budget_match_score(prod_price, budget_min, budget_max)
        signals["budget"] = b_score
        if b_score >= 0.8:
            matched_labels.append(f"within budget (₹{int(prod_price)})")
        elif b_score >= 0.5:
            matched_labels.append(f"close to budget (₹{int(prod_price)})")

    # 3. Color (weight: 2.0 — strongest visual signal)
    pref_colors = preferences.get("preferred_colors") or []
    prod_color = (product.get("color") or product.get("primary_color_name") or
                  product.get("color_family") or "").strip()
    if pref_colors:
        weights["color"] = 2.0
        c_score = _color_match_score(prod_color, pref_colors)
        signals["color"] = c_score
        if c_score >= 0.7:
            matched_labels.append(f"color: {prod_color}")
        elif c_score >= 0.3:
            matched_labels.append(f"similar color tone ({prod_color})")

    # 4. Style (weight: 1.0)
    pref_styles = [s.lower() for s in (preferences.get("preferred_styles") or [])]
    prod_style = (product.get("style") or "").lower().strip()
    if pref_styles and prod_style:
        weights["style"] = 1.0
        if prod_style in pref_styles:
            signals["style"] = 1.0
            matched_labels.append(f"style: {prod_style}")
        elif any(ps in prod_style or prod_style in ps for ps in pref_styles):
            signals["style"] = 0.6
            matched_labels.append(f"related style ({prod_style})")
        else:
            signals["style"] = 0.0

    # 5. Brand (weight: 1.0)
    pref_brands = [b.lower() for b in (preferences.get("preferred_brands") or [])]
    prod_brand = (product.get("brand") or "").lower().strip()
    if pref_brands and prod_brand:
        weights["brand"] = 1.0
        if prod_brand in pref_brands or any(pb in prod_brand for pb in pref_brands):
            signals["brand"] = 1.0
            matched_labels.append(f"brand: {product.get('brand', '')}")
        else:
            signals["brand"] = 0.0

    # Weighted average
    if weights:
        total_weight = sum(weights.values())
        weighted_sum = sum(signals.get(k, 0) * weights[k] for k in weights)
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
    else:
        score = 0.0

    return {
        "score": round(score, 4),
        "signals": {k: round(v, 4) for k, v in signals.items()},
        "matched_labels": matched_labels
    }


def get_personalized_recommendations(
    candidates: List[Dict],
    preferences: Dict,
    behavioral_profile: Optional[Dict],
    detected_category: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Return top-K products ranked by preference match + visual similarity.
    Includes same-category AND related-category products.

    Score = 0.6 × preference_match + 0.4 × visual_similarity

    Args:
        candidates: Products from search_products_v3() (large pool).
        preferences: User prefs (gender, budget, colors, styles, brands — no category).
        behavioral_profile: Aggregated purchase history (optional).
        detected_category: YOLO-detected category for finding related categories.
        top_k: Number of results.

    Returns:
        List of product dicts with preference explanation added.
    """
    from models.explanation_engine import generate_preference_explanation

    if not preferences:
        return []

    det_cat = (detected_category or "").lower().strip()
    related = RELATED_CATEGORIES.get(detected_category, [])
    allowed_cats = {det_cat} | {r.lower().strip() for r in related}

    # Filter to allowed categories
    pool = [
        p for p in candidates
        if (p.get("category") or "").lower().strip() in allowed_cats
    ]

    # Score each product
    scored = []
    for p in pool:
        visual_score = p.get("similarity_score", 0)
        pref_result = compute_preference_match(p, preferences)
        pref_score = pref_result["score"]
        pref_signals = pref_result["signals"]
        matched_labels = pref_result["matched_labels"]

        # Final score: 0.6 * preference + 0.4 * visual
        final = 0.6 * pref_score + 0.4 * visual_score

        prod = dict(p)
        prod["personalized_score"] = round(final, 4)
        prod["preference_score"] = round(pref_score, 4)
        prod["visual_score"] = round(visual_score, 4)
        prod["matched_preferences"] = matched_labels
        prod["preference_signals"] = pref_signals
        prod["explanation"] = generate_preference_explanation(
            prod, pref_signals, matched_labels, preferences
        )
        prod["score_breakdown"] = {
            "visual": round(visual_score, 4),
            "preference": round(pref_score, 4),
        }
        prod["section"] = "personalized"
        scored.append(prod)

    # Sort by personalized score descending, then by visual as tiebreaker
    scored.sort(key=lambda x: (-x["personalized_score"], -x["visual_score"]))

    return scored[:top_k]
