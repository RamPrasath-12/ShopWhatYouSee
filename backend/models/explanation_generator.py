"""
Explanation Generator — Grounded, Non-Hallucinated Product Explanations
========================================================================
Generates 1-2 sentence explanations for recommended products using:
  1. Deterministic tag builder (hard thresholds)
  2. Groq LLM call (temp=0.2, strict grounded prompt)
  3. Controlled vocabulary hallucination check
  4. Deterministic fallback if LLM fails/times out

Only called for top-5 products (performance optimization).
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
EXPLANATION_TIMEOUT = 5  # seconds per batch

# ─────────────────────────────────────────────────────
# CONTROLLED VOCABULARY — Only these tokens may appear
# ─────────────────────────────────────────────────────
ALLOWED_MENTION_TOKENS = {
    "color", "colour", "pattern", "sleeve", "visual similarity",
    "visually similar", "user preference", "user filter",
    "override", "relaxed", "broadened", "ranked", "candidates",
    "style", "material",
}

# ─────────────────────────────────────────────────────
# DETERMINISTIC TAG BUILDER (Hard Thresholds)
# ─────────────────────────────────────────────────────
def _build_tags(meta: Dict) -> List[str]:
    """
    Assign deterministic tags based on match_meta.
    Hard thresholds — no ambiguity.
    """
    tags = []

    vs = meta.get("visual_similarity", 0.0)
    if vs > 0.8:
        tags.append("high_visual_similarity")
    elif vs > 0.6:
        tags.append("moderate_visual_similarity")
    elif vs > 0.4:
        tags.append("low_visual_similarity")
    else:
        tags.append("weak_visual_similarity")

    if meta.get("color_match"):
        tags.append("strong_color_match" if meta.get("color_score", 0) > 0.5 else "color_match")
    if meta.get("pattern_match"):
        tags.append("strong_pattern_match" if meta.get("pattern_score", 0) > 0.5 else "pattern_match")
    if meta.get("sleeve_match"):
        tags.append("strong_sleeve_match" if meta.get("sleeve_score", 0) > 0.5 else "sleeve_match")

    if meta.get("override_applied"):
        tags.append("user_override_applied")

    relaxed = meta.get("relaxed_constraints", [])
    if relaxed:
        tags.append("relaxed_search")

    rank = meta.get("rank_position", 99)
    if rank == 1:
        tags.append("top_ranked")
    elif rank <= 3:
        tags.append("top_3")

    return tags


# ─────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK TEMPLATE (No LLM needed)
# ─────────────────────────────────────────────────────
def _deterministic_explanation(meta: Dict, detected_attrs: Dict, user_filters: Dict) -> str:
    """
    Pure template-based explanation. Used when LLM fails or times out.
    Guarantees zero hallucination.
    """
    parts = []
    tags = _build_tags(meta)

    vs = meta.get("visual_similarity", 0.0)
    rank = meta.get("rank_position", 0)
    total = meta.get("total_candidates", 0)
    percentile = meta.get("visual_percentile", 0.0)

    # Contrastive opening for rank-1
    if "top_ranked" in tags and total > 10:
        parts.append(f"Top-ranked among {total} similar products")
    elif "top_3" in tags:
        parts.append(f"Ranked #{rank} out of {total} candidates")

    # Visual similarity
    if "high_visual_similarity" in tags:
        parts.append("with high visual similarity to your image")
    elif "moderate_visual_similarity" in tags:
        parts.append("with moderate visual similarity")

    # Attribute matches
    matches = []
    if meta.get("color_match"):
        color_name = detected_attrs.get("color_name") or user_filters.get("color") or ""
        if color_name:
            matches.append(f"{color_name} color")
        else:
            matches.append("color")
    if meta.get("sleeve_match"):
        sleeve_val = user_filters.get("sleeve") or detected_attrs.get("sleeve") or ""
        if sleeve_val:
            matches.append(f"{sleeve_val} sleeve")
        else:
            matches.append("sleeve style")
    if meta.get("pattern_match"):
        pattern_val = user_filters.get("pattern") or detected_attrs.get("pattern") or ""
        if pattern_val:
            matches.append(f"{pattern_val} pattern")
        else:
            matches.append("pattern")

    if matches:
        parts.append("matching " + ", ".join(matches))

    # Override
    if "user_override_applied" in tags:
        parts.append("based on your filter preferences")

    # Relaxation
    relaxed = meta.get("relaxed_constraints", [])
    if relaxed:
        parts.append(f"(search broadened by relaxing {', '.join(relaxed)})")

    if not parts:
        return "Recommended based on visual similarity to your selected item."

    return "Recommended " + ", ".join(parts) + "."


# ─────────────────────────────────────────────────────
# CACHE — Deterministic tuple key
# ─────────────────────────────────────────────────────
_explanation_cache: Dict[tuple, str] = {}
_CACHE_MAX_SIZE = 500


def _cache_key(product_id: str, meta: Dict) -> tuple:
    """
    Deterministic cache key from product_id + rounded scores + bools.
    No floats or dicts — only hashable primitives.
    """
    return (
        product_id,
        round(meta.get("visual_similarity", 0.0), 3),
        meta.get("color_match", False),
        meta.get("pattern_match", False),
        meta.get("sleeve_match", False),
        meta.get("override_applied", False),
        tuple(sorted(meta.get("relaxed_constraints", []))),
        meta.get("rank_position", 0),
    )


# ─────────────────────────────────────────────────────
# CONTROLLED VOCABULARY HALLUCINATION CHECK
# ─────────────────────────────────────────────────────
def _validate_explanation(explanation: str, meta: Dict) -> bool:
    """
    Verify that the LLM explanation does not mention attributes
    that are NOT matched in the metadata.

    Returns True if explanation is valid, False if hallucinated.
    """
    text_lower = explanation.lower()

    # Check: if color not matched, explanation must not mention color
    if not meta.get("color_match") and any(w in text_lower for w in ["color", "colour"]):
        logger.warning("[ExplainCheck] Hallucination: mentions color but color_match=False")
        return False

    # Check: if pattern not matched, explanation must not mention pattern
    if not meta.get("pattern_match") and "pattern" in text_lower:
        logger.warning("[ExplainCheck] Hallucination: mentions pattern but pattern_match=False")
        return False

    # Check: if sleeve not matched, explanation must not mention sleeve
    if not meta.get("sleeve_match") and "sleeve" in text_lower:
        logger.warning("[ExplainCheck] Hallucination: mentions sleeve but sleeve_match=False")
        return False

    # Check: if no relaxation, must not mention broadened/relaxed
    if not meta.get("relaxed_constraints") and any(w in text_lower for w in ["relaxed", "broadened"]):
        logger.warning("[ExplainCheck] Hallucination: mentions relaxation but none occurred")
        return False

    return True


# ─────────────────────────────────────────────────────
# GROUNDED LLM PROMPT
# ─────────────────────────────────────────────────────
EXPLANATION_SYSTEM_PROMPT = """You generate grounded product explanations for a fashion recommendation system.

You are given structured metadata describing:
- Detected image attributes (from camera)
- User-applied filters
- Product match scores (visual similarity, color, pattern, sleeve)
- Which specific attributes matched
- Whether user override or search relaxation occurred
- Rank position and total candidate count

STRICT RULES:
- ONLY use the provided metadata. Never invent attributes.
- If color_match=false, do NOT mention color.
- If pattern_match=false, do NOT mention pattern.
- If sleeve_match=false, do NOT mention sleeve.
- If relaxation_applied, mention that results were broadened.
- If rank_position=1, mention it is the top-ranked result.
- Keep explanation to exactly 1-2 sentences.
- Use clear, simple, conversational language.
- Start with "Recommended because" or similar opener.
- Include numeric context when available (e.g. "among 500 candidates").

OUTPUT: Plain text explanation only. No JSON. No markdown."""


def _build_user_prompt(meta: Dict, detected_attrs: Dict, user_filters: Dict, tags: List[str]) -> str:
    """Build the user prompt for the LLM with structured metadata."""
    return json.dumps({
        "detected_attributes": {k: v for k, v in detected_attrs.items() if v},
        "user_filters": {k: v for k, v in user_filters.items() if v},
        "product_match_metadata": {
            "visual_similarity": meta.get("visual_similarity"),
            "color_match": meta.get("color_match"),
            "pattern_match": meta.get("pattern_match"),
            "sleeve_match": meta.get("sleeve_match"),
            "color_score": meta.get("color_score"),
            "pattern_score": meta.get("pattern_score"),
            "sleeve_score": meta.get("sleeve_score"),
            "override_applied": meta.get("override_applied"),
            "relaxed_constraints": meta.get("relaxed_constraints"),
            "rank_position": meta.get("rank_position"),
            "total_candidates": meta.get("total_candidates"),
            "visual_percentile": meta.get("visual_percentile"),
            "visual_weight": meta.get("visual_weight"),
            "override_weight": meta.get("override_weight"),
        },
        "tags": tags,
    }, indent=2)


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────
def generate_explanations(
    products: List[Dict],
    detected_attrs: Dict,
    user_filters: Dict,
    max_products: int = 5
) -> List[Dict]:
    """
    Generate grounded explanations for top-N products.

    Returns list of products (copies) with 'explanation' field added.
    Does NOT mutate original products.

    Args:
        products: Top-K products from retrieval (only first max_products get explanations)
        detected_attrs: Visual baseline from AGMAN
        user_filters: User override filters
        max_products: Max products to explain (default 5)

    Returns:
        List of product dicts with 'explanation' field
    """
    import copy
    result_products = [copy.deepcopy(p) for p in products]

    # Only explain top N
    to_explain = result_products[:max_products]

    t0 = time.time()

    # Try batch LLM call
    llm_explanations = _batch_llm_explain(to_explain, detected_attrs, user_filters)

    # Apply explanations (LLM or fallback)
    for i, prod in enumerate(to_explain):
        meta = prod.get("match_meta", {})
        product_id = prod.get("product_id", "")

        # Check cache first
        key = _cache_key(product_id, meta)
        if key in _explanation_cache:
            prod["explanation"] = _explanation_cache[key]
            continue

        # LLM explanation (if available and validated)
        if i < len(llm_explanations) and llm_explanations[i]:
            explanation = llm_explanations[i].strip()
            if _validate_explanation(explanation, meta):
                prod["explanation"] = explanation
                _explanation_cache[key] = explanation
                continue
            else:
                logger.warning(f"[Explain] Hallucination detected for product {product_id}, using fallback")

        # Deterministic fallback
        explanation = _deterministic_explanation(meta, detected_attrs, user_filters)
        prod["explanation"] = explanation
        _explanation_cache[key] = explanation

    # Products beyond max_products get no explanation
    elapsed_ms = (time.time() - t0) * 1000
    print(f"[Explain] Generated {len(to_explain)} explanations in {elapsed_ms:.0f}ms")

    # Evict old cache entries if too large
    if len(_explanation_cache) > _CACHE_MAX_SIZE:
        keys = list(_explanation_cache.keys())
        for k in keys[:len(keys) - _CACHE_MAX_SIZE]:
            del _explanation_cache[k]

    return result_products


# ─────────────────────────────────────────────────────
# BATCH LLM CALL (with timeout + rate limit guard)
# ─────────────────────────────────────────────────────
def _batch_llm_explain(
    products: List[Dict],
    detected_attrs: Dict,
    user_filters: Dict
) -> List[Optional[str]]:
    """
    Call Groq LLM for each product explanation.
    Returns list of explanation strings (or None on failure).
    Falls back to None (triggers deterministic template) on any error.
    """
    if not GROQ_API_KEY:
        logger.warning("[Explain] No GROQ_API_KEY — using deterministic fallback for all")
        return [None] * len(products)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY, timeout=EXPLANATION_TIMEOUT)
    except Exception as e:
        logger.warning(f"[Explain] Groq client init failed: {e}")
        return [None] * len(products)

    results = []
    for prod in products:
        meta = prod.get("match_meta", {})
        tags = _build_tags(meta)

        user_prompt = _build_user_prompt(meta, detected_attrs, user_filters, tags)

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=100,
            )
            explanation = response.choices[0].message.content.strip()
            results.append(explanation)
        except Exception as e:
            logger.warning(f"[Explain] Groq call failed for product {prod.get('product_id')}: {e}")
            results.append(None)

    return results
