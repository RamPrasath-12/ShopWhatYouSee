"""
Unified LLM Module - Phase 2 (Fixed: Category Override Guard)
==============================================================
Orchestrates LLM filter generation with:
- Pure visual search mode (uses AGMAN attributes)
- Query-based search mode (calls Groq LLM)
- Keyword fallback (when LLM unavailable)

FIX: Post-LLM category guard strips hallucinated category overrides
     when the user query contains no explicit product-type keyword.
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# =============================================================================
# KNOWN PRODUCT-TYPE KEYWORDS
# If none of these appear in the user query, the LLM must NOT override category.
# Keys are query substrings; values are the matching DB category.
# =============================================================================

PRODUCT_TYPE_KEYWORDS = {
    # Tops
    "jacket": "Jacket",
    "coat": "Jacket",
    "shirt": "shirts",
    "shirts": "shirts",
    "formal shirt": "shirts",
    "tshirt": "tshirt",
    "t-shirt": "tshirt",
    "t shirt": "tshirt",
    "tee": "tshirt",
    "blouse": "blouse",
    "blazer": "blazer",
    "shawl": "shawl",

    # Bottoms
    "pant": "pant",
    "pants": "pant",
    "jeans": "pant",
    "trouser": "pant",
    "trousers": "pant",
    "shorts": "shorts",
    "short": "shorts",
    "skirt": "skirt",
    "leggings": "leggings",

    # Traditional
    "churidhar": "churidhar",
    "dhoti": "dhoti",
    "saree": "saree",

    # Footwear
    "sandal": "Footwear_sandals",
    "sandals": "Footwear_sandals",
    "shoe": "Footwear_shoes",
    "shoes": "Footwear_shoes",
    "heel": "Footwear_sandals",
    "heels": "Footwear_sandals",
    "footwear": None,  # generic — allow LLM to pick sub-type

    # Accessories
    "bag": "bag",
    "purse": "purse",
    "belt": "belt",
    "tie": "tie",
    "cap": "caps",
    "caps": "caps",
    "hat": "caps",
    "watch": "watch",
    "glasses": "glasses",
    "sunglasses": "glasses",
    "eyewear": "glasses",
    "earring": "earrings",
    "earrings": "earrings",
    "necklace": "necklace",
    "bangle": "bangle",
    "bracelet": "bracelet",
    "ring": "ring",
}

# Words that look product-like but are actually attributes — never treat as category
ATTRIBUTE_ONLY_WORDS = {
    # Colors — most common source of false category overrides
    "red", "blue", "green", "black", "white", "pink", "yellow", "orange",
    "purple", "brown", "grey", "gray", "navy", "maroon", "beige", "teal",
    "turquoise", "cream", "ivory", "gold", "silver", "cyan", "coral",
    "peach", "magenta", "fuchsia", "lime", "olive", "khaki", "rust",
    "burgundy", "wine", "crimson", "mint", "lavender", "violet", "plum",
    # Sleeve types
    "full sleeve", "half sleeve", "short sleeve", "long sleeve", "sleeveless",
    "three quarter",
    # Patterns
    "solid", "striped", "checked", "plain", "printed", "floral", "plaid",
    # Styles
    "casual", "formal", "sports", "party", "ethnic",
    # Genders
    "men", "women", "male", "female",
    # Price words
    "cheap", "budget", "expensive", "affordable", "premium",
    # Materials
    "cotton", "denim", "silk", "polyester", "wool", "leather",
}


def _query_contains_product_type(query: str) -> tuple[bool, str | None]:
    """
    Check if the user query explicitly contains a product-type keyword.

    Returns:
        (True, matched_keyword)  — query names a product type
        (False, None)            — query only contains attribute words (color, sleeve, etc.)
    """
    q_lower = query.lower().strip()

    # Mask out known attribute words/phrases so they don't accidentally match product types
    # e.g., "short sleeve" contains "short", which could trigger the "shorts" category.
    for attr in sorted(ATTRIBUTE_ONLY_WORDS, key=len, reverse=True):
        # Replace matches with space to avoid merging words
        q_lower = re.sub(r'\b' + re.escape(attr) + r'\b', ' ', q_lower)

    # Check multi-word keywords first (longer matches take priority)
    for keyword in sorted(PRODUCT_TYPE_KEYWORDS.keys(), key=len, reverse=True):
        # Use regex to match exact word boundaries to prevent substring matches
        if re.search(r'\b' + re.escape(keyword) + r'\b', q_lower):
            return True, keyword

    return False, None


def _sanitize_llm_category(
    llm_filters: Dict[str, Any],
    detected_category: str,
    user_query: str
) -> Dict[str, Any]:
    """
    Post-LLM guard: Remove category from LLM output if user did not explicitly
    request a product-type change.

    This is the primary defense against the LLM hallucinating category overrides
    for pure attribute queries like "red", "full sleeve", "under 500", etc.

    Args:
        llm_filters:        The raw "add" dict from the LLM
        detected_category:  YOLO-detected category (ground truth)
        user_query:         Original user query string

    Returns:
        Sanitized "add" dict — category removed if not user-requested
    """
    if not llm_filters:
        return llm_filters

    llm_category = llm_filters.get("category")
    if not llm_category:
        return llm_filters  # LLM didn't output a category — nothing to check

    # Check if user explicitly named a product type
    query_has_product, matched_kw = _query_contains_product_type(user_query)

    if not query_has_product:
        # User did NOT name a product type → LLM hallucinated the category override
        print(
            f"[CategoryGuard] ⚠️  LLM output category='{llm_category}' "
            f"but query='{user_query}' contains NO product-type keyword. "
            f"Stripping hallucinated category. Keeping detected='{detected_category}'."
        )
        sanitized = dict(llm_filters)
        del sanitized["category"]
        return sanitized

    # User DID name a product type — verify the LLM's category matches what was asked
    # (Prevent subtle hallucinations like query="jacket" → LLM outputs "Footwear_sandals")
    from models.prompts import SYSTEM_PROMPT  # just for import path check
    expected_category = PRODUCT_TYPE_KEYWORDS.get(matched_kw)

    if expected_category is not None:
        # Normalize both for comparison
        llm_cat_lower = llm_category.lower().replace("_", "").replace(" ", "")
        exp_cat_lower = expected_category.lower().replace("_", "").replace(" ", "")
        det_cat_lower = detected_category.lower().replace("_", "").replace(" ", "")

        if llm_cat_lower == exp_cat_lower:
            # LLM output matches what user asked — allow
            print(f"[CategoryGuard] ✅ Category override confirmed: '{llm_category}' matches query keyword '{matched_kw}'")
        elif llm_cat_lower == det_cat_lower:
            # LLM echoed the detected category — that is fine, leave it
            print(f"[CategoryGuard] ✅ Category unchanged: LLM echoed detected='{detected_category}'")
        else:
            # LLM output a THIRD category that matches neither query nor detected
            print(
                f"[CategoryGuard] ⚠️  LLM category='{llm_category}' does not match "
                f"query keyword '{matched_kw}' (expected '{expected_category}'). "
                f"Correcting to '{expected_category}'."
            )
            sanitized = dict(llm_filters)
            sanitized["category"] = expected_category
            return sanitized
    else:
        # matched_kw is "footwear" (generic) — trust the LLM's sub-type
        print(f"[CategoryGuard] ✅ Generic footwear query — trusting LLM category '{llm_category}'")

    return llm_filters


class UnifiedLLM:
    """Unified LLM with visual bypass for pure image search."""

    def __init__(self):
        self._external_llm = None

    def _get_external_llm(self):
        """Lazy load Groq LLM."""
        if self._external_llm is None:
            from models.external_llm import GroqLLM
            self._external_llm = GroqLLM()
        return self._external_llm

    def generate_filters(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict] = None,
        prefer_external: bool = False
    ) -> Dict[str, Any]:
        """
        Generate product filters via Groq or visual bypass.

        Returns:
            {
                "add": {...},
                "remove": [],
                "reset_to_visual": false,
                "price_max": null,
                "reasoning": "...",
                "confidence": 0.95,
                "source": "groq" | "visual_bypass" | "fallback"
            }
        """
        print(f"\n[UnifiedLLM] Generating filters...")
        print(f"  Query: {query}")
        print(f"  Category: {category}")

        # =====================================================================
        # PURE VISUAL SEARCH MODE (No query provided)
        # =====================================================================
        if not query.strip() and not session_history:
            print("[UnifiedLLM] Pure visual search - using AGMAN attributes")

            try:
                from utils.color_utils import normalize_color_name
            except:
                normalize_color_name = lambda x: x

            try:
                from product_retrieval import COLOR_FAMILIES
            except:
                COLOR_FAMILIES = {}

            # Extract AGMAN attributes (handle both flat and structured formats)
            color = None
            if attributes.get("color_name"):
                color = attributes["color_name"]
            elif isinstance(attributes.get("color"), dict):
                color = attributes["color"].get("value")
            elif isinstance(attributes.get("color"), str):
                color = attributes["color"]

            sleeve = None
            if attributes.get("sleeve"):
                sleeve = attributes["sleeve"]
            elif isinstance(attributes.get("sleeve_structured"), dict):
                sleeve = attributes["sleeve_structured"].get("value")

            pattern = None
            if attributes.get("pattern"):
                pattern = attributes["pattern"]
            elif isinstance(attributes.get("pattern_structured"), dict):
                pattern = attributes["pattern_structured"].get("value")

            # Build filters from visual detection
            filters = {"category": category}

            if color and color.lower() not in ["none", "not_applicable", "", "unknown"]:
                filters["primary_color_name"] = normalize_color_name(color)
                color_lower = color.lower().strip()
                color_family = COLOR_FAMILIES.get(color_lower, color_lower)
                filters["color_family"] = color_family
                print(f"[UnifiedLLM] Color: {color} → {filters['primary_color_name']} (family: {color_family})")

            if sleeve and sleeve.lower() not in ["none", "not_applicable", "", "unknown"]:
                sleeve_map = {
                    "long": "long",
                    "short": "short",
                    "three_quarter": "three_quarter",
                    "sleeveless": "sleeveless",
                    "half": "half",
                }
                mapped_sleeve = sleeve_map.get(sleeve.lower(), sleeve)
                filters["sleeve_value"] = mapped_sleeve
                print(f"[UnifiedLLM] Sleeve: {sleeve} → {mapped_sleeve}")

            if pattern and pattern.lower() not in ["none", "not_applicable", "", "unknown"]:
                pattern_map = {
                    "solid": "solid",
                    "striped": "striped",
                    "checked": "checked",
                    "patterned": "printed",
                    "printed": "printed",
                    "floral": "floral",
                }
                mapped_pattern = pattern_map.get(pattern.lower(), pattern.lower())
                filters["pattern_value"] = mapped_pattern
                print(f"[UnifiedLLM] Pattern: {pattern} → {mapped_pattern}")

            print(f"[UnifiedLLM] Visual bypass filters: {filters}")

            return {
                "add": filters,
                "remove": [],
                "reset_to_visual": False,
                "price_max": None,
                "reasoning": "Pure visual search using detected attributes",
                "confidence": 0.95,
                "source": "visual_bypass"
            }

        # =====================================================================
        # QUERY-BASED SEARCH MODE (User provided query)
        # =====================================================================
        try:
            print("[UnifiedLLM] Calling Groq LLM...")
            result = self._get_external_llm().generate_filters(
                category=category,
                attributes=attributes,
                scene=scene,
                query=query,
                session_history=session_history
            )

            # =================================================================
            # POST-LLM CATEGORY GUARD
            # Strips any hallucinated category override before the filters are
            # used by the retrieval engine.
            # =================================================================
            if result and result.get("add"):
                result["add"] = _sanitize_llm_category(
                    llm_filters=result["add"],
                    detected_category=category,
                    user_query=query
                )
                print(f"[UnifiedLLM] Filters after CategoryGuard: {result['add']}")

            return result

        except Exception as e:
            print(f"[UnifiedLLM] Groq failed: {e}, using keyword fallback...")
            fallback_result = self._keyword_fallback(query)

            # Apply category guard to fallback too
            if fallback_result and fallback_result.get("add"):
                fallback_result["add"] = _sanitize_llm_category(
                    llm_filters=fallback_result["add"],
                    detected_category=category,
                    user_query=query
                )

            return fallback_result

    def _keyword_fallback(self, query: str) -> Dict[str, Any]:
        """
        Rule-based keyword extraction when LLM unavailable.
        Returns flat filters with only confidently extracted values.
        """
        filters = {}
        q = (query or "").lower()

        # Color keywords
        for color in ["red", "blue", "green", "black", "white", "pink",
                      "yellow", "orange", "purple", "brown", "grey", "gray",
                      "navy", "maroon", "beige", "teal", "turquoise"]:
            if color in q:
                filters["color_family"] = color
                filters["primary_color_name"] = color.title()
                break

        # Gender keywords
        if any(w in q for w in ["for men", "men's", "male", " man ", "mens"]):
            filters["gender"] = "Men"
        elif any(w in q for w in ["for women", "women's", "female", " woman ", "womens"]):
            filters["gender"] = "Women"

        # Style keywords
        for style in ["casual", "formal", "sports", "party", "ethnic"]:
            if style in q:
                filters["style"] = style
                break

        # Sleeve keywords
        sleeve_map = {
            "full sleeve": "long",
            "long sleeve": "long",
            "half sleeve": "half",
            "short sleeve": "short",
            "sleeveless": "sleeveless",
            "three quarter": "three_quarter",
        }
        for kw, val in sleeve_map.items():
            if kw in q:
                filters["sleeve_value"] = val
                break

        # Pattern keywords
        pattern_map = {
            "solid": "solid",
            "plain": "solid",
            "striped": "striped",
            "stripes": "striped",
            "checked": "checked",
            "check": "checked",
            "printed": "printed",
            "print": "printed",
            "floral": "floral",
        }
        for kw, val in pattern_map.items():
            if kw in q:
                filters["pattern_value"] = val
                break

        # Material keywords
        for material in ["cotton", "denim", "silk", "polyester", "wool", "leather"]:
            if material in q:
                filters["material"] = material
                break

        # Price extraction
        price_max = None
        match = re.search(r'(?:under|below|less than|<\s*)\s*(\d+)', q)
        if match:
            try:
                price_max = int(match.group(1))
            except ValueError:
                pass

        # =====================================================================
        # Category extraction in fallback
        # Only extract if query contains an explicit product keyword
        # =====================================================================
        has_product, matched_kw = _query_contains_product_type(query)
        if has_product and matched_kw:
            expected_cat = PRODUCT_TYPE_KEYWORDS.get(matched_kw)
            if expected_cat:
                filters["category"] = expected_cat

        # Reset intent
        reset_keywords = ["show original", "reset", "go back", "revert"]
        reset_to_visual = any(kw in q for kw in reset_keywords)

        # Removal intent
        remove_keys = []
        if "remove color" in q or "any color" in q:
            remove_keys.extend(["color_family", "primary_color_name"])
        if "remove sleeve" in q or "any sleeve" in q:
            remove_keys.append("sleeve_value")
        if "remove pattern" in q or "any pattern" in q:
            remove_keys.append("pattern_value")

        return {
            "add": filters if not reset_to_visual else {},
            "remove": remove_keys,
            "reset_to_visual": reset_to_visual,
            "price_max": price_max,
            "reasoning": "Keyword extraction fallback (LLM unavailable)",
            "confidence": 0.3,
            "source": "fallback"
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_unified_llm = None


def get_unified_llm() -> UnifiedLLM:
    """Get the unified LLM instance."""
    global _unified_llm
    if _unified_llm is None:
        _unified_llm = UnifiedLLM()
    return _unified_llm


def generate_filters(
    category: str,
    attributes: Dict[str, Any],
    scene: str,
    query: str,
    session_history: List[Dict] = None,
    prefer_external: bool = False
) -> Dict[str, Any]:
    """Main entry point for filter generation."""
    return get_unified_llm().generate_filters(
        category=category,
        attributes=attributes,
        scene=scene,
        query=query,
        session_history=session_history,
        prefer_external=prefer_external
    )