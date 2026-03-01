"""
Unified LLM Module - Phase 2 (Flat Filters)
=============================================
Orchestrates LLM filter generation.
Currently uses Groq directly (Phi-3 local disabled due to thread issues).

Pipeline:
  Query -> Groq (retry on fail) -> flat filters -> return
  Validation and price mapping done externally in app.py
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class UnifiedLLM:
    """Unified LLM that uses Groq for structured filter generation."""

    def __init__(self):
        self._external_llm = None

    def _get_external_llm(self):
        """Lazy load external LLM."""
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
        Generate flat product filters via Groq.

        Returns:
            {
                "filters": {"category": "shirts", "gender": "men", ...},
                "price_max": 1000 or None,
                "reasoning": "...",
                "confidence": 0.85,
                "source": "groq" | "fallback"
            }
        """
        print(f"\n[UnifiedLLM] Generating filters via Groq...")
        print(f"  Query: {query}")
        print(f"  Category: {category}")

        try:
            result = self._get_external_llm().generate_filters(
                category=category,
                attributes=attributes,
                scene=scene,
                query=query,
                session_history=session_history
            )
            return result

        except Exception as e:
            print(f"[UnifiedLLM] Groq failed: {e}, using keyword fallback...")
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> Dict[str, Any]:
        """
        Rule-based keyword extraction when LLM is unavailable.
        Returns flat filters dict with only confidently extracted values.
        """
        filters = {}
        q = (query or "").lower()

        # Color keywords
        for color in ["red", "blue", "green", "black", "white", "pink",
                       "yellow", "orange", "purple", "brown", "grey",
                       "navy", "maroon", "beige"]:
            if color in q:
                filters["color_family"] = color
                filters["primary_color_name"] = color.title()
                break

        # Gender keywords
        if any(w in q for w in ["for men", "men's", "male", " man "]):
            filters["gender"] = "men"
        elif any(w in q for w in ["for women", "women's", "female", " woman "]):
            filters["gender"] = "women"

        # Style keywords
        for style in ["casual", "formal", "sports", "party", "ethnic"]:
            if style in q:
                filters["style"] = style
                break

        # Sleeve keywords
        sleeve_map = {
            "full sleeve": "Full Sleeves", "long sleeve": "Full Sleeves",
            "half sleeve": "Half Sleeves", "short sleeve": "Short Sleeves",
            "sleeveless": "Sleeveless", "three quarter": "Three-Quarter Sleeves",
        }
        for kw, val in sleeve_map.items():
            if kw in q:
                filters["sleeve_value"] = val
                break

        # Pattern keywords
        pattern_map = {
            "solid": "Solid", "striped": "Striped", "checked": "Checked",
            "printed": "Printed", "floral": "Floral", "polka": "Polka Dots",
        }
        for kw, val in pattern_map.items():
            if kw in q:
                filters["pattern_value"] = val
                break

        # Price extraction
        price_max = None
        match = re.search(r'(?:under|below|less than|<\s*)\s*(\d+)', q)
        if match:
            try:
                price_max = int(match.group(1))
            except ValueError:
                pass

        return {
            "filters": filters,
            "price_max": price_max,
            "reasoning": "Keyword extraction fallback (LLM unavailable)",
            "confidence": 0.3,
            "source": "fallback"
        }


# ---- Singleton ----
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
