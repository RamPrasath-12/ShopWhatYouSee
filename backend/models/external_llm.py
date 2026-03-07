"""
External LLM Module - Groq API (Phase 2)
==========================================
Uses Groq's Llama 3.3 70B for structured filter generation.
Outputs FLAT filters matching retrieval_service fields exactly.

Pipeline order per query:
  raw_llm_json -> price normalization -> filter_schema.validate()
"""

import os
import re
import json
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_FALLBACK_MODEL = "llama3-8b-8192"


class GroqLLM:
    """External LLM using Groq API — flat filter output only."""

    def __init__(self):
        self.client = None
        self._initialized = False

    def _ensure_client(self) -> bool:
        """Lazy initialization of Groq client."""
        if self._initialized:
            return self.client is not None

        self._initialized = True

        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found in environment")
            return False

        try:
            from groq import Groq
            self.client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized")
            return True
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return False

    def generate_filters(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate flat product filters using Groq API.

        Returns:
            {
                "filters": {"category": "shirts", "gender": "men", ...},
                "reasoning": "...",
                "confidence": 0.85,
                "source": "groq"
            }
        """
        if not self._ensure_client():
            return self._fallback_response(query)

        # Build system prompt with dynamic allowed values
        from services.filter_schema import filter_schema
        from models.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE_TEXT_ONLY

        system_prompt = SYSTEM_PROMPT.format(
            allowed_values=filter_schema.prompt_block()
        )

        # Choose template based on whether visual context is available
        if category or attributes:
            user_message = USER_PROMPT_TEMPLATE.format(
                category=category or "unknown",
                agman_attributes=json.dumps(attributes or {}),
                scene=scene or "unknown",
                user_query=query or ""
            )
        else:
            user_message = USER_PROMPT_TEMPLATE_TEXT_ONLY.format(
                user_query=query or ""
            )

        # ---- Attempt 1 ----
        result = self._call_groq(system_prompt, user_message, GROQ_MODEL)
        if result is not None:
            return result

        # ---- Retry (attempt 2) on parse failure ----
        print("  [GroqLLM] First attempt failed, retrying...")
        result = self._call_groq(system_prompt, user_message, GROQ_MODEL)
        if result is not None:
            return result

        # ---- Fallback model ----
        print("  [GroqLLM] Main model failed, trying fallback model...")
        result = self._call_groq(system_prompt, user_message, GROQ_FALLBACK_MODEL)
        if result is not None:
            return result

        # ---- Ultimate fallback ----
        print("  [GroqLLM] All attempts failed, returning empty filters")
        return self._fallback_response(query)

    def _call_groq(self, system_prompt: str, user_message: str, model: str):
        """
        Call Groq API with strict JSON mode.
        Returns parsed result dict or None on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,  # Deterministic output
                max_tokens=500,
                response_format={"type": "json_object"},  # Strict JSON mode
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"Groq response ({model}): {content}")

            return self._parse_response(content)

        except Exception as e:
            logger.error(f"Groq API error ({model}): {e}")
            return None

    # Canonical key mapping — LLM may use DB names or display names
    CANONICAL_KEYS = {
        "color_family": "color", "primary_color_name": "color",
        "sleeve_value": "sleeve", "sleeve_length": "sleeve",
        "pattern_value": "pattern",
    }

    def _canonicalize_key(self, key):
        """Map LLM/DB filter key names to canonical internal names."""
        return self.CANONICAL_KEYS.get(key, key)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse Groq JSON response into add/remove/reset_to_visual structure.
        
        Handles both new schema ({add, remove, reset_to_visual}) and
        legacy flat schema ({category, color_family, ...}) for backward compat.
        
        Pipeline: parse -> detect schema -> canonicalize -> return
        (validation done externally in app.py)
        """
        try:
            # Clean markdown fences if any (shouldn't happen with json_object mode)
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                logger.warning(f"No JSON found in response: {content[:200]}")
                return None

            data = json.loads(json_match.group())

            # ---- Detect schema: new (add/remove) vs legacy (flat) ----
            if "add" in data or "remove" in data or "reset_to_visual" in data:
                # NEW SCHEMA: add/remove/reset_to_visual
                raw_add = data.get("add", {}) or {}
                raw_remove = data.get("remove", []) or []
                reset_to_visual = bool(data.get("reset_to_visual", False))

                # Clean null/empty values from add
                add_filters = {}
                for field, val in raw_add.items():
                    if val is not None and val != "" and isinstance(val, str):
                        add_filters[field] = val

                # Ensure remove is a list of strings
                remove_keys = [k for k in raw_remove if isinstance(k, str)]

                return {
                    "add": add_filters,
                    "remove": remove_keys,
                    "reset_to_visual": reset_to_visual,
                    "price_max": data.get("price_max"),
                    "reasoning": data.get("reasoning", ""),
                    "confidence": data.get("confidence", 0.8),
                    "source": "groq"
                }
            else:
                # LEGACY SCHEMA: flat {category, color_family, ...}
                # Convert to new format: treat all fields as "add"
                add_filters = {}
                for field in ("category", "gender", "style", "material",
                              "color_family", "price_bucket",
                              "sleeve_value", "pattern_value", "primary_color_name"):
                    val = data.get(field)
                    if val is not None and val != "" and isinstance(val, str):
                        add_filters[field] = val

                return {
                    "add": add_filters,
                    "remove": [],
                    "reset_to_visual": False,
                    "price_max": data.get("price_max"),
                    "reasoning": data.get("reasoning", ""),
                    "confidence": data.get("confidence", 0.8),
                    "source": "groq"
                }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None

    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Empty filters when all LLM attempts fail."""
        return {
            "add": {},
            "remove": [],
            "reset_to_visual": False,
            "price_max": None,
            "reasoning": "LLM unavailable, returning empty filters",
            "confidence": 0.0,
            "source": "fallback"
        }


# ---- Singleton ----
_groq_instance = None

def get_groq_llm() -> GroqLLM:
    """Get singleton Groq LLM instance."""
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqLLM()
    return _groq_instance


def generate_filters_external(
    category: str,
    attributes: Dict[str, Any],
    scene: str,
    query: str,
    session_history: List[Dict] = None
) -> Dict[str, Any]:
    """Convenience function for external LLM filter generation."""
    return get_groq_llm().generate_filters(
        category=category,
        attributes=attributes,
        scene=scene,
        query=query,
        session_history=session_history
    )
