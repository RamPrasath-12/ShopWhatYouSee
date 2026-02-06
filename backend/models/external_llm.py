"""
External LLM Module - Groq API Fallback

Uses Groq's Llama 3.3 70B model as fallback when local LLM
has low confidence or fails to load.

Groq is FREE and very fast (lowest latency inference).
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Best for structured JSON output
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"  # Faster fallback

# System prompt optimized for JSON output
SYSTEM_PROMPT = """You are a fashion product filter assistant. 

Given:
- Category: The detected product category
- Attributes: Currently detected visual attributes (color, pattern, sleeve, etc.)
- Scene: The detected scene/context
- Session History: Previous queries in this conversation
- Query: The user's current request

Your task: Generate search filters based on the user's query.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "reasoning": "Brief explanation of your filter choices",
  "filters": {
    "attribute_name": "new_value"
  },
  "confidence": 0.9,
  "preserved": ["list", "of", "unchanged", "attributes"]
}

Rules:
1. Only include filters that need to CHANGE based on the query
2. Don't repeat unchanged attributes in filters
3. Use confidence 0.8-0.95 for clear requests, 0.5-0.7 for ambiguous ones
4. For price queries: use "price_max": number (e.g., "price_max": 1500)
5. **For color: If user says "red", use "color": "red" (simple color name)**
6. **Extract color variations: "bright red" → "red", "dark blue" → "navy"**
7. Common colors: red, blue, green, black, white, pink, yellow, navy, maroon, beige, brown, grey

Respond with JSON only, no markdown code blocks or explanations."""


class GroqLLM:
    """External LLM using Groq API."""
    
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
            logger.info("✅ Groq client initialized")
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
        Generate product filters using Groq API.
        
        Returns:
            {
                "reasoning": "explanation",
                "filters": {"color": "red", ...},
                "confidence": 0.9,
                "preserved": ["pattern", "sleeve"]
            }
        """
        if not self._ensure_client():
            return self._fallback_response(query)
        
        # Build user message
        user_message = f"""Category: {category}
Attributes: {json.dumps(attributes)}
Scene: {scene}
Session History: {json.dumps(session_history or [])}
Query: {query}"""

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            
            # Extract response
            content = response.choices[0].message.content.strip()
            logger.debug(f"Groq response: {content}")
            
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Try fallback model
            return self._try_fallback_model(user_message)
    
    def _try_fallback_model(self, user_message: str) -> Dict[str, Any]:
        """Try with lighter fallback model if main model fails."""
        try:
            response = self.client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            return self._fallback_response("")
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse Groq response to standard format."""
        try:
            # Clean up response - remove markdown code blocks if present
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                logger.warning(f"No JSON in Groq response: {content[:200]}")
                return {"filters": {}, "confidence": 0.5, "reasoning": content}
            
            data = json.loads(json_match.group())
            
            # Normalize format
            return {
                "reasoning": data.get("reasoning", ""),
                "filters": data.get("filters", {}),
                "confidence": float(data.get("confidence", 0.85)),
                "preserved": data.get("preserved", []),
                "source": "groq"
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error from Groq: {e}")
            return {"filters": {}, "confidence": 0.5, "reasoning": content, "source": "groq"}
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Basic fallback when API fails."""
        return {
            "reasoning": "External LLM unavailable, using basic extraction",
            "filters": {},
            "confidence": 0.3,
            "preserved": [],
            "source": "fallback"
        }


# Singleton instance
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
