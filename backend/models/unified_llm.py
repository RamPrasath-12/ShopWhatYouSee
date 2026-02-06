"""
Unified LLM Module - Smart Fallback between Local and External LLM

This module orchestrates between:
1. Local fine-tuned Phi-3 (primary) - Fast, no API costs
2. External Groq LLM (fallback) - When local confidence is low

Flow:
    Query → Check Cache → Local LLM (timeout 3s) → Groq
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Confidence threshold for falling back to external LLM
CONFIDENCE_THRESHOLD = 0.7


class UnifiedLLM:
    """
    Unified LLM that tries local first, falls back to external.
    """
    
    def __init__(self):
        self._local_llm = None
        self._external_llm = None
        self._local_available = None  # None = not checked yet
    
    def _get_local_llm(self):
        """Lazy load local LLM - uses efficient GGUF version."""
        if self._local_llm is None:
            from models.local_llm_efficient import LocalLLMEfficient
            self._local_llm = LocalLLMEfficient()
        return self._local_llm
    
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
        Generate product filters with smart fallback.
        
        Args:
            category: Detected product category
            attributes: Detected visual attributes
            scene: Scene context
            query: User's natural language query
            session_history: Previous queries in session
            prefer_external: If True, use external LLM directly
            
        Returns:
            {
                "reasoning": "explanation",
                "filters": {"color": "red", ...},
                "confidence": 0.9,
                "preserved": [...],
                "source": "local" | "external" | "fallback"
            }\
        """
        # ================================================
        # ITERATIVE LLM WITH SESSION HISTORY
        # ================================================
        history_str = ""
        if session_history and isinstance(session_history, list) and len(session_history) > 0:
            last_queries = session_history[-3:]  # Last 3 queries
            history_str = f"\n[History] Previous queries: {last_queries}"
            print(f"[UnifiedLLM] [INFO] Iterative mode: Using {len(last_queries)} previous queries")
        
        # ================================================
        # PHI-3 MINI - PRIMARY LLM (With Timeout)
        # ================================================
        print(f"\n[UnifiedLLM] Attempting Fine-tuned Phi-3 Mini (Primary LLM)...")
        
        import threading
        result = [None]
        error = [None]
        
        def run_local():
            try:
                result[0] = self._try_local(category, attributes, scene, query + history_str, session_history)
            except Exception as e:
                error[0] = e
        
        # Run Phi-3 with 8-second timeout (increased for better success rate)
        thread = threading.Thread(target=run_local)
        thread.start()
        thread.join(timeout=8.0)  # 8 second timeout - more time for Phi-3
        
        if thread.is_alive():
            print("[UnifiedLLM] [TIMEOUT] Phi-3 timeout (8s), falling back to Groq...")
        elif error[0]:
            print(f"[UnifiedLLM] [ERR] Phi-3 error: {error[0]}, falling back to Groq...")
        elif result[0]:
            confidence = result[0].get("confidence", 0)
            print(f"[UnifiedLLM] [OK] Phi-3 succeeded! Confidence: {confidence}")
            
            if confidence >= 0.5:  # Accept lower confidence since it's our main contribution
                result[0]["source"] = "phi3_finetuned"
                result[0]["iterative"] = len(session_history) > 0 if session_history else False
                return result[0]
            else:
                print(f"[UnifiedLLM] [WARN] Phi-3 confidence low ({confidence:.2f}), using Groq...")
        
        # ================================================
        # GROQ FALLBACK (Fast & Reliable)
        # ================================================
        print("[UnifiedLLM] Using Groq (Fast External LLM)...")
        
        groq_result = self._try_external(category, attributes, scene, query + history_str, session_history)
        groq_result["source"] = "groq"
        groq_result["iterative"] = len(session_history) > 0 if session_history else False
        return groq_result
        
        if result:
            result["source"] = "local"
            
            # Check confidence - if too low, try external
            confidence = result.get("confidence", 0)
            print(f"[UnifiedLLM] Local Confidence: {confidence}")
            
            if confidence < CONFIDENCE_THRESHOLD:
                print(f"[UnifiedLLM] [WARN] Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD}). Trying Groq fallback...")
                logger.info(f"Local LLM confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}, trying external")
                external_result = self._try_external(category, attributes, scene, query, session_history)
                
                # Use external if it has better confidence
                if external_result.get("confidence", 0) > confidence:
                    print(f"[UnifiedLLM] [INFO] Groq was better ({external_result.get('confidence', 0)}). Swapping.")
                    logger.info(f"External LLM has better confidence: {external_result.get('confidence', 0):.2f}")
                    return external_result
                else:
                    print("[UnifiedLLM] [OK] Keeping Local (Groq wasn't better).")
                    logger.info("Keeping local result (external not better)")
            
            return result
        
        # Local failed, use external
        print("[UnifiedLLM] 🔴 Local Phi-3 Failed/Unavailable. Switching to Groq.")
        logger.info("Local LLM failed, using external")
        return self._try_external(category, attributes, scene, query, session_history)
    
    def _try_local(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict]
    ) -> Dict[str, Any]:
        """Try local LLM, return None on failure."""
        try:
            local_llm = self._get_local_llm()
            
            # Check if local is available
            if self._local_available is False:
                return None
            
            result = local_llm.generate_filters(
                category=category,
                attributes=attributes,
                scene=scene,
                query=query,
                session_history=session_history
            )
            
            # Check if it's a real result or fallback
            if result.get("reasoning") == "Fallback: extracted keywords from query":
                self._local_available = False
                return None
            
            self._local_available = True
            return result
            
        except Exception as e:
            logger.warning(f"Local LLM error: {e}")
            self._local_available = False
            return None
    
    def _try_local_with_timeout(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict],
        timeout: float = 3.0
    ) -> Dict[str, Any]:
        """Try local LLM with timeout to prevent blocking."""
        import concurrent.futures
        import signal
        
        try:
            # Use ThreadPoolExecutor with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._try_local,
                    category, attributes, scene, query, session_history
                )
                result = future.result(timeout=timeout)
                return result
        except concurrent.futures.TimeoutError:
            print(f"[UnifiedLLM] ⏱️ Phi-3 timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"[UnifiedLLM] ❌ Phi-3 error: {e}")
            return None
    
    def _try_external(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict]
    ) -> Dict[str, Any]:
        """Try external LLM."""
        try:
            external_llm = self._get_external_llm()
            result = external_llm.generate_filters(
                category=category,
                attributes=attributes,
                scene=scene,
                query=query,
                session_history=session_history
            )
            result["source"] = "external"
            return result
            
        except Exception as e:
            logger.error(f"External LLM error: {e}")
            return self._ultimate_fallback(query)
    
    def _ultimate_fallback(self, query: str) -> Dict[str, Any]:
        """Ultimate fallback when both LLMs fail."""
        filters = {}
        query_lower = query.lower()
        
        # Extract basic keywords
        colors = ["red", "blue", "green", "black", "white", "pink", "yellow", 
                  "purple", "orange", "brown", "grey", "gray", "navy", "beige", "maroon"]
        for color in colors:
            if color in query_lower:
                filters["color"] = color
                break
        
        patterns = ["striped", "solid", "floral", "checked", "printed", "plain", "polka"]
        for pattern in patterns:
            if pattern in query_lower:
                filters["pattern"] = pattern
                break
        
        # Price keywords
        if any(word in query_lower for word in ["cheap", "budget", "affordable", "low price"]):
            filters["price_range"] = "budget"
        elif any(word in query_lower for word in ["expensive", "premium", "luxury", "high end"]):
            filters["price_range"] = "premium"
            
        # Regex for price (e.g. under 500, < 1000)
        import re
        price_match = re.search(r'(?:under|below|less than|<\s*)\s*(\d+)', query_lower)
        if price_match:
            try:
                filters['price_max'] = float(price_match.group(1))
            except:
                pass
        
        return {
            "reasoning": "Both LLMs unavailable, using keyword extraction",
            "filters": filters,
            "confidence": 0.3,
            "preserved": [],
            "source": "fallback"
        }


# Singleton instance
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
    """
    Main entry point for filter generation.
    
    Uses local LLM (Phi-3) first, falls back to external (Groq/Llama)
    if confidence is low or local fails.
    """
    return get_unified_llm().generate_filters(
        category=category,
        attributes=attributes,
        scene=scene,
        query=query,
        session_history=session_history,
        prefer_external=prefer_external
    )
