"""
Local LLM Module - Fine-tuned Phi-3 for Fashion Filter Generation

This module loads and runs the fine-tuned Phi-3 model for generating
product filters based on user queries and detected attributes.
"""

import os
import re
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "data" / "llm"
ADAPTER_PATH = MODEL_DIR / "phi3-adapters"
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# System prompt for the LLM
SYSTEM_PROMPT = """You are a fashion product filter assistant. Given product attributes and user query, 
generate search filters with reasoning. Output valid JSON only."""


class LocalLLM:
    """Local fine-tuned LLM for fashion filter generation."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _loaded = False
    
    def __new__(cls):
        """Singleton pattern - only load model once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self) -> bool:
        """Load the fine-tuned model. Returns True if successful."""
        if self._loaded:
            return True
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import gc
            
            logger.info("Loading Phi-3 base model with memory optimization...")
            
            # Check if adapters exist
            if not ADAPTER_PATH.exists():
                logger.error(f"Adapter not found at {ADAPTER_PATH}")
                logger.info("Please extract phi3-adapters.zip to backend/data/llm/phi3-adapters/")
                return False
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try 4-bit quantization first (works on Linux/CUDA), fall back to float16
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization for lower memory")
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                )
            except Exception as quant_error:
                logger.warning(f"4-bit quantization failed ({quant_error}), using float16")
                # Fallback to float16 (works on Windows)
                self._model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            
            # Load fine-tuned adapters
            logger.info(f"Loading adapters from {ADAPTER_PATH}...")
            self._model = PeftModel.from_pretrained(
                self._model, 
                str(ADAPTER_PATH)
            )
            self._model.eval()
            
            self._loaded = True
            logger.info("✅ Local LLM loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.info("Will use external LLM as fallback")
            return False
    
    def unload(self):
        """Unload model to free memory (for lazy loading with YOLO)."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("LLM unloaded from memory")
    
    def generate_filters(
        self,
        category: str,
        attributes: Dict[str, Any],
        scene: str,
        query: str,
        session_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate product filters based on user query.
        
        Args:
            category: Detected product category (e.g., "tshirt")
            attributes: Detected attributes {"color": "blue", "pattern": "solid", ...}
            scene: Detected scene context (e.g., "casual street")
            query: User's natural language query
            session_history: Previous queries in this session
            
        Returns:
            {
                "reasoning": "explanation of filter choices",
                "filters": {"color": "red", ...},
                "confidence": 0.9,
                "preserved": ["pattern", "sleeve"]
            }
        """
        if not self._loaded:
            if not self.load():
                return self._fallback_response(query)
        
        # Build input prompt
        user_input = f"""Category: {category}
Attributes: {json.dumps(attributes)}
Scene: {scene}
Session History: {json.dumps(session_history or [])}
Query: {query}"""

        # Print scene context for visibility
        print(f"[LocalLLM] Scene in Prompt: {scene}")

        # Format as chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        try:
            # Apply chat template
            prompt = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode response (only new tokens)
            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse the response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(query)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to standard format."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning(f"No JSON found in response: {response[:200]}")
                return {"filters": {}, "confidence": 0.5, "reasoning": response}
            
            data = json.loads(json_match.group())
            
            # Normalize to standard format (handle different output formats)
            filters = data.get("filters") or data.get("changes") or {}
            reasoning = data.get("reason") or data.get("reasoning") or ""
            preserved = data.get("preserved", [])
            
            # Convert confidence_factors to numeric score
            conf_factors = str(data.get("confidence_factors", "")).lower()
            if "explicit" in conf_factors:
                confidence = 0.95
            elif "clear" in conf_factors or "direct" in conf_factors:
                confidence = 0.85
            else:
                confidence = 0.75
            
            return {
                "reasoning": reasoning,
                "filters": filters,
                "confidence": confidence,
                "preserved": preserved
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {"filters": {}, "confidence": 0.5, "reasoning": response}
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Fallback when LLM fails - basic keyword extraction."""
        filters = {}
        query_lower = query.lower()
        
        # Basic color extraction
        colors = ["red", "blue", "green", "black", "white", "pink", "yellow", "purple", "orange", "brown", "grey", "gray", "navy", "beige"]
        for color in colors:
            if color in query_lower:
                filters["color"] = color
                break
        
        # Basic pattern extraction
        patterns = ["striped", "solid", "floral", "checked", "printed", "plain"]
        for pattern in patterns:
            if pattern in query_lower:
                filters["pattern"] = pattern
                break
        
        return {
            "reasoning": "Fallback: extracted keywords from query",
            "filters": filters,
            "confidence": 0.5,
            "preserved": []
        }


# Singleton instance
_llm_instance = None

def get_llm() -> LocalLLM:
    """Get the singleton LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance


def generate_filters(
    category: str,
    attributes: Dict[str, Any],
    scene: str,
    query: str,
    session_history: List[Dict] = None
) -> Dict[str, Any]:
    """Convenience function to generate filters."""
    return get_llm().generate_filters(
        category=category,
        attributes=attributes,
        scene=scene,
        query=query,
        session_history=session_history
    )
