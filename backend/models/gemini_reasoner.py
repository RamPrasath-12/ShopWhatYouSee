# backend/llm/gemini_reasoner.py

import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
import google.generativeai as genai  # Legacy fallback (not primary LLM)
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS
)
from models.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class GeminiReasoner:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    # -------------------------------
    # SAFE RESPONSE TEXT EXTRACTION
    # -------------------------------
    @staticmethod
    def _extract_text(response) -> str:
        if not response.candidates:
            raise RuntimeError("Gemini returned no candidates")

        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            raise RuntimeError(
                f"No content parts. finish_reason={candidate.finish_reason}"
            )

        texts = []
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)

        if not texts:
            raise RuntimeError(
                f"Empty text parts. finish_reason={candidate.finish_reason}"
            )

        return "\n".join(texts)

    # -------------------------------
    # SAFE JSON PARSER (REPAIR MODE)
    # -------------------------------
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()

        # Remove markdown fences
        if text.startswith("```"):
            text = text.split("```")[1].strip()

        # HARD SAFETY: auto-close JSON if truncated
        if text.count("{") > text.count("}"):
            text += "}" * (text.count("{") - text.count("}"))

        return json.loads(text)

    # -------------------------------
    # MAIN REASONING METHOD
    # -------------------------------
    def reason(self, category, agman_attributes, scene, user_query, history=None):
        if history is None:
            history = []
            
        history_text = ""
        if history:
            history_text = "\nCONVERSATION HISTORY:\n" + "\n".join([f"- {h}" for h in history])

        prompt = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE.format(
            category=category,
            agman_attributes=json.dumps(agman_attributes, indent=2),
            scene=scene,
            user_query=f"{history_text}\nCURRENT USER REQUEST: {user_query}"
        )
        
        print("\n" + "="*40)
        print("[LLM] SENDING PROMPT TO GEMINI")
        print("="*40)
        print(f"User Query: {user_query}")
        print(f"History Depth: {len(history)}")
        # print("Full Prompt:", prompt[:500] + "...")

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,          # CRITICAL
                    "max_output_tokens": 256     # PREVENT TRUNCATION
                }
            )

            raw_text = self._extract_text(response)
            
            print("\n" + "="*40)
            print("[LLM] GEMINI RESPONSE")
            print("="*40)
            print(raw_text)
            print("="*40 + "\n")
            
            return self._parse_json(raw_text)

        except Exception as e:
        # HARD FAIL SAFE
           return {
            "final_filters": {
                "category": category,
                "color_hex": agman_attributes.get("color_hex"),
                "pattern": agman_attributes.get("pattern"),
                "sleeve": agman_attributes.get("sleeve_length"),
                "fit": None,
                "formality": None,
                "occasion": None
            },
            "confidence": 0.0,
            "llm_failed": True,
            "llm_error": str(e)
        }
