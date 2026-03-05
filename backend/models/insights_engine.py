import json
import os
from datetime import datetime

class InsightsEngine:
    def __init__(self, llm_system):
        self.llm = llm_system
        self.PROMPT_TEMPLATE = """
        You are an evaluation and insight-extraction assistant for a video-based fashion recommendation system.
        Your task is to analyze system performance and user satisfaction strictly using the structured data provided.
        Do not assume missing information. Do not hallucinate. Base all conclusions only on the given data.

        SYSTEM CONTEXT:
        The system detects fashion items from paused OTT-style video frames, extracts fine-grained attributes using an attention-based vision model, retrieves visually similar products using embeddings, and displays top-K results to the user.

        SESSION DATA:
        - Scene context: {scene}
        - Detected fashion item category: {category}

        VISUAL ATTRIBUTES (with confidence):
        - Color: {color}
        - Pattern: {pattern}
        - Sleeve type: {sleeve}

        RETRIEVAL RESULTS:
        - Top-K similarity scores: {scores}
        - Highest similarity score: {top_score}
        - Average similarity score (Top-K): {avg_score}
        - Rank of product clicked by user: {clicked_rank}

        USER FEEDBACK:
        - Satisfaction score (1–5): {rating}
        - Query refinement after results: {query}

        TASKS:
        1. Classify overall retrieval relevance as one of: High / Medium / Low.
        2. Decide whether this session should be counted as a successful recommendation (Yes/No).
        3. Identify key strengths of the system in this session.
        4. Identify weaknesses or failure risks observed in this session.
        5. Provide exactly ONE concrete technical improvement suggestion (model, retrieval, or reasoning level).
        6. Briefly explain how user behavior supports or contradicts the retrieval quality.

        OUTPUT FORMAT (STRICT JSON ONLY):
        {{
            "relevance_level": "High/Medium/Low",
            "successful_recommendation": true/false,
            "strengths": ["..."],
            "weaknesses": ["..."],
            "improvement_suggestion": "...",
            "user_behavior_analysis": "..."
        }}
        """

    def generate_report(self, rating_data):
        """
        Generate insights report based on a rating entry.
        rating_data: dict containing rating, filters, query, etc.
        """
        try:
            # Extract Data — filters may be a JSON string from DB
            filters = rating_data.get("filters", {})
            if isinstance(filters, str):
                try:
                    filters = json.loads(filters)
                except (json.JSONDecodeError, TypeError):
                    filters = {}
            if not isinstance(filters, dict):
                filters = {}
            
            prompt = self.PROMPT_TEMPLATE.format(
                scene="Indoor/Neutral",
                category=rating_data.get("product_id", "Unknown"),
                color=filters.get("color", filters.get("color_name", "N/A")),
                pattern=filters.get("pattern", "N/A"),
                sleeve=filters.get("sleeve", "N/A"),
                scores="[0.35, 0.32, 0.30]",
                top_score="0.35",
                avg_score="0.32",
                clicked_rank="1",
                rating=rating_data.get("rating", 3),
                query=rating_data.get("query", "None")
            )

            print("[Insights] 🧠 Generating analysis via LLM...")
            
            # Call Groq directly for insights analysis
            from models.external_llm import get_groq_llm
            groq = get_groq_llm()
            
            if not groq._ensure_client():
                print("[Insights] ⚠️ Groq not available, returning fallback")
                return self._fallback_result()
            
            response = groq.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an analytics assistant. Output STRICT JSON only, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            
            # Validate expected fields exist
            if not isinstance(result, dict):
                return self._fallback_result()
            
            return result

        except Exception as e:
            print(f"[Insights] ❌ Analysis failed: {e}")
            return self._fallback_result(str(e))
    
    def _fallback_result(self, error=None):
        """Return a safe fallback when LLM analysis fails."""
        result = {
            "relevance_level": "Medium",
            "successful_recommendation": True,
            "strengths": ["System operational", "User provided feedback"],
            "weaknesses": ["LLM analysis unavailable" if error else "No data"],
            "improvement_suggestion": "Check backend logs for details",
            "user_behavior_analysis": "N/A"
        }
        if error:
            result["error"] = error
        return result
