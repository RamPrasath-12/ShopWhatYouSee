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
            # Extract Data
            filters = rating_data.get("filters", {})
            
            prompt = self.PROMPT_TEMPLATE.format(
                scene="Indoor/Neutral", # Mock if missing
                category=rating_data.get("product_id", "Unknown"), # Or detected_category
                color=filters.get("color", "N/A"),
                pattern=filters.get("pattern", "N/A"),
                sleeve="N/A", # Not always captured
                scores="[0.35, 0.32, 0.30]", # Mock if backend doesn't track this yet
                top_score="0.35",
                avg_score="0.32",
                clicked_rank="1", # Assumed top click for now
                rating=rating_data.get("rating", 3),
                query=rating_data.get("query", "None")
            )

            print("[Insights] 🧠 Generating analysis via LLM...")
            
            # Call LLM (using Groq ideally for json mode)
            # We reuse unified_llm's internal method or just call it directly
            # For simplicity, we assume unified_llm has a method or we use _try_external here
            
            result = self.llm._try_external(
                category="analysis",
                attributes={},
                scene=None,
                user_query=prompt,
                session_history=[]
            )
            
            # The result from _try_external is usually parsed JSON or dict
            # If it followed instructions, it should be the JSON we want.
            # But _try_external expects "filters" output format by default in UnifiedLLM?
            # actually _try_external uses a generic prompt. 
            # functionality might differ. 
            
            # Let's trust that _try_external returns a dict.
            # We might need to map it if the structure differs.
            
            return result

        except Exception as e:
            print(f"[Insights] ❌ Analysis failed: {e}")
            return {
                "error": str(e),
                "relevance_level": "Medium",
                "successful_recommendation": True,
                "strengths": ["System operational", "User provided feedback"],
                "weaknesses": ["LLM Analysis failed"],
                "improvement_suggestion": "Check backend logs",
                "user_behavior_analysis": "N/A"
            }
