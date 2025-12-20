# # backend/models/llm_reasoner.py
# import os
# import json
# import re
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# class LLMReasoner:
#     def __init__(self):
#         os.environ["HF_HOME"] = "D:/huggingface_cache"
#         os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"

#         self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

#         print("Loading LLM model:", self.model_name)

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype="auto",
#             device_map="auto",
#             trust_remote_code=True
#         )

#     # ----------------------------------------------------
#     # HEX → Fashion Color Name
#     # ----------------------------------------------------
#     def hex_to_color_name(self, hexcode):
#         if not hexcode or not isinstance(hexcode, str) or len(hexcode) < 7:
#             return "unknown"

#         # Convert HEX → RGB
#         r = int(hexcode[1:3], 16)
#         g = int(hexcode[3:5], 16)
#         b = int(hexcode[5:7], 16)

#         brightness = (r + g + b) / 3
#         max_c = max(r, g, b)
#         min_c = min(r, g, b)

#         # ----------------------------------
#         # GRAYSCALE
#         # ----------------------------------
#         if max_c - min_c < 20:
#             if brightness < 40:
#                 return "black"
#             if brightness > 220:
#                 return "white"
#             return "gray"

#         # ----------------------------------
#         # RED FAMILY
#         # ----------------------------------
#         if r > g and r > b:
#             # Dark reds
#             if r > 120 and g < 40 and b < 40:
#                 if r < 90:
#                     return "maroon"
#                 if r < 130:
#                     return "burgundy"
#                 return "red"

#             # Coral / Peach
#             if r > 200 and 120 < g < 170 and b < 120:
#                 return "coral"
#             if r > 210 and 160 < g < 200 and b < 150:
#                 return "peach"

#             # Pink / Rose
#             if r > 200 and b > 150:
#                 if b > r * 0.8:
#                     return "rose"
#                 return "pink"

#             return "red"

#         # ----------------------------------
#         # ORANGE / YELLOW / GOLD
#         # ----------------------------------
#         if r > 180 and g > 100 and b < 80:
#             if r > 200 and g > 180:
#                 return "yellow"
#             if 150 < g < 180:
#                 return "gold"
#             return "orange"

#         # ----------------------------------
#         # GREEN FAMILY
#         # ----------------------------------
#         if g > r and g > b:
#             if g > 180 and r < 100 and b < 100:
#                 return "lime"
#             if g > 160 and r > 120 and b < 80:
#                 return "olive"
#             if g > 150 and b > 120:
#                 return "teal"
#             return "green"

#         # ----------------------------------
#         # BLUE FAMILY
#         # ----------------------------------
#         if b > r and b > g:
#             if b < 80:
#                 return "navy"
#             if b > 200 and g > 200:
#                 return "sky"
#             return "blue"

#         # ----------------------------------
#         # PURPLE / VIOLET / LAVENDER
#         # ----------------------------------
#         if b > 120 and r > 120:
#             if brightness < 120:
#                 return "violet"
#             if brightness < 180:
#                 return "purple"
#             return "lavender"

#         # ----------------------------------
#         # BROWN / BEIGE / CREAM / KHAKI
#         # ----------------------------------
#         if r > 120 and g > 100 and b < 70:
#             if brightness < 120:
#                 return "brown"
#             if brightness < 180:
#                 return "khaki"
#             if brightness < 220:
#                 return "beige"
#             return "cream"

#         return "gray"

#     # ----------------------------------------------------
#     # JSON CLEANING
#     # ----------------------------------------------------
#     def clean_json(self, text: str):
#         """
#         Extract a JSON-like structure from text and repair missing braces/keys.
#         """

#         # Try to extract {...}
#         text = text.strip()

#         # Repair missing outer braces
#         if not text.startswith("{"):
#             text = "{" + text
#         if not text.endswith("}"):
#             text = text + "}"

#         # Add missing "filters" wrapper if needed
#         if '"filters"' not in text:
#             text = '{"filters": ' + text[text.index("{") + 1:]

#         # Try to load valid JSON
#         try:
#             # Replace incorrect formatting like `"filters": "color":`
#             text = re.sub(r'"filters"\s*:\s*"color"', '"filters": {"color"', text)

#             # Ensure commas between fields
#             text = text.replace('""', '", "')
#             text = text.replace('": "', '": "')

#             j = json.loads(text)
#             return j

#         except Exception:
#             pass

#         # Fallback extraction:
#         color = re.search(r'"color"\s*:\s*"([^"]+)"', text)
#         pattern = re.search(r'"pattern"\s*:\s*"([^"]+)"', text)
#         sleeve = re.search(r'"sleeve_length"\s*:\s*"([^"]+)"', text)
#         style = re.search(r'"style"\s*:\s*"([^"]+)"', text)
#         price = re.search(r'([0-9]{2,6})', text)

#         return {
#             "filters": {
#                 "color": color.group(1) if color else None,
#                 "pattern": pattern.group(1) if pattern else None,
#                 "sleeve_length": sleeve.group(1) if sleeve else None,
#                 "style": style.group(1) if style else None,
#                 "price_max": float(price.group(1)) if price else None
#             }
#         }

#     # ----------------------------------------------------
#     # MAIN FUNCTION
#     # ----------------------------------------------------
#     def generate_filters(self, item, user_query):
#         color = self.hex_to_color_name(item.get("color_hex", "#000000"))
#         pattern = item.get("pattern", "solid")
#         sleeve = item.get("sleeve_length", "short")

#         instruction = f"""
# ### TASK
# You are a JSON-only fashion filter generator.
# You must return **only valid JSON**, nothing else.

# ### STRICT RULES
# - color → simple color word (red, blue, black, white, brown, pink, green, grey)
# - pattern → solid, striped, checked, patterned
# - sleeve_length → short, long, three_quarter
# - style → one simple word (formal, casual, party, ethnic, sports)
# - If user mentions price ("under 800", "less than 1000"), extract the NUMBER → price_max
# - price_max must be a number (no quotes)
# - If not mentioned → null

# ### RETURN FORMAT (MANDATORY)
# {{
#   "filters": {{
#     "color": null | "<color>",
#     "pattern": null | "<pattern>",
#     "sleeve_length": null | "<sleeve_length>",
#     "style": null | "<style>",
#     "price_max": null | <number>
#   }}
# }}

# ### ITEM ATTRIBUTES
# color: {color}
# pattern: {pattern}
# sleeve_length: {sleeve}

# ### USER REQUEST
# {user_query}

# ### OUTPUT JSON
# """


#         inputs = self.tokenizer(instruction, return_tensors="pt").to(self.model.device)

#         output_ids = self.model.generate(
#             **inputs,
#             max_new_tokens=200,
#             temperature=0.2,
#             do_sample=False,
#         )

#         raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         print("RAW LLM OUTPUT:", raw)

#         return self.clean_json(raw)



# backend/models/llm_reasoner.py
# backend/models/llm_reasoner.py

import re
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Canonical color space (limited, explainable)
# --------------------------------------------------
CANONICAL_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (34, 139, 34),
    "yellow": (255, 215, 0),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "grey": (128, 128, 128),
    "navy": (0, 0, 128),
    "beige": (245, 245, 220),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "cream": (255, 253, 208),
}

VALID_COLORS = set(CANONICAL_COLORS.keys())


# ==================================================
# LLM Reasoner
# ==================================================
class LLMReasoner:
    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to("cpu").eval()

        self.style_words = {
            "formal", "casual", "party", "office", "ethnic", "sports"
        }


    # --------------------------------------------------
    # HEX → nearest canonical color
    # --------------------------------------------------
    def hex_to_color(self, hexcode):
        if not hexcode or not isinstance(hexcode, str) or not hexcode.startswith("#"):
            return None

        try:
            r = int(hexcode[1:3], 16)
            g = int(hexcode[3:5], 16)
            b = int(hexcode[5:7], 16)
        except Exception:
            return None

        best_color = None
        min_dist = float("inf")

        for name, (cr, cg, cb) in CANONICAL_COLORS.items():
            dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_color = name

        # Hard cutoff → avoid incorrect black/white jumps
        if min_dist > 120:
            return "maroon"

        return best_color


    # --------------------------------------------------
    # Price extraction (rule-based, safe)
    # --------------------------------------------------
    def extract_price(self, query):
        if not query:
            return None

        m = re.search(r'(under|less than|below|<)\s*₹?\s*(\d+)', query.lower())
        if m:
            return int(m.group(2))

        return None


    # --------------------------------------------------
    # Minimal LLM call (ONLY for soft intent)
    # --------------------------------------------------
    def call_llm(self, query):
        if not query:
            return {}

        prompt = (
            "Extract ONLY style from the user request.\n"
            "Return valid JSON ONLY.\n\n"
            "Format:\n"
            "{\"style\": \"formal\" | \"casual\" | \"party\" | \"office\" | \"ethnic\" | \"sports\" | null}\n\n"
            f"User request: {query}"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=40)

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            return json.loads(re.search(r"\{.*\}", text, re.S).group())
        except Exception:
            return {}


    # ==================================================
    # MAIN FILTER GENERATION (CORE LOGIC)
    # ==================================================
    def generate_filters(self, item, scene, user_query):
        """
        Two explicit reasoning modes:

        MODE A — Auto Reasoning:
            Triggered when user_query is None
            Uses AG-MAN + Scene

        MODE B — Refinement Reasoning:
            Triggered when user_query exists
            AG-MAN is locked, user refines soft attributes
        """

        # -------------------------------
        # Base filters from AG-MAN
        # -------------------------------
        filters = {
            "color": self.hex_to_color(item.get("color_hex")),
            "pattern": item.get("pattern"),
            "sleeve_length": item.get("sleeve_length"),
            "style": None,
            "price_max": None
        }

        # ==================================================
        # MODE A — AUTO REASONING (NO USER QUERY)
        # ==================================================
        if not user_query:
            if scene and scene.get("confidence", 0) >= 0.6:
                scene_label = scene.get("scene_label", "").lower()

                if any(k in scene_label for k in ["arena", "performance", "outdoor"]):
                    filters["style"] = "casual"

                elif any(k in scene_label for k in ["office", "indoor"]):
                    filters["style"] = "formal"

            return {"filters": filters}


        # ==================================================
        # MODE B — USER REFINEMENT
        # ==================================================
        filters["price_max"] = self.extract_price(user_query)

        llm_out = self.call_llm(user_query)
        if llm_out.get("style") in self.style_words:
            filters["style"] = llm_out["style"]

        # Explicit color override only if user mentions it
        for color in VALID_COLORS:
            if re.search(rf"\b{color}\b", user_query.lower()):
                filters["color"] = color
                break

        return {"filters": filters}
