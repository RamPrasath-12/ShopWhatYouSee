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
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # or Seq2Seq if you used that
# if you're using flan-t5: use AutoModelForSeq2SeqLM and Tokenizer; if qwen causal: use AutoModelForCausalLM
# adjust imports to match the model you actually loaded in your environment.

class LLMReasoner:
    def __init__(self, model_name="google/flan-t5-small", cache_dir="D:/huggingface_cache"):
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        self.model_name = model_name
        print(f"[LLMReasoner] Loading model: {self.model_name}")
        # If your model is Seq2Seq (flan-t5)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # try seq2seq first (works for flan-t5)
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto")
            self.model_type = "seq2seq"
        except Exception:
            # fallback: load causal (for qwen-like)
            from transformers import AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, device_map="auto")
            self.model_type = "causal"

        if not torch.cuda.is_available():
            # ensure model on CPU to avoid device issues
            self.model.to("cpu")

        print("[LLMReasoner] Model loaded (or attempted). model_type:", self.model_type)

        # small dictionaries for rule-based fallback
        self.color_words = set([
            "red","blue","green","black","white","yellow","brown","pink","purple","orange","gray","grey","beige","navy"
        ])
        self.style_words = set(["formal","casual","party","ethnic","sports","business","smart","street"])
        self.pattern_words = set(["striped","checked","checked","plaid","solid","patterned","floral"])
        self.sleeve_words = {
            "short": ["short","shortsleeve","short-sleeve"],
            "long": ["long","longsleeve","long-sleeve"],
            "three_quarter": ["three_quarter","three-quarter","3/4","three quarter"]
        }

    def hex_to_color_name(self, hexcode):
        # simple approximate mapping: returns a basic family name
        if not hexcode or len(hexcode) < 7:
            return None
        try:
            r = int(hexcode[1:3], 16)
            g = int(hexcode[3:5], 16)
            b = int(hexcode[5:7], 16)
        except:
            return None
        if r > g and r > b: return "red"
        if g > r and g > b: return "green"
        if b > r and b > g: return "blue"
        if r > 200 and g > 200 and b < 150: return "yellow"
        if r > 120 and g > 80 and b < 60: return "brown"
        return "neutral"

    def clean_json(self, text):
        if not text:
            return None
        text = text.replace("```json", "").replace("```", "").strip()
        # find first JSON block
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            # sometimes the model writes keys without quotes; try to patch naive mistakes
            s = m.group(0)
            # add quotes around property names (very naive)
            s2 = re.sub(r'(\b[a-zA-Z_]+\b)\s*:', r'"\1":', s)
            try:
                return json.loads(s2)
            except Exception:
                return None

    # ---------- rule-based fallback ----------
    def rule_based_filters(self, item, scene_label, user_query):
        # item: dict containing attributes (color_hex, pattern, sleeve_length)
        # returns dictionary with filters
        color = None
        if item.get("color_hex"):
            color = self.hex_to_color_name(item.get("color_hex"))
        # attempt to get color from user_query first if explicit
        if user_query:
            q = user_query.lower()
            # color words
            for cw in self.color_words:
                if re.search(r'\b' + re.escape(cw) + r'\b', q):
                    color = cw
                    break

        pattern = item.get("pattern") or None
        if user_query:
            for p in self.pattern_words:
                if re.search(r'\b' + re.escape(p) + r'\b', user_query.lower()):
                    pattern = p
                    break

        sleeve = item.get("sleeve_length") or None
        if user_query:
            for k,vlist in self.sleeve_words.items():
                for token in vlist:
                    if token in user_query.lower():
                        sleeve = k
                        break
                if sleeve:
                    break

        # style detection
        style = None
        if user_query:
            for sw in self.style_words:
                if re.search(r'\b' + re.escape(sw) + r'\b', user_query.lower()):
                    style = sw
                    break
        # price extraction
        price_max = None
        if user_query:
            # common patterns: "under 1000", "less than 1000", "below 1000", "price < 1000"
            m = re.search(r'(?:under|less than|below|<|price under|price less than)\s*₹?\s*([0-9]{2,6})', user_query.lower())
            if not m:
                # match a bare number if prefixed by "less" or "under"
                m = re.search(r'(?:under|less than|below)\s*([0-9]{2,6})', user_query.lower())
            if m:
                try:
                    price_max = float(m.group(1))
                except:
                    price_max = None
            else:
                # also match "1000 rupees" or "1000"
                m2 = re.search(r'([0-9]{2,6})', user_query)
                if m2:
                    # only accept if user used "price", "rupee", "₹", "under", etc.
                    if re.search(r'price|₹|rupee|rs|under|less than|below', user_query.lower()):
                        try:
                            price_max = float(m2.group(1))
                        except:
                            price_max = None

        # scene-based adjustments (example): if scene suggests "beach", deprioritize heavy coats — we keep as context only
        # for fallback, we won't change style automatically but you can map scene to style suggestions:
        if not style and scene_label:
            if scene_label.lower() in ("beach","outdoor","pool"):
                # if user asked 'formal' but scene is beach, you might warn — but here keep style None
                pass

        return {
            "filters": {
                "color": color,
                "pattern": pattern,
                "sleeve_length": sleeve,
                "style": style,
                "price_max": price_max
            }
        }

    # ---------- main: try LLM then fallback ----------
    def generate_filters(self, item, scene_label, user_query):
        # Build minimal prompt (avoid very long prompt)
        color_w = self.hex_to_color_name(item.get("color_hex", "")) or "unknown"
        pattern = item.get("pattern") or "unknown"
        sleeve = item.get("sleeve_length") or "unknown"

        prompt = (
            f"Item color: {color_w}\n"
            f"Pattern: {pattern}\n"
            f"Sleeve: {sleeve}\n"
            f"Scene: {scene_label}\n"
            f"User request: {user_query}\n\n"
            "Return ONLY valid JSON with keys: filters -> color, pattern, sleeve_length, style, price_max. "
            "If nothing, use null."
        )

        try:
            # prepare tokens depending on model type
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                if getattr(self, "model_type", None) == "seq2seq":
                    outputs = self.model.generate(**inputs, max_new_tokens=80, do_sample=False, num_beams=2)
                    raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # causal LM – produce by just encoding prompt and generating
                    input_ids = inputs["input_ids"]
                    outputs = self.model.generate(input_ids=input_ids, max_new_tokens=80, do_sample=False, temperature=0.1)
                    raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw = raw.strip()
            print("[LLMReasoner] RAW LLM OUTPUT:\n", raw)
            parsed = self.clean_json(raw)
            # if parsed is None or has all nulls: run fallback
            if parsed is None:
                print("[LLMReasoner] LLM parsing failed, using rule-based fallback.")
                return self.rule_based_filters(item, scene_label, user_query)
            # validate parsed structure
            if "filters" not in parsed or not isinstance(parsed["filters"], dict):
                print("[LLMReasoner] LLM output missing filters key; fallback.")
                return self.rule_based_filters(item, scene_label, user_query)
            # check if all are null
            f = parsed["filters"]
            non_null_exists = any(v is not None for v in f.values())
            if not non_null_exists:
                print("[LLMReasoner] LLM returned all nulls; fallback.")
                return self.rule_based_filters(item, scene_label, user_query)
            return parsed
        except Exception as e:
            print("[LLMReasoner] Exception during LLM call:", e)
            print("[LLMReasoner] Using rule-based fallback.")
            return self.rule_based_filters(item, scene_label, user_query)
