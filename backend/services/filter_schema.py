"""
Filter Schema Service -- Dynamic Allowed Values from DB
========================================================
Loads SELECT DISTINCT for each filterable field from visual_attributes
at startup. Provides:
  - prompt_block(): render allowed values for LLM prompt injection
  - validate(): drop any LLM value not in allowed set
  - price_to_bucket(): deterministic price normalization
  - same_category_group(): check if two categories are in same group

NOTE: filter_schema.init() runs ONCE at startup.
      If DB values change, service restart is required.
"""

import os
import psycopg2

# Centralized DB config (Supabase in production, local fallback)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DB_CONFIG

# ---- Category Groups (same as retrieval_validation.py) ----
CATEGORY_GROUPS = {
    "upper_wear":  {"shirts", "tshirt", "blazer", "Jacket"},
    "lower_wear":  {"pant", "shorts", "skirt"},
    "jewelry":     {"earrings", "necklace"},
    "footwear":    {"Footwear_sandals", "Footwear_shoes"},
    "accessories": {"belt", "tie", "caps", "glasses", "watch"},
    "traditional": {"dhoti", "churidhar"},
}

# Reverse lookup: category -> group name
_CAT_TO_GROUP = {}
for _grp, _cats in CATEGORY_GROUPS.items():
    for _c in _cats:
        _CAT_TO_GROUP[_c] = _grp

# ---- Price bucket boundaries (INR) ----
PRICE_BOUNDARIES = [
    (500, "budget"),      # < 500
    (1500, "mid"),        # 500-1499
    (5000, "premium"),    # 1500-4999
    (float("inf"), "luxury"),  # 5000+
]

# Price keywords -> bucket
PRICE_KEYWORDS = {
    "cheap": "budget", "affordable": "budget", "low price": "budget", "budget": "budget",
    "mid": "mid", "moderate": "mid",
    "premium": "premium", "expensive": "premium", "high end": "premium",
    "luxury": "luxury", "designer": "luxury",
}

# Fields that retrieval_service accepts as filters
FILTERABLE_FIELDS = {
    "category", "gender", "style", "material", "price_bucket", "color_family",
    "sleeve_value", "pattern_value", "primary_color_name",
}


class FilterSchema:
    """Dynamic filter schema loaded from DB at startup."""

    def __init__(self):
        self._initialized = False
        self.allowed = {}  # field -> frozenset of allowed values

    @property
    def is_ready(self):
        return self._initialized

    def init(self):
        """Load distinct values for each filterable field from visual_attributes."""
        if self._initialized:
            print("[FilterSchema] Already initialized, skipping.")
            return

        print("[FilterSchema] Loading allowed values from DB...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        field_columns = {
            "category": "category",
            "gender": "gender",
            "style": "style",
            "material": "material",
            "price_bucket": "price_bucket",
            "color_family": "color_family",
            "sleeve_value": "sleeve_value",
            "pattern_value": "pattern_value",
            "primary_color_name": "primary_color_name",
        }

        for field, col in field_columns.items():
            cur.execute(f"SELECT DISTINCT {col} FROM visual_attributes WHERE {col} IS NOT NULL")
            values = set(row[0] for row in cur.fetchall())
            
            # ✅ FIX: Hardcode the canonical values we teach the LLM in prompts
            # If the DB currently has 0 products of a certain type, it won't be in the 
            # SELECT DISTINCT, causing FilterSchema to reject perfectly valid LLM output.
            if field == "sleeve_value":
                values.update(["long", "short", "half", "three_quarter", "sleeveless"])
            elif field == "pattern_value":
                values.update(["solid", "striped", "checked", "printed", "floral"])
                
            self.allowed[field] = frozenset(values)
            print(f"  {field}: {len(values)} values")

        cur.close()
        conn.close()

        self._initialized = True
        print("[FilterSchema] READY")

    # ---- Prompt Block ----
    # Values to exclude from LLM prompt (they exist in DB but shouldn't be user-facing)
    _EXCLUDED_VALUES = {"not_applicable", "Not Applicable", "not applicable"}

    def prompt_block(self):
        """Render allowed values for LLM prompt injection."""
        if not self._initialized:
            return ""

        lines = ["ALLOWED FILTER VALUES (output EXACT strings only):"]
        for field, values in sorted(self.allowed.items()):
            # Exclude not_applicable from the list shown to LLM
            filtered = sorted(v for v in values if v.lower().replace(" ", "_") != "not_applicable")
            lines.append(f"  {field}: {', '.join(filtered)}")
        return "\n".join(lines)

    # ---- Validation ----
    def validate(self, filters):
        """
        Validate LLM output filters against allowed values.
        
        Rules:
          - If value not in allowed set -> set to None, log warning
          - If field not in FILTERABLE_FIELDS -> drop it
          - If any value is a dict/list (nested) -> discard ENTIRE filter object
          - Never crash on malformed input
        
        Returns: cleaned dict with only valid filters
        """
        if not isinstance(filters, dict):
            print(f"  [FilterSchema] WARN: filters is not dict, got {type(filters)}, returning empty")
            return {}

        # Reject nested structures
        for k, v in filters.items():
            if isinstance(v, (dict, list)):
                print(f"  [FilterSchema] WARN: nested value in '{k}', discarding entire filter object")
                return {}

        cleaned = {}
        for field, value in filters.items():
            if field not in FILTERABLE_FIELDS:
                continue
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                print(f"  [FilterSchema] WARN: {field}={value} is not string, dropping")
                continue

            # Guard: never allow "not_applicable" for sleeve or pattern
            if field in ("sleeve_value", "pattern_value") and value.lower().replace(" ", "_") == "not_applicable":
                print(f"  [FilterSchema] BLOCKED: {field}='{value}' — not_applicable is never valid for user queries")
                continue

            # Case-sensitive match against DB values
            if value in self.allowed.get(field, set()):
                cleaned[field] = value
            else:
                # Try case-insensitive fallback
                match = None
                for allowed_val in self.allowed.get(field, set()):
                    if allowed_val.lower() == value.lower():
                        match = allowed_val
                        break
                if match:
                    cleaned[field] = match
                    print(f"  [FilterSchema] INFO: {field}='{value}' -> case-corrected to '{match}'")
                else:
                    print(f"  [FilterSchema] WARN: {field}='{value}' not in allowed values, dropping")

        return cleaned

    # ---- Price Mapping ----
    @staticmethod
    def price_to_bucket(price_value):
        """
        Deterministic price -> bucket mapping.
        
        Args:
            price_value: numeric price (int/float) or price keyword string
        
        Returns:
            bucket string or None
        """
        if price_value is None:
            return None

        # String keyword
        if isinstance(price_value, str):
            kw = price_value.lower().strip()
            return PRICE_KEYWORDS.get(kw)

        # Numeric
        try:
            price = float(price_value)
            for boundary, bucket in PRICE_BOUNDARIES:
                if price < boundary:
                    return bucket
            return "luxury"
        except (ValueError, TypeError):
            return None

    # ---- Category Group Check ----
    @staticmethod
    def same_category_group(cat_a, cat_b):
        """Check if two categories belong to the same semantic group."""
        if not cat_a or not cat_b:
            return False
        grp_a = _CAT_TO_GROUP.get(cat_a)
        grp_b = _CAT_TO_GROUP.get(cat_b)
        if grp_a is None or grp_b is None:
            return False
        return grp_a == grp_b

    # ---- Debug Info ----
    def get_debug_info(self):
        """Return all loaded allowed values for debugging."""
        return {
            field: sorted(values)
            for field, values in self.allowed.items()
        }


# ---- Singleton ----
filter_schema = FilterSchema()
