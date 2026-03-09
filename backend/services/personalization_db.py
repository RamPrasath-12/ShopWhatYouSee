"""
Personalization Database Layer
================================
Manages users, preferences, and purchase history tables in Supabase/PostgreSQL.
All CRUD operations for the personalization recommendation system.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from db_config import get_db_connection


# ─────────────────────────────────────────────────
# TABLE INITIALIZATION
# ─────────────────────────────────────────────────

def init_personalization_tables():
    """Create personalization tables if they don't exist."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                gender TEXT,
                size TEXT,
                budget_min REAL DEFAULT 0,
                budget_max REAL DEFAULT 99999,
                preferred_colors TEXT[] DEFAULT '{}',
                preferred_styles TEXT[] DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS purchase_history (
                purchase_id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
                product_id TEXT NOT NULL,
                category TEXT,
                price REAL,
                color TEXT,
                pattern TEXT,
                sleeve TEXT,
                style TEXT,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ph_user ON purchase_history(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ph_product ON purchase_history(product_id)")

        # Migration: add preferred_brands if missing
        try:
            cur.execute("""
                ALTER TABLE user_preferences
                ADD COLUMN IF NOT EXISTS preferred_brands TEXT[] DEFAULT '{}'
            """)
        except Exception:
            pass

        # Migration: drop preferred_categories if it exists (category is dynamic)
        try:
            cur.execute("""
                ALTER TABLE user_preferences
                DROP COLUMN IF EXISTS preferred_categories
            """)
        except Exception:
            pass  # column already gone or not supported

        conn.commit()
        cur.close()
        conn.close()
        print("[Personalization] Database tables initialized")
    except Exception as e:
        print(f"[Personalization] DB Init Error: {e}")


# ─────────────────────────────────────────────────
# USER CRUD
# ─────────────────────────────────────────────────

def create_user(user_id: str, name: str) -> bool:
    """Create a new user. Returns True on success."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users (user_id, name) VALUES (%s, %s)
               ON CONFLICT (user_id) DO UPDATE SET name = EXCLUDED.name""",
            (user_id, name)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[Personalization] create_user error: {e}")
        return False


def get_user(user_id: str) -> Optional[Dict]:
    """Get user by ID."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, name, created_at FROM users WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {"user_id": row[0], "name": row[1], "created_at": str(row[2])}
        return None
    except Exception as e:
        print(f"[Personalization] get_user error: {e}")
        return None


# ─────────────────────────────────────────────────
# PREFERENCES CRUD
# ─────────────────────────────────────────────────

def save_preferences(user_id: str, prefs: Dict[str, Any]) -> bool:
    """Save or update user preferences. Upserts on user_id."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO user_preferences
               (user_id, gender, size, budget_min, budget_max,
                preferred_colors, preferred_styles, preferred_brands, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
               ON CONFLICT (user_id) DO UPDATE SET
                   gender = EXCLUDED.gender,
                   size = EXCLUDED.size,
                   budget_min = EXCLUDED.budget_min,
                   budget_max = EXCLUDED.budget_max,
                   preferred_colors = EXCLUDED.preferred_colors,
                   preferred_styles = EXCLUDED.preferred_styles,
                   preferred_brands = EXCLUDED.preferred_brands,
                   updated_at = NOW()
            """,
            (
                user_id,
                prefs.get("gender"),
                prefs.get("size"),
                prefs.get("budget_min", 0),
                prefs.get("budget_max", 99999),
                prefs.get("preferred_colors", []),
                prefs.get("preferred_styles", []),
                prefs.get("preferred_brands", []),
            )
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[Personalization] save_preferences error: {e}")
        return False


def get_preferences(user_id: str) -> Optional[Dict]:
    """Get user preferences."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT gender, size, budget_min, budget_max,
                      preferred_colors, preferred_styles, preferred_brands
               FROM user_preferences WHERE user_id = %s""",
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "gender": row[0],
                "size": row[1],
                "budget_min": row[2] or 0,
                "budget_max": row[3] or 99999,
                "preferred_colors": row[4] or [],
                "preferred_styles": row[5] or [],
                "preferred_brands": row[6] or [],
            }
        return None
    except Exception as e:
        print(f"[Personalization] get_preferences error: {e}")
        return None


# ─────────────────────────────────────────────────
# PURCHASE HISTORY
# ─────────────────────────────────────────────────

def save_purchase(user_id: str, product: Dict[str, Any]) -> bool:
    """Record a purchase transaction."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO purchase_history
               (user_id, product_id, category, price, color, pattern, sleeve, style, timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
            (
                user_id,
                product.get("product_id", ""),
                product.get("category", ""),
                product.get("price", 0),
                product.get("color", ""),
                product.get("pattern", ""),
                product.get("sleeve", ""),
                product.get("style", ""),
            )
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[Personalization] save_purchase error: {e}")
        return False


def get_purchase_history(user_id: str, limit: int = 50) -> List[Dict]:
    """Get recent purchase history for a user."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT product_id, category, price, color, pattern, sleeve, style, timestamp
               FROM purchase_history
               WHERE user_id = %s
               ORDER BY timestamp DESC
               LIMIT %s""",
            (user_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "product_id": r[0], "category": r[1], "price": r[2],
                "color": r[3], "pattern": r[4], "sleeve": r[5],
                "style": r[6], "timestamp": str(r[7]),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[Personalization] get_purchase_history error: {e}")
        return []


def get_behavioral_profile(user_id: str) -> Dict[str, Any]:
    """
    Aggregate purchase history into a behavioral profile.
    Returns: top categories, average price, common colors.
    """
    purchases = get_purchase_history(user_id, limit=100)
    if not purchases:
        return {"top_categories": [], "avg_price": 0, "common_colors": [], "purchase_count": 0}

    # Top categories
    cat_counts = {}
    for p in purchases:
        cat = (p.get("category") or "").lower()
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]

    # Average price
    prices = [p["price"] for p in purchases if p.get("price") and p["price"] > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    # Common colors
    color_counts = {}
    for p in purchases:
        color = (p.get("color") or "").lower()
        if color:
            color_counts[color] = color_counts.get(color, 0) + 1
    top_colors = sorted(color_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "top_categories": [c[0] for c in top_cats],
        "avg_price": round(avg_price, 2),
        "common_colors": [c[0] for c in top_colors],
        "purchase_count": len(purchases),
    }
