"""
Centralized Database Configuration
=====================================
Single source of truth for DB connection across the entire backend.
Uses DATABASE_URL from .env (Supabase) with local fallback.

Usage:
    from db_config import get_db_connection, DATABASE_URL
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Supabase DSN (production) or local fallback
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@@localhost:5432/shopwhatyousee"
)

# Parsed config dict for psycopg2.connect(**DB_CONFIG) compatibility
# This parses the DSN into host/database/user/password for legacy code
def _parse_dsn(dsn: str) -> dict:
    """Parse a PostgreSQL DSN into a dict for psycopg2.connect(**config)."""
    from urllib.parse import urlparse
    parsed = urlparse(dsn)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or "shopwhatyousee",
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres123@",
    }

DB_CONFIG = _parse_dsn(DATABASE_URL)


def get_db_connection():
    """Get a new psycopg2 connection using the centralized config."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)
