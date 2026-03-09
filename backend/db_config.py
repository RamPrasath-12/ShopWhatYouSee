"""
Centralized Database Configuration
=====================================
Single source of truth for DB connection across the entire backend.
Uses DATABASE_URL from .env (Supabase) with local fallback.

Usage:
    from db_config import get_db_connection, DATABASE_URL, DB_CONFIG
"""

import os
import socket
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Supabase DSN (production) or local fallback
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@@localhost:5432/shopwhatyousee"
)


def _resolve_host(hostname: str) -> str:
    """
    Resolve a hostname to an IP address.
    Uses multiple strategies:
      1. System resolver (IPv4, then IPv6)
      2. Google Public DNS via dnspython (for blocked networks)
    Returns the original hostname if all resolution fails.
    """
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return hostname

    # Strategy 1: System resolver — IPv4
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        if infos:
            return infos[0][4][0]
    except socket.gaierror:
        pass

    # Strategy 2: System resolver — IPv6
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET6)
        if infos:
            return infos[0][4][0]
    except socket.gaierror:
        pass

    # Strategy 3: dnspython via Google Public DNS (8.8.8.8)
    try:
        import dns.resolver
        r = dns.resolver.Resolver()
        r.nameservers = ['8.8.8.8', '8.8.4.4']
        r.lifetime = 5

        # Try A record first (IPv4)
        try:
            answers = r.resolve(hostname, 'A')
            if answers:
                return str(answers[0])
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            pass

        # Try AAAA record (IPv6)
        try:
            answers = r.resolve(hostname, 'AAAA')
            if answers:
                return str(answers[0])
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            pass
    except ImportError:
        pass
    except Exception:
        pass

    return hostname


# Parsed config dict for psycopg2.connect(**DB_CONFIG) compatibility
def _parse_dsn(dsn: str) -> dict:
    """Parse a PostgreSQL DSN into a dict for psycopg2.connect(**config)."""
    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    # Resolve to IP so psycopg2 doesn't choke on DNS issues
    resolved_host = _resolve_host(host)
    return {
        "host": resolved_host,
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or "shopwhatyousee",
        "user": parsed.username or "postgres",
        "password": unquote(parsed.password) if parsed.password else "postgres123@",
    }

DB_CONFIG = _parse_dsn(DATABASE_URL)


def get_db_connection():
    """Get a new psycopg2 connection using the centralized config."""
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)
