"""
Analytics Event Logger — Batched, Async, Thread-Safe
=====================================================
Captures structured events from the ShopWhatYouSee pipeline
and writes them to the analytics_events table in batches.

Architecture:
  log_event() → in-memory queue → background thread → batch INSERT every 1s

Usage:
  from services.analytics_logger import log_event
  log_event("search", detected_category="tshirt", latency_ms=734.2)
"""

import uuid
import threading
import time
import hashlib
import psycopg2
import psycopg2.extras
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────
# DB Config (centralized — Supabase in production)
# ─────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DB_CONFIG, DATABASE_URL

SCHEMA_VERSION = 1
BATCH_INTERVAL = 1.0      # flush every 1 second
BATCH_MAX_SIZE = 100       # flush if queue reaches this size
QUEUE_MAX_SIZE = 5000      # hard cap to prevent memory leak

# ─────────────────────────────────────────────
# Valid event types (controlled vocabulary)
# ─────────────────────────────────────────────
VALID_EVENT_TYPES = {
    "detection",              # YOLO detection (per item found)
    "detection_request",      # frame processed (for success/failure rate)
    "attribute_extraction",   # AGMAN extraction
    "search",                 # product search/retrieval
    "search_impression",      # products shown (for CTR)
    "product_click",          # user clicked a product
    "filter_change",          # user added/removed filter
    "buy_click",              # user clicked Buy Now
    "explanation_view",       # user expanded "Why?"
}

# ─────────────────────────────────────────────
# All columns in analytics_events
# ─────────────────────────────────────────────
EVENT_COLUMNS = [
    "event_id", "session_id", "user_id", "event_type", "schema_version",
    "detected_category", "detected_color", "detected_pattern", "detected_sleeve",
    "scene_label", "yolo_confidence", "extraction_quality",
    "user_filters", "override_category", "pool_size", "result_count",
    "relaxation_steps", "price_max", "visual_similarity", "final_score",
    "clicked_product_id", "rank_clicked", "product_url", "explanation_shown",
    "impression_ids", "impression_ranks",
    "device_type", "user_agent", "ip_region",
    "latency_ms", "created_at",
]

# ─────────────────────────────────────────────
# Internal queue + background thread
# ─────────────────────────────────────────────
_event_queue: deque = deque(maxlen=QUEUE_MAX_SIZE)
_queue_lock = threading.Lock()
_flush_thread: Optional[threading.Thread] = None
_running = False


def _build_row(event_type: str, **kwargs) -> Dict[str, Any]:
    """Build a single event row with defaults."""
    import json as _json

    row = {
        "event_id": str(uuid.uuid4()),
        "session_id": kwargs.get("session_id"),
        "user_id": kwargs.get("user_id"),
        "event_type": event_type,
        "schema_version": SCHEMA_VERSION,
        "detected_category": kwargs.get("detected_category"),
        "detected_color": kwargs.get("detected_color"),
        "detected_pattern": kwargs.get("detected_pattern"),
        "detected_sleeve": kwargs.get("detected_sleeve"),
        "scene_label": kwargs.get("scene_label"),
        "yolo_confidence": kwargs.get("yolo_confidence"),
        "extraction_quality": kwargs.get("extraction_quality"),
        "user_filters": _json.dumps(kwargs["user_filters"]) if kwargs.get("user_filters") else None,
        "override_category": kwargs.get("override_category"),
        "pool_size": kwargs.get("pool_size"),
        "result_count": kwargs.get("result_count"),
        "relaxation_steps": kwargs.get("relaxation_steps"),
        "price_max": kwargs.get("price_max"),
        "visual_similarity": kwargs.get("visual_similarity"),
        "final_score": kwargs.get("final_score"),
        "clicked_product_id": kwargs.get("clicked_product_id"),
        "rank_clicked": kwargs.get("rank_clicked"),
        "product_url": kwargs.get("product_url"),
        "explanation_shown": kwargs.get("explanation_shown", False),
        "impression_ids": kwargs.get("impression_ids"),
        "impression_ranks": kwargs.get("impression_ranks"),
        "device_type": kwargs.get("device_type"),
        "user_agent": kwargs.get("user_agent"),
        "ip_region": kwargs.get("ip_region"),
        "latency_ms": kwargs.get("latency_ms"),
        "created_at": datetime.now(timezone.utc),
    }
    return row


def _flush_batch():
    """Flush queued events to DB in a single batch INSERT."""
    batch = []
    with _queue_lock:
        while _event_queue and len(batch) < BATCH_MAX_SIZE:
            batch.append(_event_queue.popleft())

    if not batch:
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Build batch INSERT
        cols = EVENT_COLUMNS
        placeholders = ", ".join([f"%({c})s" for c in cols])
        sql = f"INSERT INTO analytics_events ({', '.join(cols)}) VALUES ({placeholders})"

        psycopg2.extras.execute_batch(cur, sql, batch, page_size=50)
        conn.commit()
        cur.close()
        conn.close()

        print(f"[Analytics] ✅ Flushed {len(batch)} events to DB")

    except Exception as e:
        print(f"[Analytics] ⚠️ Batch flush failed (non-critical): {e}")
        # Re-queue failed events (best-effort)
        with _queue_lock:
            for evt in reversed(batch):
                if len(_event_queue) < QUEUE_MAX_SIZE:
                    _event_queue.appendleft(evt)


def _background_flusher():
    """Background thread: flush queue every BATCH_INTERVAL seconds."""
    global _running
    while _running:
        time.sleep(BATCH_INTERVAL)
        try:
            _flush_batch()
        except Exception as e:
            print(f"[Analytics] ⚠️ Background flusher error: {e}")


def _ensure_started():
    """Start the background flush thread if not already running."""
    global _flush_thread, _running
    if _running:
        return
    _running = True
    _flush_thread = threading.Thread(target=_background_flusher, daemon=True, name="analytics-flusher")
    _flush_thread.start()
    print("[Analytics] 🚀 Background event flusher started")


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def log_event(event_type: str, **kwargs) -> None:
    """
    Log an analytics event. Non-blocking, thread-safe.

    Args:
        event_type: One of VALID_EVENT_TYPES
        **kwargs: Event fields matching analytics_events columns

    Example:
        log_event("search",
            session_id="abc123",
            detected_category="tshirt",
            latency_ms=734.2,
            pool_size=541,
            result_count=20
        )
    """
    if event_type not in VALID_EVENT_TYPES:
        print(f"[Analytics] ⚠️ Unknown event type: {event_type}")
        return

    _ensure_started()

    row = _build_row(event_type, **kwargs)

    with _queue_lock:
        _event_queue.append(row)

    # If queue is getting large, trigger immediate flush
    if len(_event_queue) >= BATCH_MAX_SIZE:
        threading.Thread(target=_flush_batch, daemon=True).start()


def hash_user_id(raw_id: str) -> str:
    """Hash a raw user identifier for privacy. One-way SHA-256."""
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def parse_device_type(user_agent: str) -> str:
    """Simple device type extraction from User-Agent string."""
    ua = (user_agent or "").lower()
    if "mobile" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    elif "tablet" in ua or "ipad" in ua:
        return "tablet"
    return "desktop"


def shutdown():
    """Gracefully flush remaining events on shutdown."""
    global _running
    _running = False
    _flush_batch()  # Final flush
    print("[Analytics] 🛑 Flusher stopped, final flush complete")
