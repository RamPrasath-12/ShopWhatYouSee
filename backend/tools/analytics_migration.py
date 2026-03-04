"""
Analytics Database Migration Script
=====================================
Creates analytics_events and analytics_daily_summary tables
in the existing shopwhatyousee PostgreSQL database.

Run once: python tools/analytics_migration.py
"""

import psycopg2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DB_CONFIG

# ─────────────────────────────────────────────
# Table 1: analytics_events (Raw event log)
# ─────────────────────────────────────────────
CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS analytics_events (
    -- Identity
    event_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          TEXT,
    user_id             TEXT,                          -- hashed, nullable for anonymous

    -- Event classification
    event_type          TEXT NOT NULL,                  -- detection, search, attribute_extraction,
                                                       -- product_click, filter_change, buy_click,
                                                       -- explanation_view, search_impression
    schema_version      INT DEFAULT 1,                 -- for future-proofing

    -- Detection signals
    detected_category   TEXT,
    detected_color      TEXT,
    detected_pattern    TEXT,
    detected_sleeve     TEXT,
    scene_label         TEXT,
    yolo_confidence     FLOAT,
    extraction_quality  FLOAT,

    -- Retrieval signals
    user_filters        JSONB,                         -- user override filters
    override_category   TEXT,                          -- LLM-overridden category
    pool_size           INT,                           -- candidate pool size
    result_count        INT,                           -- products returned
    relaxation_steps    TEXT[],                        -- which constraints relaxed
    price_max           FLOAT,

    -- Scoring signals (for model health)
    visual_similarity   FLOAT,                         -- cosine sim of clicked product
    final_score         FLOAT,                         -- rank score of clicked product

    -- Click signals
    clicked_product_id  TEXT,
    rank_clicked        INT,
    product_url         TEXT,
    explanation_shown   BOOLEAN DEFAULT FALSE,

    -- Impression data (for true CTR)
    impression_ids      TEXT[],                        -- product IDs shown in results
    impression_ranks    INT[],                         -- their rank positions

    -- Device / platform metadata
    device_type         TEXT,                          -- mobile, desktop, tablet
    user_agent          TEXT,
    ip_region           TEXT,

    -- Performance
    latency_ms          FLOAT,

    -- Timestamp
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
"""

# ─────────────────────────────────────────────
# Table 2: analytics_daily_summary (Pre-aggregated)
# ─────────────────────────────────────────────
CREATE_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS analytics_daily_summary (
    summary_date                DATE PRIMARY KEY,

    -- Volume metrics (structured columns — fast queries)
    total_searches              INT DEFAULT 0,
    total_detections            INT DEFAULT 0,
    total_clicks                INT DEFAULT 0,
    total_buy_clicks            INT DEFAULT 0,
    total_impressions           INT DEFAULT 0,

    -- Performance metrics
    avg_latency_ms              FLOAT,
    p95_latency_ms              FLOAT,

    -- Model health metrics
    avg_yolo_confidence         FLOAT,
    avg_extraction_quality      FLOAT,
    avg_visual_similarity_clicked FLOAT,
    embedding_click_correlation FLOAT,

    -- CTR by rank (structured for fast dashboard)
    ctr_rank_1                  FLOAT,
    ctr_rank_2                  FLOAT,
    ctr_rank_3                  FLOAT,
    ctr_rank_4                  FLOAT,
    ctr_rank_5                  FLOAT,

    -- Relaxation & override analytics
    relaxation_rate             FLOAT,
    override_frequency          FLOAT,
    conversion_with_relaxation  FLOAT,
    conversion_without_relax    FLOAT,

    -- Demand analytics
    avg_pool_size               FLOAT,

    -- Flexible JSONB for detailed breakdowns
    category_demand             JSONB,                 -- {"tshirt": 42, "shirt": 30}
    color_demand                JSONB,                 -- {"navy_blue": 25, "red": 18}
    scene_distribution          JSONB,                 -- {"clothing_store": 15, "street": 8}
    top_filters                 JSONB,                 -- most-used filter combinations
    demand_supply_gaps          JSONB,                 -- {category: {demand: N, supply: M}}

    -- Aggregation tracking
    last_processed_event_id     UUID,                  -- for incremental aggregation
    computed_at                 TIMESTAMPTZ DEFAULT NOW()
);
"""

# ─────────────────────────────────────────────
# Indexes for fast queries
# ─────────────────────────────────────────────
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics_events (created_at);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics_events (event_type);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_category ON analytics_events (detected_category);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_session ON analytics_events (session_id);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics_events (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_type_date ON analytics_events (event_type, created_at);",
]

# ─────────────────────────────────────────────
# Data retention: auto-delete raw events > 180 days
# ─────────────────────────────────────────────
CREATE_RETENTION_FUNCTION = """
CREATE OR REPLACE FUNCTION cleanup_old_analytics_events()
RETURNS void AS $$
BEGIN
    DELETE FROM analytics_events WHERE created_at < NOW() - INTERVAL '180 days';
END;
$$ LANGUAGE plpgsql;
"""


def run_migration():
    print("=" * 60)
    print("Analytics Database Migration")
    print("=" * 60)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        print("\n[1/4] Creating analytics_events table...")
        cur.execute(CREATE_EVENTS_TABLE)
        print("  ✅ analytics_events created")

        print("\n[2/4] Creating analytics_daily_summary table...")
        cur.execute(CREATE_SUMMARY_TABLE)
        print("  ✅ analytics_daily_summary created")

        print("\n[3/4] Creating indexes...")
        for idx_sql in CREATE_INDEXES:
            cur.execute(idx_sql)
            idx_name = idx_sql.split("INDEX IF NOT EXISTS ")[1].split(" ON")[0]
            print(f"  ✅ {idx_name}")

        print("\n[4/4] Creating retention cleanup function...")
        cur.execute(CREATE_RETENTION_FUNCTION)
        print("  ✅ cleanup_old_analytics_events()")

        # Verify
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('analytics_events', 'analytics_daily_summary')
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"\n{'=' * 60}")
        print(f"✅ Migration complete. Tables: {tables}")
        print(f"{'=' * 60}")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
