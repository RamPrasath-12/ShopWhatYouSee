"""
Phase 2 — Supabase Direct Batch Insert + Index Build
=====================================================
Reads from local PostgreSQL, inserts directly to Supabase in batches of 1000.
Builds pgvector IVFFlat index + ANALYZE after full insert.

TASKS COVERED:
  - Task 2: Batch insert (batches of 1000)
  - Task 3: pgvector IVFFlat index + ANALYZE

Usage:
  python tools/supabase_batch_insert.py
"""
import os
import sys
import time
import json
import psycopg2
from psycopg2.extras import execute_values

# ─── Connection configs ──────────────────────────────────────────
LOCAL_DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "shopwhatyousee"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres123@"),
}

SUPABASE_DSN = (
    "postgresql://postgres:ShopWhatYouSee123"
    "@db.mxjpueufbooxgewqxccm.supabase.co:5432/postgres"
)

BATCH_SIZE = 1000
EXPECTED_ROWS = 34787

# Columns to migrate (must match both schemas)
COLUMNS = [
    "product_id", "category", "gender", "style", "material",
    "price_bucket", "color_family", "brand", "product_name",
    "image_url", "discounted_price", "original_price",
    "primary_color_name", "pattern_value", "sleeve_value",
    "embedding",
]


# ─── Task 2: Create schema + batch insert ────────────────────────
def create_schema(supa_cur):
    """Create table + enable pgvector in Supabase."""
    print("\n[SCHEMA] Creating table in Supabase...")
    supa_cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    supa_cur.execute("""
        CREATE TABLE IF NOT EXISTS visual_attributes (
            id SERIAL PRIMARY KEY,
            product_id TEXT NOT NULL UNIQUE,
            category TEXT,
            gender TEXT,
            style TEXT,
            material TEXT,
            price_bucket TEXT,
            color_family TEXT,
            brand TEXT,
            product_name TEXT,
            image_url TEXT,
            discounted_price NUMERIC,
            original_price NUMERIC,
            primary_color_name TEXT,
            pattern_value TEXT,
            sleeve_value TEXT,
            embedding VECTOR(512) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    # Metadata indexes
    supa_cur.execute("CREATE INDEX IF NOT EXISTS idx_va_category ON visual_attributes(category);")
    supa_cur.execute("CREATE INDEX IF NOT EXISTS idx_va_gender ON visual_attributes(gender);")
    supa_cur.execute("CREATE INDEX IF NOT EXISTS idx_va_product_id ON visual_attributes(product_id);")
    print("  Schema created")


def batch_insert(local_conn, supa_conn):
    """Read all rows from local PG, insert to Supabase in batches."""
    print(f"\n[INSERT] Batch insert ({BATCH_SIZE} rows/batch)...")

    # Count source rows
    local_cur = local_conn.cursor()
    local_cur.execute("SELECT COUNT(*) FROM visual_attributes")
    total = local_cur.fetchone()[0]
    print(f"  Source rows: {total}")

    # Use server-side cursor for memory efficiency
    local_cur = local_conn.cursor(name="migration_cursor")
    local_cur.itersize = BATCH_SIZE
    local_cur.execute(f"""
        SELECT {', '.join(COLUMNS)}
        FROM visual_attributes
        ORDER BY product_id
    """)

    supa_cur = supa_conn.cursor()

    # Disable autocommit by using manual transaction
    insert_sql = f"""
        INSERT INTO visual_attributes ({', '.join(COLUMNS)})
        VALUES %s
        ON CONFLICT (product_id) DO NOTHING
    """

    batch = []
    inserted = 0
    batch_num = 0
    t0 = time.time()

    for row in local_cur:
        # Convert embedding list to pgvector string format
        row_list = list(row)
        emb = row_list[-1]
        if isinstance(emb, list):
            row_list[-1] = "[" + ",".join(str(v) for v in emb) + "]"
        batch.append(tuple(row_list))

        if len(batch) >= BATCH_SIZE:
            batch_num += 1
            execute_values(supa_cur, insert_sql, batch, page_size=BATCH_SIZE)
            supa_conn.commit()
            inserted += len(batch)
            elapsed = time.time() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            print(f"    Batch {batch_num}: {inserted}/{total} "
                  f"({inserted/total*100:.1f}%) [{rate:.0f} rows/s]")
            batch = []

    # Final batch
    if batch:
        batch_num += 1
        execute_values(supa_cur, insert_sql, batch, page_size=BATCH_SIZE)
        supa_conn.commit()
        inserted += len(batch)
        print(f"    Batch {batch_num}: {inserted}/{total} (100%)")

    elapsed = time.time() - t0
    print(f"  Inserted {inserted} rows in {batch_num} batches ({elapsed:.1f}s)")

    local_cur.close()
    supa_cur.close()
    return inserted


# ─── Task 3: Build pgvector index + ANALYZE ──────────────────────
def build_index(supa_conn):
    """Create IVFFlat index on embedding column + ANALYZE."""
    print("\n[INDEX] Building IVFFlat index...")
    supa_cur = supa_conn.cursor()

    # lists = sqrt(N) is the recommendation; sqrt(34787) ≈ 186, use 150
    t0 = time.time()
    supa_cur.execute("""
        DROP INDEX IF EXISTS idx_va_embedding;
    """)
    supa_cur.execute("""
        CREATE INDEX idx_va_embedding ON visual_attributes
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 150);
    """)
    supa_conn.commit()
    index_time = time.time() - t0
    print(f"  IVFFlat index built in {index_time:.1f}s (lists=150)")

    # ANALYZE for query planner
    t1 = time.time()
    supa_cur.execute("ANALYZE visual_attributes;")
    supa_conn.commit()
    analyze_time = time.time() - t1
    print(f"  ANALYZE completed in {analyze_time:.1f}s")

    supa_cur.close()
    return index_time, analyze_time


# ─── Verify row count ────────────────────────────────────────────
def verify_count(supa_conn):
    """Verify final row count in Supabase."""
    print("\n[VERIFY] Checking row count...")
    supa_cur = supa_conn.cursor()
    supa_cur.execute("SELECT COUNT(*) FROM visual_attributes;")
    count = supa_cur.fetchone()[0]
    supa_cur.close()
    print(f"  Supabase rows: {count}")
    print(f"  Expected:      {EXPECTED_ROWS}")
    print(f"  Match: {'✅ PASS' if count == EXPECTED_ROWS else '❌ FAIL'}")
    return count


# ─── Test query latency ──────────────────────────────────────────
def test_query_latency(supa_conn, n=5):
    """Test vector similarity query latency."""
    print(f"\n[LATENCY] Testing {n} vector similarity queries...")
    supa_cur = supa_conn.cursor()

    # Get a sample embedding
    supa_cur.execute(
        "SELECT embedding FROM visual_attributes ORDER BY RANDOM() LIMIT 1"
    )
    sample_emb = supa_cur.fetchone()[0]

    latencies = []
    for i in range(n):
        t0 = time.time()
        supa_cur.execute("""
            SELECT product_id, category,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM visual_attributes
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (sample_emb, sample_emb))
        results = supa_cur.fetchall()
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        print(f"    [{i+1}] {lat:.0f}ms ({len(results)} results)")

    avg = sum(latencies) / len(latencies)
    print(f"  Avg latency: {avg:.0f}ms (target <150ms)")
    print(f"  Result: {'✅ PASS' if avg < 150 else '⚠ WARN'}")

    supa_cur.close()
    return avg


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 2: SUPABASE MIGRATION")
    print("=" * 60)

    report = {}

    # Connect to both databases
    print("\nConnecting to local PostgreSQL...")
    local_conn = psycopg2.connect(**LOCAL_DB)
    print("  Connected")

    print("Connecting to Supabase...")
    supa_conn = psycopg2.connect(SUPABASE_DSN)
    supa_conn.autocommit = False
    print("  Connected")

    try:
        # Task 2: Schema + batch insert
        supa_conn.autocommit = True
        supa_cur = supa_conn.cursor()
        create_schema(supa_cur)
        supa_cur.close()
        supa_conn.autocommit = False

        inserted = batch_insert(local_conn, supa_conn)
        report["inserted_rows"] = inserted

        # Task 3: Build index + ANALYZE
        supa_conn.autocommit = True
        idx_time, analyze_time = build_index(supa_conn)
        report["index_build_time_s"] = round(idx_time, 1)
        report["analyze_time_s"] = round(analyze_time, 1)

        # Verify
        count = verify_count(supa_conn)
        report["supabase_rows"] = count
        report["count_match"] = count == EXPECTED_ROWS

        # Test latency
        avg_lat = test_query_latency(supa_conn, n=5)
        report["avg_query_latency_ms"] = round(avg_lat, 1)

    finally:
        local_conn.close()
        supa_conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  Inserted:       {report.get('inserted_rows', 0)} rows")
    print(f"  Supabase count: {report.get('supabase_rows', 0)} (expected {EXPECTED_ROWS})")
    print(f"  Count match:    {'✅' if report.get('count_match') else '❌'}")
    print(f"  Index build:    {report.get('index_build_time_s', 0)}s")
    print(f"  ANALYZE:        {report.get('analyze_time_s', 0)}s")
    print(f"  Query latency:  {report.get('avg_query_latency_ms', 0)}ms")
    print("=" * 60)

    # Save report
    with open("tools/supabase_migration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: tools/supabase_migration_report.json")


if __name__ == "__main__":
    main()
