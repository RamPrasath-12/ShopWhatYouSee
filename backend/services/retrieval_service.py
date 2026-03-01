"""
Production Retrieval Service — FAISS from PostgreSQL
=====================================================
Singleton service that:
1. Loads all embeddings + metadata from visual_attributes at startup
2. Builds FAISS IndexFlatIP(512) once
3. Exposes search() with post-FAISS filtering + graceful relaxation
4. Pure cosine similarity ranking — no hybrid scoring

Usage:
    from services.retrieval_service import retrieval_service
    retrieval_service.init()
    results = retrieval_service.search(query_embedding, filters={"category": "shirts"})
"""

import os
import time
import numpy as np
import faiss
import psycopg2

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ─── DB Config ──────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "shopwhatyousee"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres123@"),
}

# ─── Filter relaxation order (never drop category) ─────────────────
RELAXATION_ORDER = ["material", "style", "price_bucket", "primary_color_name", "color_family", "gender"]
FILTERABLE_FIELDS = {
    "category", "gender", "style", "material", "price_bucket", "color_family",
    "sleeve_value", "pattern_value", "primary_color_name",
}


class RetrievalService:
    """Singleton FAISS retrieval service backed by PostgreSQL."""

    def __init__(self):
        self._initialized = False
        self.index = None
        self.n_vectors = 0

        # Parallel metadata arrays (same index as FAISS vectors)
        self.product_ids = []
        self.categories = []
        self.genders = []
        self.styles = []
        self.materials = []
        self.price_buckets = []
        self.color_families = []
        self.brands = []
        self.product_names = []
        self.image_urls = []
        self.discounted_prices = []
        self.original_prices = []
        self.primary_colors = []
        self.patterns = []
        self.sleeves = []

        # Lookup: product_id -> index
        self._pid_to_idx = {}

    @property
    def is_ready(self):
        return self._initialized

    # ─── INIT ───────────────────────────────────────────────────────
    def init(self):
        """Load all embeddings + metadata from PostgreSQL, build FAISS index.

        MEMORY-OPTIMIZED: Streams rows via server-side cursor, writes
        embeddings directly into a preallocated numpy array. Never builds
        a Python list of all embeddings simultaneously.
        """
        if self._initialized:
            print("[RetrievalService] Already initialized, skipping.")
            return

        import gc

        print("=" * 60)
        print("[RetrievalService] INITIALIZING (memory-optimized)...")
        print("=" * 60)

        rss_before = self._get_rss_mb()

        # ── Step 1: Get row count for preallocation ──────────────
        t0 = time.time()
        conn = psycopg2.connect(**DB_CONFIG)
        count_cur = conn.cursor()
        count_cur.execute("SELECT COUNT(*) FROM visual_attributes WHERE embedding IS NOT NULL")
        total_rows = count_cur.fetchone()[0]
        count_cur.close()
        print(f"  Row count: {total_rows}")

        if total_rows == 0:
            print("  [ERROR] No rows found in visual_attributes!")
            conn.close()
            return

        # ── Step 2: Preallocate numpy array ──────────────────────
        emb_np = np.empty((total_rows, 512), dtype=np.float32)
        rss_after_alloc = self._get_rss_mb()
        print(f"  Preallocated numpy: {emb_np.nbytes / (1024*1024):.1f} MB "
              f"(RSS: {rss_after_alloc:.0f} MB, +{rss_after_alloc - rss_before:.0f} MB)")

        # ── Step 3: Stream rows via server-side cursor ───────────
        # Server-side cursor avoids fetchall() — psycopg2 fetches
        # `itersize` rows at a time from PostgreSQL.
        cur = conn.cursor(name="rs_init_stream")
        cur.itersize = 2000

        cur.execute("""
            SELECT
                product_id, category, gender, style, material,
                price_bucket, color_family, brand, product_name,
                image_url, discounted_price, original_price,
                primary_color_name, pattern_value, sleeve_value,
                embedding
            FROM visual_attributes
            WHERE embedding IS NOT NULL
            ORDER BY id
        """)

        idx = 0
        for row in cur:
            (pid, cat, gen, sty, mat, pbk, cfm, brand, pname,
             img_url, dprice, oprice, pcolor, pattern, sleeve, emb) = row

            # Metadata into parallel arrays (strings — compact)
            self.product_ids.append(pid)
            self.categories.append(cat)
            self.genders.append(gen)
            self.styles.append(sty)
            self.materials.append(mat)
            self.price_buckets.append(pbk)
            self.color_families.append(cfm)
            self.brands.append(brand)
            self.product_names.append(pname)
            self.image_urls.append(img_url)
            self.discounted_prices.append(dprice)
            self.original_prices.append(oprice)
            self.primary_colors.append(pcolor)
            self.patterns.append(pattern)
            self.sleeves.append(sleeve)

            # Embedding: write DIRECTLY into preallocated numpy row.
            # `emb` is a Python list of 512 floats (from psycopg2 jsonb).
            # This creates a temporary Python list but immediately copies
            # into the numpy buffer. The Python list is GC'd next iteration.
            emb_np[idx] = emb
            idx += 1

            # Progress log every 10K rows
            if idx % 10000 == 0:
                print(f"    Streamed {idx}/{total_rows} rows (RSS: {self._get_rss_mb():.0f} MB)")

        cur.close()
        conn.close()

        db_time = time.time() - t0
        self.n_vectors = idx
        rss_after_stream = self._get_rss_mb()
        print(f"  Streamed {idx} rows in {db_time:.1f}s "
              f"(RSS: {rss_after_stream:.0f} MB, +{rss_after_stream - rss_after_alloc:.0f} MB)")

        # Handle case where actual rows < total_rows (shouldn't happen)
        if idx < total_rows:
            emb_np = emb_np[:idx]

        # Build product_id → index map
        for i, pid in enumerate(self.product_ids):
            self._pid_to_idx[pid] = i

        # ── Step 4: Validate embeddings ──────────────────────────
        dim = emb_np.shape[1]
        norms = np.linalg.norm(emb_np, axis=1)
        mean_norm = norms.mean()
        std_norm = norms.std()

        print(f"  Embeddings: {emb_np.shape[0]} x {dim}-D")
        print(f"  Norm: mean={mean_norm:.4f}, std={std_norm:.6f}")

        assert dim == 512, f"Expected 512-D embeddings, got {dim}"

        if abs(mean_norm - 1.0) > 0.01:
            print("  [WARN] Norms not ~1.0, re-normalizing...")
            faiss.normalize_L2(emb_np)
        else:
            print("  Norms ~= 1.0, skipping re-normalization.")

        # ── Step 5: Build FAISS index ────────────────────────────
        t1 = time.time()
        self.index = faiss.IndexFlatIP(512)
        self.index.add(emb_np)  # IndexFlatIP copies data internally
        build_time = time.time() - t1

        rss_after_faiss = self._get_rss_mb()
        faiss_est_mb = self.index.ntotal * 512 * 4 / (1024 * 1024)
        numpy_mb = emb_np.nbytes / (1024 * 1024)

        print(f"  FAISS build: {build_time:.3f}s")
        print(f"  Index size: {self.index.ntotal} vectors")
        print(f"  FAISS estimated: {faiss_est_mb:.1f} MB")
        print(f"  numpy array: {numpy_mb:.1f} MB (will be freed)")
        print(f"  RSS after FAISS: {rss_after_faiss:.0f} MB")

        # ── Step 6: Cleanup — delete numpy array ─────────────────
        # IndexFlatIP.add() copies data, so numpy array is now redundant.
        del emb_np
        del norms
        gc.collect()

        rss_final = self._get_rss_mb()
        print(f"  RSS after cleanup: {rss_final:.0f} MB "
              f"(freed {rss_after_faiss - rss_final:.0f} MB)")

        # ── Summary ─────────────────────────────────────────────
        print(f"\n  Memory breakdown:")
        print(f"    Baseline:       {rss_before:.0f} MB")
        print(f"    After stream:   {rss_after_stream:.0f} MB")
        print(f"    After FAISS:    {rss_after_faiss:.0f} MB")
        print(f"    After cleanup:  {rss_final:.0f} MB")
        print(f"    FAISS index:    ~{faiss_est_mb:.0f} MB")
        print(f"    Total growth:   +{rss_final - rss_before:.0f} MB")

        target_met = rss_final < 400
        print(f"    Target <400MB:  {'✅ PASS' if target_met else '❌ FAIL'}")
        print("=" * 60)
        print("[RetrievalService] READY")
        print("=" * 60)

        self._initialized = True

    # ─── SEARCH ─────────────────────────────────────────────────────
    def search(self, query_embedding, filters=None, top_k=20, exclude_product_id=None):
        """
        Search for similar products.

        Args:
            query_embedding: list or np.array of 512 floats
            filters: dict of {field: value} — all optional
                     category, gender, style, material, price_bucket, color_family
            top_k: number of results to return (default 20)
            exclude_product_id: product_id to exclude (self-match)

        Returns:
            list of dicts with product metadata + similarity_score
        """
        if not self._initialized:
            raise RuntimeError("RetrievalService not initialized. Call init() first.")

        # Prepare query vector
        q_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        # FAISS search top-50 candidates
        faiss_k = max(50, top_k * 3)
        D, I = self.index.search(q_vec, faiss_k)

        # Build candidate list (exclude self-match)
        candidates = []
        exclude_idx = self._pid_to_idx.get(exclude_product_id, -1) if exclude_product_id else -1

        for j in range(faiss_k):
            idx = int(I[0][j])
            if idx < 0 or idx >= self.n_vectors:
                continue
            if idx == exclude_idx:
                continue
            candidates.append({
                "_idx": idx,
                "similarity_score": round(float(D[0][j]), 4),
            })

        # Apply filters with graceful relaxation
        # DO NOT modify filter relaxation order or ranking logic.
        # Only structured logging and metadata return added here.
        filter_meta = {"active_filters": 0, "relaxations": 0, "relaxed_fields": [], "result_count": 0}

        if filters:
            results, filter_meta = self._apply_filters(candidates, filters, top_k)
        else:
            results = candidates[:top_k]
            filter_meta["result_count"] = len(results)

        # Build full response objects
        built = [self._build_result(c) for c in results]

        # Compute similarity range
        if built:
            scores = [r["similarity_score"] for r in built]
            filter_meta["cosine_similarity_range"] = [
                round(min(scores), 4), round(max(scores), 4)
            ]
        else:
            filter_meta["cosine_similarity_range"] = [0.0, 0.0]

        # Observability logging (no logic changes)
        if filter_meta["active_filters"] > 0:
            print(f"  [Observability] filters={filter_meta['active_filters']}, "
                  f"relaxations={filter_meta['relaxations']}, "
                  f"relaxed={filter_meta['relaxed_fields']}, "
                  f"results={filter_meta['result_count']}")

        return built, filter_meta

    # ─── FILTER LOGIC ───────────────────────────────────────────────
    def _apply_filters(self, candidates, filters, top_k):
        """Apply filters with graceful relaxation.
        DO NOT modify filter relaxation order or ranking logic.
        Only structured logging added."""
        # Normalize filter dict — remove None/empty values
        active_filters = {
            k: v for k, v in filters.items()
            if v is not None and v != "" and k in FILTERABLE_FIELDS
        }

        meta = {"active_filters": len(active_filters), "relaxations": 0,
                "relaxed_fields": [], "result_count": 0}

        if not active_filters:
            meta["result_count"] = min(len(candidates), top_k)
            return candidates[:top_k], meta

        # First attempt: all filters
        filtered = self._filter_candidates(candidates, active_filters)
        if len(filtered) >= top_k:
            meta["result_count"] = top_k
            return filtered[:top_k], meta

        # Graceful relaxation
        relaxed = []
        current_filters = dict(active_filters)

        for drop_field in RELAXATION_ORDER:
            if drop_field not in current_filters:
                continue
            if drop_field == "category":
                continue  # Never drop category

            del current_filters[drop_field]
            relaxed.append(drop_field)
            meta["relaxations"] += 1

            filtered = self._filter_candidates(candidates, current_filters)
            if len(filtered) >= top_k:
                if relaxed:
                    print(f"  [Filter] Relaxed: {relaxed}")
                meta["relaxed_fields"] = list(relaxed)
                meta["result_count"] = top_k
                return filtered[:top_k], meta

        # If still not enough, return what we have
        if relaxed:
            print(f"  [Filter] Relaxed all: {relaxed}, got {len(filtered)} results")
        meta["relaxed_fields"] = list(relaxed)
        final = filtered[:top_k] if filtered else candidates[:top_k]
        meta["result_count"] = len(final)
        return final, meta

    def _filter_candidates(self, candidates, filters):
        """Filter candidates against active filters."""
        result = []
        for c in candidates:
            idx = c["_idx"]
            match = True
            for field, value in filters.items():
                actual = self._get_field(idx, field)
                if actual is None:
                    # NULL fields don't match explicit filters
                    match = False
                    break
                if actual.lower() != value.lower():
                    match = False
                    break
            if match:
                result.append(c)
        return result

    def _get_field(self, idx, field):
        """Get metadata field value by index."""
        field_map = {
            "category": self.categories,
            "gender": self.genders,
            "style": self.styles,
            "material": self.materials,
            "price_bucket": self.price_buckets,
            "color_family": self.color_families,
            "sleeve_value": self.sleeves,
            "pattern_value": self.patterns,
            "primary_color_name": self.primary_colors,
        }
        arr = field_map.get(field)
        if arr is None:
            return None
        return arr[idx]

    # ─── RESULT BUILDER ─────────────────────────────────────────────
    def _build_result(self, candidate):
        """Build full result dict from candidate."""
        idx = candidate["_idx"]
        return {
            "product_id": self.product_ids[idx],
            "category": self.categories[idx],
            "brand": self.brands[idx],
            "product_name": self.product_names[idx],
            "image_url": self.image_urls[idx],
            "discounted_price": self.discounted_prices[idx],
            "original_price": self.original_prices[idx],
            "primary_color_name": self.primary_colors[idx],
            "pattern_value": self.patterns[idx],
            "sleeve_value": self.sleeves[idx],
            "gender": self.genders[idx],
            "style": self.styles[idx],
            "material": self.materials[idx],
            "price_bucket": self.price_buckets[idx],
            "color_family": self.color_families[idx],
            "similarity_score": candidate["similarity_score"],
        }

    # ─── UTILITY ────────────────────────────────────────────────────
    def _get_rss_mb(self):
        if _HAS_PSUTIL:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        return 0.0


# ─── Singleton instance ─────────────────────────────────────────────
retrieval_service = RetrievalService()
