"""
Retrieval Metrics Tracker - Aggregate-Only (No Per-Query Logs)

DEMO-ONLY: In-memory metrics for review defense.
Stores only running aggregates (counts/averages) to prevent memory growth.

Tracks:
  - stage_a_hit_rate: % of queries where Stage A returns > 0 results
  - avg_relaxed_count: average soft attrs relaxed per query
  - avg_top_similarity: average top FAISS similarity score
"""

import threading


class RetrievalMetrics:
    """Thread-safe aggregate-only metrics tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_queries = 0
        self._stage_a_hits = 0  # queries where Stage A > 0
        self._total_relaxed_attrs = 0
        self._total_top_similarity = 0.0
        self._low_confidence_count = 0  # top sim below threshold

    def record_query(self, stage_a_count: int, relaxed_count: int, top_similarity: float):
        """Record a single query's metrics into running aggregates."""
        with self._lock:
            self._total_queries += 1
            if stage_a_count > 0:
                self._stage_a_hits += 1
            self._total_relaxed_attrs += relaxed_count
            self._total_top_similarity += top_similarity

    def record_low_confidence(self):
        """Record a query where top similarity was below threshold."""
        with self._lock:
            self._low_confidence_count += 1

    def get_summary(self) -> dict:
        """Return aggregate metrics summary. Safe for API response."""
        with self._lock:
            total = self._total_queries
            if total == 0:
                return {
                    "total_queries": 0,
                    "stage_a_hit_rate": 0.0,
                    "avg_relaxed_count": 0.0,
                    "avg_top_similarity": 0.0,
                    "low_confidence_queries": 0
                }
            return {
                "total_queries": total,
                "stage_a_hit_rate": round(self._stage_a_hits / total, 4),
                "avg_relaxed_count": round(self._total_relaxed_attrs / total, 2),
                "avg_top_similarity": round(self._total_top_similarity / total, 4),
                "low_confidence_queries": self._low_confidence_count
            }


# Singleton instance
_metrics = RetrievalMetrics()


def get_metrics() -> RetrievalMetrics:
    return _metrics
