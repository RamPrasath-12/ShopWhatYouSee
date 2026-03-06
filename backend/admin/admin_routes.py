"""
ShopWhatYouSee — Admin Dashboard API
=====================================
Flask Blueprint with JWT-protected endpoints.
Each endpoint returns data for a single dashboard section.

Usage:
    from admin.admin_routes import admin_bp
    app.register_blueprint(admin_bp)
"""

import json
import os
import jwt
import psycopg2
import psycopg2.extras
from datetime import date, datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify, g

from db_config import DATABASE_URL
from dotenv import load_dotenv
load_dotenv()

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

ADMIN_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "swys2026")
JWT_SECRET = os.getenv("JWT_SECRET", "swys-admin-secret-key-2026")
JWT_EXPIRY_HOURS = 24


# Prevent browser caching of admin API responses
@admin_bp.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def _ensure_aggregated():
    """Run today's aggregation so dashboard data is fresh."""
    try:
        from tools.analytics_aggregation import aggregate_day
        aggregate_day(date.today())
    except Exception as e:
        print(f"[AdminAPI] Aggregation warning: {e}")


# ─────────────────────────────────────────────
# AUTH: JWT middleware
# ─────────────────────────────────────────────
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not token:
            return jsonify({"error": "Missing authorization token"}), 401

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.admin_user = payload.get("role", "admin")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _query(sql, params=None):
    """Execute SQL and return list of dicts."""
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()
        # Convert RealDictRow to regular dicts for JSON serialization
        result = [dict(r) for r in rows]
        cur.close()
        return result
    finally:
        conn.close()


def _parse_json_field(value):
    """Parse a JSON field that may be a string, dict, or None."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _get_date_range():
    """Extract date range from query params with 30-day default."""
    d1_str = request.args.get("from")
    d2_str = request.args.get("to")
    try:
        d1 = datetime.strptime(d1_str, "%Y-%m-%d").date() if d1_str else date.today() - timedelta(days=30)
        d2 = datetime.strptime(d2_str, "%Y-%m-%d").date() if d2_str else date.today()
    except ValueError:
        d1 = date.today() - timedelta(days=30)
        d2 = date.today()
    return d1, d2


def _get_summary_rows(d1, d2):
    """Fetch daily summary rows for the date range."""
    return _query(
        "SELECT * FROM analytics_daily_summary WHERE summary_date >= %s AND summary_date <= %s ORDER BY summary_date",
        (d1, d2)
    )


def _aggregate_json_column(rows, column_name, top_n=10):
    """Aggregate a JSON dict column across multiple days. Returns top N items."""
    agg = {}
    for row in rows:
        data = _parse_json_field(row.get(column_name))
        for k, v in data.items():
            if k.startswith("_"):
                continue
            agg[k] = agg.get(k, 0) + (v if isinstance(v, (int, float)) else 0)
    sorted_items = sorted(agg.items(), key=lambda x: -x[1])[:top_n]
    return [{"name": k, "value": v} for k, v in sorted_items]


# ─────────────────────────────────────────────
# POST /api/admin/login
# ─────────────────────────────────────────────
@admin_bp.route("/login", methods=["POST"])
def admin_login():
    body = request.get_json() or {}
    password = body.get("password", "")

    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid password"}), 401

    token = jwt.encode(
        {
            "role": "admin",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        },
        JWT_SECRET,
        algorithm="HS256",
    )

    return jsonify({"token": token, "expires_in": JWT_EXPIRY_HOURS * 3600})


# ─────────────────────────────────────────────
# POST /api/admin/refresh — Force re-aggregation
# ─────────────────────────────────────────────
@admin_bp.route("/refresh", methods=["POST"])
@require_admin
def admin_refresh():
    _ensure_aggregated()
    return jsonify({"status": "ok", "message": "Aggregation complete"})


# ─────────────────────────────────────────────
# GET /api/admin/kpis
# ─────────────────────────────────────────────
@admin_bp.route("/kpis", methods=["GET"])
@require_admin
def admin_kpis():
    d1, d2 = _get_date_range()
    _ensure_aggregated()

    # Real-time event counts directly from analytics_events (always fresh)
    raw_counts = {}
    try:
        raw_rows = _query("""
            SELECT event_type, COUNT(*) as cnt
            FROM analytics_events
            WHERE created_at >= %s AND created_at < %s
            GROUP BY event_type
        """, (d1, d2 + timedelta(days=1)))
        for r in raw_rows:
            raw_counts[r["event_type"]] = r["cnt"]
    except Exception as e:
        print(f"[AdminAPI] Raw counts query failed: {e}")

    rows = _get_summary_rows(d1, d2)
    if not rows:
        # No aggregated summary yet — return raw event counts if available
        if raw_counts:
            return jsonify({
                "has_data": True,
                "kpis": {
                    "searches": raw_counts.get("search", 0),
                    "detections": raw_counts.get("detection", 0),
                    "clicks": raw_counts.get("product_click", 0),
                    "buys": raw_counts.get("buy_click", 0),
                    "impressions": raw_counts.get("search_impression", 0),
                    "detection_success": 0, "search_click_rate": 0, "top1_rate": 0,
                    "conversion_rate": 0, "no_click_rate": 0, "search_fail_rate": 0,
                    "override_rate": 0, "relaxation_rate": 0,
                },
                "funnel": [
                    {"stage": "Detection", "count": raw_counts.get("detection", 0)},
                    {"stage": "Search", "count": raw_counts.get("search", 0)},
                    {"stage": "Product Click", "count": raw_counts.get("product_click", 0)},
                    {"stage": "Buy Click", "count": raw_counts.get("buy_click", 0)},
                ],
                "raw_event_counts": raw_counts,
                "data_source": "raw_events",
            })
        return jsonify({"kpis": {}, "funnel": [], "has_data": False})

    # Sum numeric KPIs across range — use max of aggregated vs raw counts
    searches = max(sum(_safe_int(r.get("total_searches")) for r in rows), raw_counts.get("search", 0))
    detections = max(sum(_safe_int(r.get("total_detections")) for r in rows), raw_counts.get("detection", 0))
    clicks = max(sum(_safe_int(r.get("total_clicks")) for r in rows), raw_counts.get("product_click", 0))
    buys = max(sum(_safe_int(r.get("total_buy_clicks")) for r in rows), raw_counts.get("buy_click", 0))
    impressions = max(sum(_safe_int(r.get("total_impressions")) for r in rows), raw_counts.get("search_impression", 0))

    # Latest day for rate metrics
    L = rows[-1]
    det_success = _safe_float(L.get("detection_success_rate")) * 100
    search_fail = _safe_float(L.get("search_failure_rate")) * 100
    top1_rate = _safe_float(L.get("top1_click_rate")) * 100
    conv_rate = _safe_float(L.get("conversion_rate")) * 100
    no_click = _safe_float(L.get("no_click_rate")) * 100
    override_freq = _safe_float(L.get("override_frequency")) * 100
    relax_rate = _safe_float(L.get("relaxation_rate")) * 100

    search_click_rate = round(clicks / max(searches, 1) * 100, 1)

    return jsonify({
        "has_data": True,
        "kpis": {
            "searches": searches,
            "detections": detections,
            "clicks": clicks,
            "buys": buys,
            "impressions": impressions,
            "detection_success": round(det_success, 1),
            "search_click_rate": search_click_rate,
            "top1_rate": round(top1_rate, 1),
            "conversion_rate": round(conv_rate, 1),
            "no_click_rate": round(no_click, 1),
            "search_fail_rate": round(search_fail, 1),
            "override_rate": round(override_freq, 1),
            "relaxation_rate": round(relax_rate, 1),
        },
        "funnel": [
            {"stage": "Detection", "count": detections},
            {"stage": "Search", "count": searches},
            {"stage": "Product Click", "count": clicks},
            {"stage": "Buy Click", "count": buys},
        ],
        "raw_event_counts": raw_counts,
        "data_source": "aggregated+raw",
    })


# ─────────────────────────────────────────────
# GET /api/admin/system-health
# ─────────────────────────────────────────────
@admin_bp.route("/system-health", methods=["GET"])
@require_admin
def admin_system_health():
    d1, d2 = _get_date_range()
    rows = _get_summary_rows(d1, d2)
    if not rows:
        return jsonify({"has_data": False})

    L = rows[-1]

    # Latency
    latency = {
        "avg": round(_safe_float(L.get("avg_latency_ms")), 0),
        "p50": round(_safe_float(L.get("p50_latency_ms")), 0),
        "p95": round(_safe_float(L.get("p95_latency_ms")), 0),
    }

    # Latency trend
    latency_trend = [
        {
            "date": str(r["summary_date"]),
            "avg": round(_safe_float(r.get("avg_latency_ms")), 0),
            "p50": round(_safe_float(r.get("p50_latency_ms")), 0),
            "p95": round(_safe_float(r.get("p95_latency_ms")), 0),
        }
        for r in rows
    ]

    # YOLO confidence
    yolo_conf = round(_safe_float(L.get("avg_yolo_confidence")) * 100, 1)

    # System rates
    search_fail = round(_safe_float(L.get("search_failure_rate")) * 100, 1)
    override_rate = round(_safe_float(L.get("override_frequency")) * 100, 1)
    relax_rate = round(_safe_float(L.get("relaxation_rate")) * 100, 1)

    # Detection by category
    det_cat = _parse_json_field(L.get("detection_by_category"))
    categories = []
    summary_stats = det_cat.pop("_summary", {})
    for cat, data in det_cat.items():
        if isinstance(data, dict):
            categories.append({
                "category": cat,
                "detections": data.get("detections", 0),
                "avg_confidence": round(data.get("avg_conf", 0) * 100, 1),
                "pct_of_total": round(data.get("pct_of_detections", 0) * 100, 1),
            })
    categories.sort(key=lambda x: -x["avg_confidence"])

    return jsonify({
        "has_data": True,
        "yolo_confidence": yolo_conf,
        "search_fail_rate": search_fail,
        "override_rate": override_rate,
        "relaxation_rate": relax_rate,
        "latency": latency,
        "latency_trend": latency_trend,
        "detection_categories": categories,
        "detection_summary": {
            "total_frames": summary_stats.get("total_frames", 0),
            "with_detection": summary_stats.get("frames_with_detection", 0),
            "no_detection": summary_stats.get("frames_no_detection", 0),
        },
    })


# ─────────────────────────────────────────────
# GET /api/admin/recommendation-quality
# ─────────────────────────────────────────────
@admin_bp.route("/recommendation-quality", methods=["GET"])
@require_admin
def admin_recommendation_quality():
    d1, d2 = _get_date_range()
    rows = _get_summary_rows(d1, d2)
    if not rows:
        return jsonify({"has_data": False})

    L = rows[-1]

    # CTR by rank
    ctr_by_rank = [
        {"rank": f"Rank {i}", "ctr": round(_safe_float(L.get(f"ctr_rank_{i}")) * 100, 1)}
        for i in range(1, 6)
    ]

    # Top-1 trend
    top1_trend = [
        {
            "date": str(r["summary_date"]),
            "rate": round(_safe_float(r.get("top1_click_rate")) * 100, 1),
        }
        for r in rows
    ]

    avg_vis_sim = round(_safe_float(L.get("avg_visual_similarity_clicked")), 3)

    return jsonify({
        "has_data": True,
        "ctr_by_rank": ctr_by_rank,
        "top1_trend": top1_trend,
        "avg_visual_similarity": avg_vis_sim,
    })


# ─────────────────────────────────────────────
# GET /api/admin/business-insights
# ─────────────────────────────────────────────
@admin_bp.route("/business-insights", methods=["GET"])
@require_admin
def admin_business_insights():
    d1, d2 = _get_date_range()
    rows = _get_summary_rows(d1, d2)
    if not rows:
        return jsonify({"has_data": False})

    L = rows[-1]

    # Aggregated demand data
    top_categories = _aggregate_json_column(rows, "category_demand")
    top_colors = _aggregate_json_column(rows, "color_demand")
    scene_distribution = _aggregate_json_column(rows, "scene_distribution")

    # Scene-click correlation (from latest day)
    sc_corr = _parse_json_field(L.get("scene_click_corr"))
    scene_click = []
    for scene, data in sc_corr.items():
        if isinstance(data, dict):
            for cat, cnt in data.get("categories", {}).items():
                scene_click.append({"scene": scene, "category": cat, "clicks": cnt})

    # Demand-supply gap
    gap_data = _parse_json_field(L.get("demand_supply_gaps"))
    demand_supply = []
    for cat, d in gap_data.items():
        if isinstance(d, dict):
            demand_supply.append({
                "category": cat,
                "demand": d.get("demand", 0),
                "catalog_size": d.get("catalog_size", 0),
                "avg_results": d.get("avg_results_shown", 0),
                "gap_score": d.get("gap_score", 0),
            })
    demand_supply.sort(key=lambda x: -x["gap_score"])

    return jsonify({
        "has_data": True,
        "top_categories": top_categories,
        "top_colors": top_colors,
        "scene_distribution": scene_distribution,
        "scene_click_correlation": scene_click,
        "demand_supply_gap": demand_supply,
    })


# ─────────────────────────────────────────────
# GET /api/admin/user-behaviour
# ─────────────────────────────────────────────
@admin_bp.route("/user-behaviour", methods=["GET"])
@require_admin
def admin_user_behaviour():
    d1, d2 = _get_date_range()
    rows = _get_summary_rows(d1, d2)
    if not rows:
        return jsonify({"has_data": False})

    L = rows[-1]

    avg_sps = round(_safe_float(L.get("avg_searches_per_session")), 1)
    avg_fps = round(_safe_float(L.get("avg_filters_per_search")), 1)
    impressions = sum(_safe_int(r.get("total_impressions")) for r in rows)

    # Top filters
    top_filters = _aggregate_json_column(rows, "top_filters")

    # Session depth trend
    session_trend = [
        {
            "date": str(r["summary_date"]),
            "searches_per_session": round(_safe_float(r.get("avg_searches_per_session")), 1),
            "filters_per_search": round(_safe_float(r.get("avg_filters_per_search")), 1),
        }
        for r in rows
    ]

    return jsonify({
        "has_data": True,
        "avg_searches_per_session": avg_sps,
        "avg_filters_per_search": avg_fps,
        "impressions": impressions,
        "top_filters": top_filters,
        "session_trend": session_trend,
    })


# ─────────────────────────────────────────────
# GET /api/admin/trends
# ─────────────────────────────────────────────
@admin_bp.route("/trends", methods=["GET"])
@require_admin
def admin_trends():
    d1, d2 = _get_date_range()
    rows = _get_summary_rows(d1, d2)
    if not rows:
        return jsonify({"has_data": False})

    volume_trend = [
        {
            "date": str(r["summary_date"]),
            "searches": _safe_int(r.get("total_searches")),
            "clicks": _safe_int(r.get("total_clicks")),
            "buys": _safe_int(r.get("total_buy_clicks")),
        }
        for r in rows
    ]

    latency_trend = [
        {
            "date": str(r["summary_date"]),
            "avg": round(_safe_float(r.get("avg_latency_ms")), 0),
            "p95": round(_safe_float(r.get("p95_latency_ms")), 0),
        }
        for r in rows
    ]

    return jsonify({
        "has_data": True,
        "volume": volume_trend,
        "latency": latency_trend,
    })


# ─────────────────────────────────────────────
# GET /api/admin/raw-events
# ─────────────────────────────────────────────
@admin_bp.route("/raw-events", methods=["GET"])
@require_admin
def admin_raw_events():
    d1, d2 = _get_date_range()
    event_type = request.args.get("event_type", "All")

    where = "created_at >= %s AND created_at < %s"
    params = [d1, d2 + timedelta(days=1)]
    if event_type != "All":
        where += " AND event_type = %s"
        params.append(event_type)

    rows = _query(f"""
        SELECT event_id, event_type, detected_category, detected_color,
               scene_label, clicked_product_id, rank_clicked,
               visual_similarity, final_score, latency_ms,
               device_type, created_at
        FROM analytics_events
        WHERE {where}
        ORDER BY created_at DESC LIMIT 100
    """, params)

    # Serialize dates
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = str(r["created_at"])

    return jsonify({
        "events": rows,
        "count": len(rows),
        "event_types": [
            "All", "detection_request", "detection", "search", "product_click",
            "buy_click", "attribute_extraction", "explanation_view", "search_impression"
        ],
    })


# ─────────────────────────────────────────────
# GET /api/admin/ratings
# ─────────────────────────────────────────────
@admin_bp.route("/ratings", methods=["GET"])
@require_admin
def admin_ratings():
    rows = _query("SELECT * FROM ratings ORDER BY id DESC LIMIT 10")

    # Serialize and parse
    for r in rows:
        for key in ("created_at", "timestamp"):
            if r.get(key):
                r[key] = str(r[key])
        if r.get("filters"):
            r["filters"] = _parse_json_field(r["filters"])

    # Summary stats
    ratings_vals = [r.get("rating", 0) for r in rows if r.get("rating")]
    avg_rating = round(sum(ratings_vals) / max(len(ratings_vals), 1), 1)
    high_satisfaction = sum(1 for v in ratings_vals if v >= 4)

    return jsonify({
        "ratings": rows,
        "count": len(rows),
        "summary": {
            "avg_rating": avg_rating,
            "total_sessions": len(rows),
            "high_satisfaction": high_satisfaction,
        },
    })


# ─────────────────────────────────────────────
# POST /api/admin/insights
# ─────────────────────────────────────────────
@admin_bp.route("/insights", methods=["POST"])
@require_admin
def admin_insights():
    body = request.get_json() or {}
    rating_data = body.get("rating_data")

    if not rating_data:
        return jsonify({"error": "rating_data required"}), 400

    try:
        from models.insights_engine import InsightsEngine
        engine = InsightsEngine(None)
        analysis = engine.generate_report(rating_data)
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
