"""
Daily Analytics Aggregation Script (Incremental)
==================================================
Processes analytics_events → analytics_daily_summary.
Uses last_processed_event_id for incremental processing.

Run: python tools/analytics_aggregation.py [YYYY-MM-DD]
Default: aggregates yesterday. Pass 'today' for current day.
"""

import sys
import json
import psycopg2
import psycopg2.extras
import os
from datetime import date, timedelta, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DB_CONFIG, DATABASE_URL


def aggregate_day(target_date: date):
    """Aggregate events for a single day into analytics_daily_summary."""
    print(f"\n{'='*60}")
    print(f"Aggregating analytics for: {target_date}")
    print(f"{'='*60}")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── Check incremental state ──
    cur.execute(
        "SELECT last_processed_event_id FROM analytics_daily_summary WHERE summary_date = %s",
        (target_date,)
    )
    existing = cur.fetchone()

    day_start = datetime(target_date.year, target_date.month, target_date.day)
    day_end = day_start + timedelta(days=1)

    extra_where = ""
    if existing and existing.get("last_processed_event_id"):
        last_id = existing["last_processed_event_id"]
        extra_where = f"AND event_id > '{last_id}'"
        print(f"  Incremental: after {last_id}")

    # ── Fetch events ──
    cur.execute(f"""
        SELECT * FROM analytics_events
        WHERE created_at >= %s AND created_at < %s {extra_where}
        ORDER BY created_at ASC
    """, (day_start, day_end))
    events = cur.fetchall()

    if not events:
        print("  No new events.")
        cur.close(); conn.close()
        return

    print(f"  Processing {len(events)} events...")

    # ── Categorize events ──
    by_type = {}
    for e in events:
        t = e["event_type"]
        by_type.setdefault(t, []).append(e)

    det_requests = by_type.get("detection_request", [])
    detections = by_type.get("detection", [])
    searches = by_type.get("search", [])
    clicks = by_type.get("product_click", [])
    buy_clicks = by_type.get("buy_click", [])
    impressions = by_type.get("search_impression", [])
    extractions = by_type.get("attribute_extraction", [])

    total_frames = len(det_requests)
    frames_ok = len([d for d in det_requests if (d.get("result_count") or 0) > 0])
    total_searches = len(searches)
    total_detections = len(detections)
    total_clicks = len(clicks)
    total_buy_clicks = len(buy_clicks)
    total_impressions = sum(len(e.get("impression_ids") or []) for e in impressions)

    # ── Rate metrics ──
    detection_success_rate = round(frames_ok / max(total_frames, 1), 4)

    relaxed = len([s for s in searches if s.get("relaxation_steps")])
    search_failure_rate = round(relaxed / max(total_searches, 1), 4)

    top1 = len([c for c in clicks if c.get("rank_clicked") == 1])
    top1_click_rate = round(top1 / max(total_clicks, 1), 4)

    conversion_rate = round(total_buy_clicks / max(total_searches, 1), 4)

    override_count = len([s for s in searches if s.get("override_category")])
    override_frequency = round(override_count / max(total_searches, 1), 4)
    relaxation_rate = search_failure_rate  # same thing

    # ── Latency ──
    lats = sorted([s["latency_ms"] for s in searches if s.get("latency_ms")])
    avg_lat = sum(lats) / len(lats) if lats else 0
    p50_lat = lats[len(lats) // 2] if lats else 0
    p95_lat = lats[int(len(lats) * 0.95)] if len(lats) > 1 else p50_lat

    # ── Model health ──
    yolo_confs = [d["yolo_confidence"] for d in detections if d.get("yolo_confidence")]
    avg_yolo = sum(yolo_confs) / len(yolo_confs) if yolo_confs else None

    ext_quals = [e["extraction_quality"] for e in extractions if e.get("extraction_quality")]
    avg_ext = sum(ext_quals) / len(ext_quals) if ext_quals else None

    click_vis = [c["visual_similarity"] for c in clicks if c.get("visual_similarity")]
    avg_vis_clicked = sum(click_vis) / len(click_vis) if click_vis else None

    # ── CTR by rank ──
    ctr = {}
    for rank_pos in range(1, 6):
        rc = len([c for c in clicks if c.get("rank_clicked") == rank_pos])
        ctr[f"ctr_rank_{rank_pos}"] = round(rc / max(len(impressions), 1), 4)

    # ── Pool sizes ──
    pools = [s["pool_size"] for s in searches if s.get("pool_size")]
    avg_pool = sum(pools) / len(pools) if pools else 0

    # ── Category demand ──
    cat_demand = {}
    for s in searches:
        cat = s.get("detected_category") or "unknown"
        cat_demand[cat] = cat_demand.get(cat, 0) + 1

    # ── Color demand ──
    color_demand = {}
    for e in extractions:
        c = e.get("detected_color") or "unknown"
        color_demand[c] = color_demand.get(c, 0) + 1

    # ── Scene distribution ──
    scene_dist = {}
    for d in detections:
        s = d.get("scene_label") or "unknown"
        scene_dist[s] = scene_dist.get(s, 0) + 1

    # ── Scene-click correlation ──
    scene_clicks = {}
    for c in clicks:
        scene = c.get("scene_label") or "unknown"
        scene_clicks.setdefault(scene, {"clicks": 0, "categories": {}})
        scene_clicks[scene]["clicks"] += 1
        cat = c.get("detected_category") or "unknown"
        scene_clicks[scene]["categories"][cat] = scene_clicks[scene]["categories"].get(cat, 0) + 1

    # ── Session behavior ──
    sessions = {}
    for e in events:
        sid = e.get("session_id")
        if sid:
            sessions.setdefault(sid, {"searches": 0, "filter_count": 0})
            if e["event_type"] == "search":
                sessions[sid]["searches"] += 1
                uf = e.get("user_filters")
                if uf:
                    fcount = len(uf) if isinstance(uf, dict) else len(json.loads(uf)) if isinstance(uf, str) else 0
                    sessions[sid]["filter_count"] += fcount

    sess_list = list(sessions.values())
    avg_searches_per_session = round(
        sum(s["searches"] for s in sess_list) / max(len(sess_list), 1), 2
    )
    total_filters = sum(s["filter_count"] for s in sess_list)
    avg_filters_per_search = round(total_filters / max(total_searches, 1), 2)

    # ── Top filters ──
    filter_counts = {}
    for s in searches:
        uf = s.get("user_filters")
        if uf:
            filters = uf if isinstance(uf, dict) else json.loads(uf) if isinstance(uf, str) else {}
            for k in filters:
                filter_counts[k] = filter_counts.get(k, 0) + 1

    # ── Detection success by category (root cause analysis) ──
    # Track detections per detection_request per category (unbiased)
    det_by_cat = {}
    # Count how many times each category was detected
    for d in detections:
        cat = d.get("detected_category") or "unknown"
        det_by_cat.setdefault(cat, {"detections": 0, "avg_conf": []})
        det_by_cat[cat]["detections"] += 1
        if d.get("yolo_confidence"):
            det_by_cat[cat]["avg_conf"].append(d["yolo_confidence"])
    # Also count failures (frames with nothing detected)
    no_det = len([r for r in det_requests if (r.get("result_count") or 0) == 0])
    # Compute rates: detections / total_detection_requests (not per-category total)
    for cat in det_by_cat:
        confs = det_by_cat[cat]["avg_conf"]
        det_by_cat[cat]["avg_conf"] = round(sum(confs) / max(len(confs), 1), 4) if confs else 0
        det_by_cat[cat]["pct_of_detections"] = round(
            det_by_cat[cat]["detections"] / max(total_detections, 1), 4
        )
    det_by_cat["_summary"] = {
        "total_frames": total_frames,
        "frames_with_detection": frames_ok,
        "frames_no_detection": no_det,
        "success_rate": detection_success_rate,
    }

    # ── No-click rate (retrieval failure) with context ──
    search_sessions_with_click = set()
    for c in clicks:
        sid = c.get("session_id")
        if sid:
            search_sessions_with_click.add(sid)
    search_sessions = set(s.get("session_id") for s in searches if s.get("session_id"))
    sessions_without_click = search_sessions - search_sessions_with_click
    no_click_rate = round(len(sessions_without_click) / max(len(search_sessions), 1), 4)

    # Avg results returned (context for no-click interpretation)
    result_counts = [s.get("result_count", 0) for s in searches]
    avg_results_returned = round(sum(result_counts) / max(len(result_counts), 1), 1)

    # ── Product supply from actual DB (visual_attributes) ──
    product_supply = {}
    try:
        cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category")
        for row in cur.fetchall():
            product_supply[row["category"]] = row["count"]
    except Exception as e:
        print(f"  ⚠️ Could not query product supply: {e}")

    # ── Demand-supply gap (gap = demand / avg_results_returned per category) ──
    demand_supply = {}
    for cat, demand in cat_demand.items():
        actual_supply = product_supply.get(cat, 0)
        cat_searches = [s for s in searches if s.get("detected_category") == cat]
        avg_results = sum(s.get("result_count", 0) for s in cat_searches) / max(len(cat_searches), 1)
        # gap_score = demand / avg_results_returned (high = users want more than system shows)
        gap_score = round(demand / max(avg_results, 1), 1)
        demand_supply[cat] = {
            "demand": demand,
            "catalog_size": actual_supply,
            "avg_results_shown": round(avg_results, 1),
            "gap_score": gap_score
        }

    last_id = str(events[-1]["event_id"])

    # ── UPSERT ──
    cur2 = conn.cursor()
    cur2.execute("""
        INSERT INTO analytics_daily_summary (
            summary_date, total_searches, total_detections, total_clicks, total_buy_clicks,
            total_impressions, avg_latency_ms, p50_latency_ms, p95_latency_ms,
            avg_yolo_confidence, avg_extraction_quality, avg_visual_similarity_clicked,
            ctr_rank_1, ctr_rank_2, ctr_rank_3, ctr_rank_4, ctr_rank_5,
            relaxation_rate, override_frequency, avg_pool_size,
            detection_success_rate, search_failure_rate, top1_click_rate, conversion_rate,
            scene_click_corr, avg_searches_per_session, avg_filters_per_search,
            category_demand, color_demand, scene_distribution,
            top_filters, demand_supply_gaps,
            detection_by_category, no_click_rate, product_supply,
            last_processed_event_id, computed_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (summary_date) DO UPDATE SET
            total_searches = EXCLUDED.total_searches,
            total_detections = EXCLUDED.total_detections,
            total_clicks = EXCLUDED.total_clicks,
            total_buy_clicks = EXCLUDED.total_buy_clicks,
            total_impressions = EXCLUDED.total_impressions,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            p50_latency_ms = EXCLUDED.p50_latency_ms,
            p95_latency_ms = EXCLUDED.p95_latency_ms,
            avg_yolo_confidence = EXCLUDED.avg_yolo_confidence,
            avg_extraction_quality = EXCLUDED.avg_extraction_quality,
            avg_visual_similarity_clicked = EXCLUDED.avg_visual_similarity_clicked,
            ctr_rank_1 = EXCLUDED.ctr_rank_1,
            ctr_rank_2 = EXCLUDED.ctr_rank_2,
            ctr_rank_3 = EXCLUDED.ctr_rank_3,
            ctr_rank_4 = EXCLUDED.ctr_rank_4,
            ctr_rank_5 = EXCLUDED.ctr_rank_5,
            relaxation_rate = EXCLUDED.relaxation_rate,
            override_frequency = EXCLUDED.override_frequency,
            avg_pool_size = EXCLUDED.avg_pool_size,
            detection_success_rate = EXCLUDED.detection_success_rate,
            search_failure_rate = EXCLUDED.search_failure_rate,
            top1_click_rate = EXCLUDED.top1_click_rate,
            conversion_rate = EXCLUDED.conversion_rate,
            scene_click_corr = EXCLUDED.scene_click_corr,
            avg_searches_per_session = EXCLUDED.avg_searches_per_session,
            avg_filters_per_search = EXCLUDED.avg_filters_per_search,
            category_demand = EXCLUDED.category_demand,
            color_demand = EXCLUDED.color_demand,
            scene_distribution = EXCLUDED.scene_distribution,
            top_filters = EXCLUDED.top_filters,
            demand_supply_gaps = EXCLUDED.demand_supply_gaps,
            detection_by_category = EXCLUDED.detection_by_category,
            no_click_rate = EXCLUDED.no_click_rate,
            product_supply = EXCLUDED.product_supply,
            last_processed_event_id = EXCLUDED.last_processed_event_id,
            computed_at = NOW()
    """, (
        target_date, total_searches, total_detections, total_clicks, total_buy_clicks,
        total_impressions, round(avg_lat, 1), round(p50_lat, 1), round(p95_lat, 1),
        avg_yolo, avg_ext, avg_vis_clicked,
        ctr.get("ctr_rank_1", 0), ctr.get("ctr_rank_2", 0),
        ctr.get("ctr_rank_3", 0), ctr.get("ctr_rank_4", 0), ctr.get("ctr_rank_5", 0),
        relaxation_rate, override_frequency, round(avg_pool, 1),
        detection_success_rate, search_failure_rate, top1_click_rate, conversion_rate,
        json.dumps(scene_clicks), avg_searches_per_session, avg_filters_per_search,
        json.dumps(cat_demand), json.dumps(color_demand), json.dumps(scene_dist),
        json.dumps(filter_counts), json.dumps(demand_supply),
        json.dumps(det_by_cat), no_click_rate, json.dumps(product_supply),
        last_id,
    ))

    conn.commit()
    cur.close(); cur2.close(); conn.close()

    print(f"  ✅ Summary written for {target_date}")
    print(f"     Searches: {total_searches} | Clicks: {total_clicks} | Buys: {total_buy_clicks}")
    print(f"     Detection Success: {detection_success_rate:.1%} | Search Failure: {search_failure_rate:.1%}")
    print(f"     Top-1 Rate: {top1_click_rate:.1%} | Conversion: {conversion_rate:.1%}")
    print(f"     No-Click Rate: {no_click_rate:.1%}")
    print(f"     Latency p50: {p50_lat:.0f}ms p95: {p95_lat:.0f}ms")
    print(f"     Sessions: {len(sess_list)} | Avg searches/session: {avg_searches_per_session}")
    if det_by_cat:
        print(f"     Detection by category:")
        for cat, d in sorted(det_by_cat.items(), key=lambda x: x[1]['rate']):
            if cat != '_no_detection':
                print(f"       {cat}: {d['rate']:.0%} ({d['success']}/{d['total']})")
    if product_supply:
        print(f"     Product catalog: {sum(product_supply.values())} items across {len(product_supply)} categories")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "today":
            target = date.today()
        else:
            target = date.fromisoformat(arg)
    else:
        target = date.today() - timedelta(days=1)

    aggregate_day(target)
    print(f"\n{'='*60}\nAggregation complete.")
