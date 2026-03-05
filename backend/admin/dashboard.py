"""
ShopWhatYouSee — Commerce Intelligence Dashboard
===================================================
Run: streamlit run admin/dashboard.py --server.port 8501

Architecture:
  KPIs + Trends → analytics_daily_summary (pre-aggregated, fast)
  Drill-down → analytics_events (raw, only for exploration)

Sections:
  1. KPI Overview + Conversion Funnel
  2. AI System Health
  3. Recommendation Quality
  4. Business Insights
  5. User Behaviour Analytics
  6. Daily Trends
  7. Raw Event Explorer
"""

import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db_config import DATABASE_URL
from dotenv import load_dotenv
load_dotenv()

ADMIN_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "swys2026")


# ── AUTH ──
def check_auth():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if not st.session_state.auth:
        st.title("🔒 ShopWhatYouSee Admin")
        pwd = st.text_input("Password:", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid password")
        st.stop()


# ── DB ──
@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def q(sql, params=None):
    """Query → DataFrame."""
    try:
        return pd.read_sql(sql, get_conn(), params=params)
    except Exception:
        return pd.read_sql(sql, psycopg2.connect(DATABASE_URL), params=params)


def safe_pct(n, d):
    return round(n / max(d, 1) * 100, 1)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    check_auth()
    st.set_page_config(page_title="SWYS Analytics", page_icon="📊", layout="wide")
    st.title("📊 ShopWhatYouSee — Commerce Intelligence Dashboard")

    # ── SIDEBAR ──
    st.sidebar.header("🔍 Filters")
    dr = st.sidebar.date_input("Date Range",
        value=(date.today() - timedelta(days=30), date.today()),
        max_value=date.today())
    if isinstance(dr, tuple) and len(dr) == 2:
        d1, d2 = dr
    else:
        d1 = d2 = date.today()

    # ═══════════════════════════════════════════════
    # AUTO-AGGREGATE TODAY'S DATA (INCREMENTAL)
    # ═══════════════════════════════════════════════
    try:
        from tools.analytics_aggregation import aggregate_day
        aggregate_day(date.today())
    except Exception as e:
        print(f"⚠️ Auto-aggregation error: {e}")

    # ═══════════════════════════════════════════════
    # LOAD FROM analytics_daily_summary (fast!)
    # ═══════════════════════════════════════════════
    summary = q("""
        SELECT * FROM analytics_daily_summary
        WHERE summary_date >= %s AND summary_date <= %s
        ORDER BY summary_date
    """, (d1, d2))

    has_summary = not summary.empty

    if has_summary:
        # Aggregate across the selected range
        S = summary.sum(numeric_only=True)
        L = summary.iloc[-1]  # latest day for rate metrics

        searches = int(S.get("total_searches", 0) or 0)
        detections = int(S.get("total_detections", 0) or 0)
        clicks = int(S.get("total_clicks", 0) or 0)
        buys = int(S.get("total_buy_clicks", 0) or 0)
        impressions = int(S.get("total_impressions", 0) or 0)

        # Use latest-day rates (not summed)
        det_success = float(L.get("detection_success_rate", 0) or 0) * 100
        search_fail = float(L.get("search_failure_rate", 0) or 0) * 100
        top1_rate = float(L.get("top1_click_rate", 0) or 0) * 100
        conv_rate = float(L.get("conversion_rate", 0) or 0) * 100
        override_freq = float(L.get("override_frequency", 0) or 0) * 100
        relax_rate = float(L.get("relaxation_rate", 0) or 0) * 100
        avg_lat = float(L.get("avg_latency_ms", 0) or 0)
        p50_lat = float(L.get("p50_latency_ms", 0) or 0)
        p95_lat = float(L.get("p95_latency_ms", 0) or 0)
        avg_yolo = float(L.get("avg_yolo_confidence", 0) or 0) * 100
        avg_vis = float(L.get("avg_visual_similarity_clicked", 0) or 0)
        avg_sps = float(L.get("avg_searches_per_session", 0) or 0)
        avg_fps = float(L.get("avg_filters_per_search", 0) or 0)
    else:
        st.warning("⚠️ No aggregated data found. Run `python tools/analytics_aggregation.py today` first.")
        searches = detections = clicks = buys = impressions = 0
        det_success = search_fail = top1_rate = conv_rate = override_freq = relax_rate = 0
        avg_lat = p50_lat = p95_lat = avg_yolo = avg_vis = avg_sps = avg_fps = 0
        no_click = 0
        L = {}
        S = pd.Series(dtype=float)

    # ═══════════════════════════════════════════════
    # SECTION 1: KPI OVERVIEW + CONVERSION FUNNEL
    # ═══════════════════════════════════════════════
    st.header("📈 KPI Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔍 Searches", searches)
    c2.metric("👁️ Detections", detections)
    c3.metric("👆 Clicks", clicks)
    c4.metric("🛒 Buy Clicks", buys)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🎯 Detection Success", f"{det_success:.1f}%")
    c6.metric("📊 Search → Click", f"{safe_pct(clicks, searches)}%")
    c7.metric("🏆 Top-1 Click Rate", f"{top1_rate:.1f}%")
    c8.metric("💰 Conversion Rate", f"{conv_rate:.1f}%")

    # Row 3: failure metrics
    no_click = float(L.get("no_click_rate", 0) or 0) * 100 if has_summary else 0
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("🚫 No-Click Rate", f"{no_click:.1f}%",
              delta="High = retrieval quality weak" if no_click > 40 else "OK")
    c10.metric("❌ Search Failure", f"{search_fail:.1f}%")
    c11.metric("✏️ Override Rate", f"{override_freq:.1f}%")
    c12.metric("🔄 Relaxation Rate", f"{relax_rate:.1f}%")

    # Conversion Funnel
    st.subheader("🔄 Conversion Funnel")
    funnel = pd.DataFrame({
        "Stage": ["Detection", "Search", "Product Click", "Buy Click"],
        "Count": [detections, searches, clicks, buys],
    })
    st.bar_chart(funnel.set_index("Stage"))

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 2: AI SYSTEM HEALTH
    # ═══════════════════════════════════════════════
    st.header("🤖 AI System Health")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 YOLO Confidence", f"{avg_yolo:.1f}%")
    c2.metric("❌ Search Failure Rate", f"{search_fail:.1f}%",
              delta="High = weak catalog" if search_fail > 25 else "OK")
    c3.metric("✏️ Override Rate", f"{override_freq:.1f}%",
              delta="High = AGMAN weak" if override_freq > 30 else "OK")
    c4.metric("🔄 Relaxation Rate", f"{relax_rate:.1f}%")

    # Latency distribution
    st.subheader("⚡ Search Latency Distribution")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg", f"{avg_lat:.0f}ms")
    c2.metric("p50 (Median)", f"{p50_lat:.0f}ms")
    c3.metric("p95", f"{p95_lat:.0f}ms")

    # Latency trend from daily summary
    if has_summary:
        lat_trend = summary[["summary_date", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms"]].copy()
        lat_trend = lat_trend.set_index("summary_date")
        lat_trend.columns = ["avg", "p50", "p95"]
        st.line_chart(lat_trend)

    # Detection analysis by category (root cause)
    st.subheader("🔬 Detection Analysis by Category")
    st.caption("Shows YOLO performance per object type — low confidence = model weakness")
    if has_summary:
        det_cat = L.get("detection_by_category")
        if det_cat:
            dc = det_cat if isinstance(det_cat, dict) else json.loads(det_cat) if isinstance(det_cat, str) else {}
            rows = []
            for cat, d in dc.items():
                if cat != "_summary" and isinstance(d, dict):
                    rows.append({
                        "Category": cat,
                        "Detections": d.get("detections", 0),
                        "Avg Confidence": f"{d.get('avg_conf', 0) * 100:.1f}%",
                        "% of Total": f"{d.get('pct_of_detections', 0) * 100:.1f}%",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("Avg Confidence"),
                             use_container_width=True, hide_index=True)
            # Summary stats
            summ = dc.get("_summary", {})
            if summ:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Frames", summ.get("total_frames", 0))
                c2.metric("✅ With Detection", summ.get("frames_with_detection", 0))
                c3.metric("❌ No Detection", summ.get("frames_no_detection", 0))

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 3: RECOMMENDATION QUALITY
    # ═══════════════════════════════════════════════
    st.header("🎯 Recommendation Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CTR by Rank Position")
        if has_summary:
            ctr_data = {
                "Rank 1": float(L.get("ctr_rank_1", 0) or 0) * 100,
                "Rank 2": float(L.get("ctr_rank_2", 0) or 0) * 100,
                "Rank 3": float(L.get("ctr_rank_3", 0) or 0) * 100,
                "Rank 4": float(L.get("ctr_rank_4", 0) or 0) * 100,
                "Rank 5": float(L.get("ctr_rank_5", 0) or 0) * 100,
            }
            ctr_df = pd.DataFrame({"Rank": ctr_data.keys(), "CTR %": ctr_data.values()})
            st.bar_chart(ctr_df.set_index("Rank"))
        else:
            st.info("No CTR data.")

    with col2:
        st.subheader("Top-1 Success Rate Trend")
        if has_summary and "top1_click_rate" in summary.columns:
            t1_trend = summary[["summary_date", "top1_click_rate"]].copy()
            t1_trend["top1_click_rate"] = t1_trend["top1_click_rate"].fillna(0) * 100
            st.line_chart(t1_trend.set_index("summary_date"))
            st.caption("Higher = retrieval ranks best product first")
        else:
            st.info("No data.")

    st.metric("👁️ Avg Visual Similarity (Clicked)", f"{avg_vis:.3f}")

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 4: BUSINESS INSIGHTS
    # ═══════════════════════════════════════════════
    st.header("💼 Business & Market Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏷️ Top Categories")
        if has_summary:
            cat_agg = {}
            for _, row in summary.iterrows():
                cd = row.get("category_demand")
                if cd:
                    d = cd if isinstance(cd, dict) else json.loads(cd) if isinstance(cd, str) else {}
                    for k, v in d.items():
                        cat_agg[k] = cat_agg.get(k, 0) + v
            if cat_agg:
                cd_df = pd.DataFrame(sorted(cat_agg.items(), key=lambda x: -x[1])[:10],
                                     columns=["Category", "Searches"])
                st.bar_chart(cd_df.set_index("Category"))

    with col2:
        st.subheader("🎨 Top Colors")
        if has_summary:
            col_agg = {}
            for _, row in summary.iterrows():
                cd = row.get("color_demand")
                if cd:
                    d = cd if isinstance(cd, dict) else json.loads(cd) if isinstance(cd, str) else {}
                    for k, v in d.items():
                        col_agg[k] = col_agg.get(k, 0) + v
            if col_agg:
                cl_df = pd.DataFrame(sorted(col_agg.items(), key=lambda x: -x[1])[:10],
                                     columns=["Color", "Count"])
                st.bar_chart(cl_df.set_index("Color"))

    with col3:
        st.subheader("🏙️ Scene Distribution")
        if has_summary:
            sc_agg = {}
            for _, row in summary.iterrows():
                sd = row.get("scene_distribution")
                if sd:
                    d = sd if isinstance(sd, dict) else json.loads(sd) if isinstance(sd, str) else {}
                    for k, v in d.items():
                        sc_agg[k] = sc_agg.get(k, 0) + v
            if sc_agg:
                sc_df = pd.DataFrame(sorted(sc_agg.items(), key=lambda x: -x[1])[:10],
                                     columns=["Scene", "Count"])
                st.bar_chart(sc_df.set_index("Scene"))

    # Scene-click correlation
    st.subheader("🏙️→👕 Scene → Category Click Correlation")
    st.caption("Which scenes drive clicks on which categories")
    if has_summary:
        sc_corr = L.get("scene_click_corr")
        if sc_corr:
            corr = sc_corr if isinstance(sc_corr, dict) else json.loads(sc_corr) if isinstance(sc_corr, str) else {}
            if corr:
                rows = []
                for scene, data in corr.items():
                    if isinstance(data, dict):
                        for cat, cnt in data.get("categories", {}).items():
                            rows.append({"Scene": scene, "Category": cat, "Clicks": cnt})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Demand-Supply Gap (using actual catalog counts)
    st.subheader("📊 Demand vs Supply Gap")
    st.caption("gap_score = (demand / catalog_size) × 100. High = inventory shortage.")
    if has_summary:
        gap = L.get("demand_supply_gaps")
        if gap:
            g = gap if isinstance(gap, dict) else json.loads(gap) if isinstance(gap, str) else {}
            if g:
                rows = []
                for cat, d in g.items():
                    if isinstance(d, dict):
                        rows.append({
                            "Category": cat,
                            "Search Demand": d.get("demand", 0),
                            "Catalog Size": d.get("catalog_size", 0),
                            "Avg Results Shown": d.get("avg_results_shown", 0),
                            "Gap Score": d.get("gap_score", 0)
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows).sort_values("Gap Score", ascending=False),
                                 use_container_width=True, hide_index=True)

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 5: USER BEHAVIOUR ANALYTICS
    # ═══════════════════════════════════════════════
    st.header("👤 User Behaviour Analytics")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔄 Avg Searches / Session", f"{avg_sps:.1f}")
    c2.metric("🏷️ Avg Filters / Search", f"{avg_fps:.1f}")
    c3.metric("💡 Impressions", impressions)

    # Top filters used
    st.subheader("🏷️ Most Used Filters")
    if has_summary:
        filt_agg = {}
        for _, row in summary.iterrows():
            tf = row.get("top_filters")
            if tf:
                d = tf if isinstance(tf, dict) else json.loads(tf) if isinstance(tf, str) else {}
                for k, v in d.items():
                    filt_agg[k] = filt_agg.get(k, 0) + v
        if filt_agg:
            f_df = pd.DataFrame(sorted(filt_agg.items(), key=lambda x: -x[1])[:10],
                                columns=["Filter", "Usage Count"])
            st.bar_chart(f_df.set_index("Filter"))

    # Session trends
    if has_summary and "avg_searches_per_session" in summary.columns:
        st.subheader("Session Depth Trend")
        sess_trend = summary[["summary_date", "avg_searches_per_session", "avg_filters_per_search"]].copy()
        sess_trend = sess_trend.set_index("summary_date")
        sess_trend.columns = ["Searches/Session", "Filters/Search"]
        st.line_chart(sess_trend)

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 6: DAILY TRENDS (from daily_summary)
    # ═══════════════════════════════════════════════
    st.header("📉 Daily Trends")

    if has_summary:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volume")
            vol = summary[["summary_date", "total_searches", "total_clicks", "total_buy_clicks"]].copy()
            vol = vol.set_index("summary_date")
            vol.columns = ["Searches", "Clicks", "Buys"]
            st.line_chart(vol)
        with col2:
            st.subheader("Latency")
            lat = summary[["summary_date", "avg_latency_ms", "p95_latency_ms"]].copy()
            lat = lat.set_index("summary_date")
            lat.columns = ["Avg", "p95"]
            st.line_chart(lat)
    else:
        st.info("Run aggregation to see trends.")

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 7: RAW EVENT EXPLORER (only section using raw table)
    # ═══════════════════════════════════════════════
    st.header("🔎 Raw Event Explorer")
    st.caption("⚠️ This section queries raw events. Use sparingly on large datasets.")

    evt_filter = st.selectbox("Event Type",
        ["All", "detection_request", "detection", "search", "product_click",
         "buy_click", "attribute_extraction", "explanation_view", "search_impression"])

    raw_where = "created_at >= %s AND created_at < %s"
    raw_params = [d1, d2 + timedelta(days=1)]
    if evt_filter != "All":
        raw_where += " AND event_type = %s"
        raw_params.append(evt_filter)

    raw = q(f"""
        SELECT event_id, event_type, detected_category, detected_color,
               scene_label, clicked_product_id, rank_clicked,
               visual_similarity, final_score, latency_ms,
               device_type, created_at
        FROM analytics_events
        WHERE {raw_where}
        ORDER BY created_at DESC LIMIT 100
    """, raw_params)

    if not raw.empty:
        st.dataframe(raw, use_container_width=True, height=400)
        st.caption(f"Latest {len(raw)} events (max 100)")
    else:
        st.info("No events.")

    st.divider()

    # ═══════════════════════════════════════════════
    # SECTION 8: LLM-POWERED SESSION INSIGHTS
    # ═══════════════════════════════════════════════
    st.header("🧠 LLM-Powered Session Insights")
    st.caption("AI-generated analysis of user satisfaction and system performance based on rating data.")

    # Fetch latest ratings
    ratings_df = q("""
        SELECT * FROM ratings
        ORDER BY id DESC LIMIT 10
    """)

    if ratings_df.empty:
        st.info("No ratings data found. Users need to submit ratings from the product page.")
    else:
        st.success(f"Found {len(ratings_df)} rating entries. Showing latest sessions below.")

        # Summary stats from ratings
        col_a, col_b, col_c = st.columns(3)
        avg_r = ratings_df["rating"].mean() if "rating" in ratings_df.columns else 0
        total_r = len(ratings_df)
        high_r = len(ratings_df[ratings_df["rating"] >= 4]) if "rating" in ratings_df.columns else 0
        col_a.metric("⭐ Avg Rating", f"{avg_r:.1f}")
        col_b.metric("📊 Total Sessions", total_r)
        col_c.metric("✅ High Satisfaction (≥4)", high_r)

        # LLM Analysis toggle
        if st.button("🤖 Generate LLM Key Findings", type="primary"):
            with st.spinner("Generating LLM analysis..."):
                try:
                    from models.insights_engine import InsightsEngine

                    engine = InsightsEngine(None)

                    # Analyze latest rating
                    latest = ratings_df.iloc[0].to_dict()
                    analysis = engine.generate_report(latest)

                    if analysis and not analysis.get("error"):
                        st.subheader("🔍 Key Findings")

                        # Relevance & Success badges
                        rel_level = analysis.get("relevance_level", "N/A")
                        success = analysis.get("successful_recommendation", False)
                        badge_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

                        f1, f2 = st.columns(2)
                        f1.metric("Relevance Level", f"{badge_colors.get(rel_level, '⚪')} {rel_level}")
                        f2.metric("Successful Recommendation", "✅ Yes" if success else "❌ No")

                        # Strengths
                        strengths = analysis.get("strengths", [])
                        if strengths:
                            st.subheader("💪 Strengths")
                            for s in strengths:
                                st.markdown(f"- {s}")

                        # Weaknesses
                        weaknesses = analysis.get("weaknesses", [])
                        if weaknesses:
                            st.subheader("⚠️ Weaknesses & Risks")
                            for w in weaknesses:
                                st.markdown(f"- :red[{w}]")

                        # Improvement suggestion
                        suggestion = analysis.get("improvement_suggestion", "")
                        if suggestion:
                            st.subheader("💡 Technical Improvement")
                            st.info(suggestion)

                        # User behavior
                        behavior = analysis.get("user_behavior_analysis", "")
                        if behavior:
                            st.subheader("👤 User Behavior Analysis")
                            st.write(behavior)
                    else:
                        st.warning(f"LLM analysis returned an error: {analysis.get('error', 'Unknown')}")

                except Exception as e:
                    st.error(f"Failed to generate LLM insights: {e}")

        # Individual rating entries
        st.subheader("📝 Recent Rating Entries")
        for idx, row in ratings_df.iterrows():
            rating_val = row.get("rating", "N/A")
            prod_id = row.get("product_id", "Unknown")
            query_val = row.get("query", "")
            emoji = "⭐" * int(rating_val) if isinstance(rating_val, (int, float)) else ""
            with st.expander(f"Session #{row.get('id', idx)} — {emoji} ({rating_val}/5) — Product: {prod_id}"):
                st.json(row.to_dict())



if __name__ == "__main__":
    main()
