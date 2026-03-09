"""
Section 8 — End-to-End User Study (Simulated)
===============================================
Simulates 100 user interactions through the full pipeline.
Tracks: Detection → Search → Click → Buy conversion funnel.
"""

import os
import sys
import csv
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

NUM_SESSIONS = 100


def _simulate_user_sessions():
    """
    Simulate user sessions based on system capabilities.
    Models realistic conversion rates from detection to purchase.
    """
    random.seed(42)

    # Based on system performance analysis:
    # - Detection success rate: ~94% (from YOLO mAP)
    # - Search relevance: ~85% (from retrieval P@10)
    # - Click rate (given relevant results): ~60%
    # - Buy rate (given click): ~15%
    detection_rate = 0.94
    search_relevance_rate = 0.85
    click_rate_given_relevant = 0.60
    buy_rate_given_click = 0.15

    sessions = []
    for i in range(NUM_SESSIONS):
        session = {"id": i + 1}

        # Stage 1: Detection
        detected = random.random() < detection_rate
        session["detected"] = detected

        # Stage 2: Search
        searched = detected and (random.random() < search_relevance_rate)
        session["search_relevant"] = searched

        # Stage 3: Click
        clicked = searched and (random.random() < click_rate_given_relevant)
        session["clicked"] = clicked

        # Stage 4: Buy
        bought = clicked and (random.random() < buy_rate_given_click)
        session["bought"] = bought

        # Top-1 click
        session["top1_click"] = clicked and (random.random() < 0.35)

        sessions.append(session)

    return sessions


def run():
    print("\n" + "=" * 60)
    print("📊 SECTION 8: END-TO-END USER STUDY (SIMULATED)")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    sessions = _simulate_user_sessions()

    # Compute metrics
    total = len(sessions)
    detected = sum(1 for s in sessions if s["detected"])
    searched = sum(1 for s in sessions if s["search_relevant"])
    clicked = sum(1 for s in sessions if s["clicked"])
    bought = sum(1 for s in sessions if s["bought"])
    top1 = sum(1 for s in sessions if s.get("top1_click"))

    metrics = {
        "Total Sessions": total,
        "Detection Success Rate": round(detected / total, 4),
        "Search → Click Rate": round(clicked / searched, 4) if searched else 0,
        "Click → Buy Rate": round(bought / clicked, 4) if clicked else 0,
        "Top-1 Click Rate": round(top1 / total, 4),
        "Overall Conversion": round(bought / total, 4),
    }

    # Print
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    # ── CSV ──
    csv_path = os.path.join(TABLES_DIR, "user_study_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"  ✅ {csv_path}")

    # ── Conversion Funnel Chart ──
    fig, ax = plt.subplots(figsize=(10, 7))
    stages = ["Detection", "Relevant Search", "Click", "Purchase"]
    counts = [detected, searched, clicked, bought]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]

    # Funnel visualization
    bar_widths = [c / total for c in counts]
    y_positions = range(len(stages))

    for i, (stage, count, width, color) in enumerate(zip(stages, counts, bar_widths, colors)):
        left = (1 - width) / 2
        ax.barh(i, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=2)
        ax.text(0.5, i, f"{stage}\n{count}/{total} ({count/total*100:.1f}%)",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("User Conversion Funnel", fontsize=15, fontweight="bold")
    ax.invert_yaxis()

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "conversion_funnel.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {os.path.join(GRAPHS_DIR, 'conversion_funnel.png')}")

    # ── JSON ──
    json_path = os.path.join(METRICS_DIR, "user_study_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ {json_path}")

    return metrics


if __name__ == "__main__":
    run()
