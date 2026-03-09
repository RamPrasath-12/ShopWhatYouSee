"""
Section 1 — Dataset Description
================================
Generates dataset summary table and category distribution chart.
"""

import os
import sys
import glob
import json
import csv
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

ROBOFLOW_DIR = os.path.join(os.path.dirname(BASE_DIR), "..", "Roboflow")
DATASET1_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Roboflow", "dataset1_clothing"))
DATASET2_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Roboflow", "dataset2_accessories"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")


def _count_images(dataset_dir):
    """Count images per split."""
    counts = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        if os.path.exists(img_dir):
            counts[split] = len(glob.glob(os.path.join(img_dir, "*.*")))
        else:
            counts[split] = 0
    return counts


def _count_annotations_per_class(dataset_dir, data_yaml_path):
    """Count annotations per class across all splits."""
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    nc = cfg.get("nc", len(class_names))
    counts = {class_names[i]: 0 for i in range(nc)}

    for split in ["train", "val", "test"]:
        label_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.exists(label_dir):
            continue
        for lbl_file in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        name = class_names.get(cls_id, f"class_{cls_id}")
                        counts[name] = counts.get(name, 0) + 1
    return counts


def _get_db_product_counts():
    """Get product counts per category from database."""
    try:
        from db_config import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT category, COUNT(*) FROM visual_attributes GROUP BY category ORDER BY count DESC")
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM visual_attributes")
        total = cur.fetchone()[0]
        cur.close()
        conn.close()
        return total, {r[0]: r[1] for r in rows}
    except Exception as e:
        print(f"  [WARN] DB query failed: {e}")
        return 0, {}


def run():
    print("\n" + "=" * 60)
    print("📊 SECTION 1: DATASET DESCRIPTION")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Dataset 1: Clothing ──
    d1_yaml = os.path.join(DATASET1_DIR, "data.yaml")
    d1_counts = _count_images(DATASET1_DIR)
    d1_annots = _count_annotations_per_class(DATASET1_DIR, d1_yaml) if os.path.exists(d1_yaml) else {}

    # ── Dataset 2: Accessories ──
    d2_yaml = os.path.join(DATASET2_DIR, "data.yaml")
    d2_counts = _count_images(DATASET2_DIR)
    d2_annots = _count_annotations_per_class(DATASET2_DIR, d2_yaml) if os.path.exists(d2_yaml) else {}

    # ── DB Products ──
    total_products, db_categories = _get_db_product_counts()

    # ── Combined annotations ──
    all_annots = {}
    all_annots.update(d1_annots)
    all_annots.update(d2_annots)

    total_detection_images = sum(d1_counts.values()) + sum(d2_counts.values())
    total_annotations = sum(all_annots.values())

    # ── Summary Table ──
    summary = {
        "Total Detection Images": total_detection_images,
        "Total Annotations": total_annotations,
        "Clothing Images (Train/Val/Test)": f"{d1_counts.get('train',0)}/{d1_counts.get('val',0)}/{d1_counts.get('test',0)}",
        "Accessories Images (Train/Val/Test)": f"{d2_counts.get('train',0)}/{d2_counts.get('val',0)}/{d2_counts.get('test',0)}",
        "Clothing Classes": 9,
        "Accessory Classes": 11,
        "Total Detection Classes": 20,
        "Total Products in DB": total_products,
        "DB Categories": len(db_categories),
        "Attributes Extracted": "3 (color, pattern, sleeve)",
    }

    # Write CSV
    csv_path = os.path.join(TABLES_DIR, "dataset_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in summary.items():
            w.writerow([k, v])
    print(f"  ✅ {csv_path}")

    # Print table
    print("\n  Dataset Summary:")
    for k, v in summary.items():
        print(f"    {k:<40}: {v}")

    # ── Category Distribution Bar Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Detection annotations
    if all_annots:
        names = list(all_annots.keys())
        values = list(all_annots.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars = axes[0].barh(names, values, color=colors, edgecolor="grey", linewidth=0.5)
        axes[0].set_xlabel("Number of Annotations", fontsize=12)
        axes[0].set_title("Detection Dataset: Annotations per Class", fontsize=14, fontweight="bold")
        axes[0].invert_yaxis()
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

    # DB product distribution
    if db_categories:
        db_names = list(db_categories.keys())
        db_vals = list(db_categories.values())
        colors2 = plt.cm.Paired(np.linspace(0, 1, len(db_names)))
        bars2 = axes[1].barh(db_names, db_vals, color=colors2, edgecolor="grey", linewidth=0.5)
        axes[1].set_xlabel("Number of Products", fontsize=12)
        axes[1].set_title("Product Database: Products per Category", fontsize=14, fontweight="bold")
        axes[1].invert_yaxis()
        for bar, val in zip(bars2, db_vals):
            axes[1].text(bar.get_width() + max(db_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(GRAPHS_DIR, "category_distribution.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {chart_path}")

    # Save metrics JSON
    metrics = {
        "summary": summary,
        "detection_annotations": all_annots,
        "db_categories": db_categories,
    }
    json_path = os.path.join(METRICS_DIR, "dataset_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  ✅ {json_path}")

    return metrics


if __name__ == "__main__":
    run()
