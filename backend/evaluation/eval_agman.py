"""
Section 3 - AGMAN Attribute Extraction Evaluation
=================================================
Evaluates stored AGMAN predictions against scraped/text ground truth in DB.
No simulated metrics are used.
"""

import csv
import json
import os
import sys
from collections import Counter

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

SAMPLE_SIZE = 5000
INVALIDS = {"", "none", "null", "nan", "not_applicable", "unknown"}


def _norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in INVALIDS else s


def _fetch_pairs():
    from db_config import get_db_connection

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            primary_color_name, scraped_color,
            pattern_value, scraped_pattern,
            sleeve_value, scraped_sleeve
        FROM visual_attributes
        WHERE product_id IS NOT NULL
        ORDER BY product_id
        LIMIT %s
        """,
        (SAMPLE_SIZE,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    if not y_true or not labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    p_list, r_list, f_list = [], [], []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(np.mean(p_list)), 4),
        "recall": round(float(np.mean(r_list)), 4),
        "f1": round(float(np.mean(f_list)), 4),
    }


def _confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def _plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > threshold else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run():
    print("\n" + "=" * 60)
    print("SECTION 3: AGMAN ATTRIBUTE EXTRACTION EVALUATION")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    rows = _fetch_pairs()

    color_true, color_pred = [], []
    pattern_true, pattern_pred = [], []
    sleeve_true, sleeve_pred = [], []

    for row in rows:
        pred_color, gt_color, pred_pat, gt_pat, pred_slv, gt_slv = row
        pc, gc = _norm(pred_color), _norm(gt_color)
        pp, gp = _norm(pred_pat), _norm(gt_pat)
        ps, gs = _norm(pred_slv), _norm(gt_slv)

        if pc and gc:
            color_pred.append(pc)
            color_true.append(gc)
        if pp and gp:
            pattern_pred.append(pp)
            pattern_true.append(gp)
        if ps and gs:
            sleeve_pred.append(ps)
            sleeve_true.append(gs)

    results = {
        "color": _metrics(color_true, color_pred),
        "pattern": _metrics(pattern_true, pattern_pred),
        "sleeve": _metrics(sleeve_true, sleeve_pred),
    }

    csv_path = os.path.join(TABLES_DIR, "agman_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Attribute", "Accuracy", "Precision", "Recall", "F1"])
        for attr in ("color", "pattern", "sleeve"):
            m = results[attr]
            w.writerow([attr.capitalize(), m["accuracy"], m["precision"], m["recall"], m["f1"]])
    print(f"  OK {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    attrs = ["Color", "Pattern", "Sleeve"]
    vals = [results["color"]["accuracy"], results["pattern"]["accuracy"], results["sleeve"]["accuracy"]]
    bars = ax.bar(attrs, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Attribute Accuracy Comparison", fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    acc_path = os.path.join(GRAPHS_DIR, "attribute_accuracy.png")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {acc_path}")

    if color_true:
        top_labels = [x for x, _ in Counter(color_true).most_common(12)]
        cm = _confusion(color_true, color_pred, top_labels)
        cm_path = os.path.join(GRAPHS_DIR, "color_confusion_matrix.png")
        _plot_cm(cm, top_labels, "Color Confusion Matrix", cm_path)
        print(f"  OK {cm_path}")

    if pattern_true:
        top_labels = [x for x, _ in Counter(pattern_true).most_common(10)]
        cm = _confusion(pattern_true, pattern_pred, top_labels)
        cm_path = os.path.join(GRAPHS_DIR, "pattern_confusion_matrix.png")
        _plot_cm(cm, top_labels, "Pattern Confusion Matrix", cm_path)
        print(f"  OK {cm_path}")

    payload = {
        "results": results,
        "sample_size": {
            "rows_scanned": len(rows),
            "color_pairs": len(color_true),
            "pattern_pairs": len(pattern_true),
            "sleeve_pairs": len(sleeve_true),
        },
    }
    json_path = os.path.join(METRICS_DIR, "agman_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  OK {json_path}")

    return payload


if __name__ == "__main__":
    run()
