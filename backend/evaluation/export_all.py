"""
Section 10 - Export All Results
===============================
Collects generated outputs, creates required file names, and writes
paper/PPT-ready summary text.
"""

import csv
import glob
import json
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_if_exists(src, dst):
    if os.path.exists(src):
        if os.path.abspath(src) == os.path.abspath(dst):
            return True
        shutil.copy2(src, dst)
        return True
    return False


def _ensure_plot(path, title):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\nNo data generated yet", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _generate_paper_text(summary):
    out_path = os.path.join(RESULTS_DIR, "paper_results_content.md")
    yolo = summary.get("yolo_metrics", {})
    ret = summary.get("retrieval_metrics", {})
    llm = summary.get("llm_metrics", {})
    agm = summary.get("agman_metrics", {})
    lat = summary.get("latency_metrics", {})
    user = summary.get("user_study_metrics", {})

    text = []
    text.append("# Results Section Draft (Springer Style)")
    text.append("")
    text.append("## Detection and Retrieval Performance")
    text.append(
        f"The YOLO module achieved mAP@0.5 = {yolo.get('combined', {}).get('combined_mAP50', 'N/A')} and precision = {yolo.get('combined', {}).get('combined_precision', 'N/A')}. "
        f"The retrieval engine reported Precision@10 = {ret.get('Precision@10', 'N/A')} and mAP = {ret.get('mAP', 'N/A')}, indicating reliable top-rank retrieval quality."
    )
    text.append("")
    text.append("## Attribute and Reasoning Quality")
    text.append(
        f"AGMAN attribute evaluation shows Color accuracy = {agm.get('results', {}).get('color', {}).get('accuracy', 'N/A')}, "
        f"Pattern accuracy = {agm.get('results', {}).get('pattern', {}).get('accuracy', 'N/A')}, and "
        f"Sleeve accuracy = {agm.get('results', {}).get('sleeve', {}).get('accuracy', 'N/A')}. "
        f"For text reasoning, the LLM JSON validity rate is {llm.get('summary', {}).get('JSON Validity', 'N/A')} with hallucination rate {llm.get('summary', {}).get('Hallucination Rate', 'N/A')}."
    )
    text.append("")
    text.append("## Runtime and User Conversion")
    text.append(
        f"Total pipeline latency is {lat.get('Total Pipeline', 'N/A')} ms. "
        f"In simulated user study, Search->Click rate is {user.get('Search -> Click Rate', 'N/A')} and Click->Buy rate is {user.get('Click -> Buy Rate', 'N/A')}."
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def _generate_ppt_text(summary):
    out_path = os.path.join(RESULTS_DIR, "ppt_results_points.md")
    ret = summary.get("retrieval_metrics", {})
    llm = summary.get("llm_metrics", {})
    ab = summary.get("ablation_metrics", {})
    lat = summary.get("latency_metrics", {})

    lines = [
        "# PPT Results Bullets",
        "",
        "- Retrieval quality:",
        f"  Precision@1={ret.get('Precision@1', 'N/A')}, Precision@5={ret.get('Precision@5', 'N/A')}, Precision@10={ret.get('Precision@10', 'N/A')}",
        f"  mAP={ret.get('mAP', 'N/A')}, NDCG@10={ret.get('NDCG@10', 'N/A')}",
        "- LLM reasoning:",
        f"  JSON validity={llm.get('summary', {}).get('JSON Validity', 'N/A')}, Schema compliance={llm.get('summary', {}).get('Schema Compliance', 'N/A')}",
        f"  Hallucination rate={llm.get('summary', {}).get('Hallucination Rate', 'N/A')}",
        "- Ablation insight:",
        f"  Visual only={ab.get('Visual Only', 'N/A')}, Full system={ab.get('Full System', 'N/A')} (Precision@10)",
        "- Efficiency:",
        f"  Total pipeline latency={lat.get('Total Pipeline', 'N/A')} ms",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _ensure_required_csvs():
    required = {
        "yolo_metrics.csv": [["Metric", "Value"], ["status", "not_generated"]],
        "retrieval_metrics.csv": [["Metric", "Value"], ["status", "not_generated"]],
        "llm_metrics.csv": [["Metric", "Value"], ["status", "not_generated"]],
        "evaluation_tables.csv": [["Metric", "Value"], ["status", "not_generated"]],
    }
    for name, rows in required.items():
        path = os.path.join(TABLES_DIR, name)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(rows)


def run():
    print("\n" + "=" * 60)
    print("SECTION 10: EXPORT ALL RESULTS")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    summary = {
        "dataset_metrics": _load_json(os.path.join(METRICS_DIR, "dataset_metrics.json")),
        "yolo_metrics": _load_json(os.path.join(METRICS_DIR, "yolo_metrics.json")),
        "agman_metrics": _load_json(os.path.join(METRICS_DIR, "agman_metrics.json")),
        "llm_metrics": _load_json(os.path.join(METRICS_DIR, "llm_metrics.json")),
        "retrieval_metrics": _load_json(os.path.join(METRICS_DIR, "retrieval_metrics.json")),
        "ablation_metrics": _load_json(os.path.join(METRICS_DIR, "ablation_metrics.json")),
        "latency_metrics": _load_json(os.path.join(METRICS_DIR, "latency_metrics.json")),
        "user_study_metrics": _load_json(os.path.join(METRICS_DIR, "user_study_metrics.json")),
        "comparison_metrics": _load_json(os.path.join(METRICS_DIR, "comparison_metrics.json")),
    }

    combined_csv = os.path.join(TABLES_DIR, "evaluation_tables.csv")
    csv_files = [p for p in sorted(glob.glob(os.path.join(TABLES_DIR, "*.csv"))) if not p.endswith("evaluation_tables.csv")]
    with open(combined_csv, "w", newline="", encoding="utf-8") as dst:
        w = csv.writer(dst)
        for path in csv_files:
            w.writerow([f"=== {os.path.basename(path)} ==="])
            with open(path, "r", encoding="utf-8") as src:
                for row in csv.reader(src):
                    w.writerow(row)
            w.writerow([])
    print(f"  OK {combined_csv}")

    _copy_if_exists(os.path.join(GRAPHS_DIR, "per_class_ap.png"), os.path.join(GRAPHS_DIR, "detection_ap_chart.png"))
    _copy_if_exists(os.path.join(GRAPHS_DIR, "precision_at_k.png"), os.path.join(GRAPHS_DIR, "precision_at_k.png"))
    _copy_if_exists(os.path.join(GRAPHS_DIR, "system_latency.png"), os.path.join(GRAPHS_DIR, "system_latency.png"))
    _copy_if_exists(os.path.join(GRAPHS_DIR, "attribute_accuracy.png"), os.path.join(GRAPHS_DIR, "attribute_accuracy.png"))
    _copy_if_exists(os.path.join(GRAPHS_DIR, "conversion_funnel.png"), os.path.join(GRAPHS_DIR, "conversion_funnel.png"))
    _ensure_plot(os.path.join(GRAPHS_DIR, "precision_at_k.png"), "Precision@K")
    _ensure_plot(os.path.join(GRAPHS_DIR, "detection_ap_chart.png"), "Detection AP Chart")
    _ensure_plot(os.path.join(GRAPHS_DIR, "attribute_accuracy.png"), "Attribute Accuracy")
    _ensure_plot(os.path.join(GRAPHS_DIR, "system_latency.png"), "System Latency")
    _ensure_plot(os.path.join(GRAPHS_DIR, "conversion_funnel.png"), "Conversion Funnel")

    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _generate_paper_text(summary)
    _generate_ppt_text(summary)
    _ensure_required_csvs()

    print("  OK paper_results_content.md")
    print("  OK ppt_results_points.md")
    return summary


if __name__ == "__main__":
    run()
