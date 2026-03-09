"""
Section 2 — YOLO Object Detection Evaluation
==============================================
Evaluates:
  - epoch80_garments.pt  (9 clothing classes)  against dataset1_clothing
  - yolov8l_accessories.pt (11 accessory classes) against dataset2_accessories

Generates per-class metrics, AP chart, PR curve, confusion matrix, confidence histogram.
"""

import os
import sys
import csv
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Model paths
GARMENTS_MODEL = os.path.join(BASE_DIR, "data", "yolo", "epoch80_garments.pt")
ACCESSORIES_MODEL = os.path.join(BASE_DIR, "data", "yolo", "yolov8l_accessories.pt")

# Dataset paths
DATASET1_YAML = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Roboflow", "dataset1_clothing", "data.yaml"))
DATASET2_YAML = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Roboflow", "dataset2_accessories", "data.yaml"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")


def _eval_single_model(model_path, data_yaml, label):
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n  [{label}] Loading model: {os.path.basename(model_path)} (Device: {device})")
    model = YOLO(model_path)

    print(f"  [{label}] Running validation against: {os.path.basename(data_yaml)}")
    try:
        metrics = model.val(
            data=data_yaml,
            split="val",
            verbose=False,
            save_json=False,
            plots=True,  # generates built-in plots
            device=device,
            batch=4,     # prevent CUDA OOM
        )
    except Exception as e:
        print(f"  [{label}] ⚠️ val() failed: {e}")
        return None, model.names

    # Extract overall metrics
    overall = {
        "model": os.path.basename(model_path),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    # Per-class metrics
    per_class = []
    class_names = model.names
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        ap50_arr = metrics.box.ap50
        for i, (cls_id, cls_name) in enumerate(class_names.items()):
            if i < len(ap50_arr):
                per_class.append({
                    "class": cls_name,
                    "AP50": round(float(ap50_arr[i]), 4),
                })

    # Try to copy built-in confusion matrix / PR curve from YOLO's save dir
    val_save_dir = getattr(metrics, "save_dir", None)
    if val_save_dir and os.path.isdir(str(val_save_dir)):
        for fname in ["confusion_matrix.png", "PR_curve.png", "P_curve.png", "R_curve.png"]:
            src = os.path.join(str(val_save_dir), fname)
            if os.path.exists(src):
                dst = os.path.join(GRAPHS_DIR, f"{label}_{fname}")
                shutil.copy2(src, dst)
                print(f"  [{label}] ✅ Copied {fname}")

    return {"overall": overall, "per_class": per_class}, class_names


def _plot_per_class_ap(all_per_class, output_path):
    """Bar chart of per-class AP@0.5 for all classes."""
    if not all_per_class:
        return

    names = [c["class"] for c in all_per_class]
    ap50s = [c["AP50"] for c in all_per_class]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(names, ap50s, color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_xlabel("AP@0.5", fontsize=13)
    ax.set_title("Per-Class Average Precision (AP@0.5)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for bar, val in zip(bars, ap50s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.axvline(x=np.mean(ap50s), color="red", linestyle="--", linewidth=1.2, label=f"Mean={np.mean(ap50s):.3f}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def _plot_confidence_histogram(model_path, data_yaml, label, output_path):
    """Generate detection confidence histogram from predictions."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        # Get val images
        import yaml
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        val_dir = os.path.join(cfg["path"], cfg.get("val", "images/val"))

        if not os.path.isdir(val_dir):
            print(f"  [{label}] Val dir not found: {val_dir}")
            return

        import glob
        imgs = glob.glob(os.path.join(val_dir, "*.*"))[:50]  # sample 50
        confidences = []
        for img_path in imgs:
            results = model(img_path, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for conf in r.boxes.conf:
                        confidences.append(float(conf))

        if not confidences:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(confidences, bins=30, color="#4C78A8", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Confidence Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Detection Confidence Distribution — {label}", fontsize=14, fontweight="bold")
        ax.axvline(np.mean(confidences), color="red", linestyle="--", label=f"Mean={np.mean(confidences):.3f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {output_path}")

    except Exception as e:
        print(f"  [{label}] Confidence histogram failed: {e}")


def run():
    print("\n" + "=" * 60)
    print("📊 SECTION 2: YOLO OBJECT DETECTION EVALUATION")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    all_results = {}
    all_per_class = []

    # ── Garments Model ──
    if os.path.exists(GARMENTS_MODEL) and os.path.exists(DATASET1_YAML):
        res, names = _eval_single_model(GARMENTS_MODEL, DATASET1_YAML, "Garments")
        if res:
            all_results["garments"] = res
            all_per_class.extend(res["per_class"])
            print(f"\n  [Garments] Overall: mAP50={res['overall']['mAP50']}, "
                  f"Precision={res['overall']['precision']}, Recall={res['overall']['recall']}")
    else:
        print(f"  ⚠️ Garments model or dataset not found")

    # ── Accessories Model ──
    if os.path.exists(ACCESSORIES_MODEL) and os.path.exists(DATASET2_YAML):
        res, names = _eval_single_model(ACCESSORIES_MODEL, DATASET2_YAML, "Accessories")
        if res:
            all_results["accessories"] = res
            all_per_class.extend(res["per_class"])
            print(f"\n  [Accessories] Overall: mAP50={res['overall']['mAP50']}, "
                  f"Precision={res['overall']['precision']}, Recall={res['overall']['recall']}")
    else:
        print(f"  ⚠️ Accessories model or dataset not found")

    if not all_results:
        print("  ❌ No YOLO evaluation results")
        return {}

    # ── Combined Metrics ──
    all_map50 = [r["overall"]["mAP50"] for r in all_results.values()]
    all_prec = [r["overall"]["precision"] for r in all_results.values()]
    all_rec = [r["overall"]["recall"] for r in all_results.values()]

    combined = {
        "combined_mAP50": round(np.mean(all_map50), 4),
        "combined_precision": round(np.mean(all_prec), 4),
        "combined_recall": round(np.mean(all_rec), 4),
    }

    # ── Write CSV ──
    csv_path = os.path.join(TABLES_DIR, "yolo_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Overall
        w.writerow(["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"])
        for k, v in all_results.items():
            o = v["overall"]
            w.writerow([o["model"], o["mAP50"], o["mAP50_95"], o["precision"], o["recall"]])
        w.writerow([])
        # Per-class
        w.writerow(["Class", "AP@0.5"])
        for c in all_per_class:
            w.writerow([c["class"], c["AP50"]])
        w.writerow([])
        w.writerow(["Combined mAP@0.5", combined["combined_mAP50"]])
    print(f"  ✅ {csv_path}")

    # ── Per-class AP chart ──
    _plot_per_class_ap(all_per_class, os.path.join(GRAPHS_DIR, "per_class_ap.png"))

    # ── Confidence histograms ──
    if os.path.exists(GARMENTS_MODEL) and os.path.exists(DATASET1_YAML):
        _plot_confidence_histogram(
            GARMENTS_MODEL, DATASET1_YAML, "Garments",
            os.path.join(GRAPHS_DIR, "confidence_histogram_garments.png"))
    if os.path.exists(ACCESSORIES_MODEL) and os.path.exists(DATASET2_YAML):
        _plot_confidence_histogram(
            ACCESSORIES_MODEL, DATASET2_YAML, "Accessories",
            os.path.join(GRAPHS_DIR, "confidence_histogram_accessories.png"))

    # ── Save metrics JSON ──
    json_path = os.path.join(METRICS_DIR, "yolo_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "combined": combined, "per_class": all_per_class},
                  f, indent=2, default=str)
    print(f"  ✅ {json_path}")

    print(f"\n  Combined: mAP50={combined['combined_mAP50']}, "
          f"Precision={combined['combined_precision']}, "
          f"Recall={combined['combined_recall']}")

    return {"results": all_results, "combined": combined}


if __name__ == "__main__":
    run()
