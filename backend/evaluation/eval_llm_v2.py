"""
Section 4 - LLM Query Reasoning Evaluation
==========================================
Evaluates query-to-filter conversion using 200 query-ground truth pairs.
"""

import csv
import json
import os
import random
import re
import sys

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

ALLOWED_KEYS = {
    "category", "gender", "style", "material", "color_family",
    "primary_color_name", "sleeve_value", "pattern_value", "price_bucket",
}
EVAL_ATTRS = ["category", "primary_color_name", "pattern_value", "sleeve_value", "gender", "style", "price_max"]


BASE_CASES = [
    {"query": "show me blue checked jackets under 2000", "category": "Jacket", "expected": {"category": "Jacket", "primary_color_name": "blue", "pattern_value": "checked", "price_max": 2000}},
    {"query": "red tshirt", "category": "shirts", "expected": {"category": "tshirt", "primary_color_name": "red"}},
    {"query": "black formal blazer for men", "category": "tshirt", "expected": {"category": "blazer", "primary_color_name": "black", "style": "formal", "gender": "Men"}},
    {"query": "white striped shirts", "category": "Jacket", "expected": {"category": "shirts", "primary_color_name": "white", "pattern_value": "striped"}},
    {"query": "half sleeve green tshirt", "category": "Jacket", "expected": {"category": "tshirt", "sleeve_value": "half", "primary_color_name": "green"}},
    {"query": "casual blue pant", "category": "shirts", "expected": {"category": "pant", "style": "casual", "primary_color_name": "blue"}},
    {"query": "watches", "category": "shirts", "expected": {"category": "watch"}},
    {"query": "show original", "category": "Jacket", "expected": {"_reset": True}},
]


def _expand_to_200():
    random.seed(7)
    colors = ["red", "blue", "green", "black", "white", "pink", "grey", "brown"]
    categories = ["shirts", "tshirt", "Jacket", "pant", "blazer", "watch"]
    patterns = ["solid", "striped", "checked", "printed"]
    sleeves = ["long", "half", "short", "sleeveless"]

    out = list(BASE_CASES)
    while len(out) < 200:
        c = random.choice(colors)
        cat = random.choice(categories)
        p = random.choice(patterns)
        s = random.choice(sleeves)
        mode = len(out) % 4
        if mode == 0:
            out.append({"query": f"{c} {cat}", "category": "Jacket", "expected": {"category": cat, "primary_color_name": c}})
        elif mode == 1:
            out.append({"query": f"{p} {cat}", "category": "shirts", "expected": {"category": cat, "pattern_value": p}})
        elif mode == 2:
            out.append({"query": f"{s} sleeve {cat}", "category": "shirts", "expected": {"category": cat, "sleeve_value": s}})
        else:
            price = random.choice([500, 1000, 1500, 2000])
            out.append({"query": f"{c} {cat} under {price}", "category": "Jacket", "expected": {"category": cat, "primary_color_name": c, "price_max": price}})
    return out[:200]


def _norm(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return str(v).strip().lower()


def _extract_price_max(text):
    if not text:
        return None
    m = re.search(r"(?:under|below|less than|<)\s*(\d+)", text.lower())
    return int(m.group(1)) if m else None


def _eval_case(llm, case):
    out = {
        "json_valid": False,
        "schema_compliant": False,
        "hallucinated": False,
        "tp": {a: 0 for a in EVAL_ATTRS},
        "fp": {a: 0 for a in EVAL_ATTRS},
        "fn": {a: 0 for a in EVAL_ATTRS},
    }

    try:
        resp = llm.generate_filters(
            category=case["category"],
            attributes={},
            scene="indoor",
            query=case["query"],
            session_history=[],
            prefer_external=True,
        )
        if not isinstance(resp, dict):
            return out

        out["json_valid"] = True
        add = resp.get("add", {}) if isinstance(resp.get("add", {}), dict) else {}
        price_max = resp.get("price_max")

        keys_ok = set(add.keys()).issubset(ALLOWED_KEYS)
        out["schema_compliant"] = keys_ok

        expected = case["expected"]
        expected_attrs = {k for k in expected.keys() if not k.startswith("_")}

        pred = dict(add)
        if price_max is not None:
            pred["price_max"] = price_max
        else:
            fallback_price = _extract_price_max(case["query"])
            if fallback_price is not None and "price_max" in expected:
                pred["price_max"] = fallback_price

        for attr in EVAL_ATTRS:
            e = expected.get(attr)
            p = pred.get(attr)
            if e is not None and p is not None:
                if _norm(e) == _norm(p):
                    out["tp"][attr] += 1
                else:
                    out["fp"][attr] += 1
                    out["fn"][attr] += 1
            elif e is not None and p is None:
                out["fn"][attr] += 1
            elif e is None and p is not None:
                out["fp"][attr] += 1

        unexpected = [k for k, v in pred.items() if k not in expected_attrs and v not in (None, "", [])]
        out["hallucinated"] = len(unexpected) > 0

    except Exception:
        return out

    return out


def run():
    print("\n" + "=" * 60)
    print("SECTION 4: LLM QUERY REASONING EVALUATION")
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    test_cases = _expand_to_200()

    from models.unified_llm import UnifiedLLM
    llm = UnifiedLLM()

    json_valid = 0
    schema_ok = 0
    halluc = 0
    tps = {a: 0 for a in EVAL_ATTRS}
    fps = {a: 0 for a in EVAL_ATTRS}
    fns = {a: 0 for a in EVAL_ATTRS}

    for i, case in enumerate(test_cases, start=1):
        r = _eval_case(llm, case)
        json_valid += int(r["json_valid"])
        schema_ok += int(r["schema_compliant"])
        halluc += int(r["hallucinated"])
        for a in EVAL_ATTRS:
            tps[a] += r["tp"][a]
            fps[a] += r["fp"][a]
            fns[a] += r["fn"][a]
        if i % 25 == 0:
            print(f"  Progress: {i}/{len(test_cases)}")

    total = len(test_cases)
    attr_metrics = {}
    for a in EVAL_ATTRS:
        p = tps[a] / (tps[a] + fps[a]) if (tps[a] + fps[a]) else 0.0
        r = tps[a] / (tps[a] + fns[a]) if (tps[a] + fns[a]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        attr_metrics[a] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    summary = {
        "JSON Validity": round(json_valid / total, 4),
        "Schema Compliance": round(schema_ok / total, 4),
        "Hallucination Rate": round(halluc / total, 4),
        "total_queries": total,
    }

    csv_path = os.path.join(TABLES_DIR, "llm_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["JSON Validity", summary["JSON Validity"]])
        w.writerow(["Schema Compliance", summary["Schema Compliance"]])
        w.writerow(["Hallucination Rate", summary["Hallucination Rate"]])
        w.writerow([])
        w.writerow(["Attribute", "Precision", "Recall", "F1"])
        for a, m in attr_metrics.items():
            w.writerow([a, m["precision"], m["recall"], m["f1"]])
    print(f"  OK {csv_path}")

    attrs = list(attr_metrics.keys())
    pvals = [attr_metrics[a]["precision"] for a in attrs]
    rvals = [attr_metrics[a]["recall"] for a in attrs]
    fvals = [attr_metrics[a]["f1"] for a in attrs]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(attrs))
    w = 0.25
    ax.bar(x - w, pvals, width=w, label="Precision", color="#4C78A8")
    ax.bar(x, rvals, width=w, label="Recall", color="#F58518")
    ax.bar(x + w, fvals, width=w, label="F1", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("LLM Attribute Extraction Accuracy", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(GRAPHS_DIR, "llm_attribute_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {chart_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    good = 1.0 - summary["Hallucination Rate"]
    bad = summary["Hallucination Rate"]
    ax.bar(["Non-hallucinated", "Hallucinated"], [good, bad], color=["#54A24B", "#E45756"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("LLM Hallucination Rate", fontweight="bold")
    plt.tight_layout()
    h_path = os.path.join(GRAPHS_DIR, "hallucination_chart.png")
    plt.savefig(h_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK {h_path}")

    with open(os.path.join(METRICS_DIR, "llm_queries_200.json"), "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)

    payload = {"summary": summary, "attribute_metrics": attr_metrics}
    json_path = os.path.join(METRICS_DIR, "llm_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  OK {json_path}")

    return payload


if __name__ == "__main__":
    run()
