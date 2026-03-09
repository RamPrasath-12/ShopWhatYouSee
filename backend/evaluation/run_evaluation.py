"""
ShopWhatYouSee - Complete Evaluation Pipeline
============================================
Run all (or selected) evaluation sections.

Usage:
  python evaluation/run_evaluation.py
  python evaluation/run_evaluation.py 1 2 5 10
"""

import datetime
import os
import sys
import time
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_ablation
import eval_agman
import eval_comparison
import eval_dataset
import eval_latency
import eval_llm_v2
import eval_retrieval_v2
import eval_user_study
import eval_yolo_v2
import export_all

SECTIONS = {
    1: ("Dataset Description", eval_dataset),
    2: ("YOLO Object Detection", eval_yolo_v2),
    3: ("AGMAN Attribute Extraction", eval_agman),
    4: ("LLM Query Reasoning", eval_llm_v2),
    5: ("Retrieval Engine", eval_retrieval_v2),
    6: ("Ablation Study", eval_ablation),
    7: ("System Performance", eval_latency),
    8: ("User Study", eval_user_study),
    9: ("Comparison with Existing Methods", eval_comparison),
    10: ("Export Results", export_all),
}


def main():
    print("\n" + "=" * 70)
    print("SHOPWHATYOUSEE - COMPLETE EVALUATION PIPELINE")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    if len(sys.argv) > 1:
        sections_to_run = [int(s) for s in sys.argv[1:] if s.isdigit()]
    else:
        sections_to_run = sorted(SECTIONS.keys())

    print(f"Sections to run: {sections_to_run}")

    results = {}
    total_start = time.time()

    for sid in sections_to_run:
        if sid not in SECTIONS:
            print(f"[WARN] Unknown section {sid}")
            continue

        name, module = SECTIONS[sid]
        print("\n" + "-" * 70)
        print(f"Section {sid}: {name}")
        print("-" * 70)

        start = time.time()
        try:
            data = module.run()
            elapsed = time.time() - start
            results[sid] = {"status": "ok", "time_s": round(elapsed, 2), "data": data}
            print(f"[OK] Section {sid} done in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[sid] = {"status": "failed", "time_s": round(elapsed, 2), "error": str(e)}
            print(f"[FAIL] Section {sid} failed in {elapsed:.2f}s: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for sid in sections_to_run:
        if sid in results:
            print(f"{sid:>2} | {SECTIONS[sid][0]:<34} | {results[sid]['status']:<7} | {results[sid]['time_s']}s")
    print(f"Total time: {total_elapsed:.2f}s")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
