
import sys
import os
import datetime

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_yolo import evaluate_yolo
from eval_retrieval import evaluate_retrieval
from eval_llm import evaluate_llm

def main():
    print("\n" + "="*60)
    print("🚀 SHOPWHATYOUSEE - SYSTEM EVALUATION REPORT")
    print(f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # 1. Detection
    try:
        results['detection'] = evaluate_yolo() or {}
    except Exception as e:
        print(f" Detection Eval Failed: {e}")
        results['detection'] = {}

    # 2. Retrieval (Embeddings + Filtering)
    try:
        results['retrieval'] = evaluate_retrieval() or {}
    except Exception as e:
        print(f" Retrieval Eval Failed: {e}")
        results['retrieval'] = {}

    # 3. LLM Reasoning
    try:
        results['llm'] = evaluate_llm() or {}
    except Exception as e:
        print(f" LLM Eval Failed: {e}")
        results['llm'] = {}
        
    # --- REPORT GENERATION ---
    print("\n\n")
    print("="*60)
    print("      SHOPWHATYOUSEE - PERFORMANCE METRICS (FINAL)")
    print("="*60)
    
    # 1. Detection
    det = results.get('detection', {})
    print("\n1. COMPUTER VISION (YOLOv8 Ensemble - 3 Models)")
    print("-" * 50)
    if det:
        print(f"{'mAP@50':<30} : {det.get('mAP50', 0):.4f}")
        print(f"{'Recall':<30} : {det.get('Recall', 0):.4f}")
        print(f"{'Avg Inference Latency':<30} : {det.get('Inference_ms', 0):.2f} ms")
    else:
        print("Status: Detection benchmarks failed.")

    # 2. Retrieval
    ret = results.get('retrieval', {})
    print("\n2. SEARCH ENGINE (Hybrid Retrieval)")
    print("-" * 50)
    if ret:
        print(f"{'Recall@5':<30} : {ret.get('Recall@K', 0):.4f}")
        print(f"{'Avg Cosine Similarity':<30} : {ret.get('Avg_Sim', 0):.4f}")
    else:
        print("Status: Retrieval benchmarks failed.")

    # 3. LLM
    llm = results.get('llm', {})
    print("\n3. GEN-AI REASONING (Unified LLM)")
    print("-" * 50)
    if llm:
        print(f"{'Filter Accuracy (Jaccard)':<30} : {llm.get('Filter_Acc', 0):.4f}")
        print(f"{'Reasoning Score (BLEU)':<30} : {llm.get('BLEU', 0):.4f}")
    else:
        print("Status: LLM benchmarks failed.")
        
    print("\n" + "="*60)
    print("report generated successfully.")

if __name__ == "__main__":
    main()
