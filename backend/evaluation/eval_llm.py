
import sys
import os
import json
from eval_metrics import calculate_bleu_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock UnifiedLLM import if not running in full backend env, 
# but usually we want to test the REAL model.
try:
    from models.unified_llm import UnifiedLLM
except ImportError:
    print("[WARN] UnifiedLLM not found. Ensure you are in backend environment.")
    UnifiedLLM = None

def evaluate_llm():
    print("\n" + "="*50)
    print("LLM REASONING EVALUATION (BLEU & Precision)")
    print("="*50)
    
    if not UnifiedLLM:
        print("[ERR] UnifiedLLM module missing.")
        return

    # TEST SUITE (Ground Truth)
    test_cases = [
        {
            "query": "show me red shirts",
            "expected_filters": {"color": "red", "category": "shirt"},
            "attributes": {}
        },
        {
            "query": "blue denim jeans",
            "expected_filters": {"color": "blue", "category": "jeans"},
            "attributes": {"color_hex": "#0000FF"} # Visual cue
        },
        {
            "query": "black shoes",
            "expected_filters": {"color": "black", "category": "shoe"},
            "attributes": {}
        }
    ]
    
    llm = UnifiedLLM()
    # Preload
    llm._get_local_llm() 
    
    total_jaccard = 0
    total_bleu = 0
    passed = 0
    
    print(f"[TEST] Running {len(test_cases)} test cases...")
    
    # Verbose output
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')
    
    try:
        for i, case in enumerate(test_cases):
            # Run LLM
            result = llm.generate_filters(
                category=case['expected_filters'].get('category', 'unknown'), 
                attributes=case['attributes'],
                scene="test_scene",
                query=case['query'],
                session_history=[],
                prefer_external=True 
            )
            
            predicted_filters = result.get('filters', {})
            reasoning = result.get('reasoning', '')
            
            # Normalize keys
            norm_pred = {}
            for k, v in predicted_filters.items():
                if k == 'color_name': k = 'color'
                norm_pred[k] = v.lower() if isinstance(v, str) else v

            norm_expected = {}
            for k, v in case['expected_filters'].items():
                if k == 'color_name': k = 'color'
                norm_expected[k] = v.lower() if isinstance(v, str) else v
                
            expected_set = {f"{k}:{v}" for k, v in norm_expected.items()}
            pred_set = {f"{k}:{v}" for k, v in norm_pred.items() if k in norm_expected}
            
            intersection = expected_set.intersection(pred_set)
            union = expected_set.union(pred_set)
            jaccard = len(intersection) / len(union) if union else 1.0
            total_jaccard += jaccard
            
            # BLEU
            reference = f"User is looking for {case['query']}"
            bleu = calculate_bleu_score(reasoning, reference)
            total_bleu += bleu
    except Exception:
        pass
    finally:
        pass
        # sys.stdout = original_stdout # Restore
            
    avg_jaccard = total_jaccard / len(test_cases)
    avg_bleu = total_bleu / len(test_cases)
    
    avg_jaccard = total_jaccard / len(test_cases)
    avg_bleu = total_bleu / len(test_cases)
    
    print("\n[OK] RESULTS:")
    print(f"   Filter Acc: {avg_jaccard:.4f}")
    print(f"   BLEU Score: {avg_bleu:.4f}")
    
    return {"Filter_Acc": avg_jaccard, "BLEU": avg_bleu}

if __name__ == "__main__":
    evaluate_llm()
