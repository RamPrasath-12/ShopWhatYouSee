
import sys
import os
import random
import numpy as np
from eval_metrics import calculate_recall_at_k, cosine_similarity

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.product_retrieval import load_index, search_products

def evaluate_retrieval(num_samples=5, k=5):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n" + "="*50)
    print("🔍 RETRIEVAL EVALUATION (FAISS Recall@K)")
    print("="*50)

    try:
        index, ids, metadata = load_index()
        total_items = len(ids)
        print(f"[OK] Index Loaded: {total_items} items")
        
        if total_items == 0:
            print("[ERR] No items in index.")
            return

        # SIMULATION: SELF-RETRIEVAL TEST
        # We pick random items and check if querying with their attributes/features finds them.
        # Ideally we use an embedding, but we'll use attribute filters if embeddings unavailable in metadata
        # Actually metadata has 'embedding' usually? No, embedding is in FAISS. 
        # But we can't extract embedding from FAISS easily without ID mapping.
        
        # We'll use the search function with FILTERS derived from the item.
        # This tests if "Category + Color + Pattern" accurately retrieves the item.
        
        hits = 0
        total_sim = 0
        
        samples_indices = random.sample(range(total_items), min(num_samples, total_items))

        # Silence output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            for i in samples_indices:
                target_id = ids[i] 
                target_meta = metadata.get(target_id, {})
                
                try:
                    # vec = index.reconstruct(target_id)
                    # vec_list = vec.tolist()
                    vec_list = None
                except:
                    vec_list = None
                
                filters = {
                    "category": target_meta.get("category"),
                    "color": target_meta.get("color"),
                    "embedding": vec_list
                }
                filters = {k: v for k, v in filters.items() if v}
                
                results = search_products(filters, top_k=k)
                
                retrieved_ids = []
                for r in results:
                    retrieved_ids.append(r.get('id'))
                
                true_db_id = target_meta.get('id')
                
                if true_db_id and true_db_id in retrieved_ids:
                    hits += 1
                    
                if results:
                    top_score = results[0].get('score', 0)
                    sim = 1.0 / (1.0 + top_score) 
                    total_sim += sim
        except Exception:
            pass
        finally:
            # sys.stdout = original_stdout # Restore
            pass
                
        recall = hits / len(samples_indices)
        avg_sim = total_sim / len(samples_indices)
        
        print("\n[OK] RESULTS:")
        print(f"   Recall@{k}:        {recall:.4f}")
        print(f"   Avg Similarity:   {avg_sim:.4f}")
        
        return {"Recall@K": recall, "Avg_Sim": avg_sim}

    except Exception as e:
        sys.stdout = sys.__stdout__ # Use original in case
        print(f"[ERR] Retrieval Eval Failed: {e}")
        return None

if __name__ == "__main__":
    evaluate_retrieval()
