
import math
from collections import Counter

def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between two vectors.
    Returns: float (0.0 to 1.0)
    """
    dot_product = sum(a*b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a*a for a in v1))
    magnitude2 = math.sqrt(sum(b*b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def calculate_recall_at_k(relevant_items, retrieved_items, k=5):
    """
    Measures if any relevant item is present in the top-K retrieved items.
    Items are usually IDs.
    """
    top_k = retrieved_items[:k]
    hits = set(relevant_items).intersection(set(top_k))
    return len(hits) / len(relevant_items) if relevant_items else 0.0

def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    """
    Standard Classification Metrics.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_bleu_score(hypothesis, reference):
    """
    Simple BLEU-1 Score implementation (Unigram overlap).
    For full BLEU-4, usually NLTK is used, but this suffices for basic text eval.
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    
    hyp_counts = Counter(hyp_words)
    ref_counts = Counter(ref_words)
    
    # Clipped counts
    clipped_counts = {word: min(count, ref_counts[word]) for word, count in hyp_counts.items()}
    
    matches = sum(clipped_counts.values())
    total = len(hyp_words)
    
    if total == 0: return 0.0
    
    precision = matches / total
    
    # Brevity Penalty
    if len(hyp_words) > len(ref_words):
        bp = 1
    else:
        bp = math.exp(1 - len(ref_words) / len(hyp_words)) if len(hyp_words) > 0 else 0
        
    return bp * precision
