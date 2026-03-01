"""
Retrieval Validation Pipeline — All 5 Steps
============================================
Validates retrieval quality of the 34,787 visual_attributes embeddings
before proceeding to Supabase deployment.

Steps:
  1. Build FAISS IndexFlatIP from all embeddings
  2. Controlled retrieval tests (Top-1/5 same-cat, same-group, Recall@10)
  3. Full intra/inter-category similarity + weak category deep-dive
  4. Hybrid scoring prototype: A (default) + B (weak-cat boost)
  5. Speed benchmarks (load, build, query, memory via psutil)

Corrections applied:
  - Color normalization + unmapped % tracking
  - Adaptive hybrid weighting for weak categories
  - Inter-similarity excludes same-group categories
  - Explicit memory via psutil
  - Top-1 accuracy
  - Confusion pairs for weak categories
  - Go/No-Go: same-cat >= 65%, same-group >= 85%, Recall@10 >= 90%
"""

import os
import sys
import time
import math
import tracemalloc
import numpy as np
import psycopg2
import faiss
from collections import Counter, defaultdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed — memory metrics will use tracemalloc fallback")

# ─── Configuration ──────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@",
}
EMB_DIM = 512
RNG_SEED = 42

# ─── Semantic Category Groups ──────────────────────────────────────
CATEGORY_GROUPS = {
    "upper_wear":  {"shirts", "tshirt", "blazer", "Jacket"},
    "lower_wear":  {"pant", "shorts", "skirt"},
    "jewelry":     {"earrings", "necklace"},
    "footwear":    {"Footwear_sandals", "Footwear_shoes"},
    "accessories": {"belt", "tie", "caps", "glasses", "watch"},
    "traditional": {"dhoti", "churidhar"},
}

UPPER_WEAR = CATEGORY_GROUPS["upper_wear"]

# Build reverse lookup: category → group name
_CAT_TO_GROUP = {}
for grp, cats in CATEGORY_GROUPS.items():
    for c in cats:
        _CAT_TO_GROUP[c] = grp


def get_group(category):
    return _CAT_TO_GROUP.get(category, "unknown")


# ─── Color Family Mapping ──────────────────────────────────────────
COLOR_FAMILIES = {
    "neutral": {"Black", "White", "Grey", "Charcoal", "Silver", "Off White", "Cream", "Multi"},
    "red":     {"Red", "Maroon", "Burgundy", "Wine", "Rust"},
    "pink":    {"Pink", "Hot Pink", "Rose", "Magenta", "Peach", "Coral"},
    "orange":  {"Orange", "Mustard", "Gold"},
    "yellow":  {"Yellow", "Lime Yellow"},
    "green":   {"Green", "Olive", "Khaki", "Lime Green", "Sea Green"},
    "blue":    {"Blue", "Navy Blue", "Sky Blue", "Turquoise", "Teal", "Teal Blue", "Steel Blue"},
    "purple":  {"Purple", "Lavender", "Violet", "Mauve"},
    "brown":   {"Brown", "Coffee Brown", "Chocolate", "Taupe", "Beige", "Tan"},
}

_COLOR_TO_FAMILY = {}
for fam, color_set in COLOR_FAMILIES.items():
    for c in color_set:
        _COLOR_TO_FAMILY[c] = fam


def normalize_color(color):
    """Normalize color name: strip whitespace, title-case."""
    if not color:
        return None
    return color.strip().title()


def get_color_family(color_name):
    """Returns family string. Normalizes before lookup."""
    normalized = normalize_color(color_name)
    if not normalized:
        return "unknown"
    return _COLOR_TO_FAMILY.get(normalized, "unknown")


def color_family_match(c1, c2):
    """1.0 if same color family, else 0.0. Unknown → 0.0."""
    f1 = get_color_family(c1)
    f2 = get_color_family(c2)
    if f1 == "unknown" or f2 == "unknown":
        return 0.0
    return 1.0 if f1 == f2 else 0.0


def pattern_match(p1, p2):
    """1.0 if both None, or both equal non-null; else 0.0."""
    if p1 is None and p2 is None:
        return 1.0
    if p1 is not None and p2 is not None:
        return 1.0 if p1 == p2 else 0.0
    return 0.0


def tier_label(rate):
    """Hard failure thresholds."""
    if rate >= 70:
        return "✅ Strong"
    elif rate >= 60:
        return "🟡 Acceptable"
    elif rate >= 50:
        return "⚠️ Risk"
    else:
        return "🔴 Failure"


def get_memory_mb():
    """Get current RSS in MB using psutil, or tracemalloc fallback."""
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    else:
        current, _ = tracemalloc.get_traced_memory()
        return current / 1024 / 1024


# ─── Data Loading ──────────────────────────────────────────────────
def load_all_embeddings():
    """Load all rows from visual_attributes. Returns parallel arrays."""
    print("Loading embeddings from PostgreSQL...")
    t0 = time.time()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT product_id, category, primary_color_name,
               pattern_value, sleeve_value, embedding
        FROM visual_attributes
        ORDER BY id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    product_ids = []
    categories = []
    colors = []
    patterns = []
    sleeves = []
    embeddings = []

    for r in rows:
        product_ids.append(r[0])
        categories.append(r[1])
        colors.append(r[2])
        patterns.append(r[3])
        sleeves.append(r[4])
        embeddings.append(list(r[5]))

    emb_np = np.array(embeddings, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"  Loaded {len(product_ids)} rows in {elapsed:.2f}s")

    # ---- Color unmapped % tracking ----
    total_colors = len(colors)
    unmapped = 0
    unmapped_names = Counter()
    for c in colors:
        fam = get_color_family(c)
        if fam == "unknown" and c is not None:
            unmapped += 1
            unmapped_names[normalize_color(c)] += 1

    null_colors = sum(1 for c in colors if c is None)
    unmapped_pct = (unmapped / total_colors * 100) if total_colors > 0 else 0
    print(f"\n  Color mapping health:")
    print(f"    Total colors:     {total_colors}")
    print(f"    NULL colors:      {null_colors}")
    print(f"    Unmapped (known): {unmapped} ({unmapped_pct:.1f}%)"
          f"  {'⚠️ > 10% — color family logic unreliable!' if unmapped_pct > 10 else '✅'}")
    if unmapped_names:
        print(f"    Top unmapped names: {unmapped_names.most_common(10)}")

    return product_ids, categories, colors, patterns, sleeves, emb_np


# =====================================================================
# STEP 1: Build FAISS Index
# =====================================================================
def step1_build_index(emb_np):
    """Build FAISS IndexFlatIP. Returns index and timing."""
    print("\n" + "=" * 70)
    print("STEP 1: BUILD FAISS IndexFlatIP")
    print("=" * 70)

    mem_before = get_memory_mb()

    # ---- Check norm before normalizing ----
    norms = np.linalg.norm(emb_np, axis=1)
    mean_norm = norms.mean()
    std_norm = norms.std()
    print(f"\n  Pre-normalization norms: mean={mean_norm:.6f}  std={std_norm:.6f}")

    if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
        print("  → Embeddings already unit-normalized, skipping re-normalization")
    else:
        print("  → Normalizing embeddings with faiss.normalize_L2")
        faiss.normalize_L2(emb_np)
        norms2 = np.linalg.norm(emb_np, axis=1)
        print(f"  Post-normalization norms: mean={norms2.mean():.6f}  std={norms2.std():.6f}")

    t0 = time.time()
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(emb_np)
    build_time = time.time() - t0

    mem_after = get_memory_mb()

    print(f"\n  Index vectors: {index.ntotal}")
    print(f"  Build time:    {build_time:.3f}s")
    print(f"  Memory delta:  {mem_after - mem_before:.1f} MB (before={mem_before:.1f}, after={mem_after:.1f})")
    return index, build_time


# =====================================================================
# STEP 2: Controlled Retrieval Tests
# =====================================================================
def step2_retrieval_tests(index, emb_np, categories, colors, product_ids):
    """Per-category retrieval evaluation with Top-1, Top-5, same-group, Recall@10."""
    print("\n" + "=" * 70)
    print("STEP 2: CONTROLLED RETRIEVAL TESTS")
    print("=" * 70)

    rng = np.random.RandomState(RNG_SEED)
    unique_cats = sorted(set(categories))
    cat_arr = np.array(categories)
    n_queries = 10
    top_k = 12  # 10 neighbours + self + margin

    results = []
    confusion_pairs = defaultdict(Counter)  # cat → {confused_cat: count}

    for cat in unique_cats:
        cat_mask = np.where(cat_arr == cat)[0]
        if len(cat_mask) < 2:
            continue
        if len(cat_mask) < n_queries:
            sample_idx = cat_mask
        else:
            sample_idx = rng.choice(cat_mask, size=n_queries, replace=False)

        top1_hits = 0
        same_cat_hits = 0
        same_grp_hits = 0
        total_top5 = 0
        recall10_hits = 0
        q_group = get_group(cat)

        for idx in sample_idx:
            q_vec = emb_np[idx:idx+1]
            D, I = index.search(q_vec, top_k)

            # Build neighbours list, skip self
            neighbours = []
            for j in range(top_k):
                nb = int(I[0][j])
                if nb == idx:
                    continue
                neighbours.append((float(D[0][j]), nb))
                if len(neighbours) >= 10:
                    break

            # Top-1 accuracy
            if neighbours and categories[neighbours[0][1]] == cat:
                top1_hits += 1

            # Top-5 same-category and same-group
            for sim, nb_idx in neighbours[:5]:
                nb_cat = categories[nb_idx]
                if nb_cat == cat:
                    same_cat_hits += 1
                else:
                    confusion_pairs[cat][nb_cat] += 1
                if get_group(nb_cat) == q_group:
                    same_grp_hits += 1
                total_top5 += 1

            # Recall@10: at least 1 same-category in top-10
            top10_cats = [categories[nb_idx] for _, nb_idx in neighbours[:10]]
            if cat in top10_cats:
                recall10_hits += 1

        n_q = len(sample_idx)
        top1_pct = (top1_hits / n_q * 100) if n_q > 0 else 0
        cat_rate = (same_cat_hits / total_top5 * 100) if total_top5 > 0 else 0
        grp_rate = (same_grp_hits / total_top5 * 100) if total_top5 > 0 else 0
        recall10 = (recall10_hits / n_q * 100) if n_q > 0 else 0

        results.append({
            "category": cat, "queries": n_q,
            "top1_pct": top1_pct, "same_cat_pct": cat_rate,
            "same_grp_pct": grp_rate, "recall10": recall10,
        })

    # Print table
    print(f"\n  {'Category':25s} | {'Top1%':>5s} | {'Cat%':>5s} | {'Grp%':>5s} | {'Rcl@10':>6s} | Tier")
    print("  " + "-" * 75)

    totals = {"top1": 0, "cat": 0, "grp": 0, "rcl": 0}
    for r in results:
        tier = tier_label(r["same_cat_pct"])
        print(f"  {r['category']:25s} | {r['top1_pct']:4.0f}% | {r['same_cat_pct']:4.0f}% | "
              f"{r['same_grp_pct']:4.0f}% | {r['recall10']:5.0f}% | {tier}")
        totals["top1"] += r["top1_pct"]
        totals["cat"]  += r["same_cat_pct"]
        totals["grp"]  += r["same_grp_pct"]
        totals["rcl"]  += r["recall10"]

    n = len(results)
    avg = {k: v / n for k, v in totals.items()}
    print("  " + "-" * 75)
    print(f"  {'AVERAGE':25s} | {avg['top1']:4.0f}% | {avg['cat']:4.0f}% | "
          f"{avg['grp']:4.0f}% | {avg['rcl']:5.0f}% | {tier_label(avg['cat'])}")

    print(f"\n  Go/No-Go:")
    print(f"    Same-category avg: {avg['cat']:.1f}%  {'✅ PASS' if avg['cat'] >= 65 else '❌ BELOW 65%'}")
    print(f"    Same-group avg:    {avg['grp']:.1f}%  {'✅ PASS' if avg['grp'] >= 85 else '❌ BELOW 85%'}")
    print(f"    Recall@10 avg:     {avg['rcl']:.1f}%  {'✅ PASS' if avg['rcl'] >= 90 else '❌ BELOW 90%'}")

    # --- Confusion pairs for weak categories ---
    weak_cats = [r["category"] for r in results if r["same_cat_pct"] < 60]
    if weak_cats:
        print(f"\n  --- Confusion Pairs (categories with < 60% same-cat) ---")
        for cat in weak_cats:
            if confusion_pairs[cat]:
                top_conf = confusion_pairs[cat].most_common(3)
                conf_str = ", ".join(f"{c}({n})" for c, n in top_conf)
                print(f"    {cat:25s} → confused with: {conf_str}")

    return results, confusion_pairs


# =====================================================================
# STEP 3: Full Intra/Inter Similarity + Weak Category Deep-Dive
# =====================================================================
def step3_similarity_analysis(index, emb_np, categories, colors, patterns, sleeves, product_ids):
    """Compute intra/inter similarity for ALL 18 categories.
    Inter-similarity excludes same-group categories (Correction #3)."""
    print("\n" + "=" * 70)
    print("STEP 3: INTRA / INTER SIMILARITY + EMBEDDING DISTRIBUTION")
    print("=" * 70)

    cat_arr = np.array(categories)
    unique_cats = sorted(set(categories))
    rng = np.random.RandomState(RNG_SEED)

    # ---- Embedding distribution check (10K random pairs) ----
    n_total = len(emb_np)
    n_pairs = 10000
    idx_a = rng.randint(0, n_total, size=n_pairs * 2)
    idx_b = rng.randint(0, n_total, size=n_pairs * 2)
    # Filter a != b
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask][:n_pairs], idx_b[mask][:n_pairs]
    pair_sims = np.sum(emb_np[idx_a] * emb_np[idx_b], axis=1)
    print(f"\n  Embedding distribution ({len(pair_sims)} random pairs):")
    print(f"    Mean cosine sim: {pair_sims.mean():.4f}")
    print(f"    Std deviation:   {pair_sims.std():.4f}")
    print(f"    Min:             {pair_sims.min():.4f}")
    print(f"    Max:             {pair_sims.max():.4f}")
    if pair_sims.std() < 0.02:
        print("    ⚠️  Very narrow spread — embeddings may lack discriminative power")
    else:
        print("    ✅ Healthy spread")

    # ---- Intra / Inter similarity (sampled) ----
    SAMPLE_SIZE = 200
    sim_results = []

    for cat in unique_cats:
        cat_mask = np.where(cat_arr == cat)[0]
        cat_group = get_group(cat)

        # Intra-similarity
        if len(cat_mask) > SAMPLE_SIZE:
            intra_sample = rng.choice(cat_mask, size=SAMPLE_SIZE, replace=False)
        else:
            intra_sample = cat_mask

        if len(intra_sample) > 1:
            intra_vecs = emb_np[intra_sample]
            sim_mat = intra_vecs @ intra_vecs.T
            n = len(intra_sample)
            diag_mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(diag_mask, False)
            intra_sim = float(sim_mat[diag_mask].mean())
        else:
            intra_sim = 0.0

        # Inter-similarity: ALL other categories
        other_all_mask = np.where(cat_arr != cat)[0]
        if len(other_all_mask) > SAMPLE_SIZE:
            inter_all_sample = rng.choice(other_all_mask, size=SAMPLE_SIZE, replace=False)
        else:
            inter_all_sample = other_all_mask

        if len(intra_sample) > 0 and len(inter_all_sample) > 0:
            q_vecs = emb_np[intra_sample[:min(50, len(intra_sample))]]
            i_vecs = emb_np[inter_all_sample]
            inter_all_sim = float((q_vecs @ i_vecs.T).mean())
        else:
            inter_all_sim = 0.0

        # Inter-similarity: CROSS-GROUP ONLY (Correction #3)
        cross_group_mask = np.array([
            i for i in range(n_total)
            if categories[i] != cat and get_group(categories[i]) != cat_group
        ])
        if len(cross_group_mask) > SAMPLE_SIZE:
            cross_sample = rng.choice(cross_group_mask, size=SAMPLE_SIZE, replace=False)
        else:
            cross_sample = cross_group_mask

        if len(intra_sample) > 0 and len(cross_sample) > 0:
            q_vecs = emb_np[intra_sample[:min(50, len(intra_sample))]]
            c_vecs = emb_np[cross_sample]
            inter_cross_sim = float((q_vecs @ c_vecs.T).mean())
        else:
            inter_cross_sim = 0.0

        sep_all = intra_sim - inter_all_sim
        sep_cross = intra_sim - inter_cross_sim

        sim_results.append({
            "category": cat, "count": len(cat_mask),
            "intra": intra_sim,
            "inter_all": inter_all_sim, "sep_all": sep_all,
            "inter_xgrp": inter_cross_sim, "sep_xgrp": sep_cross,
        })

    print(f"\n  {'Category':25s} | {'N':>5s} | {'Intra':>6s} | {'InterAll':>8s} | {'SepAll':>6s} | {'InterXG':>7s} | {'SepXG':>6s} | Status")
    print("  " + "-" * 100)

    weak_cats = []
    for r in sim_results:
        if r["sep_xgrp"] < 0.05:
            status = "🔴 Weak"
            weak_cats.append(r["category"])
        elif r["sep_xgrp"] < 0.10:
            status = "⚠️ Low"
        else:
            status = "✅ Good"
        print(f"  {r['category']:25s} | {r['count']:5d} | {r['intra']:.4f} | "
              f"{r['inter_all']:8.4f} | {r['sep_all']:.4f} | "
              f"{r['inter_xgrp']:7.4f} | {r['sep_xgrp']:.4f} | {status}")

    # ---- Deep-dive on weak / risky categories ----
    focus_cats = set(weak_cats) | {"earrings", "necklace", "tie", "glasses", "dhoti"}
    focus_cats = focus_cats & set(unique_cats)

    if focus_cats:
        print(f"\n  --- Weak Category Deep-Dive ({len(focus_cats)} categories) ---")

        for cat in sorted(focus_cats):
            cat_mask = np.where(cat_arr == cat)[0]
            if len(cat_mask) < 3:
                continue

            sample_q = rng.choice(cat_mask, size=min(3, len(cat_mask)), replace=False)
            print(f"\n  [{cat}] 3 retrieval examples:")

            for qi in sample_q:
                q_vec = emb_np[qi:qi+1]
                D, I = index.search(q_vec, 6)
                print(f"    Query: {product_ids[qi]} | color={colors[qi]} | pattern={patterns[qi]}")
                for rank in range(1, 6):
                    nb = int(I[0][rank])
                    sim = float(D[0][rank])
                    same = "✓" if categories[nb] == cat else "✗"
                    print(f"      #{rank} sim={sim:.4f} | cat={categories[nb]:15s} {same} | "
                          f"color={colors[nb]} | pattern={patterns[nb]}")

    return sim_results, weak_cats


# =====================================================================
# STEP 4: Hybrid Scoring Prototype (A + B)
# =====================================================================
def step4_hybrid_scoring(index, emb_np, categories, colors, patterns, sleeves, product_ids, weak_cats):
    """Re-rank with hybrid scoring.
    Hybrid A: default weights
    Hybrid B: weak-category boost (more color/pattern weight)
    """
    print("\n" + "=" * 70)
    print("STEP 4: HYBRID SCORING (A=default, B=weak-cat boost)")
    print("=" * 70)

    rng = np.random.RandomState(RNG_SEED)
    cat_arr = np.array(categories)
    unique_cats = sorted(set(categories))
    weak_set = set(weak_cats)
    fetch_k = 22  # Top-20 + self + margin

    pure_results = []
    hybrid_a_results = []
    hybrid_b_results = []

    for cat in unique_cats:
        cat_mask = np.where(cat_arr == cat)[0]
        if len(cat_mask) < 2:
            continue
        sample_idx = rng.choice(cat_mask, size=min(10, len(cat_mask)), replace=False)

        is_upper = cat in UPPER_WEAR
        is_weak = cat in weak_set

        counts = {"pure_cat": 0, "ha_cat": 0, "hb_cat": 0,
                  "pure_grp": 0, "ha_grp": 0, "hb_grp": 0, "total": 0}
        q_group = get_group(cat)

        for idx in sample_idx:
            q_vec = emb_np[idx:idx+1]
            D, I = index.search(q_vec, fetch_k)

            candidates = []
            for j in range(fetch_k):
                nb = int(I[0][j])
                if nb == idx:
                    continue
                cos_sim = float(D[0][j])
                nb_cat = categories[nb]
                nb_grp = get_group(nb_cat)

                c_score = color_family_match(colors[idx], colors[nb])
                p_score = pattern_match(patterns[idx], patterns[nb])

                # --- Hybrid A: default ---
                if is_upper:
                    s_score = 1.0 if sleeves[idx] == sleeves[nb] else 0.0
                    h_a = 0.6 * cos_sim + 0.2 * c_score + 0.1 * p_score + 0.1 * s_score
                else:
                    h_a = 0.7 * cos_sim + 0.2 * c_score + 0.1 * p_score

                # --- Hybrid B: weak-category boost ---
                if is_weak:
                    h_b = 0.5 * cos_sim + 0.3 * c_score + 0.2 * p_score
                elif is_upper:
                    s_score = 1.0 if sleeves[idx] == sleeves[nb] else 0.0
                    h_b = 0.6 * cos_sim + 0.2 * c_score + 0.1 * p_score + 0.1 * s_score
                else:
                    h_b = 0.7 * cos_sim + 0.2 * c_score + 0.1 * p_score

                candidates.append({
                    "idx": nb, "cos": cos_sim, "h_a": h_a, "h_b": h_b,
                    "cat": nb_cat, "grp": nb_grp,
                })

            if not candidates:
                continue

            # Pure top-5
            pure_top5 = sorted(candidates, key=lambda x: x["cos"], reverse=True)[:5]
            for c in pure_top5:
                if c["cat"] == cat: counts["pure_cat"] += 1
                if c["grp"] == q_group: counts["pure_grp"] += 1
                counts["total"] += 1

            # Hybrid A top-5
            ha_top5 = sorted(candidates, key=lambda x: x["h_a"], reverse=True)[:5]
            for c in ha_top5:
                if c["cat"] == cat: counts["ha_cat"] += 1
                if c["grp"] == q_group: counts["ha_grp"] += 1

            # Hybrid B top-5
            hb_top5 = sorted(candidates, key=lambda x: x["h_b"], reverse=True)[:5]
            for c in hb_top5:
                if c["cat"] == cat: counts["hb_cat"] += 1
                if c["grp"] == q_group: counts["hb_grp"] += 1

        t = counts["total"]
        if t == 0:
            continue

        pure_results.append({"category": cat,
                             "cat_pct": counts["pure_cat"] / t * 100,
                             "grp_pct": counts["pure_grp"] / t * 100})
        hybrid_a_results.append({"category": cat,
                                  "cat_pct": counts["ha_cat"] / t * 100,
                                  "grp_pct": counts["ha_grp"] / t * 100})
        hybrid_b_results.append({"category": cat,
                                  "cat_pct": counts["hb_cat"] / t * 100,
                                  "grp_pct": counts["hb_grp"] / t * 100})

    # Print comparison table
    weak_mark = lambda cat: " ★" if cat in weak_set else ""
    print(f"\n  {'Category':25s} | {'Pure%':>5s} | {'HybA%':>5s} | {'ΔA':>5s} | {'HybB%':>5s} | {'ΔB':>5s} | {'Best':>5s}")
    print("  " + "-" * 85)

    totals_p = totals_a = totals_b = 0
    for p, ha, hb in zip(pure_results, hybrid_a_results, hybrid_b_results):
        da = ha["cat_pct"] - p["cat_pct"]
        db = hb["cat_pct"] - p["cat_pct"]
        best = "B" if db > da else ("A" if da > 0 else "Pure")
        da_s = f"{da:+.0f}" if da != 0 else "  0"
        db_s = f"{db:+.0f}" if db != 0 else "  0"
        wm = weak_mark(p["category"])
        print(f"  {p['category'] + wm:25s} | {p['cat_pct']:4.0f}% | {ha['cat_pct']:4.0f}% | {da_s:>5s} | "
              f"{hb['cat_pct']:4.0f}% | {db_s:>5s} | {best}")
        totals_p += p["cat_pct"]
        totals_a += ha["cat_pct"]
        totals_b += hb["cat_pct"]

    n = len(pure_results)
    avg_p = totals_p / n
    avg_a = totals_a / n
    avg_b = totals_b / n
    imp_a = avg_a - avg_p
    imp_b = avg_b - avg_p

    print("  " + "-" * 85)
    print(f"  {'AVERAGE':25s} | {avg_p:4.0f}% | {avg_a:4.0f}% | {imp_a:+4.0f} | {avg_b:4.0f}% | {imp_b:+4.0f} |")
    print(f"\n  Hybrid A improvement: {imp_a:+.1f}%  {'✅ ≥ 5%' if imp_a >= 5 else '⚠️ < 5%'}")
    print(f"  Hybrid B improvement: {imp_b:+.1f}%  {'✅ ≥ 5%' if imp_b >= 5 else '⚠️ < 5%'}")
    print(f"  Recommended config:   {'B (weak-cat boost)' if imp_b > imp_a else 'A (default)'}")


# =====================================================================
# STEP 5: Speed Benchmarks
# =====================================================================
def step5_speed_benchmark(index, emb_np, db_load_time, build_time):
    """Measure query latency and memory usage (psutil)."""
    print("\n" + "=" * 70)
    print("STEP 5: SPEED BENCHMARKS")
    print("=" * 70)

    rng = np.random.RandomState(RNG_SEED)

    # Single query latency (100 queries)
    n_single = 100
    idxs = rng.randint(0, len(emb_np), size=n_single)
    t0 = time.time()
    for i in idxs:
        index.search(emb_np[i:i+1], 10)
    single_total = time.time() - t0
    single_avg_ms = single_total / n_single * 1000

    # Batch query latency (10 batches of 10)
    n_batch = 10
    batch_size = 10
    t0 = time.time()
    for _ in range(n_batch):
        batch_idx = rng.randint(0, len(emb_np), size=batch_size)
        batch_vecs = emb_np[batch_idx]
        index.search(batch_vecs, 10)
    batch_total = time.time() - t0
    batch_avg_ms = batch_total / n_batch * 1000

    # Memory
    mem_rss = get_memory_mb()

    print(f"\n  DB load time:       {db_load_time:.2f}s")
    print(f"  Index build time:   {build_time:.3f}s  {'✅ < 2s' if build_time < 2 else '⚠️ ≥ 2s'}")
    print(f"  Single query avg:   {single_avg_ms:.2f}ms  {'✅ < 5ms' if single_avg_ms < 5 else '⚠️ ≥ 5ms'}")
    print(f"  Batch query avg:    {batch_avg_ms:.2f}ms (10 vectors)")
    print(f"  Process RSS:        {mem_rss:.1f} MB")


# =====================================================================
# MAIN
# =====================================================================
def main():
    tracemalloc.start()

    print("=" * 70)
    print("RETRIEVAL VALIDATION PIPELINE")
    print("=" * 70)

    # Load data
    t_load_start = time.time()
    product_ids, categories, colors, patterns, sleeves, emb_np = load_all_embeddings()
    db_load_time = time.time() - t_load_start

    print(f"\n  Total vectors: {len(product_ids)}")
    print(f"  Embedding dim: {emb_np.shape[1]}")
    print(f"  Categories:    {len(set(categories))}")

    # Step 1
    index, build_time = step1_build_index(emb_np)

    # Step 2
    step2_results, confusion_pairs = step2_retrieval_tests(index, emb_np, categories, colors, product_ids)

    # Step 3
    step3_results, weak_cats = step3_similarity_analysis(index, emb_np, categories, colors, patterns, sleeves, product_ids)

    # Step 4 — pass weak_cats for adaptive weighting
    step4_hybrid_scoring(index, emb_np, categories, colors, patterns, sleeves, product_ids, weak_cats)

    # Step 5
    step5_speed_benchmark(index, emb_np, db_load_time, build_time)

    tracemalloc.stop()
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
