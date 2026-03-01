"""
AG-MAN Preprocessing + Evaluation Pipeline
==========================================
Reduces distribution shift between scraped Myntra images (model-worn, cluttered)
and AG-MAN's training data (product-only, white background).

3 Pipelines compared:
  1. RAW       - No preprocessing, direct to AG-MAN
  2. CROP_ONLY - Category-aware vertical crop, then AG-MAN
  3. CROP_REMBG- Category-aware crop + rembg background removal + white bg, then AG-MAN

All preprocessing operates on ORIGINAL resolution images (420x560).
AG-MAN transform handles final resize to 224x224.
AG-MAN code is NOT modified.

Usage:
  python tools/preprocessing_pipeline.py
"""

import os
import sys
import io
import csv
import time
import base64
import logging
import traceback
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ===================================================================
# PATHS
# ===================================================================
IMAGE_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase\myntra"
OUTPUT_DIR = r"D:\Final_Year_Project\docs\Review_2\test_supabase\evaluation_results"

# Add backend to path so we can import AG-MAN
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BACKEND_DIR)

# ===================================================================
# CATEGORY GROUPS + CROP RULES (domain-informed, safety-checked)
# ===================================================================
# Crop rules are informed by fashion photography conventions:
# - Large garments: face at top, shoes at bottom → trim both
# - Footwear: product in lower half → trim upper
# - Head accessories: product in upper half → trim lower
# - Middle accessories: product in middle → trim both ends
# - Small jewelry: often model-worn → light trim only
#
# crop_top = fraction to remove from top
# crop_bottom = fraction to remove from bottom
# apply_rembg = whether to apply background removal
# ===================================================================

CATEGORY_CONFIG = {
    # ---- Large Fabric Garments ----
    "shirts":           {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "tshirt":           {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "blazer":           {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "Jacket":           {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "pant":             {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "shorts":           {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "skirt":            {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "churidhar":        {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    "dhoti":            {"group": "large_fabric",      "crop_top": 0.08, "crop_bottom": 0.08, "apply_rembg": True},
    # ---- Footwear ----
    "Footwear_sandals": {"group": "footwear",          "crop_top": 0.18, "crop_bottom": 0.08, "apply_rembg": True},
    "Footwear_shoes":   {"group": "footwear",          "crop_top": 0.18, "crop_bottom": 0.08, "apply_rembg": True},
    # ---- Head Accessories ----
    "caps":             {"group": "head_accessories",   "crop_top": 0.05, "crop_bottom": 0.22, "apply_rembg": False},
    "glasses":          {"group": "head_accessories",   "crop_top": 0.05, "crop_bottom": 0.22, "apply_rembg": False},
    # ---- Middle Accessories ----
    "belt":             {"group": "middle_accessories", "crop_top": 0.13, "crop_bottom": 0.13, "apply_rembg": False},
    "tie":              {"group": "middle_accessories", "crop_top": 0.13, "crop_bottom": 0.13, "apply_rembg": False},
    # ---- Small Jewelry ----
    "earrings":         {"group": "small_jewelry",      "crop_top": 0.08, "crop_bottom": 0.06, "apply_rembg": False},
    "necklace":         {"group": "small_jewelry",      "crop_top": 0.08, "crop_bottom": 0.06, "apply_rembg": False},
    "watch":            {"group": "small_jewelry",      "crop_top": 0.08, "crop_bottom": 0.06, "apply_rembg": False},
}

# Map folder name to AG-MAN category name
FOLDER_TO_AGMAN_CATEGORY = {
    "shirts": "shirt",
    "tshirt": "tshirt",
    "blazer": "blazer",
    "Jacket": "jacket",
    "pant": "pant",
    "shorts": "shorts",
    "skirt": "skirt",
    "churidhar": "churidhar",
    "dhoti": "dhoti",
    "Footwear_sandals": "footwear_sandals",
    "Footwear_shoes": "footwear_shoes",
    "caps": "cap",
    "glasses": "glass",
    "belt": "belt",
    "tie": "tie",
    "earrings": "earring",
    "necklace": "necklace",
    "watch": "watch",
}


# ===================================================================
# PREPROCESSING FUNCTIONS
# ===================================================================

def category_crop(pil_img, category):
    """
    Apply category-aware vertical crop.
    Uses minimum safe crop bands (domain-informed).
    Returns cropped PIL image.
    """
    config = CATEGORY_CONFIG.get(category)
    if config is None:
        log.warning(f"Unknown category '{category}', skipping crop")
        return pil_img

    w, h = pil_img.size
    top_frac = config["crop_top"]
    bot_frac = config["crop_bottom"]

    y_top = int(h * top_frac)
    y_bot = int(h * (1.0 - bot_frac))

    # Safety: ensure we keep at least 50% of the image
    if y_bot - y_top < h * 0.5:
        log.warning(f"Crop too aggressive for {category}: {top_frac}/{bot_frac}, reducing")
        y_top = int(h * 0.05)
        y_bot = int(h * 0.95)

    cropped = pil_img.crop((0, y_top, w, y_bot))
    return cropped


def apply_rembg_with_validation(pil_img, category):
    """
    Apply rembg background removal with mask coverage validation.
    
    Rejects masks where:
      - Foreground coverage < 15% (rembg failed, removed too much)
      - Foreground coverage > 95% (rembg did nothing useful)
    
    Returns:
      (result_pil_img, mask_info_dict)
    """
    try:
        from rembg import remove
    except ImportError:
        log.error("rembg not installed. Run: pip install rembg")
        return pil_img, {"status": "rembg_not_installed", "coverage": None}

    try:
        # Apply rembg — returns RGBA image
        rgba = remove(pil_img)

        # Extract alpha channel to compute mask coverage
        alpha = np.array(rgba)[:, :, 3]
        total_pixels = alpha.size
        fg_pixels = np.sum(alpha > 128)
        coverage = fg_pixels / total_pixels

        log.info(f"[rembg] {category}: mask coverage = {coverage:.1%}")

        # Validate mask
        if coverage < 0.15:
            log.warning(f"[rembg] REJECTED: coverage {coverage:.1%} < 15% — rembg removed too much")
            return pil_img, {"status": "rejected_low_coverage", "coverage": round(coverage, 4)}
        
        if coverage > 0.95:
            log.warning(f"[rembg] REJECTED: coverage {coverage:.1%} > 95% — rembg did nothing")
            return pil_img, {"status": "rejected_high_coverage", "coverage": round(coverage, 4)}

        # Compose on white background
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, rgba)
        result = composite.convert("RGB")

        return result, {"status": "applied", "coverage": round(coverage, 4)}

    except Exception as e:
        log.error(f"[rembg] Failed for {category}: {e}")
        return pil_img, {"status": f"error: {e}", "coverage": None}


def preprocess_for_agman(pil_img, category, pipeline="crop_rembg"):
    """
    Main preprocessing entry point.
    
    Args:
        pil_img: PIL Image (RGB, original resolution)
        category: Category folder name
        pipeline: "raw" | "crop_only" | "crop_rembg"
    
    Returns:
        (preprocessed_pil_img, preprocessing_info)
    """
    info = {
        "pipeline": pipeline,
        "category": category,
        "original_size": pil_img.size,
        "crop_applied": False,
        "rembg_info": None,
    }

    if pipeline == "raw":
        return pil_img, info

    # Step 1: Category-aware crop
    img = category_crop(pil_img, category)
    info["crop_applied"] = True
    info["cropped_size"] = img.size

    if pipeline == "crop_only":
        return img, info

    # Step 2: rembg (only for configured categories)
    config = CATEGORY_CONFIG.get(category, {})
    if config.get("apply_rembg", False):
        img, rembg_info = apply_rembg_with_validation(img, category)
        info["rembg_info"] = rembg_info
    else:
        info["rembg_info"] = {"status": "skipped_by_config", "coverage": None}

    return img, info


# ===================================================================
# AG-MAN INTEGRATION
# ===================================================================

def pil_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_agman_on_image(image_path, category, pipeline="crop_rembg"):
    """
    Full pipeline: load → preprocess → AG-MAN extract.
    
    Args:
        image_path: Path to image file
        category: Category folder name
        pipeline: "raw" | "crop_only" | "crop_rembg"
    
    Returns:
        dict with embedding, attributes, extraction_quality, preprocessing_info
        None if image is corrupted/missing
    """
    from models.agman_extractor import process_crop_base64

    agman_category = FOLDER_TO_AGMAN_CATEGORY.get(category, category.lower())

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"Failed to load {image_path}: {e}")
        return None

    # Preprocess
    preprocessed, preproc_info = preprocess_for_agman(pil_img, category, pipeline)

    # Convert to base64 and call AG-MAN
    b64 = pil_to_base64(preprocessed)
    
    try:
        result = process_crop_base64(b64, agman_category)
    except Exception as e:
        log.error(f"AG-MAN failed for {image_path}: {e}")
        return None

    return {
        "embedding": result.get("embedding"),
        "attributes": result.get("attributes", {}),
        "extraction_quality": result.get("attributes", {}).get("extraction_quality", 0.0),
        "preprocessing_info": preproc_info,
    }


# ===================================================================
# EVALUATION FUNCTIONS
# ===================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_intra_similarity(embeddings):
    """
    Compute average pairwise cosine similarity within a set of embeddings.
    Returns (mean_similarity, std_similarity, n_pairs).
    """
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0
    
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    
    return float(np.mean(sims)), float(np.std(sims)), len(sims)


def compute_inter_similarity(emb_a, emb_b):
    """
    Compute average cosine similarity between two sets of embeddings.
    Returns (mean_similarity, std_similarity, n_pairs).
    """
    sims = []
    for a in emb_a:
        for b in emb_b:
            sims.append(cosine_similarity(a, b))
    
    if not sims:
        return 0.0, 0.0, 0
    return float(np.mean(sims)), float(np.std(sims)), len(sims)


def get_image_files(folder, n=50):
    """Get up to n image files, sorted for determinism."""
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    return files[:n]


def evaluate_category(folder_path, category, n=50):
    """
    Evaluate AG-MAN on a single category across all 3 pipelines.
    
    Computes per-pipeline:
      - Intra-category cosine similarity (mean, std)
      - Mean extraction_quality
      - Sleeve prediction distribution
      - Per-attribute confidence stats
    
    Returns dict with results for all 3 pipelines.
    """
    files = get_image_files(folder_path, n)
    if not files:
        log.warning(f"No images found in {folder_path}")
        return None

    log.info(f"Evaluating {category}: {len(files)} images × 3 pipelines")

    pipelines = ["raw", "crop_only", "crop_rembg"]
    results = {}

    for pipeline in pipelines:
        embeddings = []
        qualities = []
        sleeve_preds = []
        pattern_preds = []
        color_confs = []
        pattern_confs = []
        sleeve_confs = []
        rembg_coverages = []
        errors = 0

        for f in files:
            fpath = os.path.join(folder_path, f)
            out = run_agman_on_image(fpath, category, pipeline)
            
            if out is None:
                errors += 1
                continue

            emb = out.get("embedding")
            attrs = out.get("attributes", {})

            if emb:
                embeddings.append(emb)
            
            eq = attrs.get("extraction_quality", 0.0)
            qualities.append(eq)

            # Sleeve distribution
            sleeve = attrs.get("sleeve") or attrs.get("sleeve_length")
            if sleeve:
                sleeve_preds.append(sleeve)

            # Pattern distribution
            pattern = attrs.get("pattern")
            if pattern:
                pattern_preds.append(pattern)

            # Confidence tracking
            color_struct = attrs.get("color", {})
            if isinstance(color_struct, dict) and "confidence" in color_struct:
                color_confs.append(color_struct["confidence"])

            pattern_struct = attrs.get("pattern_structured", {})
            if isinstance(pattern_struct, dict) and pattern_struct and "confidence" in pattern_struct:
                pattern_confs.append(pattern_struct["confidence"])

            sleeve_struct = attrs.get("sleeve_structured", {})
            if isinstance(sleeve_struct, dict) and sleeve_struct and "confidence" in sleeve_struct:
                sleeve_confs.append(sleeve_struct["confidence"])

            # rembg coverage
            preproc = out.get("preprocessing_info", {})
            rembg = preproc.get("rembg_info", {})
            if rembg and rembg.get("coverage") is not None:
                rembg_coverages.append(rembg["coverage"])

        # Compute intra-category similarity
        intra_mean, intra_std, n_pairs = compute_intra_similarity(embeddings)

        # Sleeve distribution as counts
        sleeve_dist = {}
        for s in sleeve_preds:
            sleeve_dist[s] = sleeve_dist.get(s, 0) + 1

        # Pattern distribution as counts
        pattern_dist = {}
        for p in pattern_preds:
            pattern_dist[p] = pattern_dist.get(p, 0) + 1

        results[pipeline] = {
            "n_images": len(files),
            "n_processed": len(embeddings),
            "n_errors": errors,
            "intra_similarity_mean": round(intra_mean, 4),
            "intra_similarity_std": round(intra_std, 4),
            "extraction_quality_mean": round(float(np.mean(qualities)) if qualities else 0.0, 4),
            "extraction_quality_std": round(float(np.std(qualities)) if qualities else 0.0, 4),
            # Sleeve distribution
            "sleeve_distribution": sleeve_dist,
            # Pattern distribution
            "pattern_distribution": pattern_dist,
            # Confidence stats
            "color_confidence_mean": round(float(np.mean(color_confs)) if color_confs else 0.0, 4),
            "color_confidence_var": round(float(np.var(color_confs)) if color_confs else 0.0, 6),
            "pattern_confidence_mean": round(float(np.mean(pattern_confs)) if pattern_confs else 0.0, 4),
            "pattern_confidence_var": round(float(np.var(pattern_confs)) if pattern_confs else 0.0, 6),
            "sleeve_confidence_mean": round(float(np.mean(sleeve_confs)) if sleeve_confs else 0.0, 4),
            "sleeve_confidence_var": round(float(np.var(sleeve_confs)) if sleeve_confs else 0.0, 6),
            # rembg stats
            "rembg_coverage_mean": round(float(np.mean(rembg_coverages)) if rembg_coverages else None or 0.0, 4),
            "rembg_reject_count": sum(1 for c in rembg_coverages if c < 0.15 or c > 0.95),
            # Raw embeddings for cross-category analysis
            "_embeddings": embeddings,
        }

        log.info(
            f"  [{pipeline:12s}] {category}: "
            f"intra_sim={intra_mean:.4f}±{intra_std:.4f}, "
            f"quality={results[pipeline]['extraction_quality_mean']:.3f}, "
            f"n={len(embeddings)}"
        )

    return results


def evaluate_cross_category(all_category_results, pipeline="raw"):
    """
    Compute inter-category cosine similarity for every category pair.
    
    Args:
        all_category_results: dict of {category: evaluate_category_result}
        pipeline: which pipeline's embeddings to use
    
    Returns:
        dict with:
          - per_pair: {(cat_a, cat_b): {mean, std}}
          - per_category_cluster_separation: {cat: intra - avg_inter}
    """
    categories = list(all_category_results.keys())
    
    # Extract embeddings per category
    cat_embeddings = {}
    for cat in categories:
        res = all_category_results[cat]
        if res and pipeline in res:
            embs = res[pipeline].get("_embeddings", [])
            if embs:
                cat_embeddings[cat] = embs

    cats_with_data = list(cat_embeddings.keys())
    if len(cats_with_data) < 2:
        log.warning("Need at least 2 categories with embeddings for cross-category analysis")
        return {"per_pair": {}, "per_category_cluster_separation": {}}

    # Per-pair inter-category similarity
    per_pair = {}
    for cat_a, cat_b in combinations(cats_with_data, 2):
        mean_sim, std_sim, n_pairs = compute_inter_similarity(
            cat_embeddings[cat_a], cat_embeddings[cat_b]
        )
        pair_key = f"{cat_a} vs {cat_b}"
        per_pair[pair_key] = {
            "mean": round(mean_sim, 4),
            "std": round(std_sim, 4),
            "n_pairs": n_pairs,
        }

    # Per-category cluster separation = intra_sim - avg(inter_sim with all others)
    per_cat_separation = {}
    for cat in cats_with_data:
        intra_mean = all_category_results[cat][pipeline]["intra_similarity_mean"]

        # Average inter-similarity with all other categories
        inter_sims = []
        for other_cat in cats_with_data:
            if other_cat == cat:
                continue
            pair_key = f"{cat} vs {other_cat}" if f"{cat} vs {other_cat}" in per_pair else f"{other_cat} vs {cat}"
            if pair_key in per_pair:
                inter_sims.append(per_pair[pair_key]["mean"])

        avg_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
        separation = intra_mean - avg_inter

        per_cat_separation[cat] = {
            "intra_similarity": round(intra_mean, 4),
            "avg_inter_similarity": round(avg_inter, 4),
            "cluster_separation": round(separation, 4),
        }

    return {
        "per_pair": per_pair,
        "per_category_cluster_separation": per_cat_separation,
    }


def compute_confidence_variance_reduction(all_category_results):
    """
    Compare per-attribute confidence variance across pipelines.
    Lower variance after preprocessing = more consistent extraction.
    """
    report = {}
    for cat, cat_results in all_category_results.items():
        if cat_results is None:
            continue
        cat_report = {}
        for attr in ["color", "pattern", "sleeve"]:
            var_key = f"{attr}_confidence_var"
            raw_var = cat_results.get("raw", {}).get(var_key, None)
            crop_var = cat_results.get("crop_only", {}).get(var_key, None)
            rembg_var = cat_results.get("crop_rembg", {}).get(var_key, None)
            cat_report[attr] = {
                "raw_variance": raw_var,
                "crop_only_variance": crop_var,
                "crop_rembg_variance": rembg_var,
                "crop_reduction": round(raw_var - crop_var, 6) if raw_var is not None and crop_var is not None else None,
                "rembg_reduction": round(raw_var - rembg_var, 6) if raw_var is not None and rembg_var is not None else None,
            }
        report[cat] = cat_report
    return report


# ===================================================================
# SLEEVE DISTRIBUTION ANALYSIS
# ===================================================================

def analyze_sleeve_shift(all_category_results):
    """
    Check if cropping causes systematic shift in sleeve prediction.
    Compare sleeve distributions across pipelines.
    
    Returns per-category comparison of sleeve distributions.
    """
    report = {}
    for cat, cat_results in all_category_results.items():
        if cat_results is None:
            continue
        
        cat_report = {}
        for pipeline in ["raw", "crop_only", "crop_rembg"]:
            if pipeline in cat_results:
                cat_report[pipeline] = cat_results[pipeline].get("sleeve_distribution", {})
        
        # Check for systematic shift
        raw_dist = cat_report.get("raw", {})
        crop_dist = cat_report.get("crop_only", {})
        
        if raw_dist and crop_dist:
            # Check if dominant sleeve type changed
            raw_dominant = max(raw_dist, key=raw_dist.get) if raw_dist else None
            crop_dominant = max(crop_dist, key=crop_dist.get) if crop_dist else None
            cat_report["shift_detected"] = raw_dominant != crop_dominant
            cat_report["raw_dominant"] = raw_dominant
            cat_report["crop_dominant"] = crop_dominant
        else:
            cat_report["shift_detected"] = None
        
        report[cat] = cat_report
    
    return report


# ===================================================================
# CSV EXPORT
# ===================================================================

def save_results_csv(all_category_results, cross_results, sleeve_report, 
                     confidence_report, output_dir):
    """Save all evaluation results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Per-category, per-pipeline summary
    summary_path = os.path.join(output_dir, "pipeline_comparison.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "pipeline", "n_processed", "n_errors",
            "intra_sim_mean", "intra_sim_std",
            "extraction_quality_mean", "extraction_quality_std",
            "color_conf_mean", "color_conf_var",
            "pattern_conf_mean", "pattern_conf_var",
            "sleeve_conf_mean", "sleeve_conf_var",
            "rembg_coverage_mean", "rembg_reject_count",
        ])
        for cat, cat_results in all_category_results.items():
            if cat_results is None:
                continue
            for pipeline in ["raw", "crop_only", "crop_rembg"]:
                r = cat_results.get(pipeline, {})
                writer.writerow([
                    cat, pipeline,
                    r.get("n_processed", 0), r.get("n_errors", 0),
                    r.get("intra_similarity_mean", ""), r.get("intra_similarity_std", ""),
                    r.get("extraction_quality_mean", ""), r.get("extraction_quality_std", ""),
                    r.get("color_confidence_mean", ""), r.get("color_confidence_var", ""),
                    r.get("pattern_confidence_mean", ""), r.get("pattern_confidence_var", ""),
                    r.get("sleeve_confidence_mean", ""), r.get("sleeve_confidence_var", ""),
                    r.get("rembg_coverage_mean", ""), r.get("rembg_reject_count", ""),
                ])
    log.info(f"Saved: {summary_path}")

    # 2. Cross-category similarity (per pair)
    for pipeline in ["raw", "crop_only", "crop_rembg"]:
        cross_path = os.path.join(output_dir, f"cross_category_{pipeline}.csv")
        cross_data = cross_results.get(pipeline, {})
        per_pair = cross_data.get("per_pair", {})
        if per_pair:
            with open(cross_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["category_pair", "inter_sim_mean", "inter_sim_std", "n_pairs"])
                for pair, vals in per_pair.items():
                    writer.writerow([pair, vals["mean"], vals["std"], vals["n_pairs"]])
            log.info(f"Saved: {cross_path}")

    # 3. Per-category cluster separation
    separation_path = os.path.join(output_dir, "cluster_separation.csv")
    with open(separation_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "category", "intra_similarity", "avg_inter_similarity", "cluster_separation"])
        for pipeline in ["raw", "crop_only", "crop_rembg"]:
            cross_data = cross_results.get(pipeline, {})
            sep = cross_data.get("per_category_cluster_separation", {})
            for cat, vals in sep.items():
                writer.writerow([
                    pipeline, cat,
                    vals["intra_similarity"], vals["avg_inter_similarity"],
                    vals["cluster_separation"],
                ])
    log.info(f"Saved: {separation_path}")

    # 4. Sleeve distribution
    sleeve_path = os.path.join(output_dir, "sleeve_distribution.csv")
    with open(sleeve_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "pipeline", "sleeve_type", "count", "shift_detected"])
        for cat, cat_report in sleeve_report.items():
            shift = cat_report.get("shift_detected", "")
            for pipeline in ["raw", "crop_only", "crop_rembg"]:
                dist = cat_report.get(pipeline, {})
                if dist:
                    for sleeve_type, count in dist.items():
                        writer.writerow([cat, pipeline, sleeve_type, count, shift])
    log.info(f"Saved: {sleeve_path}")

    # 5. Confidence variance reduction
    conf_path = os.path.join(output_dir, "confidence_variance.csv")
    with open(conf_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "attribute",
            "raw_variance", "crop_only_variance", "crop_rembg_variance",
            "crop_reduction", "rembg_reduction",
        ])
        for cat, cat_report in confidence_report.items():
            for attr, vals in cat_report.items():
                writer.writerow([
                    cat, attr,
                    vals.get("raw_variance", ""),
                    vals.get("crop_only_variance", ""),
                    vals.get("crop_rembg_variance", ""),
                    vals.get("crop_reduction", ""),
                    vals.get("rembg_reduction", ""),
                ])
    log.info(f"Saved: {conf_path}")


# ===================================================================
# AG-MAN REAL-WORLD ASSESSMENT
# ===================================================================

def assess_agman_realworld(all_category_results):
    """
    Evaluate whether AG-MAN is suitable for real-world scraped fashion data.
    Analyses the RAW pipeline results.
    """
    print("\n" + "=" * 70)
    print("AG-MAN REAL-WORLD SUITABILITY ASSESSMENT")
    print("=" * 70)

    issues = []
    strengths = []

    for cat, cat_results in all_category_results.items():
        if cat_results is None:
            continue
        raw = cat_results.get("raw", {})

        eq = raw.get("extraction_quality_mean", 0)
        intra = raw.get("intra_similarity_mean", 0)
        errors = raw.get("n_errors", 0)
        n = raw.get("n_processed", 0)

        if eq < 0.5:
            issues.append(f"  [{cat}] Low extraction quality: {eq:.3f} (< 0.5)")
        elif eq > 0.7:
            strengths.append(f"  [{cat}] Good extraction quality: {eq:.3f}")

        if intra < 0.3:
            issues.append(f"  [{cat}] Low intra-category similarity: {intra:.4f} (< 0.3) — embeddings not discriminative")
        elif intra > 0.5:
            strengths.append(f"  [{cat}] Good intra-category similarity: {intra:.4f}")

        if errors > 0 and n > 0:
            error_rate = errors / (errors + n)
            if error_rate > 0.1:
                issues.append(f"  [{cat}] High error rate: {error_rate:.1%}")

    print("\nSTRENGTHS:")
    for s in strengths:
        print(s)

    print("\nISSUES:")
    for i in issues:
        print(i)

    # Overall assessment
    all_qualities = []
    all_intras = []
    for cat, cr in all_category_results.items():
        if cr and "raw" in cr:
            all_qualities.append(cr["raw"].get("extraction_quality_mean", 0))
            all_intras.append(cr["raw"].get("intra_similarity_mean", 0))

    avg_quality = np.mean(all_qualities) if all_qualities else 0
    avg_intra = np.mean(all_intras) if all_intras else 0

    print(f"\nOVERALL:")
    print(f"  Avg extraction quality (raw):       {avg_quality:.4f}")
    print(f"  Avg intra-category similarity (raw): {avg_intra:.4f}")

    if avg_quality >= 0.6 and avg_intra >= 0.4:
        print("  VERDICT: AG-MAN is SUITABLE for real-world use with preprocessing")
    elif avg_quality >= 0.5:
        print("  VERDICT: AG-MAN is MARGINALLY suitable — preprocessing is REQUIRED")
    else:
        print("  VERDICT: AG-MAN may struggle on scraped images — consider fine-tuning")

    print("=" * 70)


# ===================================================================
# MAIN BATCH RUNNER
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AG-MAN Preprocessing Pipeline")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Categories to evaluate (folder names). Default: all")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of images per category per pipeline")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory for CSV results")
    args = parser.parse_args()

    categories = args.categories or list(CATEGORY_CONFIG.keys())
    n_per_cat = args.n
    output_dir = args.output

    print("=" * 70)
    print("AG-MAN PREPROCESSING PIPELINE — 3-WAY COMPARISON")
    print(f"Categories: {len(categories)}")
    print(f"Images per category: {n_per_cat}")
    print(f"Pipelines: raw, crop_only, crop_rembg")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Phase 1: Evaluate each category
    all_results = {}
    for cat in categories:
        folder = os.path.join(IMAGE_BASE, cat)
        if not os.path.isdir(folder):
            log.warning(f"Folder not found: {folder}, skipping")
            continue
        
        t0 = time.time()
        result = evaluate_category(folder, cat, n=n_per_cat)
        elapsed = time.time() - t0
        log.info(f"  {cat} done in {elapsed:.1f}s")
        all_results[cat] = result

    # Phase 2: Cross-category analysis (per pipeline)
    print("\n" + "=" * 70)
    print("CROSS-CATEGORY ANALYSIS")
    print("=" * 70)
    
    cross_results = {}
    for pipeline in ["raw", "crop_only", "crop_rembg"]:
        log.info(f"Computing cross-category similarity for [{pipeline}]...")
        cross_results[pipeline] = evaluate_cross_category(all_results, pipeline)
        
        # Print per-category cluster separation
        sep = cross_results[pipeline].get("per_category_cluster_separation", {})
        for cat, vals in sep.items():
            log.info(
                f"  [{pipeline}] {cat}: "
                f"intra={vals['intra_similarity']:.4f}, "
                f"inter={vals['avg_inter_similarity']:.4f}, "
                f"separation={vals['cluster_separation']:.4f}"
            )

    # Phase 3: Sleeve distribution analysis
    print("\n" + "=" * 70)
    print("SLEEVE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    sleeve_report = analyze_sleeve_shift(all_results)
    for cat, report in sleeve_report.items():
        if report.get("shift_detected") is not None:
            log.info(f"  {cat}: shift_detected={report['shift_detected']}, "
                     f"raw_dominant={report.get('raw_dominant')}, "
                     f"crop_dominant={report.get('crop_dominant')}")

    # Phase 4: Confidence variance reduction
    print("\n" + "=" * 70)
    print("CONFIDENCE VARIANCE ANALYSIS")
    print("=" * 70)
    confidence_report = compute_confidence_variance_reduction(all_results)
    for cat, cat_report in confidence_report.items():
        for attr, vals in cat_report.items():
            crop_red = vals.get("crop_reduction")
            if crop_red is not None:
                direction = "REDUCED" if crop_red > 0 else "INCREASED"
                log.info(f"  {cat}/{attr}: crop variance {direction} by {abs(crop_red):.6f}")

    # Phase 5: AG-MAN real-world assessment
    assess_agman_realworld(all_results)

    # Phase 6: Save everything to CSV
    save_results_csv(all_results, cross_results, sleeve_report, confidence_report, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Category':20s} | {'raw intra':>10s} | {'crop intra':>10s} | {'rembg intra':>11s} | {'Best':>6s}")
    print("-" * 70)
    for cat in categories:
        cr = all_results.get(cat)
        if cr is None:
            continue
        raw_s = cr.get("raw", {}).get("intra_similarity_mean", 0)
        crop_s = cr.get("crop_only", {}).get("intra_similarity_mean", 0)
        rembg_s = cr.get("crop_rembg", {}).get("intra_similarity_mean", 0)
        best = max([(raw_s, "raw"), (crop_s, "crop"), (rembg_s, "rembg")], key=lambda x: x[0])
        print(f"  {cat:20s} | {raw_s:10.4f} | {crop_s:10.4f} | {rembg_s:11.4f} | {best[1]:>6s}")

    print(f"\nResults saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
