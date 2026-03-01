"""
Image Analysis Script for AG-MAN Preprocessing Pipeline.
Analyses 20-30 sample images per category to determine:
  - Resolution distribution (width, height, aspect ratio)
  - Brightness/contrast distribution (detect cluttered backgrounds)
  - Vertical intensity profile (where product region is)
  - Recommended crop bands per category group

Output: CSV report + console summary
"""

import os
import sys
import csv
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict

# ===================================================================
# CONFIGURATION
# ===================================================================
IMAGE_BASE = r"D:\Final_Year_Project\docs\Review_2\test_supabase\myntra"
SAMPLES_PER_CATEGORY = 25  # Analyse 25 images per category
OUTPUT_CSV = r"D:\Final_Year_Project\docs\Review_2\test_supabase\image_analysis_report.csv"

# Category groups (from spec)
CATEGORY_GROUPS = {
    "large_fabric": ["shirts", "tshirt", "blazer", "Jacket", "pant", "shorts", "skirt", "churidhar", "dhoti"],
    "footwear": ["Footwear_sandals", "Footwear_shoes"],
    "head_accessories": ["caps", "glasses"],
    "middle_accessories": ["belt", "tie"],
    "small_jewelry": ["earrings", "necklace", "watch"],
}

# Reverse map: category folder name -> group
FOLDER_TO_GROUP = {}
for group, cats in CATEGORY_GROUPS.items():
    for cat in cats:
        FOLDER_TO_GROUP[cat] = group


def get_image_files(folder, n=SAMPLES_PER_CATEGORY):
    """Get up to n image files from folder, sorted for determinism."""
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    return files[:n]


def compute_vertical_intensity_profile(img_rgb):
    """
    Compute vertical intensity variance per horizontal band.
    Returns a 1D array of length H, where each value is the
    standard deviation of pixel intensity in that row.
    Higher values = more texture/detail = likely product region.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Compute per-row statistics
    row_std = np.std(gray, axis=1)  # std dev per row
    row_mean = np.mean(gray, axis=1)
    
    return row_std, row_mean


def find_product_band(row_std, threshold_ratio=0.5):
    """
    Find the vertical band where product dominates.
    Uses intensity variance: product region has higher texture/variance
    than empty/skin background.
    
    Returns (top_pct, bottom_pct) - the fraction of image height
    where product starts and ends.
    """
    h = len(row_std)
    if h == 0:
        return 0.0, 1.0
    
    # Smooth the profile
    kernel_size = max(h // 20, 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(row_std, np.ones(kernel_size)/kernel_size, mode='same')
    
    threshold = np.max(smoothed) * threshold_ratio
    
    # Find first and last rows above threshold
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return 0.0, 1.0
    
    top_row = above[0]
    bottom_row = above[-1]
    
    return top_row / h, bottom_row / h


def analyze_single_image(filepath):
    """Analyse a single image and return stats dict."""
    try:
        pil = Image.open(filepath).convert("RGB")
        w, h = pil.size
        img_rgb = np.array(pil)
        
        # Intensity profile
        row_std, row_mean = compute_vertical_intensity_profile(img_rgb)
        top_pct, bottom_pct = find_product_band(row_std)
        
        # Brightness stats
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Edge density (proxy for clutter)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Skin pixel ratio (YCrCb)
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        skin_mask = (
            (ycrcb[:,:,1] >= 133) & (ycrcb[:,:,1] <= 173) &
            (ycrcb[:,:,2] >= 77) & (ycrcb[:,:,2] <= 127)
        )
        skin_ratio = np.mean(skin_mask)
        
        # White pixel ratio (background proxy)
        white_mask = np.all(img_rgb > 230, axis=2)
        white_ratio = np.mean(white_mask)
        
        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "mean_brightness": round(float(mean_brightness), 1),
            "edge_density": round(float(edge_density), 4),
            "skin_ratio": round(float(skin_ratio), 4),
            "white_ratio": round(float(white_ratio), 4),
            "product_top_pct": round(float(top_pct), 3),
            "product_bottom_pct": round(float(bottom_pct), 3),
            "product_height_pct": round(float(bottom_pct - top_pct), 3),
        }
    except Exception as e:
        print(f"  [SKIP] {filepath}: {e}")
        return None


def analyze_category(category_folder, category_name, group_name):
    """Analyse a category, return list of per-image stats."""
    folder = os.path.join(IMAGE_BASE, category_folder)
    files = get_image_files(folder)
    
    if not files:
        print(f"  [WARN] No images found in {folder}")
        return []
    
    results = []
    for f in files:
        fpath = os.path.join(folder, f)
        stats = analyze_single_image(fpath)
        if stats:
            stats["category"] = category_name
            stats["group"] = group_name
            stats["filename"] = f
            results.append(stats)
    
    return results


def summarize_group(results, group_name):
    """Print summary statistics for a category group."""
    if not results:
        return {}
    
    widths = [r["width"] for r in results]
    heights = [r["height"] for r in results]
    top_pcts = [r["product_top_pct"] for r in results]
    bottom_pcts = [r["product_bottom_pct"] for r in results]
    skin_ratios = [r["skin_ratio"] for r in results]
    white_ratios = [r["white_ratio"] for r in results]
    
    summary = {
        "group": group_name,
        "n_images": len(results),
        "resolution_range": f"{min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}",
        "median_width": int(np.median(widths)),
        "median_height": int(np.median(heights)),
        "mean_product_top": round(float(np.mean(top_pcts)), 3),
        "mean_product_bottom": round(float(np.mean(bottom_pcts)), 3),
        "recommended_crop_top": round(float(np.percentile(top_pcts, 25)), 3),
        "recommended_crop_bottom": round(float(1.0 - np.percentile([1.0 - b for b in bottom_pcts], 25)), 3),
        "mean_skin_ratio": round(float(np.mean(skin_ratios)), 4),
        "mean_white_ratio": round(float(np.mean(white_ratios)), 4),
    }
    
    print(f"\n{'='*60}")
    print(f"  GROUP: {group_name.upper()} ({summary['n_images']} images)")
    print(f"{'='*60}")
    print(f"  Resolution: {summary['resolution_range']}")
    print(f"  Median: {summary['median_width']}x{summary['median_height']}")
    print(f"  Product band (mean): {summary['mean_product_top']:.1%} - {summary['mean_product_bottom']:.1%}")
    print(f"  Recommended crop:    top {summary['recommended_crop_top']:.1%}, keep to {summary['recommended_crop_bottom']:.1%}")
    print(f"  Mean skin ratio: {summary['mean_skin_ratio']:.2%}")
    print(f"  Mean white bg ratio: {summary['mean_white_ratio']:.2%}")
    
    return summary


def main():
    print("="*60)
    print("AG-MAN Image Analysis")
    print(f"Base directory: {IMAGE_BASE}")
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}")
    print("="*60)
    
    all_results = []
    group_results = defaultdict(list)
    
    # Scan all category folders
    for group_name, categories in CATEGORY_GROUPS.items():
        for cat_folder in categories:
            print(f"\n[Analysing] {cat_folder} ({group_name})...")
            cat_results = analyze_category(cat_folder, cat_folder, group_name)
            all_results.extend(cat_results)
            group_results[group_name].extend(cat_results)
            print(f"  Analysed {len(cat_results)} images")
    
    # Per-group summaries
    print("\n\n" + "="*60)
    print("GROUP SUMMARIES — RECOMMENDED CROP BANDS")
    print("="*60)
    
    group_summaries = []
    for group_name in CATEGORY_GROUPS:
        if group_results[group_name]:
            summary = summarize_group(group_results[group_name], group_name)
            group_summaries.append(summary)
    
    # Write detailed per-image CSV
    if all_results:
        fieldnames = ["group", "category", "filename", "width", "height", "aspect_ratio",
                       "mean_brightness", "edge_density", "skin_ratio", "white_ratio",
                       "product_top_pct", "product_bottom_pct", "product_height_pct"]
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\n[SAVED] Detailed analysis: {OUTPUT_CSV}")
    
    # Print final crop band recommendations
    print("\n\n" + "="*60)
    print("FINAL CROP BAND RECOMMENDATIONS")
    print("="*60)
    for s in group_summaries:
        crop_top = s["recommended_crop_top"]
        crop_bottom = s["recommended_crop_bottom"]
        print(f"  {s['group']:25s} → remove top {crop_top:.1%}, keep to {crop_bottom:.1%}  "
              f"(skin: {s['mean_skin_ratio']:.1%}, white_bg: {s['mean_white_ratio']:.1%})")
    
    print(f"\nTotal images analysed: {len(all_results)}")
    return all_results, group_summaries


if __name__ == "__main__":
    main()
