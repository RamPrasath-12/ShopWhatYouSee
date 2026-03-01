import csv
import numpy as np
from collections import defaultdict

data = defaultdict(lambda: {'tops': [], 'bottoms': [], 'skin': [], 'white': []})
with open(r'D:\Final_Year_Project\docs\Review_2\test_supabase\image_analysis_report.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        g = row['group']
        data[g]['tops'].append(float(row['product_top_pct']))
        data[g]['bottoms'].append(float(row['product_bottom_pct']))
        data[g]['skin'].append(float(row['skin_ratio']))
        data[g]['white'].append(float(row['white_ratio']))

print("MINIMUM SAFE CROP BANDS (5th/95th percentile — no product truncation)")
print("=" * 100)
print(f"  {'Group':25s} | {'P5 Top':>7s} | {'P95 Bot':>7s} | {'Safe Top':>8s} | {'Safe Bot':>8s} | {'Skin%':>6s} | {'White%':>6s} | N")
print("-" * 100)
for g in ['large_fabric', 'footwear', 'head_accessories', 'middle_accessories', 'small_jewelry']:
    d = data[g]
    n = len(d['tops'])
    # 5th percentile of top: the point where 95% of products start BELOW
    # This is the safe crop-from-top boundary
    p5_top = np.percentile(d['tops'], 5)
    # 95th percentile of bottom: the point where 95% of products end ABOVE
    p95_bot = np.percentile(d['bottoms'], 95)
    
    # Safe crop = use floor/ceil to be conservative
    safe_top = max(0.0, p5_top - 0.02)  # 2% safety margin
    safe_bot = min(1.0, p95_bot + 0.02)  # 2% safety margin
    
    mean_skin = np.mean(d['skin'])
    mean_white = np.mean(d['white'])
    
    print(f"  {g:25s} | {p5_top:7.3f} | {p95_bot:7.3f} | {safe_top:8.3f} | {safe_bot:8.3f} | {mean_skin:5.1%} | {mean_white:5.1%} | {n}")

print()
print("Interpretation: safe_top = start of crop, safe_bot = end of crop")
print("Effective crop removes [0, safe_top] from top and [safe_bot, 1.0] from bottom")
print("2% safety margin added to protect outlier product regions")
