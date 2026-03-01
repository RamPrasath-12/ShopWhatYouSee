"""
AGMAN Model Comparison Script
Compares two AGMAN models for embedding quality and product similarity
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Paths
MODEL_DIR = Path("models")
MODEL_1_PATH = MODEL_DIR / "agman_model_best.pth"  # Your model
MODEL_2_PATH = MODEL_DIR / "agman_best_final.pth"  # Friend's model

print("=" * 70)
print("AGMAN MODEL COMPARISON")
print("=" * 70)

# ============================================================================
# STEP 1: Check model file sizes and basic info
# ============================================================================
print("\n[1] MODEL FILE INFO")
print("-" * 50)
print(f"Model 1 (agman_model_best.pth): {MODEL_1_PATH.stat().st_size / 1024 / 1024:.2f} MB")
print(f"Model 2 (agman_best_final.pth): {MODEL_2_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# ============================================================================
# STEP 2: Load and analyze model structures
# ============================================================================
print("\n[2] MODEL STRUCTURE ANALYSIS")
print("-" * 50)

# Load checkpoints
print("Loading Model 1...")
checkpoint1 = torch.load(MODEL_1_PATH, map_location='cpu')
print("Loading Model 2...")
checkpoint2 = torch.load(MODEL_2_PATH, map_location='cpu')

# Check what's in each checkpoint
print(f"\nModel 1 checkpoint keys: {list(checkpoint1.keys()) if isinstance(checkpoint1, dict) else 'state_dict only'}")
print(f"Model 2 checkpoint keys: {list(checkpoint2.keys()) if isinstance(checkpoint2, dict) else 'state_dict only'}")

# Get state dicts
if isinstance(checkpoint1, dict) and 'model_state_dict' in checkpoint1:
    state_dict1 = checkpoint1['model_state_dict']
    epoch1 = checkpoint1.get('epoch', 'N/A')
    loss1 = checkpoint1.get('loss', checkpoint1.get('val_loss', 'N/A'))
else:
    state_dict1 = checkpoint1 if isinstance(checkpoint1, dict) else checkpoint1
    epoch1 = 'N/A'
    loss1 = 'N/A'

if isinstance(checkpoint2, dict) and 'model_state_dict' in checkpoint2:
    state_dict2 = checkpoint2['model_state_dict']
    epoch2 = checkpoint2.get('epoch', 'N/A')
    loss2 = checkpoint2.get('loss', checkpoint2.get('val_loss', 'N/A'))
else:
    state_dict2 = checkpoint2 if isinstance(checkpoint2, dict) else checkpoint2
    epoch2 = 'N/A'
    loss2 = 'N/A'

print(f"\nModel 1 - Epoch: {epoch1}, Loss: {loss1}")
print(f"Model 2 - Epoch: {epoch2}, Loss: {loss2}")

# Count parameters
def count_params(state_dict):
    total = sum(p.numel() for p in state_dict.values())
    return total

params1 = count_params(state_dict1)
params2 = count_params(state_dict2)
print(f"\nModel 1 parameters: {params1:,}")
print(f"Model 2 parameters: {params2:,}")

# ============================================================================
# STEP 3: Load actual models and test embedding generation
# ============================================================================
print("\n[3] EMBEDDING QUALITY TEST")
print("-" * 50)

try:
    from models.agman_extractor import AGMANExtractor
    
    # Create two extractors
    print("Creating extractors...")
    extractor1 = AGMANExtractor()
    extractor2 = AGMANExtractor()
    
    # Load respective weights
    print("Loading Model 1 weights...")
    extractor1.load_state_dict(state_dict1)
    extractor1.eval()
    
    print("Loading Model 2 weights...")
    extractor2.load_state_dict(state_dict2)
    extractor2.eval()
    
    # Create a test input (224x224 RGB image tensor)
    print("\nGenerating test embeddings...")
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Get embeddings from both models
        emb1 = extractor1(test_input)
        emb2 = extractor2(test_input)
    
    print(f"Model 1 embedding shape: {emb1.shape}")
    print(f"Model 2 embedding shape: {emb2.shape}")
    
    # Check embedding statistics
    print(f"\nModel 1 embedding stats: mean={emb1.mean():.4f}, std={emb1.std():.4f}, min={emb1.min():.4f}, max={emb1.max():.4f}")
    print(f"Model 2 embedding stats: mean={emb2.mean():.4f}, std={emb2.std():.4f}, min={emb2.min():.4f}, max={emb2.max():.4f}")
    
    # Check for NaN or Inf
    print(f"\nModel 1 has NaN: {torch.isnan(emb1).any()}, has Inf: {torch.isinf(emb1).any()}")
    print(f"Model 2 has NaN: {torch.isnan(emb2).any()}, has Inf: {torch.isinf(emb2).any()}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("\nTrying alternative loading method...")

# ============================================================================
# STEP 4: Test with multiple random inputs for consistency
# ============================================================================
print("\n[4] EMBEDDING CONSISTENCY TEST")
print("-" * 50)

try:
    # Generate multiple test inputs
    test_inputs = [torch.randn(1, 3, 224, 224) for _ in range(5)]
    
    embeddings1 = []
    embeddings2 = []
    
    with torch.no_grad():
        for inp in test_inputs:
            embeddings1.append(extractor1(inp))
            embeddings2.append(extractor2(inp))
    
    # Check variance across embeddings (should be diverse but consistent)
    emb_stack1 = torch.cat(embeddings1, dim=0)
    emb_stack2 = torch.cat(embeddings2, dim=0)
    
    # Calculate cosine similarities between embeddings
    from torch.nn.functional import cosine_similarity
    
    # Average similarity of different inputs (should be moderate - not too high or too low)
    sims1 = []
    sims2 = []
    for i in range(len(embeddings1)):
        for j in range(i+1, len(embeddings1)):
            sims1.append(cosine_similarity(embeddings1[i], embeddings1[j], dim=1).item())
            sims2.append(cosine_similarity(embeddings2[i], embeddings2[j], dim=1).item())
    
    avg_sim1 = np.mean(sims1)
    avg_sim2 = np.mean(sims2)
    
    print(f"Model 1 - Avg similarity between random inputs: {avg_sim1:.4f}")
    print(f"Model 2 - Avg similarity between random inputs: {avg_sim2:.4f}")
    print("(Good models should have moderate similarity ~0.3-0.7, not too high or too low)")
    
except Exception as e:
    print(f"Consistency test error: {e}")

# ============================================================================
# STEP 5: Determinism test (same input should give same output)
# ============================================================================
print("\n[5] DETERMINISM TEST")
print("-" * 50)

try:
    torch.manual_seed(42)
    test_img = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out1_a = extractor1(test_img)
        out1_b = extractor1(test_img)
        out2_a = extractor2(test_img)
        out2_b = extractor2(test_img)
    
    diff1 = (out1_a - out1_b).abs().max().item()
    diff2 = (out2_a - out2_b).abs().max().item()
    
    print(f"Model 1 - Same input difference: {diff1:.10f} {'✅' if diff1 == 0 else '⚠️'}")
    print(f"Model 2 - Same input difference: {diff2:.10f} {'✅' if diff2 == 0 else '⚠️'}")
    
except Exception as e:
    print(f"Determinism test error: {e}")

# ============================================================================
# STEP 6: Summary and Recommendation
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATION")
print("=" * 70)

print(f"""
Model 1 (agman_model_best.pth):
  - Size: {MODEL_1_PATH.stat().st_size / 1024 / 1024:.2f} MB
  - Parameters: {params1:,}
  - Training Epoch: {epoch1}
  - Loss: {loss1}

Model 2 (agman_best_final.pth):
  - Size: {MODEL_2_PATH.stat().st_size / 1024 / 1024:.2f} MB  
  - Parameters: {params2:,}
  - Training Epoch: {epoch2}
  - Loss: {loss2}
""")

print("\nRun this script to see full comparison results!")
