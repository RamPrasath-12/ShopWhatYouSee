"""
AGMAN Model Deep Comparison - Tests embeddings on real images
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

# ============================================================================
# Model paths
# ============================================================================
MODEL_DIR = Path("models")
MODEL_1_PATH = MODEL_DIR / "agman_model_best.pth"  # Your model
MODEL_2_PATH = MODEL_DIR / "agman_best_final.pth"  # Friend's model

print("=" * 70)
print("AGMAN MODEL DEEP COMPARISON")
print("=" * 70)

# ============================================================================
# Load and inspect checkpoints
# ============================================================================
print("\n[1] LOADING CHECKPOINTS")
print("-" * 50)

checkpoint1 = torch.load(MODEL_1_PATH, map_location='cpu', weights_only=False)
checkpoint2 = torch.load(MODEL_2_PATH, map_location='cpu', weights_only=False)

# Analyze checkpoint structure
def analyze_checkpoint(ckpt, name):
    print(f"\n{name}:")
    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            if key == 'model_state_dict':
                print(f"  - {key}: {len(ckpt[key])} layers")
            elif key == 'optimizer_state_dict':
                print(f"  - {key}: present")
            elif isinstance(ckpt[key], (int, float, str)):
                print(f"  - {key}: {ckpt[key]}")
            else:
                print(f"  - {key}: {type(ckpt[key]).__name__}")
    else:
        print(f"  - Direct state_dict with {len(ckpt)} layers")

analyze_checkpoint(checkpoint1, "Model 1 (agman_model_best.pth)")
analyze_checkpoint(checkpoint2, "Model 2 (agman_best_final.pth)")

# Get state dicts
def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            return ckpt['state_dict']
    return ckpt

state_dict1 = get_state_dict(checkpoint1)
state_dict2 = get_state_dict(checkpoint2)

# ============================================================================
# Analyze model architecture from state dict
# ============================================================================
print("\n[2] ARCHITECTURE ANALYSIS")
print("-" * 50)

def analyze_architecture(state_dict, name):
    # Find unique layer prefixes
    prefixes = set()
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefixes.add(parts[0])
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    
    # Check for specific heads
    has_color_head = any('color' in k for k in state_dict.keys())
    has_pattern_head = any('pattern' in k for k in state_dict.keys())
    has_sleeve_head = any('sleeve' in k for k in state_dict.keys())
    has_embedding = any('fc' in k or 'embedding' in k for k in state_dict.keys())
    
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Top-level modules: {sorted(prefixes)}")
    print(f"  Has color head: {has_color_head}")
    print(f"  Has pattern head: {has_pattern_head}")
    print(f"  Has sleeve head: {has_sleeve_head}")
    print(f"  Has embedding layer: {has_embedding}")
    
    return total_params, prefixes

params1, modules1 = analyze_architecture(state_dict1, "Model 1")
params2, modules2 = analyze_architecture(state_dict2, "Model 2")

# ============================================================================
# Check layer dimensions for embedding quality
# ============================================================================
print("\n[3] EMBEDDING LAYER ANALYSIS")
print("-" * 50)

def find_embedding_layer(state_dict, name):
    # Look for final fully connected layer that would produce embeddings
    embedding_keys = [k for k in state_dict.keys() if 'fc' in k.lower() or 'embed' in k.lower()]
    
    print(f"\n{name} embedding-related layers:")
    for key in embedding_keys:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
    
    # Also check attention layers
    attn_keys = [k for k in state_dict.keys() if 'attn' in k.lower()]
    if attn_keys:
        print(f"\n{name} attention layers:")
        for key in attn_keys[:5]:  # Show first 5
            print(f"  {key}: {state_dict[key].shape}")

find_embedding_layer(state_dict1, "Model 1")
find_embedding_layer(state_dict2, "Model 2")

# ============================================================================
# Key layer comparison
# ============================================================================
print("\n[4] KEY LAYER COMPARISON")
print("-" * 50)

# Sample a few key layers to compare
def sample_layer_stats(state_dict, layer_name):
    if layer_name in state_dict:
        tensor = state_dict[layer_name]
        return {
            'shape': tensor.shape,
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item()
        }
    return None

# Find common layers
common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
unique_to_1 = set(state_dict1.keys()) - set(state_dict2.keys())
unique_to_2 = set(state_dict2.keys()) - set(state_dict1.keys())

print(f"\nCommon layers: {len(common_keys)}")
print(f"Unique to Model 1: {len(unique_to_1)}")
print(f"Unique to Model 2: {len(unique_to_2)}")

if unique_to_1:
    print(f"\nSample unique to Model 1: {list(unique_to_1)[:5]}")
if unique_to_2:
    print(f"\nSample unique to Model 2: {list(unique_to_2)[:5]}")

# ============================================================================
# Check training metadata if available
# ============================================================================
print("\n[5] TRAINING METADATA")
print("-" * 50)

def get_training_info(ckpt, name):
    if not isinstance(ckpt, dict):
        print(f"{name}: No training metadata (raw state_dict)")
        return
    
    epoch = ckpt.get('epoch', 'N/A')
    loss = ckpt.get('loss', ckpt.get('val_loss', ckpt.get('best_loss', 'N/A')))
    acc = ckpt.get('accuracy', ckpt.get('val_accuracy', ckpt.get('best_acc', 'N/A')))
    
    print(f"\n{name}:")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss}")
    print(f"  Accuracy: {acc}")

get_training_info(checkpoint1, "Model 1")
get_training_info(checkpoint2, "Model 2")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ Model 1 (agman_model_best.pth) - YOUR MODEL                        │
├─────────────────────────────────────────────────────────────────────┤
│ Size: {MODEL_1_PATH.stat().st_size / 1024 / 1024:.2f} MB                                                  │
│ Parameters: {params1:,}                                          │
│ Modules: {', '.join(sorted(modules1)[:5])}                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Model 2 (agman_best_final.pth) - FRIEND'S MODEL                    │
├─────────────────────────────────────────────────────────────────────┤
│ Size: {MODEL_2_PATH.stat().st_size / 1024 / 1024:.2f} MB                                                  │
│ Parameters: {params2:,}                                          │
│ Modules: {', '.join(sorted(modules2)[:5])}                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# Make recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if params2 > params1 * 1.5:
    print("""
Model 2 (Friend's model) has significantly MORE parameters (~2x).
This typically means:
  ✅ More capacity to learn complex patterns
  ✅ Better feature representation
  ⚠️ Slower inference
  ⚠️ More memory usage

For EMBEDDING QUALITY, larger models usually produce BETTER embeddings
if properly trained.
""")
else:
    print("Both models have similar parameter counts.")

print("To make a FINAL DECISION, we need to test both on actual product images.")
print("Run the next test script to compare embeddings on real products.")
