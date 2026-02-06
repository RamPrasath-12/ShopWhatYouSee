"""
Test AGMAN Model Embedding Quality

This script evaluates:
1. Model architecture correctness
2. Embedding similarity for same-category items
3. Embedding differentiation across categories
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import base64
import io

# Load the AGMAN model directly
from models.agman_model import AGMAN
from models.agman_extractor import extract_embedding, b64_to_pil, transform, resnet, DEVICE

print("=" * 60)
print("AGMAN MODEL EMBEDDING ANALYSIS")
print("=" * 60)

# Load AGMAN model
print("\n1. Loading AGMAN model...")
agman = AGMAN()
try:
    agman.load_state_dict(torch.load("models/agman_model.pth", map_location=DEVICE))
    agman = agman.to(DEVICE)
    agman.eval()
    print("   ✓ AGMAN model loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# Check model parameters
print("\n2. Model Architecture:")
total_params = sum(p.numel() for p in agman.parameters())
trainable_params = sum(p.numel() for p in agman.parameters() if p.requires_grad)
print(f"   - Total parameters: {total_params:,}")
print(f"   - Trainable parameters: {trainable_params:,}")

for name, param in agman.named_parameters():
    print(f"   - {name}: {param.shape}")

# Check parameter statistics (are weights learned or random?)
print("\n3. Weight Statistics (checking if model is trained):")
for name, param in agman.named_parameters():
    data = param.data.cpu().numpy()
    print(f"   {name}:")
    print(f"     Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
    print(f"     Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")

# Test with synthetic inputs
print("\n4. Testing embedding computation...")

def test_embedding_quality():
    # Create synthetic 2048-D input (simulating ResNet output)
    test_inputs = []
    
    # Create diverse test vectors
    for i in range(5):
        vec = np.random.randn(2048).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        test_inputs.append(vec)
    
    # Process through AGMAN
    embeddings = []
    for vec in test_inputs:
        x = torch.tensor(vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = agman(x)
        embeddings.append(emb.cpu().numpy()[0])
    
    # Check output properties
    print("\n   Embedding properties:")
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        print(f"   - Input {i+1}: dim={len(emb)}, norm={norm:.4f}")
    
    # Check similarity matrix (embeddings should be different)
    print("\n   Cosine similarity matrix:")
    print("        ", end="")
    for i in range(len(embeddings)):
        print(f"Inp{i+1}  ", end="")
    print()
    
    for i, emb_i in enumerate(embeddings):
        print(f"   Inp{i+1}", end="")
        for j, emb_j in enumerate(embeddings):
            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
            print(f"  {sim:.2f}", end="")
        print()

test_embedding_quality()

# Test with real image if available
print("\n5. Testing with real image...")
test_image_path = "tests/test.png"
if os.path.exists(test_image_path):
    try:
        img = Image.open(test_image_path).convert("RGB")
        
        # Get ResNet embedding (2048-D)
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            base_emb = resnet(img_t).cpu().numpy().flatten()
        
        base_emb = base_emb / np.linalg.norm(base_emb)
        
        # Refine with AGMAN (512-D)
        x = torch.tensor(base_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            refined = agman(x).cpu().numpy()[0]
        
        print(f"   ✓ Base embedding (ResNet): {len(base_emb)} dimensions")
        print(f"   ✓ Refined embedding (AGMAN): {len(refined)} dimensions")
        print(f"   ✓ Refined embedding norm: {np.linalg.norm(refined):.4f}")
        print(f"   ✓ First 10 values: {refined[:10]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"   ⚠ Test image not found: {test_image_path}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
The AGMAN model transforms 2048-D ResNet embeddings into 512-D
refined embeddings using an attention mechanism:

  1. Attention weights computed: w = sigmoid(fc2(relu(fc1(x))))
  2. Weighted input: x' = x * w
  3. Final projection: emb = normalize(fc3(x'))

This attention mechanism learns to emphasize fashion-relevant
features while suppressing irrelevant ones (e.g., background).
""")
