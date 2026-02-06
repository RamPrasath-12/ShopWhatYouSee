# PHI-3 MINI FINE-TUNING FOR FASHION FILTER GENERATION
# =====================================================
# Kaggle Notebook - Run with T4 GPU
# Upload llm_training_data.jsonl to Kaggle as dataset first
# Training Time: ~2-3 hours

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES (Run first, restart kernel if needed)
# ============================================================================
# !pip install -q transformers==4.40.0 datasets peft accelerate bitsandbytes trl

# ============================================================================
# CELL 2: IMPORTS
# ============================================================================
import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("=" * 70)
print("PHI-3 MINI FINE-TUNING FOR FASHION FILTER GENERATION")
print("=" * 70)
print(f"Time: {datetime.now()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# CELL 3: CONFIGURATION
# ============================================================================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./phi3-fashion-filters"

# Training params
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-5
MAX_LEN = 1024

# LoRA params
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

print(f"\nConfig: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}x{GRAD_ACCUM}, lr={LR}")

# ============================================================================
# CELL 4: LOAD DATASET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

DATASET_PATH = "/kaggle/input/fashion-llm-dataset/llm_training_data.jsonl"
paths = [DATASET_PATH, "llm_training_data.jsonl", "../input/fashion-llm-dataset/llm_training_data.jsonl"]
for p in paths:
    if os.path.exists(p):
        DATASET_PATH = p
        break

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]
print(f"Loaded {len(raw_data)} samples from {DATASET_PATH}")

# Distribution
types = {}
for s in raw_data:
    t = s.get("type", "unknown")
    types[t] = types.get(t, 0) + 1
print("Distribution:", dict(sorted(types.items(), key=lambda x: -x[1])[:5]), "...")

# ============================================================================
# CELL 5: FORMAT FOR PHI-3
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: FORMATTING DATA")
print("=" * 70)

SYSTEM_PROMPT = """You are a fashion filter generation assistant for ShopWhatYouSee.
Given detected visual attributes, scene context, and user query, generate product filters with reasoning.
Always output valid JSON with 'reasoning' and 'filters' fields.
The reasoning must explain why each filter was chosen or changed."""

def format_sample(sample):
    inp = sample["input"]
    out = sample["output"]
    
    # Build user message
    user_parts = [
        f"Category: {inp['category']}",
        f"Attributes: {json.dumps(inp['attributes'])}",
        f"Scene: {inp['scene']}",
    ]
    if inp.get('session_history'):
        user_parts.append(f"Session History: {json.dumps(inp['session_history'])}")
    user_parts.append(f"User Query: \"{inp['user_query']}\"")
    user_parts.append("\nGenerate JSON with reasoning and filters.")
    user_msg = "\n".join(user_parts)
    
    # Assistant response (the target output)
    assistant_msg = json.dumps(out, indent=2)
    
    # Phi-3 format
    text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n
