# PHI-3 MINI FINE-TUNING FOR FASHION FILTER GENERATION
# Kaggle Notebook - Run with T4 GPU
# Upload llm_training_data.jsonl first. Training: 2-3 hours

# CELL 1: Install (uncomment in Kaggle)
# !pip install -q transformers==4.40.0 datasets peft accelerate bitsandbytes trl

# CELL 2: Imports
import os, json, random, numpy as np, torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("="*70)
print("PHI-3 MINI FINE-TUNING FOR FASHION FILTERS")
print("="*70)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# CELL 3: Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./phi3-fashion-filters"
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-5
MAX_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32

# CELL 4: Load Dataset
print("\n[STEP 1] Loading dataset...")
paths = ["/kaggle/input/fashion-llm-dataset/llm_training_data.jsonl", "llm_training_data.jsonl"]
DATASET_PATH = next((p for p in paths if os.path.exists(p)), paths[0])
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]
print(f"Loaded {len(raw_data)} samples from {DATASET_PATH}")

# Train/Val split
random.shuffle(raw_data)
split_idx = int(0.9 * len(raw_data))
train_raw, val_raw = raw_data[:split_idx], raw_data[split_idx:]
print(f"Train: {len(train_raw)}, Validation: {len(val_raw)}")

# CELL 5: Format for Phi-3
print("\n[STEP 2] Formatting data for Phi-3...")

SYS_PROMPT = "You are a fashion filter generation assistant. Given visual attributes and user query, generate JSON with reasoning and filters."

def format_sample(sample):
    inp = sample["input"]
    out = sample["output"]
    user = f"Category: {inp['category']}\nAttributes: {json.dumps(inp['attributes'])}\nScene: {inp['scene']}\nQuery: {inp['user_query']}"
    assistant = json.dumps(out, ensure_ascii=False)
    # Phi-3 chat format with special tokens
    formatted = "<|system|>\n" + SYS_PROMPT + "<|end|>\n"
    formatted += "
