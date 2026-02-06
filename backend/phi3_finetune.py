# PHI-3 MINI FINE-TUNING FOR FASHION FILTER GENERATION
# Run on Kaggle with T4 GPU. Training time: ~2-3 hours
# Upload llm_training_data.jsonl to Kaggle first

# CELL 1: Install (uncomment in Kaggle)
# !pip install -q transformers datasets peft accelerate bitsandbytes trl

import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("PHI-3 FASHION FILTER FINE-TUNING")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# CELL 2: Config
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "/kaggle/input/fashion-llm-dataset/llm_training_data.jsonl"
if not os.path.exists(DATASET_PATH): DATASET_PATH = "llm_training_data.jsonl"

# CELL 3: Load data
with open(DATASET_PATH, 'r') as f:
    raw_data = [json.loads(line) for line in f]
print(f"Loaded {len(raw_data)} samples")

# CELL 4: Format for Phi-3
def format_sample(sample):
    i, o = sample["input"], sample["output"]
    sys = "You are a fashion filter assistant. Output JSON with reasoning and filters."
    usr = f"Category: {i['category']}, Attrs: {json.dumps(i['attributes'])}, Scene: {i['scene']}, Query: {i['user_query']}"
    ast = json.dumps(o)
    return {"text": f"<|system|>\n{sys}<|end|>\n
