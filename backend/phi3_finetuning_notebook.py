# PHI-3 MINI FINE-TUNING FOR FASHION FILTER GENERATION
# =====================================================
# Kaggle Notebook - Run with T4 GPU
# Upload llm_training_data.jsonl as dataset first
# Training Time: 2-3 hours

# CELL 1: Install dependencies (run first, restart kernel)
# !pip install -q transformers==4.40.0 datasets peft accelerate bitsandbytes trl

# CELL 2: Imports and Setup
import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("=" * 70)
print("PHI-3 MINI FINE-TUNING FOR FASHION FILTERS")
print("=" * 70)
print(f"Time: {datetime.now()}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Seeds for reproducibility
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
print("\n" + "=" * 70)
print("[STEP 1] LOADING DATASET")
print("=" * 70)

paths = [
    "/kaggle/input/fashion-llm-dataset/llm_training_data.jsonl",
    "llm_training_data.jsonl",
    "../input/fashion-llm-dataset/llm_training_data.jsonl"
]
DATASET_PATH = next((p for p in paths if os.path.exists(p)), paths[0])

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]

print(f"Loaded {len(raw_data)} samples from {DATASET_PATH}")

# Show distribution
types = {}
for s in raw_data:
    t = s.get("type", "unknown")
    types[t] = types.get(t, 0) + 1
print("Sample types:", dict(sorted(types.items(), key=lambda x: -x[1])[:5]))

# Train/Val split (90/10)
random.shuffle(raw_data)
split_idx = int(0.9 * len(raw_data))
train_raw, val_raw = raw_data[:split_idx], raw_data[split_idx:]
print(f"Train: {len(train_raw)}, Validation: {len(val_raw)}")

# CELL 5: Format for Phi-3 Instruction Tuning
print("\n" + "=" * 70)
print("[STEP 2] FORMATTING DATA")
print("=" * 70)

SYS_PROMPT = """You are a fashion filter generation assistant for ShopWhatYouSee.
Given visual attributes, scene, and user query, generate product filters with reasoning.
Output valid JSON with 'reasoning' and 'filters' fields."""

def format_sample(sample):
    inp = sample["input"]
    out = sample["output"]
    
    # User message with context
    user_msg = f"""Category: {inp['category']}
Attributes: {json.dumps(inp['attributes'])}
Scene: {inp['scene']}
Session: {json.dumps(inp.get('session_history', []))}
Query: "{inp['user_query']}"

Generate JSON with reasoning and filters."""
    
    # Assistant response
    assistant_msg = json.dumps(out, indent=2, ensure_ascii=False)
    
    # Phi-3 chat template
    # Using concatenation to avoid special char issues
    sys_tag = "<" + "|system|" + ">"
    end_tag = "<" + "|end|" + ">"
    user_tag = "<" + "|user|" + ">"
    asst_tag = "<" + "|assistant|" + ">"
    
    text = f"{sys_tag}\n{SYS_PROMPT}{end_tag}\n"
    text += f"{user_tag}\n{user_msg}{end_tag}\n"
    text += f"{asst_tag}\n{assistant_msg}{end_tag}"
    
    return {"text": text}

# Convert to HuggingFace datasets
train_formatted = [format_sample(s) for s in train_raw]
val_formatted = [format_sample(s) for s in val_raw]

train_dataset = Dataset.from_list(train_formatted)
val_dataset = Dataset.from_list(val_formatted)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
print("\nExample formatted sample:")
print(train_formatted[0]["text"][:500] + "...")

# CELL 6: Load Model with 4-bit Quantization
print("\n" + "=" * 70)
print("[STEP 3] LOADING MODEL")
print("=" * 70)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (4-bit quantized)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

print(f"Model loaded: {MODEL_NAME}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# CELL 7: Setup LoRA
print("\n" + "=" * 70)
print("[STEP 4] CONFIGURING LoRA")
print("=" * 70)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# CELL 8: Training Arguments
print("\n" + "=" * 70)
print("[STEP 5] SETTING UP TRAINING")
print("=" * 70)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    packing=False,
)

print("Training configuration ready!")

# CELL 9: Train
print("\n" + "=" * 70)
print("[STEP 6] TRAINING")
print("=" * 70)
print(f"Starting training at {datetime.now()}")

trainer.train()

print(f"\nTraining completed at {datetime.now()}")

# CELL 10: Save Model
print("\n" + "=" * 70)
print("[STEP 7] SAVING MODEL")
print("=" * 70)

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Merge and save full model
print("Merging LoRA weights...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-merged")
print(f"Merged model saved to {OUTPUT_DIR}-merged")

# CELL 11: Evaluation
print("\n" + "=" * 70)
print("[STEP 8] EVALUATION")
print("=" * 70)

def evaluate_model(model, tokenizer, test_samples, max_new_tokens=512):
    results = {
        "json_valid": 0,
        "has_reasoning": 0,
        "has_filters": 0,
        "total": len(test_samples)
    }
    
    model.eval()
    
    for i, sample in enumerate(test_samples[:50]):  # Evaluate 50 samples
        inp = sample["input"]
        expected = sample["output"]
        
        # Build prompt
        sys_tag = "<" + "|system|" + ">"
        end_tag = "<" + "|end|" + ">"
        user_tag = "<" + "|user|" + ">"
        asst_tag = "<" + "|assistant|" + ">"
        
        prompt = f"{sys_tag}\n{SYS_PROMPT}{end_tag}\n"
        prompt += f"{user_tag}\nCategory: {inp['category']}\n"
        prompt += f"Attributes: {json.dumps(inp['attributes'])}\n"
        prompt += f"Scene: {inp['scene']}\n"
        prompt += f"Query: \"{inp['user_query']}\"\n"
        prompt += f"Generate JSON with reasoning and filters.{end_tag}\n{asst_tag}\n"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if asst_tag.replace("<", "").replace(">", "") in response:
            response = response.split(asst_tag.replace("<", "").replace(">", ""))[-1].strip()
        
        # Check JSON validity
        try:
            parsed = json.loads(response)
            results["json_valid"] += 1
            if "reasoning" in parsed:
                results["has_reasoning"] += 1
            if "filters" in parsed:
                results["has_filters"] += 1
        except json.JSONDecodeError:
            pass
        
        if i % 10 == 0:
            print(f"Evaluated {i+1}/{min(50, len(test_samples))}")
    
    return results

print("Running evaluation on validation set...")
eval_results = evaluate_model(model, tokenizer, val_raw)

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print(f"JSON Valid:     {eval_results['json_valid']}/{eval_results['total']} ({100*eval_results['json_valid']/min(50, eval_results['total']):.1f}%)")
print(f"Has Reasoning:  {eval_results['has_reasoning']}/{eval_results['total']} ({100*eval_results['has_reasoning']/min(50, eval_results['total']):.1f}%)")
print(f"Has Filters:    {eval_results['has_filters']}/{eval_results['total']} ({100*eval_results['has_filters']/min(50, eval_results['total']):.1f}%)")

# CELL 12: Test Inference
print("\n" + "=" * 70)
print("[STEP 9] TEST INFERENCE")
print("=" * 70)

test_cases = [
    {"category": "shirt", "attributes": {"color": "blue", "pattern": "solid"}, "scene": "office", "user_query": "same but in red"},
    {"category": "tshirt", "attributes": {"color": "black", "sleeve": "short"}, "scene": "mall", "user_query": "full sleeve please"},
    {"category": "shirt", "attributes": {"color": "white", "pattern": "striped"}, "scene": "wedding_hall", "user_query": "something suitable for the occasion"},
]

for i, test in enumerate(test_cases):
    print(f"\n--- Test Case {i+1} ---")
    print(f"Input: {json.dumps(test, indent=2)}")
    
    sys_tag = "<" + "|system|" + ">"
    end_tag = "<" + "|end|" + ">"
    user_tag = "<" + "|user|" + ">"
    asst_tag = "<" + "|assistant|" + ">"
    
    prompt = f"{sys_tag}\n{SYS_PROMPT}{end_tag}\n"
    prompt += f"{user_tag}\nCategory: {test['category']}\n"
    prompt += f"Attributes: {json.dumps(test['attributes'])}\n"
    prompt += f"Scene: {test['scene']}\n"
    prompt += f"Query: \"{test['user_query']}\"\n"
    prompt += f"Generate JSON with reasoning and filters.{end_tag}\n{asst_tag}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output:\n{response[-500:]}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Download the model from: {OUTPUT_DIR}-merged")
print("Copy to backend/models/phi3-fashion-filters/")
