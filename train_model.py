import os
import torch
import gc
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===========================
# SETUP
# ===========================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ===========================
# LOAD MODEL
# ===========================
print("Loading Mistral-7B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True
)

print("Model loaded!")

# ===========================
# ADD LORA
# ===========================
print("Adding LoRA...")
 
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===========================
# LOAD DATASET
# ===========================
print("Loading dataset...")

SYSTEM_PROMPT = "You are an expert personal trainer and nutritionist. Give specific, practical, science-based advice."

examples = []
with open("gym_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["input"]},
                {"role": "assistant", "content": item["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            examples.append({"text": text})
        except:
            continue

print(f"Loaded {len(examples)} examples!")

split = int(len(examples) * 0.9)
train_data = Dataset.from_list(examples[:split])
test_data  = Dataset.from_list(examples[split:])
print(f"Train: {len(train_data)} | Test: {len(test_data)}")

# ===========================
# TRAIN
# ===========================
print("Training on RTX 5070 Ti...")

torch.cuda.empty_cache()
gc.collect()

training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,          # bfloat16 — RTX 5070 Ti supports it, faster + more stable
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_steps=50,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_data,
    args=training_args,
)

trainer.train()

# ===========================
# SAVE
# ===========================
print("Saving model...")
model.save_pretrained("gym_ai_model")
tokenizer.save_pretrained("gym_ai_model")
print("Saved to gym_ai_model/")

# ===========================
# TEST
# ===========================
print("Testing...")

torch.cuda.empty_cache()

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "I am a 22 year old male, 75kg, 178cm, intermediate bodybuilder with full gym access. How do I improve my bench press?"},
]

input_ids = tokenizer.apply_chat_template(
    test_messages,
    return_tensors="pt",
    add_generation_prompt=True
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

print("\n" + "="*50)
print("TEST OUTPUT:")
print("="*50)
print(response)
print("Fine-tuning complete!")
