# Training/train_lora.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROC_DIR / "train.jsonl"

OUT_DIR = ROOT / "models" / "tinyllama-financial-lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model / device ----------
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ---------- Dataset ----------
class QADataset(Dataset):
    """
    Reads train.jsonl and turns each Q/A pair into a single
    chat-style training string.
    """

    def __init__(self, path: Path, tokenizer, max_length: int = 512):
        self.examples: List[Dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                if len(msgs) != 2:
                    continue
                user_msg = msgs[0].get("content", "").strip()
                assistant_msg = msgs[1].get("content", "").strip()
                if not user_msg or not assistant_msg:
                    continue

                text = self.build_prompt(user_msg, assistant_msg)
                self.examples.append({"text": text})

        print(f"Loaded {len(self.examples)} training examples from {path}")

    @staticmethod
    def build_prompt(question: str, answer: str) -> str:
        system = (
            "You are a helpful financial education assistant. "
            "Explain concepts clearly and concisely for beginners."
        )
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{question}\n"
            f"<|assistant|>\n{answer}"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        text = self.examples[idx]["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        # For causal LM, labels == input_ids
        enc["labels"] = enc["input_ids"].copy()
        return enc


# ---------- Main ----------
def main():
    # 1) Load tokenizer & base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # make sure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
        device_map=None,
    ).to(DEVICE)

    # 2) Wrap with LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3) Build dataset
    dataset = QADataset(TRAIN_PATH, tokenizer, max_length=512)

    # 4) Data collator (handles padding & label shifting)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) TrainingArguments
    training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=2,               # more epochs = stronger fine-tune
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="epoch",            # <-- supported
    save_total_limit=2,
    warmup_ratio=0.05,
    fp16=False,                       # mps does NOT support fp16
    bf16=False,                       # mps does NOT support bf16
    remove_unused_columns=False,
)




    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 7) Train
    print("Starting LoRA fine-tuning ...")
    trainer.train()

    # 8) Save the adapter + tokenizer
    print(f"Saving LoRA adapter and tokenizer to {OUT_DIR} ...")
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print("🎉 Training complete.")


if __name__ == "__main__":
    main()
    