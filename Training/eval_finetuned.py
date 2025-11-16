from __future__ import annotations

# ---- SPEED SETTINGS ----
MAX_EXAMPLES = 3          # evaluate only 3 examples (fast)
MAX_NEW_TOKENS = 80       # shorter answers = MUCH faster
# ------------------------

import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

EVAL_PATH = PROC_DIR / "eval.jsonl"
OUT_PATH = PROC_DIR / "finetuned_predictions.jsonl"

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = ROOT / "models" / "tinyllama-financial-lora"

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_eval() -> List[Dict]:
    rows = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(question: str) -> str:
    system = (
        "You are a helpful financial education assistant. "
        "Answer clearly, accurately, and concisely in 2–3 sentences."
    )
    return f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n"


def main():
    print(f"Loading eval set from {EVAL_PATH} ...")
    eval_rows = load_eval()
    print(f"Loaded {len(eval_rows)} eval examples.")

    # FAST MODE
    eval_rows = eval_rows[:MAX_EXAMPLES]
    print(f"Evaluating finetuned model on {len(eval_rows)} examples (fast mode).")

    print(f"Loading base model: {BASE_MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(DEVICE)

    print(f"Loading LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    predictions = []

    for row in tqdm(
        eval_rows,
        desc="Evaluating finetuned (fast)",
        mininterval=1.0,
    ):
        question = row["question"]
        ref_answer = row["answer"]

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # ---- SUPER FAST GENERATION ----
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                # greedy decoding (fast)
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        # --------------------------------

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract assistant part
        if full_text.startswith(prompt):
            generated_answer = full_text[len(prompt):].strip()
        else:
            split_token = "<|assistant|>"
            if split_token in full_text:
                generated_answer = full_text.split(split_token)[-1].strip()
            else:
                generated_answer = full_text.strip()

        predictions.append(
            {
                "question": question,
                "reference": ref_answer,
                "prediction": generated_answer,
            }
        )

    # Write outputs
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ wrote {len(predictions)} finetuned predictions → {OUT_PATH}")
    print("Compare this with baseline_predictions.jsonl to see improvements.")


if __name__ == "__main__":
    main()