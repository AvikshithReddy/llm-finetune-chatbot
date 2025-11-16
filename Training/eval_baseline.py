# Training/eval_baseline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

EVAL_PATH = PROC_DIR / "eval.jsonl"
OUT_PATH = PROC_DIR / "baseline_predictions.jsonl"

# 🔁 Use a much smaller model so your Mac can handle it quickly
# You can later switch back to: "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Device selection: prefer mps, else cuda, else cpu
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
    """
    Simple chat-style prompt; TinyLlama is instruction-tuned, so this is fine.
    """
    system = (
        "You are a helpful financial education assistant. "
        "Answer clearly and concisely in 3–5 sentences. "
        "Focus on basic personal finance concepts."
    )
    return f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n"


def main():
    print(f"Loading eval set from {EVAL_PATH} ...")
    eval_rows = load_eval()
    print(f"Loaded {len(eval_rows)} eval examples.")

    # ⚠️ Limit to a few examples for speed (you can increase later)
    eval_rows = eval_rows[:5]
    print(f"Evaluating on {len(eval_rows)} examples for this quick baseline run.")

    print(f"Loading model {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
        device_map=None,
    ).to(DEVICE)

    predictions = []

    for row in tqdm(eval_rows, desc="Evaluating baseline"):
        question = row["question"]
        ref_answer = row["answer"]

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,   # deterministic
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Take only the part after the prompt
        if full_text.startswith(prompt):
            generated_answer = full_text[len(prompt):].strip()
        else:
            # Fallback: take last part after assistant tag if present
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

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ wrote {len(predictions)} baseline predictions → {OUT_PATH}")
    print("You can open this file to qualitatively compare baseline vs your ground truth.")


if __name__ == "__main__":
    main()