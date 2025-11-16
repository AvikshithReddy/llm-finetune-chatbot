# Training/chat_finetuned.py

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

MAX_NEW_TOKENS = 90  # shorter answers → faster

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models" / "tinyllama-financial-lora"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Pick device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def build_prompt(history, new_user_msg: str) -> str:
    system = (
    "You are a friendly financial literacy tutor for beginners. "
    "You ONLY answer questions about personal finance topics such as saving, budgeting, debt, credit scores, loans, interest, and fraud prevention. "
    "If a question is outside these topics or the answer is unclear, say you don't know and suggest a safer question. "
    "When you do answer, use simple language, 2–4 short sentences, and include 1 small practical example."
)
    text = f"<|system|>\n{system}\n"
    for turn in history:
        text += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
    text += f"<|user|>\n{new_user_msg}\n<|assistant|>\n"
    return text


def main():
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

    history = []
    print("💬 Finetuned Financial Tutor — type 'exit' to quit.\n")

    while True:
        user_msg = input("You: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Bye! 👋")
            break

        prompt = build_prompt(history, user_msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        print("...generating answer...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy = faster & more stable
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "<|assistant|>" in full_text:
            answer = full_text.split("<|assistant|>")[-1].strip()
        else:
            answer = full_text.strip()

        print(f"\nBot: {answer}\n")
        history.append({"user": user_msg, "assistant": answer})
        



if __name__ == "__main__":
    main()