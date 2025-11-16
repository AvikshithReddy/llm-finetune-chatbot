# Training/dataset_prep.py

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
from openai import OpenAI
import random

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

HTML_DIR = RAW_DIR / "html"
PDF_DIR = RAW_DIR / "pdfs"
TXT_DIR = RAW_DIR / "txt"

CLEANED_PATH = PROC_DIR / "cleaned_text.jsonl"
QA_PATH = PROC_DIR / "qa_pairs.jsonl"
TRAIN_PATH = PROC_DIR / "train.jsonl"
EVAL_PATH = PROC_DIR / "eval.jsonl"

PROC_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_html_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # drop scripts/headers/etc
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return clean_whitespace(text)


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return clean_whitespace(" ".join(pages))


def load_txt_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return clean_whitespace(text)


def iter_raw_docs() -> Iterable[dict]:
    """Yield {id, title, source, text} for each raw document."""
    # HTML
    for p in sorted(HTML_DIR.glob("*.html")):
        yield {
            "id": f"html::{p.name}",
            "title": p.stem,
            "source": str(p),
            "text": load_html_file(p),
        }

    # PDFs
    for p in sorted(PDF_DIR.glob("*.pdf")):
        yield {
            "id": f"pdf::{p.name}",
            "title": p.stem,
            "source": str(p),
            "text": load_pdf_file(p),
        }

    # TXT
    for p in sorted(TXT_DIR.glob("*")):
        if p.is_file():
            yield {
                "id": f"txt::{p.name}",
                "title": p.stem,
                "source": str(p),
                "text": load_txt_file(p),
            }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- 1. Build cleaned_text.jsonl ----------
def build_cleaned() -> list[dict]:
    docs = []
    for doc in iter_raw_docs():
        # skip super short docs
        if len(doc["text"]) < 400:
            continue
        docs.append(doc)

    write_jsonl(CLEANED_PATH, docs)
    print(f"✅ wrote {len(docs)} cleaned docs → {CLEANED_PATH}")
    return docs


# ---------- 2. Call OpenAI to make QA pairs ----------
load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a friendly financial education tutor. "
    "You ONLY use information from the provided text. "
    "Your job is to create 4–6 high-quality question–answer pairs that would help a BEGINNER. "
    "Make questions specific (not too broad), and answers clear, in 2–4 short sentences. "
    "Explain with simple examples when helpful. "
    "Return ONLY valid JSON with a top-level key 'qa', where each item has 'question' and 'answer'."
)


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """Very simple character-based chunking."""
    chunks = []
    while text:
        chunk = text[:max_chars]
        text = text[max_chars:]
        chunks.append(chunk)
    return chunks

def generate_qa_for_chunk(doc_id: str, chunk: str) -> list[dict]:
    """Use OpenAI to generate QA pairs for a given text chunk."""
    if not chunk.strip():
        return []

    user_prompt = (
    "Read the text below and create 4–6 question-answer pairs based ONLY on that text.\n"
    "Focus on financial literacy concepts: definitions, why it matters, practical tips, do's and don'ts.\n"
    "Each question should be answerable directly from the text (no outside knowledge).\n"
    "Each answer should be beginner-friendly, 2–4 short sentences, and include a simple example if appropriate.\n\n"
    f"TEXT:\n{chunk}\n\n"
    "Respond ONLY as minified JSON, no commentary, exactly in this shape:\n"
    '{"qa": [{"question": "...", "answer": "..."}, ...]}'
)

    resp = client.responses.create(
        model="gpt-4.1-mini",  # or "gpt-4o-mini"
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    # ---- robust extraction of text ----
    block = resp.output[0].content[0]
    if hasattr(block, "text"):
        # block.text might be a string OR an object with .value
        t = block.text
        content = t.value if hasattr(t, "value") else str(t)
    else:
        content = str(block)
    # -----------------------------------

    # try to parse JSON out of model response
    try:
        data = json.loads(content)
        qa_list = data.get("qa", [])
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            qa_list = data.get("qa", [])
        except Exception:
            return []

    out = []
    for qa in qa_list:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if len(q) < 5 or len(a) < 10:
            continue
        out.append(
            {
                "doc_id": doc_id,
                "question": q,
                "answer": a,
            }
        )
    return out

def build_qa_pairs(cleaned_docs: list[dict]) -> list[dict]:
    all_pairs: list[dict] = []

    for doc in tqdm(cleaned_docs, desc="Generating QA pairs"):
        chunks = chunk_text(doc["text"])
        for ch in chunks[:3]:  # limit per doc so we don't explode tokens
            pairs = generate_qa_for_chunk(doc["id"], ch)
            all_pairs.extend(pairs)

    write_jsonl(QA_PATH, all_pairs)
    print(f"✅ wrote {len(all_pairs)} QA pairs → {QA_PATH}")
    return all_pairs


# ---------- 3. Split into train / eval ----------
def split_train_eval(pairs: list[dict], eval_ratio: float = 0.2):
    random.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * eval_ratio))
    eval_pairs = pairs[:n_eval]
    train_pairs = pairs[n_eval:]

    train_rows = [
        {
            "messages": [
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": p["answer"]},
            ]
        }
        for p in train_pairs
    ]

    eval_rows = [
        {
            "question": p["question"],
            "answer": p["answer"],
        }
        for p in eval_pairs
    ]

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)
    print(f"✅ train samples: {len(train_rows)} → {TRAIN_PATH}")
    print(f"✅ eval samples:  {len(eval_rows)} → {EVAL_PATH}")


# ---------- main ----------
def main():
    print("Step 1: building cleaned_text.jsonl …")
    cleaned_docs = build_cleaned()

    print("\nStep 2: generating QA pairs with OpenAI …")
    qa_pairs = build_qa_pairs(cleaned_docs)

    print("\nStep 3: splitting into train/eval …")
    split_train_eval(qa_pairs)

    print("\n🎉 Dataset prep complete.")


if __name__ == "__main__":
    main()