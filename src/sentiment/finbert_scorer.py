# pre-scores all ProQuest articles using FinBERT

# uses homework 3 logic
# run this once overnight — output CSVs are then used by collect_ft.py
# which just does fast pandas joins, no FinBERT in the market loop
#

# to score a single topic:
# python -m src.sentiment.finbert_scorer

import os
from pathlib import Path

# fix OpenMP duplicate runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


TOPICS = ["russia_ukraine"]  # test

PQ_DIR     = Path("data/processed/proquest")
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 512


# load model once 
print(f"Loading FinBERT ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Ready on {device}\n")


# chunking logic
def get_token_chunks(text, max_len = MAX_LEN):
    ids  = tokenizer.encode(str(text), add_special_tokens=False)
    step = max_len - 2
    return [ids[i:i + step] for i in range(0, len(ids), step)]


# scoring logic
def score_article(text):
    chunks = get_token_chunks(text)

    if not chunks:
        return {"positive": 0.33, 
                "negative": 0.33, 
                "neutral": 0.34, 
                "net_score": 0.0}

    probs_list = []

    for ids in chunks:
        input_ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # pad to MAX_LEN
        pad_id = tokenizer.pad_token_id or 0
        if len(input_ids) < MAX_LEN:
            pad_len  = MAX_LEN - len(input_ids)
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        input_ids      = torch.tensor([input_ids],      device=device)
        attention_mask = torch.tensor([attention_mask], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            probs_list.append(probs.cpu().numpy()[0])

    avg = np.mean(probs_list, axis=0)

    return {
        "positive": float(avg[0]),
        "negative": float(avg[1]),
        "neutral": float(avg[2]),
        "net_score": float(avg[0] - avg[1]),  # positive - negative
    }


# main loop
total_scored = 0

for topic in TOPICS:
    in_path  = PQ_DIR / f"{topic}_articles.csv"
    out_path = PQ_DIR / f"{topic}_articles_scored.csv"

    if not in_path.exists():
        print(f"[{topic}] no input file found — skipping")
        continue

    # skip if already scored - delete file to re-run
    if out_path.exists():
        print(f"[{topic}] already scored — skipping (delete file to re-run)")
        continue

    df = pd.read_csv(in_path)
    df = df.dropna(subset=["text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # scoring text = title + body
    scoring_texts = (
        df["title"].fillna("") + ". " + df["text"].fillna("")
    ).tolist()

    print(f"[{topic}] scoring {len(df):,} articles...")

    pos_list  = []
    neg_list  = []
    neu_list  = []
    net_list  = []

    for text in tqdm(scoring_texts, desc=topic):
        result = score_article(text)
        pos_list.append(result["positive"])
        neg_list.append(result["negative"])
        neu_list.append(result["neutral"])
        net_list.append(result["net_score"])

    df["positive"]  = pos_list
    df["negative"]  = neg_list
    df["neutral"]   = neu_list
    df["net_score"] = net_list

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[{topic}] saved {len(df):,} scored articles → {out_path}\n")
    total_scored += len(df)

print(f"Done: {total_scored:,} total articles scored across {len(TOPICS)} topics")