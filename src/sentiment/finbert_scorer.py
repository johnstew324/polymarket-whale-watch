# -*- coding: utf-8 -*-
"""
finbert_scorer.py

Loads each parsed ProQuest CSV (ft_{topic}_articles_full.csv),
runs FinBERT sentiment scoring with chunking for long articles,
and saves ft_{topic}_articles_scored.csv.

Usage:
    python src/sentiment/finbert_scorer.py                   # scores all topics
    python src/sentiment/finbert_scorer.py --topic ukraine   # scores one topic

Input:  data/processed/ft_{topic}_articles_full.csv
Output: data/processed/ft_{topic}_articles_scored.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
TOPICS = [
    "ukraine",
    "russia",
    "israel",
    "iran",
    "gaza",
    "lebanon",
    "hezbollah",
    "hamas",
    "yemen",
    "oil",
    "venezuela",
    "trump",
    "putin",
    "khamenei",
    "elections",
    "geopolitics",
    "politics",
    "world",
    "middle-east",
    "mention-markets",
    "foreign-policy",
    "world-affairs",
    "military-action",
    "global-elections",
    "world-elections",
    "trump-presidency",
    "iranian-leadership-regime",
    "israel-x-iran",
    "daily-strikes",
    "ukraine-map",
]

PROCESSED_DIR = "data/processed"
MODEL_NAME    = "ProsusAI/finbert"
MAX_TOKENS    = 450   # leave headroom below BERT's 512 limit
BATCH_SIZE    = 16    # articles logged per progress update

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
def load_model():
    print(f"Loading FinBERT ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  Model ready on {device}.")
    return tokenizer, model, device


# -------------------------------------------------------
# CHUNKING
# Sentence-aware split so we never cut mid-sentence.
# -------------------------------------------------------
def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list:
    sentences = str(text).replace("\n", " ").split(". ")
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        token_len = len(tokenizer.tokenize(sentence))
        if current_len + token_len > max_tokens:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_len   = token_len
        else:
            current_chunk.append(sentence)
            current_len  += token_len

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks


# -------------------------------------------------------
# SCORING
# -------------------------------------------------------
def score_chunk(text: str, tokenizer, model, device) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze().tolist()

    # FinBERT label order: positive=0, negative=1, neutral=2
    return {
        "positive": probs[0],
        "negative": probs[1],
        "neutral":  probs[2],
        "net_score": probs[0] - probs[1]
    }


def score_article(text: str, tokenizer, model, device) -> dict:
    """Score one article by chunking and averaging chunk scores."""
    chunks = chunk_text(text, tokenizer)

    if not chunks:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "net_score": 0.0}

    chunk_scores = [score_chunk(c, tokenizer, model, device) for c in chunks]

    return {
        "positive":  float(np.mean([s["positive"]  for s in chunk_scores])),
        "negative":  float(np.mean([s["negative"]  for s in chunk_scores])),
        "neutral":   float(np.mean([s["neutral"]   for s in chunk_scores])),
        "net_score": float(np.mean([s["net_score"] for s in chunk_scores])),
    }


# -------------------------------------------------------
# MAIN SCORING LOOP
# -------------------------------------------------------
def score_topic(topic: str, tokenizer, model, device):
    in_path  = os.path.join(PROCESSED_DIR, f"ft_{topic}_articles_full.csv")
    out_path = os.path.join(PROCESSED_DIR, f"ft_{topic}_articles_scored.csv")

    if not os.path.exists(in_path):
        print(f"[{topic}] No input file found at '{in_path}' — skipping.")
        return

    # Check if already scored — skip unless re-running intentionally
    if os.path.exists(out_path):
        print(f"[{topic}] Already scored at '{out_path}' — skipping. Delete file to re-run.")
        return

    raw = pd.read_csv(in_path)

    # Keep only useful columns, preserve raw body as body_text
    df = raw[["source_file", "date", "title", "author",
              "publication", "document_type", "proquest_id", "text"]].copy()

    df = df.rename(columns={"text": "body_text"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "body_text"]).reset_index(drop=True)

    # Build scoring text: title + body — used for FinBERT only, not saved as a column
    scoring_text = (df["title"].fillna("") + ". " + df["body_text"].fillna("")).tolist()

    print(f"\n[{topic}] Scoring {len(df):,} articles...")

    results = []
    for text in tqdm(scoring_text, total=len(df), desc=topic):
        results.append(score_article(text, tokenizer, model, device))

    scores_df = pd.DataFrame(results)

    # Final clean output — no duplicate or null columns
    scored = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    scored.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[{topic}] Saved {len(scored):,} scored articles → {out_path}")
    print(f"  Columns: {scored.columns.tolist()}")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FinBERT scorer for ProQuest topic CSVs")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help=f"Topic to score. Options: {TOPICS}. Omit to score all."
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    if args.topic:
        topic = args.topic.lower()
        if topic not in TOPICS:
            print(f"Unknown topic '{topic}'. Available: {TOPICS}")
            return
        score_topic(topic, tokenizer, model, device)
    else:
        print("Scoring all topics...")
        for topic in TOPICS:
            score_topic(topic, tokenizer, model, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()