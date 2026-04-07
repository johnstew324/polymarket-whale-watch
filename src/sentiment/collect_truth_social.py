# Matches Trump Truth Social posts to Polymarket markets using tag-base  keyword matching,
# runs FinBERT sentiment scoring, and aggregates toweekly scores per market.
#
# Output: data/processed/sentiment/truth_social_sentiment.csv
# Schema: condition_id, source, week_start, sentiment_score, sentiment_direction, post_count
import pandas as pd
import re
import pandas as pdp
import numpy as np
import torch
import torch.nn.functional as F
import duckdb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from truth_social_keywords import TAG_KEYWORDS

#  config and constants
ARCHIVE_PATH = Path("data/raw/truth_social_archive.csv")
OUT_PATH     = Path("data/processed/truth_social_sentiment.csv")
DB_PATH      = Path("data/analytical/polymarket.ddb")
MODEL_NAME   = "ProsusAI/finbert"
MIN_TEXT_LEN = 30
DIRECTION_THRESHOLD = 0.05

# load and clean archive
def clean_post(text: str) -> str:
    text = str(text)
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"RT @\w+:?\s*", "", text)      # remove retweets
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_archive():
    print(f"Loading Truth Social archive from {ARCHIVE_PATH}")
    df = pd.read_csv(ARCHIVE_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["text"] = df["content"].fillna("").apply(clean_post)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN]
    df = df[~df["content"].fillna("").str.startswith("RT @")]
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} posts after cleaning")
    print(f"  Date range: {df['created_at'].min().date()} → {df['created_at'].max().date()}")
    return df


#  load markets from DuckDB 
def load_markets():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    markets = con.execute("""
        SELECT conditionId, question, startDate, endDate, tags, resolvedOutcome
        FROM markets
        WHERE resolvedOutcome IS NOT NULL
          AND startDate IS NOT NULL
          AND endDate IS NOT NULL
        LIMIT 20
    """).fetchdf() # ADDED LIMIT FOR TESTING
    con.close()
    
    # normalise to UTC once
    markets["startDate"] = pd.to_datetime(markets["startDate"], utc=True)
    markets["endDate"]   = pd.to_datetime(markets["endDate"],   utc=True)
    
    print(f"Loaded {len(markets):,} resolved markets from DuckDB")
    return markets


#   build keyword pattern per market from tags 
def get_keywords_for_market(tags: list) -> list:
    keywords = []
    for tag in tags:
        if tag in TAG_KEYWORDS:
            keywords.extend(TAG_KEYWORDS[tag])
    return list(set(keywords))  # deduplicate


# FinBERT scoring 
def load_finbert():
    print(f"\nLoading FinBERT ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("  Model ready.")
    return tokenizer, model


def score_posts(texts: list, tokenizer, model, batch_size: int = 16) -> list:
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        batch_scores = (probs[:, 0] - probs[:, 1]).tolist()
        scores.extend(batch_scores)
    return scores



# aggregate to weekly scores
def aggregate_weekly(scored_posts: pd.DataFrame, condition_id: str) -> list:
    if scored_posts.empty:
        return []

    rows = []
    scored_posts = scored_posts.copy()
    scored_posts["week_start"] = scored_posts["created_at"].dt.to_period("W").apply(
        lambda p: p.start_time.tz_localize("UTC")
    )

    for week_start, group in scored_posts.groupby("week_start"):
        avg_score = float(group["net_score"].mean())

        if avg_score > DIRECTION_THRESHOLD:
            direction = 1
        elif avg_score < -DIRECTION_THRESHOLD:
            direction = -1
        else:
            direction = 0

        rows.append({
            "condition_id":        condition_id,
            "source":              "truth_social",
            "week_start":          week_start,
            "sentiment_score":     round(avg_score, 6),
            "sentiment_direction": direction,
            "post_count":          len(group),
        })

    return rows


#full pipeline 
def run():
    posts    = load_archive()
    markets  = load_markets()
    print(type(markets["tags"].iloc[0]))
    print(markets["tags"].iloc[0])

    tokenizer, model = load_finbert()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    matched_markets = 0

    for i, market in markets.iterrows():
        condition_id = market["conditionId"]
        tags = list(market["tags"]) if market["tags"] is not None else []
        start        = market["startDate"]
        end          = market["endDate"]

        # get keywords first — skip if none
        keywords = get_keywords_for_market(tags)
        if not keywords:
            continue

        # then filter posts
        window = posts[
            (posts["created_at"] >= start) &
            (posts["created_at"] <= end)
        ]
        if i < 5:
            print(f"tags: {tags} | keywords: {keywords} | window: {len(window)}")

        if window.empty:
            continue

        pattern = "|".join(re.escape(k) for k in keywords)
        matched = window[
            window["text"].str.contains(pattern, case=False, na=False)
        ].copy()

        if matched.empty:
            continue

        matched_markets += 1
        if matched_markets % 50 == 0:
            print(f"  {matched_markets} markets matched so far...")

        matched["net_score"] = score_posts(matched["text"].tolist(), tokenizer, model)
        rows = aggregate_weekly(matched, condition_id)
        all_rows.extend(rows)


    # save
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nDone: {len(out_df):,} weekly sentiment rows across {matched_markets} markets")
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    run()