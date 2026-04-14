# aggregates pre-scored FT ProQuest articles into weekly sentiment per market
# requires finbert_scorer.py to have been run first to generate scored CSVs
#
# input:  data/processed/proquest/{topic}_articles_scored.csv
#         data/analytical/polymarket.ddb
# output: data/processed/sentiment/ft_sentiment.csv
# schema: condition_id, source, week_start, sentiment_score, sentiment_direction, post_count

from pathlib import Path
from datetime import timezone


import pandas as pd
import duckdb
from src.sentiment.ner_keywords import extract_keywords, keywords_to_pattern

DB       = Path("data/analytical/polymarket.ddb")
PQ_DIR   = Path("data/processed/proquest")
OUT_DIR  = Path("data/processed/sentiment")
OUT_FILE = OUT_DIR / "ft_sentiment.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# corpora available locally add new topics here as files are downloaded
from src.sentiment.pipeline_config import AVAILABLE_CORPORA, KEYWORD_TO_CORPORA


# need to update 
# min articles per week to include - fewer is too noisy
MIN_POSTS = 2

# direction threshold for FinBERT net_score
DIRECTION_THRESHOLD = 0.05


def get_corpora_for_keywords(keywords):
    corpora = set()
    for kw in keywords:
        for corpus in KEYWORD_TO_CORPORA.get(kw, []):
            if corpus in AVAILABLE_CORPORA:
                corpora.add(corpus)
    return list(corpora)

def compound_to_direction(score):
    if score > DIRECTION_THRESHOLD:
        return 1
    if score < -DIRECTION_THRESHOLD:
        return -1
    return 0


#  load pre-scored corpora (lazy cache)
_corpus_cache = {}

def load_scored_corpus(topic):
    if topic in _corpus_cache:
        return _corpus_cache[topic]

    path = PQ_DIR / f"{topic}_articles_scored.csv"
    if not path.exists():
        print(f"  warning: no scored CSV for '{topic}' — run finbert_scorer.py first")
        _corpus_cache[topic] = None
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "net_score"])
    _corpus_cache[topic] = df
    return df


def get_combined_corpus(corpora):
    frames = []
    for topic in corpora:
        df = load_scored_corpus(topic)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "proquest_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["proquest_id"])
    return combined


# load markets
print("Loading markets from DuckDB...")
con = duckdb.connect(str(DB), read_only=True)
markets = con.execute("""
    SELECT m.conditionId, m.question, m.startDate, m.endDate, m.tags
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (
        SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId
    )
    ORDER BY m.startDate
""").fetchdf()
con.close()
print(f"{len(markets)} markets loaded")


#  main loop 
print(f"\nProcessing {len(markets)} markets...")

output_rows = []
empty_count = 0

for i, market in enumerate(markets.itertuples(), 1):
    cid  = market.conditionId
    question = market.question
    start  = market.startDate
    end  = market.endDate
    tags = list(market.tags) if market.tags is not None else []

    if i % 100 == 0 or i == len(markets):
        print(f"[{i}/{len(markets)}] processed ({empty_count} markets with no matches so far)")

    keywords = extract_keywords(question)
    if not keywords:
        empty_count += 1
        continue

    # get relevant corpora from tags
    corpora = get_corpora_for_keywords(keywords)
    if not corpora:
        empty_count += 1
        continue

    # load pre-scored corpus
    corpus = get_combined_corpus(corpora)
    if corpus is None or corpus.empty:
        empty_count += 1
        continue


    # ensure start/end are timezone-aware
    start = pd.Timestamp(start, tz="UTC") if pd.notna(start) else None
    end   = pd.Timestamp(end,   tz="UTC") if pd.notna(end)   else None

    if start is None or end is None:
        empty_count += 1
        continue

    # filter corpus to market date window
    in_window = corpus[
        (corpus["date"] >= start) &
        (corpus["date"] <= end)
    ]

    if in_window.empty:
        empty_count += 1
        continue

    # filter to articles mentioning market keywords
    pattern = keywords_to_pattern(keywords)
    matched = in_window[
        in_window["text"].astype(str).str.contains(pattern, na=False)
    ].copy()

    if matched.empty:
        empty_count += 1
        continue

    # floor to week start for aggregation
    matched["week_start"] = matched["date"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.tz_localize("UTC")
    )

    # aggregate pre-scored net_score to weekly scores  no FinBERT here
    weekly = (
        matched.groupby("week_start")
        .agg(
            sentiment_score=("net_score", "mean"),
            post_count=("net_score", "count"),
        )
        .reset_index()
    )

    # drop weeks below minimum post threshold
    weekly = weekly[weekly["post_count"] >= MIN_POSTS]

    if weekly.empty:
        empty_count += 1
        continue

    weekly["sentiment_direction"] = weekly["sentiment_score"].apply(compound_to_direction)
    weekly["condition_id"] = cid
    weekly["source"] = "ft_proquest"

    weekly = weekly[[
        "condition_id", "source", "week_start",
        "sentiment_score", "sentiment_direction", "post_count"
    ]]

    output_rows.append(weekly)


print(f"\nDone. Writing output...")
print(f"Markets with sentiment data: {len(markets) - empty_count}")
print(f"Markets with no matches:{empty_count}")

if output_rows:
    result = pd.concat(output_rows, ignore_index=True)
    result["week_start"] = result["week_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["sentiment_score"] = result["sentiment_score"].round(4)
    result.to_csv(OUT_FILE, index=False)
    print(f"  Rows written: {len(result)}")
    print(f"  Output: {OUT_FILE}")
else:
    print("No output rows - check scored CSVs exist and market date ranges align")