# scores Truth Social posts against Polymarket markets
# for each market, finds Trump posts within the market's date (startdate to enddate) 
# each that mention the market's keywords, ->  runs VADER sentiment ->  and aggregates to weekly scores

# schema: condition_id, source, week_start, sentiment_score, sentiment_direction, post_count
#

from pathlib import Path
from datetime import timezone

import pandas as pd
import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.sentiment.ner_keywords import extract_keywords, keywords_to_pattern


DB       = Path("data/analytical/polymarket.ddb")
ARCHIVE  = Path("data/raw/truth_social_archive.csv")
OUT_DIR  = Path("data/processed/sentiment")
OUT_FILE = OUT_DIR / "truth_social_sentiment.csv"

# keywords that match the archive author — useless for single-author archives
# since every post is by Trump, "Trump" matches everything and dilutes market-specific signal
AUTHOR_STOPWORDS = {"trump", "donald trump"}

#  load markets (resolved Yes/No with trades only) 
print("Loading markets from DuckDB...")
con = duckdb.connect(str(DB), read_only=True)
markets = con.execute("""
    SELECT m.conditionId, m.question, m.startDate, m.endDate
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (
        SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId
    )
    ORDER BY m.startDate
""").fetchdf()
con.close()
print(f"  {len(markets)} markets loaded")


# load truth social archive 
print("Loading Truth Social archive...")
archive = pd.read_csv(ARCHIVE, parse_dates=["created_at"])
# ensure created_at is UTC-aware for comparison with market dates
if archive["created_at"].dt.tz is None:
    archive["created_at"] = archive["created_at"].dt.tz_localize("UTC")

# floor each post to its week start (Monday) for aggregation
archive["week_start"] = archive["created_at"].dt.to_period("W-SUN").apply(
    lambda p: p.start_time.tz_localize("UTC")
)

print(f"{len(archive)} posts loaded ({archive['created_at'].min().date()} -> {archive['created_at'].max().date()})")


# vader 
vader = SentimentIntensityAnalyzer()

def score_text(text):
    return vader.polarity_scores(str(text))["compound"]

# Convert compound score to direction using VADER's recommended thresholds 
def compound_to_direction(score):
    if score > 0.05:
        return 1
    if score < -0.05:
        return -1
    return 0

# main loop 
print(f"\nProcessing {len(markets)} markets:")


output_rows = []
empty_count = 0

for i, market in enumerate(markets.itertuples(), 1):
    cid = market.conditionId
    question = market.question
    start = market.startDate
    end = market.endDate

    # progress counter every 100 markets
    if i % 100 == 0 or i == len(markets):
        print(f"[{i}/{len(markets)}] processed ({empty_count} markets with no matches so far)")

    # extract keywords - skip market if none found
    keywords = extract_keywords(question)

    # remove author because "Trump" matches all 29k posts and dilutes signal
    keywords = [
        kw for kw in keywords
        if kw.lower() not in AUTHOR_STOPWORDS
    ]

    if not keywords:
        empty_count += 1
        continue

    # ensure start/end are timezone-aware for comparison
    if hasattr(start, "tzinfo") and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if hasattr(end, "tzinfo") and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # filter archive to posts within this market's date window
    in_window = archive[
        (archive["created_at"] >= start) &
        (archive["created_at"] <= end)
    ]

    if in_window.empty:
        empty_count += 1
        continue

    # filter to posts mentioning any of this market's keywords
    pattern = keywords_to_pattern(keywords)
    matched = in_window[
        in_window["content"].astype(str).str.contains(pattern, na=False)
    ]

    if matched.empty:
        empty_count += 1
        continue

    # drop posts that are just URLs — no text for VADER to score, adds noise
    matched = matched[
        matched["content"].astype(str).str.match(r"^\s*https?://\S+\s*$") == False
    ]

    if matched.empty:
        empty_count += 1
        continue

    # score each matched post
    matched = matched.copy()
    matched["compound"] = matched["content"].apply(score_text)

    # aggregate to weekly scores
    weekly = (
        matched.groupby("week_start")
        .agg(
            sentiment_score=("compound", "mean"),
            post_count=("compound", "count"),
        ).reset_index()
    )

    # drop weeks with only 1 post — single post drives whole week score, too noisy
    weekly = weekly[weekly["post_count"] >= 2]

    if weekly.empty:
        empty_count += 1
        continue

    weekly["sentiment_direction"] = weekly["sentiment_score"].apply(compound_to_direction)
    weekly["condition_id"] = cid
    weekly["source"] = "truth_social"

    # reorder columns to match schema
    weekly = weekly[[
        "condition_id", "source", "week_start",
        "sentiment_score", "sentiment_direction", "post_count"
    ]]
    output_rows.append(weekly)

# write output
print(f"Markets with sentiment data: {len(markets) - empty_count}")
print(f"Markets with no matches:{empty_count}")

if output_rows:
    result = pd.concat(output_rows, ignore_index=True)
    result["week_start"] = result["week_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["sentiment_score"] = result["sentiment_score"].round(5)
    result.to_csv(OUT_FILE, index=False)
    print(f"Rows written: {len(result)}")
    print(f"Output: {OUT_FILE}")
else:
    print("No output rows")