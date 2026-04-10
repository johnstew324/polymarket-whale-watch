import pandas as pd
from src.sentiment.ner_keywords import extract_keywords
import re

df = pd.read_csv("data/processed/sentiment/truth_social_sentiment.csv")


cid = "0x8ee2f1640386310eb5e7ffa596ba9335f2d324e303d21b0dfea6998874445791" # # Russia x Ukraine ceasefire in 2025?
rows = df[df["condition_id"] == cid].sort_values("week_start")
print(f"Russia x Ukraine ceasefire in 2025? — {len(rows)} weeks with coverage")
print(rows[["week_start", "sentiment_score", "sentiment_direction", "post_count"]].to_string(index=False))


archive = pd.read_csv("data/raw/truth_social_archive.csv", parse_dates=["created_at"])

pattern = re.compile(r"Russia|Ukraine", re.IGNORECASE)

# check a negative week (Feb 10) and a positive week (Mar 10)
for week_start, week_end, label in [
    ("2025-02-10", "2025-02-17", "NEGATIVE week Feb 10 (-0.26)"),
    ("2025-03-10", "2025-03-17", "POSITIVE week Mar 10 (+0.29)"),
]:
    start = pd.Timestamp(week_start, tz="UTC")
    end   = pd.Timestamp(week_end, tz="UTC")

    matched = archive[
        (archive["created_at"] >= start) &
        (archive["created_at"] < end) &
        (archive["content"].astype(str).str.contains(pattern, na=False)) &
        (~archive["content"].astype(str).str.match(r"^\s*https?://\S+\s*$"))
    ]

    print(f"\n--- {label} ({len(matched)} posts) ---")
    for _, row in matched.iterrows():
        print(f"[{row['created_at'].date()}] {str(row['content'])[:300]}")
        print()