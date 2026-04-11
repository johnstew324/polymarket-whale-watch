import pandas as pd
import duckdb

df = pd.read_csv("data/processed/sentiment/ft_sentiment.csv")
con = duckdb.connect("data/analytical/polymarket.ddb", read_only=True)

# get top Yemen markets by volume
top = con.execute("""
    SELECT conditionId, question, resolvedOutcome, volume
    FROM markets
    WHERE resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (SELECT 1 FROM trades t WHERE t.condition_id = conditionId)
    AND array_contains(tags, 'yemen')
    ORDER BY volume DESC
    LIMIT 3
""").fetchdf()
con.close()

for _, row in top.iterrows():
    cid = row["conditionId"]
    rows = df[df["condition_id"] == cid].sort_values("week_start")
    print(f"\n--- {row['question']} (resolved: {row['resolvedOutcome']}) ---")
    from src.sentiment.ner_keywords import extract_keywords
    print(f"keywords: {extract_keywords(row['question'])}")
    if rows.empty:
        print("NO COVERAGE")
    else:
        print(rows[["week_start", "sentiment_score", "sentiment_direction", "post_count"]].to_string(index=False))