# loads sentiment and financial signals into polymarket.ddb
# run after collect_truth_social.py, collect_ft.py, collect_financial.py

import duckdb
import pandas as pd
from pathlib import Path

DB = Path("data/analytical/polymarket.ddb")
TS_FILE  = Path("data/processed/sentiment/truth_social_sentiment.csv")
FT_FILE  = Path("data/processed/sentiment/ft_sentiment.csv")
FIN_FILE = Path("data/processed/financial/financial_signals_weekly.csv")

con = duckdb.connect(str(DB))

# sentiment - combine truth social + ft into one table
ts = pd.read_csv(TS_FILE)
ft = pd.read_csv(FT_FILE)
sentiment = pd.concat([ts, ft], ignore_index=True)
sentiment = sentiment.drop_duplicates(subset=["condition_id", "source", "week_start"])

con.execute("""
    CREATE OR REPLACE TABLE sentiment AS
    SELECT
        condition_id,
        source,
        week_start::TIMESTAMP AS week_start,
        sentiment_score,
        sentiment_direction,
        post_count
    FROM sentiment
""")
print(f"sentiment: {con.execute('SELECT COUNT(*) FROM sentiment').fetchone()[0]:,} rows")

# financial signals
financial = pd.read_csv(FIN_FILE)

con.execute("""
    CREATE OR REPLACE TABLE financial AS
    SELECT
        condition_id,
        ticker,
        week_start::TIMESTAMP AS week_start,
        weekly_return,
        mean_z_score,
        max_abs_z_score,
        trading_days
    FROM financial
""")
print(f"financial: {con.execute('SELECT COUNT(*) FROM financial').fetchone()[0]:,} rows")

con.close()
print(f"\nDone: database saved to {DB}")