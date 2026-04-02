# loads processed data into DuckDB for analysis.
# tables includ trades, markets, token_map 
# Duckdb is better than sqlite for time-series data
import duckdb
import json
from pathlib import Path
import pandas as pd

DB  = Path("data/analytical/polymarket.ddb")
con = duckdb.connect(str(DB))

# trades 
con.execute("""
    CREATE OR REPLACE TABLE trades AS
    SELECT
        epoch_ms(timestamp::BIGINT * 1000) AS timestamp,
        wallet,
        side,
        outcomes,
        usd_amount,
        token_amount,
        price,
        condition_id,
        transactionHash
    FROM read_csv_auto('data/processed/trades_clean.csv',
        types={'outcomes': 'VARCHAR'})
""")
print(f"trades: {con.execute('SELECT COUNT(*) FROM trades').fetchone()[0]:,} rows")



# markets 
markets_raw = json.loads(Path("data/raw/markets.json").read_text())
markets = []
for m in markets_raw:
    markets.append({
        "conditionId":     m.get("conditionId"),
        "question":        m.get("question"),
        "eventTitle":      m.get("eventTitle"),
        "startDate":       m.get("startDate"),
        "endDate":         m.get("endDate"),
        "closedTime":      m.get("closedTime"),
        "volume":          m.get("volume"),
        "resolvedOutcome": m.get("resolvedOutcome"),
        "tags":            m.get("tags", []),
    })

df = pd.DataFrame(markets)

def clean_tags(tags):
    return tags if isinstance(tags, list) else []
df['tags'] = df['tags'].apply(clean_tags)

con.execute("""
    CREATE OR REPLACE TABLE markets AS
    SELECT
        conditionId,
        question,
        eventTitle,
        startDate::TIMESTAMP  AS startDate,
        endDate::TIMESTAMP    AS endDate,
        closedTime::TIMESTAMP AS closedTime,
        volume,
        resolvedOutcome,
        tags
    FROM df
""")
print(f"markets: {con.execute('SELECT COUNT(*) FROM markets').fetchone()[0]:,} rows")



# token_map 
token_raw = json.loads(Path("data/raw/token_to_condition.json").read_text())

rows = []
for token_id, info in token_raw.items():
    rows.append({
        "token_id":     token_id,
        "condition_id": info["condition_id"],
        "outcome":      info["outcome"]
    }) 
token_df = pd.DataFrame(rows)

con.execute("""
    CREATE OR REPLACE TABLE token_map AS
    SELECT token_id, condition_id, outcome
    FROM token_df
""")
print(f"token_map: {con.execute('SELECT COUNT(*) FROM token_map').fetchone()[0]:,} rows")

con.close()
print(f"\nDone: database saved to {DB}")