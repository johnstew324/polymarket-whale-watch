# Analytical queries for Polymarket insider trading analysis.
# Run after load_db.py to validate the database and explore the data.
import duckdb # using duckdb for analysis since its faster than sqlite on large datasets and is better for time-series data
from pathlib import Path

DB  = Path("data/analytical/polymarket.ddb")
con = duckdb.connect(str(DB), read_only=True)


# data overview

# database structure
print("Database structure:")
print(con.execute("""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'main'
    ORDER BY table_name, ordinal_position
""").fetchdf().to_string(index=False))  


# row counts
print("\nrow counts:")
print(con.execute("""
    SELECT 'trades'    AS table_name, COUNT(*) AS rows FROM trades    UNION ALL
    SELECT 'markets'   AS table_name, COUNT(*) AS rows FROM markets   UNION ALL
    SELECT 'token_map' AS table_name, COUNT(*) AS rows FROM token_map
""").fetchdf().to_string(index=False))


# date ranges
print("\ntrades date range:")
print(con.execute("""
    SELECT
        MIN(timestamp) AS earliest_trade,
        MAX(timestamp) AS latest_trade,
        COUNT(DISTINCT DATE_TRUNC('month', timestamp)) AS months_covered
    FROM trades
""").fetchdf().to_string(index=False))


# market date ranges 
# 8 distinct outcomes - small number named !!!
print("\nmarkets date range:")
print(con.execute("""
    SELECT
        MIN(startDate) AS earliest_market,
        MAX(endDate)   AS latest_market,
        COUNT(*)       AS total_markets,
        COUNT(DISTINCT resolvedOutcome) AS distinct_outcomes 
    FROM markets
""").fetchdf().to_string(index=False))



# resolved outcome breakdown
print("\nresolved outcome breakdown:")
print(con.execute("""
    SELECT resolvedOutcome, COUNT(*) AS markets
    FROM markets
    GROUP BY resolvedOutcome
    ORDER BY markets DESC
""").fetchdf().to_string(index=False))


# outcomes breakdown in trades - should be mostly Yes/No but we have some outliers - putin, zelenskyy gaza ukraine something . 
print("\noutcomes breakdown in trades (should be mostly Yes/No):")
print(con.execute("""
    SELECT outcomes, COUNT(*) AS trades
    FROM trades
    GROUP BY outcomes
    ORDER BY trades DESC
    LIMIT 20
""").fetchdf().to_string(index=False))



print("\ntrades coverage: how many markets have trade data:")
print(con.execute("""
    SELECT
        COUNT(DISTINCT t.condition_id) AS markets_with_trades,
        COUNT(DISTINCT m.conditionId)  AS total_markets,
    FROM trades t
    RIGHT JOIN markets m ON t.condition_id = m.conditionId
""").fetchdf().to_string(index=False))


# token map coverage - how many trades can we link to a market condition using the token_map?
print("\nnull check on trades:")
print(con.execute("""
    SELECT
        COUNT(*) FILTER (WHERE wallet IS NULL)       AS null_wallet,
        COUNT(*) FILTER (WHERE price IS NULL)        AS null_price,
        COUNT(*) FILTER (WHERE outcomes IS NULL)     AS null_outcomes,
        COUNT(*) FILTER (WHERE condition_id IS NULL) AS null_condition_id
    FROM trades
""").fetchdf().to_string(index=False))



# price sanity check 
print("\nprice sanity: should be between 0 and 1 on a prediction market:")
print(con.execute("""
    SELECT
        MIN(price)  AS min_price,
        MAX(price)  AS max_price,
        AVG(price)  AS avg_price,
        COUNT(*) FILTER (WHERE price < 0 OR price > 1) AS out_of_range
    FROM trades
""").fetchdf().to_string(index=False))


con.close()


## other ideas for data exploration and validation:


# markey summary
# top markets by volume, trade count, unique traders, trades by year, volume by tag, etc



# wallet summary 
# - distribution of activity 
# top wallets by volume



# wallet hit rates - how often do they bet on the correct outcome in resolved markets - key for insider trading analysis



# domain specialisation (top wallets by volume)") - so like which tags do the biggest whales focus on?


# trading timing relative to resolution (do they trade early or late in the market lifecycle?)
