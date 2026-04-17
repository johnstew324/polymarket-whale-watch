import duckdb
import numpy as np
import pandas as pd
from src.machine_learning.config import DB_PATH, GEO_TAGS
from pathlib import Path



CACHE_PATH = Path('data/processed/trades_enriched.parquet')


def load_resolved_geopolitical_trades(use_cache=True):
    if use_cache and CACHE_PATH.exists():
        print(f'Loading from cache: {CACHE_PATH}')
        return pd.read_parquet(CACHE_PATH)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT
            t.timestamp,
            t.wallet,
            t.side,
            t.outcomes                                           AS bet_outcome,
            t.usd_amount,
            t.token_amount,
            t.price,
            t.condition_id,
            t.transactionHash                                    AS transaction_hash,
            DATEDIFF('hour', t.timestamp, m.closedTime)          AS hours_before_resolution,
            CASE
                WHEN t.side = 'BUY' AND t.outcomes = 'Yes' AND t.price < 0.5 THEN 1
                WHEN t.side = 'BUY' AND t.outcomes = 'No'  AND t.price < 0.5 THEN 1
                ELSE 0
            END                                                  AS bet_vs_market,
            CASE WHEN t.outcomes = m.resolvedOutcome THEN 1 ELSE 0 END AS outcome_correct,
            CASE WHEN t.outcomes = 'Yes' THEN 1 ELSE -1 END     AS wallet_direction,
            m.closedTime                                         AS resolution_date,
            m.endDate                                            AS market_end_date,
            m.question,
            m.volume                                             AS market_volume,
            m.tags                                               AS tags
        FROM trades t
        JOIN markets m ON t.condition_id = m.conditionId
        WHERE
            m.resolvedOutcome IN ('Yes', 'No')
            AND m.closedTime IS NOT NULL
            AND t.side = 'BUY'
            AND list_has_any(m.tags, ?)
    """, [list(GEO_TAGS)]).fetchdf()
    con.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')
    df['resolution_date'] = pd.to_datetime(df['resolution_date'], utc=True, format='mixed')
    df['market_end_date'] = pd.to_datetime(df['market_end_date'], utc=True, format='mixed')

    n_raw = len(df)
    df = df.drop_duplicates(subset=[
        'timestamp', 'wallet', 'bet_outcome', 'usd_amount', 'token_amount',
        'price', 'condition_id', 'transaction_hash'
    ]).copy()
    print(f'Dropped exact duplicate trade rows: {n_raw - len(df):,}')

    print(f'Geopolitical trades: {len(df):,}')
    print(f'Unique wallets: {df["wallet"].nunique():,}')
    print(f'Unique markets: {df["condition_id"].nunique():,}')

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f'Cached to {CACHE_PATH}')

    return df