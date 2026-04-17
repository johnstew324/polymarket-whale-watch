import numpy as np
import pandas as pd


def collapse_to_positions(df):
    positions = (
        df.groupby(
            ['wallet', 'condition_id', 'resolution_date', 'question',
             'bet_outcome', 'wallet_direction'],
            as_index=False,
        )
        .agg(
            gross_usd=('usd_amount', 'sum'),
            total_tokens=('token_amount', 'sum'),
            num_trades=('usd_amount', 'size'),
            hours_before=('hours_before_resolution', 'median'),
            entry_date=('timestamp', 'min'),
            outcome_correct=('outcome_correct', 'first'),
            bet_vs_market=('bet_vs_market', 'first'),
            market_volume=('market_volume', 'first'),
        )
        .reset_index(drop=True)
    )

    positions['resolution_date'] = pd.to_datetime(positions['resolution_date'], utc=True, format='mixed')
    positions['entry_date'] = pd.to_datetime(positions['entry_date'], utc=True, format='mixed')

    positions['avg_price'] = (
        positions['gross_usd'] / positions['total_tokens'].replace(0, np.nan)
    )
    positions['avg_price'] = positions['avg_price'].fillna(positions['avg_price'].median())
    positions['net_usd'] = positions['gross_usd']

    return positions