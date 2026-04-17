import numpy as np
import pandas as pd

from  src.machine_learning.config import FM_COLS
from  .load_trades import load_resolved_geopolitical_trades
from .positions import collapse_to_positions
from .wallet_edge import compute_wallet_edge
from .wallet_history import attach_point_in_time_features
from .sentiment_join import attach_sentiment_features
from .divergence import add_divergence_and_transforms


def build_feature_matrix(use_cache=True):
    trades = load_resolved_geopolitical_trades(use_cache=use_cache)
    positions = collapse_to_positions(trades)
    positions, cutoff = compute_wallet_edge(positions)
    positions = attach_point_in_time_features(positions)
    positions = attach_sentiment_features(positions)
    # positions = attach_financial_features(positions)  # restore when financial_signals.csv is back
    positions = add_divergence_and_transforms(positions)

    fm = positions[FM_COLS].copy()
    fm['resolution_date'] = pd.to_datetime(fm['resolution_date'], utc=True, format='mixed')
    fm['entry_date'] = pd.to_datetime(fm['entry_date'], utc=True, format='mixed')
    fm['outcome_correct'] = fm['outcome_correct'].astype(int)
    fm['market_implied_prob'] = fm['avg_price'].clip(1e-6, 1 - 1e-6)
    fm['realized_edge'] = fm['outcome_correct'] - fm['market_implied_prob']

    print(f'Feature matrix shape: {fm.shape}')
    print(f'Positive rate: {fm["informed_label"].mean()*100:.1f}%')
    print(f'Weekly TS coverage:  {fm["ts_has_coverage"].mean()*100:.1f}%')
    print(f'Market TS coverage:  {fm["ts_market_has_coverage"].mean()*100:.1f}%')
    print(f'FT coverage:         {fm["ft_has_coverage"].mean()*100:.1f}%')

    return fm, cutoff