import numpy as np
import pandas as pd
from scipy.stats import norm

from src.machine_learning.config import MIN_BETS_FOR_EDGE, CUTOFF_QUANTILE


def compute_wallet_edge(positions):
    split_index = positions.groupby(
        ['wallet', 'condition_id', 'resolution_date', 'question', 'bet_outcome'],
        as_index=False,
    ).agg(entry_date=('entry_date', 'min'))
    split_index['resolution_date'] = pd.to_datetime(split_index['resolution_date'], utc=True, format='mixed')
    cutoff = split_index['resolution_date'].quantile(CUTOFF_QUANTILE)
    print(f'Global modelling cutoff: {cutoff}')

    edge_train = positions.loc[
        positions['resolution_date'] <= cutoff,
        ['wallet', 'condition_id', 'bet_outcome', 'gross_usd', 'avg_price', 'outcome_correct'],
    ].copy()

    wallet_edge_stats = (
        edge_train.groupby('wallet', as_index=False)
        .agg(
            n_bets=('condition_id', 'size'),
            total_usd=('gross_usd', 'sum'),
            actual_wins=('outcome_correct', 'sum'),
            expected_wins=('avg_price', 'sum'),
            variance=('avg_price', lambda s: (s * (1 - s)).sum()),
            avg_implied_prob=('avg_price', 'mean'),
            hit_rate=('outcome_correct', 'mean'),
        )
    )
    wallet_edge_stats['z_score'] = (
        (wallet_edge_stats['actual_wins'] - wallet_edge_stats['expected_wins'])/ np.sqrt(wallet_edge_stats['variance'].clip(lower=1e-9)) )
    
    wallet_edge_stats['raw_edge'] = wallet_edge_stats['hit_rate'] - wallet_edge_stats['avg_implied_prob']
    wallet_edge_stats = wallet_edge_stats[wallet_edge_stats['n_bets'] >= MIN_BETS_FOR_EDGE].copy()

    n_tests = len(wallet_edge_stats)
    wallet_edge_stats['p_value'] = norm.sf(wallet_edge_stats['z_score'])
    wallet_edge_stats['p_bonferroni'] = (wallet_edge_stats['p_value'] * n_tests).clip(upper=1.0)
    wallet_edge_stats['sig_bonferroni_01'] = wallet_edge_stats['p_bonferroni'] < 0.01

    positions = positions.merge(
        wallet_edge_stats[['wallet', 'z_score', 'raw_edge', 'p_bonferroni', 'sig_bonferroni_01']], on='wallet', how='left', validate='m:1',)
    positions['sig_bonferroni_01'] = positions['sig_bonferroni_01'].fillna(False)
    positions['informed_label'] = ( positions['sig_bonferroni_01'] & (positions['outcome_correct'] == 1)).astype(int)

    return positions, cutoff