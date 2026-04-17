import numpy as np
import pandas as pd

from src.machine_learning.config import MIN_PRIOR_BETS


def attach_point_in_time_features(positions):
    positions = positions.copy()
    positions['wallet'] = positions['wallet'].astype(str)
    positions['entry_date'] = pd.to_datetime(positions['entry_date'], utc=True, format='mixed')
    positions['resolution_date'] = pd.to_datetime(positions['resolution_date'], utc=True, format='mixed')

    position_history = positions[['wallet', 'entry_date', 'resolution_date', 'avg_price', 'outcome_correct']].copy()
    position_history = position_history.sort_values(['wallet', 'resolution_date', 'entry_date']).reset_index(drop=True)
    position_history['win_component'] = position_history['outcome_correct']
    position_history['expected_component'] = position_history['avg_price']
    position_history['variance_component'] = position_history['avg_price'] * (1 - position_history['avg_price'])
    position_history['cum_actual_wins'] = position_history.groupby('wallet')['win_component'].cumsum()
    position_history['cum_expected_wins'] = position_history.groupby('wallet')['expected_component'].cumsum()
    position_history['cum_variance'] = position_history.groupby('wallet')['variance_component'].cumsum()
    position_history['cum_n_bets'] = position_history.groupby('wallet').cumcount() + 1

    history_state = (
        position_history.groupby(['wallet', 'resolution_date'], as_index=False)
        .agg(
            cum_actual_wins=('cum_actual_wins', 'last'),
            cum_expected_wins=('cum_expected_wins', 'last'),
            cum_variance=('cum_variance', 'last'),
            cum_n_bets=('cum_n_bets', 'last'),
        )
        .rename(columns={'resolution_date': 'history_asof_date'}))
    
    history_state['history_asof_date'] = pd.to_datetime(history_state['history_asof_date'], utc=True, format='mixed')

    positions = positions.dropna(subset=['entry_date']).copy()
    history_state = history_state.dropna(subset=['history_asof_date']).copy()
    positions = positions.sort_values(['entry_date', 'wallet']).reset_index(drop=True)
    history_state = history_state.sort_values(['history_asof_date', 'wallet']).reset_index(drop=True)

    n_before = len(positions)
    positions = pd.merge_asof(
        positions,
        history_state,
        by='wallet',
        left_on='entry_date',
        right_on='history_asof_date',
        direction='backward',
        allow_exact_matches=False,
    )
    assert len(positions) == n_before

    positions['n_prior_bets'] = positions['cum_n_bets'].fillna(0).astype(int)
    positions['hit_rate_pct'] = positions['cum_actual_wins'] / positions['cum_n_bets']
    positions['raw_edge_roll'] = (
        (positions['cum_actual_wins'] / positions['cum_n_bets'])
        - (positions['cum_expected_wins'] / positions['cum_n_bets'])
    )
    
    positions['z_score_roll'] = (
        (positions['cum_actual_wins'] - positions['cum_expected_wins'])/ np.sqrt(positions['cum_variance'].clip(lower=1e-9)))

    positions['hit_rate_pct'] = positions['hit_rate_pct'].fillna(positions['hit_rate_pct'].median())
    positions['raw_edge_roll'] = positions['raw_edge_roll'].fillna(positions['raw_edge_roll'].median())
    positions['z_score_roll'] = positions['z_score_roll'].fillna(positions['z_score_roll'].median())
    positions = positions[positions['n_prior_bets'] >= MIN_PRIOR_BETS].copy()

    return positions