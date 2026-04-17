import numpy as np
import pandas as pd


def _winsorise(series, lower=0.01, upper=0.99):
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))


def add_divergence_and_transforms(positions):
    positions = positions.copy()

    positions['ts_weekly_divergence'] = (
        positions['wallet_direction'] * positions['ts_weekly_score'] * -1
    )
    positions['ts_market_divergence'] = (
        positions['wallet_direction'] * positions['ts_market_direction'] * -1
    )
    positions['divergence_score'] = np.where(
        positions['ts_market_has_coverage'] == 1,
        positions['ts_market_divergence'],
        positions['ts_weekly_divergence'],
    )

    ft_mean = positions['ft_sentiment_score'].mean()
    ft_std = positions['ft_sentiment_score'].std()
    if pd.isna(ft_std) or ft_std == 0:
        positions['ft_sentiment_score_z'] = 0.0
    else:
        positions['ft_sentiment_score_z'] = (positions['ft_sentiment_score'] - ft_mean) / ft_std
    positions['ft_divergence'] = (
        positions['wallet_direction'] * positions['ft_sentiment_score_z'] * -1
    )
    positions['divergence_score'] = np.where(
        positions['ft_has_coverage'] == 1,
        positions['ft_divergence'],
        positions['divergence_score'],
    )

    positions['log_net_usd'] = np.log1p(_winsorise(positions['net_usd']))
    positions['log_market_volume'] = np.log1p(_winsorise(positions['market_volume']))
    positions['hours_int'] = positions['hours_before'].clip(0, 8760).astype(int)
    positions['log_hours_before'] = np.log1p(positions['hours_before'])

    return positions