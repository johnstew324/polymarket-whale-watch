import numpy as np
import pandas as pd
import duckdb

from src.machine_learning.config import DB_PATH


def _load_sentiment():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT condition_id, source, week_start, sentiment_score, post_count
        FROM sentiment
    """).fetchdf()
    con.close()
    df['week_start'] = pd.to_datetime(df['week_start'], utc=True, format='mixed')
    df['condition_id'] = df['condition_id'].astype(str)
    return df


def attach_sentiment_features(positions):
    positions = positions.copy()
    positions['entry_date'] = pd.to_datetime(positions['entry_date'], utc=True, format='mixed')
    positions['condition_id'] = positions['condition_id'].astype(str)

    # Monday 00:00 UTC for the trade's calendar week
    positions['trade_week_start'] = (
        positions['entry_date'].dt.normalize()
        - pd.to_timedelta(positions['entry_date'].dt.weekday, unit='D')
    )
    # Only allow the most recently completed week
    positions['sentiment_week_start'] = positions['trade_week_start'] - pd.Timedelta(days=7)

    sentiment = _load_sentiment()
    ts_df = sentiment[sentiment['source'] == 'truth_social'].copy()
    ft_df = sentiment[sentiment['source'] == 'ft_proquest'].copy()

    weekly_sentiment = (
        ts_df.groupby('week_start', as_index=False)
        .agg(
            ts_weekly_score=('sentiment_score', 'mean'),
            ts_weekly_post_count=('post_count', 'sum'),
            ts_weekly_volatility=('sentiment_score', 'std'),
        )
        .sort_values('week_start')
        .reset_index(drop=True)
    )
    weekly_sentiment['ts_weekly_volatility'] = weekly_sentiment['ts_weekly_volatility'].fillna(0.0)

    ts_market_weekly = (
        ts_df.groupby(['condition_id', 'week_start'], as_index=False)
        .agg(
            ts_market_score=('sentiment_score', 'mean'),
            ts_market_post_count=('post_count', 'sum'),
        )
        .sort_values(['condition_id', 'week_start'])
        .reset_index(drop=True)
    )
    ts_market_weekly['ts_market_direction'] = np.sign(ts_market_weekly['ts_market_score']).astype(int)

    ft_weekly = (
        ft_df.groupby(['condition_id', 'week_start'], as_index=False)
        .agg(
            ft_sentiment_score=('sentiment_score', 'mean'),
            ft_post_count=('post_count', 'sum'),
        )
        .sort_values(['condition_id', 'week_start'])
        .reset_index(drop=True)
    )
    ft_weekly['ft_sentiment_direction'] = np.sign(ft_weekly['ft_sentiment_score']).astype(int)
    ft_weekly['ft_sentiment_volatility'] = (
        ft_weekly.groupby('condition_id')['ft_sentiment_score']
        .transform(lambda s: s.rolling(4, min_periods=1).std())
        .fillna(0.0)
    )

    positions = positions.merge(
        weekly_sentiment[
            ['week_start', 'ts_weekly_score', 'ts_weekly_post_count', 'ts_weekly_volatility']
        ],
        left_on='sentiment_week_start',
        right_on='week_start',
        how='left',
    ).drop(columns=['week_start'])

    positions['ts_weekly_score'] = positions['ts_weekly_score'].fillna(0.0)
    positions['ts_weekly_post_count'] = positions['ts_weekly_post_count'].fillna(0).astype(int)
    positions['ts_weekly_volatility'] = positions['ts_weekly_volatility'].fillna(0.0)
    positions['ts_has_coverage'] = (positions['ts_weekly_post_count'] > 0).astype(int)


    positions = positions.merge(
        ts_market_weekly[
            ['condition_id', 'week_start', 'ts_market_score', 'ts_market_post_count', 'ts_market_direction']
        ],
        left_on=['condition_id', 'sentiment_week_start'],
        right_on=['condition_id', 'week_start'],
        how='left',
    ).drop(columns=['week_start'])

    positions['ts_market_score'] = positions['ts_market_score'].fillna(0.0)
    positions['ts_market_post_count'] = positions['ts_market_post_count'].fillna(0).astype(int)
    positions['ts_market_direction'] = positions['ts_market_direction'].fillna(0).astype(int)
    positions['ts_market_has_coverage'] = (positions['ts_market_post_count'] > 0).astype(int)

    positions = positions.merge(
        ft_weekly[
            ['condition_id', 'week_start', 'ft_sentiment_score', 'ft_post_count',
             'ft_sentiment_direction', 'ft_sentiment_volatility']
        ],
        left_on=['condition_id', 'sentiment_week_start'],
        right_on=['condition_id', 'week_start'],
        how='left',
    ).drop(columns=['week_start'])

    positions['ft_sentiment_score'] = positions['ft_sentiment_score'].fillna(0.0)
    positions['ft_post_count'] = positions['ft_post_count'].fillna(0).astype(int)
    positions['ft_sentiment_direction'] = positions['ft_sentiment_direction'].fillna(0).astype(int)
    positions['ft_sentiment_volatility'] = positions['ft_sentiment_volatility'].fillna(0.0)
    positions['ft_has_coverage'] = (positions['ft_post_count'] > 0).astype(int)

    positions = positions.drop(columns=['trade_week_start', 'sentiment_week_start'])

    return positions