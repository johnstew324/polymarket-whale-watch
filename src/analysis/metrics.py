import numpy as np
import pandas as pd

from .config import (
    EXCLUDED_CONTRACTS, MIN_MARKETS, HIT_RATE_THRESHOLD,
    VOLUME_QUANTILE, EARLY_QUANTILE, BOOTSTRAP_ITER, BOOTSTRAP_SEED,
)


def filter_experienced(hit_rate_df, min_markets=MIN_MARKETS):
    df = hit_rate_df[~hit_rate_df['wallet'].isin(EXCLUDED_CONTRACTS)].copy()
    return df[df['markets_traded'] >= min_markets].copy()


def threshold_sensitivity(hit_rate_df):
    df = hit_rate_df[~hit_rate_df['wallet'].isin(EXCLUDED_CONTRACTS)].copy()
    rows = []
    for m in [5, 10, 20]:
        sub = df[df['markets_traded'] >= m]
        rows.append({
            'min_markets': m,
            'wallets': len(sub),
            'median_hit_pct': sub['hit_rate_pct'].median(),
            'wallets_above_65': (sub['hit_rate_pct'] >= HIT_RATE_THRESHOLD).sum(),
        })
    return pd.DataFrame(rows)


def build_whale_quadrant(experienced, wallet_edge):
    vol_threshold = experienced['total_volume'].quantile(VOLUME_QUANTILE)
    whales = (
        experienced[
            (experienced['hit_rate_pct'] >= HIT_RATE_THRESHOLD)
            & (experienced['total_volume'] >= vol_threshold)
        ]
        .merge(wallet_edge[['wallet', 'mean_realised_edge']], on='wallet', how='left')
        .sort_values('total_volume', ascending=False)
    )
    return whales, vol_threshold


def per_wallet_timing(timing_df):
    return (
        timing_df
        .groupby('wallet')
        .agg(
            avg_days_before=pd.NamedAgg(column='days_before_resolution', aggfunc='mean'),
            median_days_before=pd.NamedAgg(column='days_before_resolution', aggfunc='median'),
            num_buys=pd.NamedAgg(column='days_before_resolution', aggfunc='count'),
        )
        .reset_index()
    )


def build_early_accurate(experienced, wallet_timing):
    merged = experienced.merge(wallet_timing, on='wallet', how='inner')
    timing_p75 = merged['avg_days_before'].quantile(EARLY_QUANTILE)
    early_accurate = merged[
        (merged['hit_rate_pct'] >= HIT_RATE_THRESHOLD)
        & (merged['avg_days_before'] >= timing_p75)
    ].sort_values('hit_rate_pct', ascending=False)
    return early_accurate, timing_p75, merged


def build_specialisation(domain_df, experienced):
    wallet_total = domain_df.groupby('wallet')['tag_volume'].sum().rename('wallet_total')
    domain_df = domain_df.merge(wallet_total, on='wallet')
    domain_df['tag_share'] = domain_df['tag_volume'] / domain_df['wallet_total']

    herf = (
        domain_df.groupby('wallet')
        .apply(lambda g: (g['tag_share'] ** 2).sum(), include_groups=False)
        .rename('herfindahl')
        .reset_index()
    )

    top_tag = (
        domain_df.sort_values('tag_share', ascending=False)
        .groupby('wallet').first().reset_index()
        [['wallet', 'tag', 'tag_share']]
        .rename(columns={'tag': 'top_tag', 'tag_share': 'top_tag_share'})
    )

    specialisation = herf.merge(top_tag, on='wallet').merge(
        experienced[['wallet', 'hit_rate_pct', 'total_volume', 'markets_traded']],
        on='wallet'
    )
    return specialisation, domain_df


def build_shortlist(experienced, wallet_timing, specialisation, wallet_edge):
    smart = (
        experienced
        .merge(wallet_timing, on='wallet', how='left')
        .merge(specialisation[['wallet', 'herfindahl', 'top_tag', 'top_tag_share']],
               on='wallet', how='left')
        .merge(wallet_edge[['wallet', 'mean_realised_edge']], on='wallet', how='left')
    )
    smart = smart[~smart['wallet'].isin(EXCLUDED_CONTRACTS)]
    vol_median = smart['total_volume'].median()
    shortlist = smart[
        (smart['hit_rate_pct'] >= HIT_RATE_THRESHOLD)
        & (smart['total_volume'] >= vol_median)
    ].sort_values(['hit_rate_pct', 'total_volume'], ascending=[False, False])
    return shortlist


def survival_analysis(whales, post_cutoff_wallets):
    whales_set = set(whales['wallet'])
    post_set = set(post_cutoff_wallets['wallet'])
    survived = whales_set & post_set
    return len(whales_set), len(post_set), len(survived)


def wallet_bootstrap(values, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED, stat='mean'):
    """Wallet-clustered bootstrap. `values` is one observation per wallet."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    samples = rng.choice(arr, size=(n_iter, len(arr)), replace=True)
    if stat == 'mean':
        draws = samples.mean(axis=1)
    elif stat == 'median':
        draws = np.median(samples, axis=1)
    elif stat == 'count':
        draws = samples.sum(axis=1)
    else:
        raise ValueError(stat)
    return (
        float(draws.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def whale_count_bootstrap(experienced, vol_threshold, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    arr = experienced[['hit_rate_pct', 'total_volume']].values
    draws = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(arr), len(arr))
        s = arr[idx]
        draws.append(((s[:, 0] >= HIT_RATE_THRESHOLD) & (s[:, 1] >= vol_threshold)).sum())
    draws = np.array(draws)
    return int(draws.mean()), int(np.quantile(draws, 0.025)), int(np.quantile(draws, 0.975))


def early_accurate_count_bootstrap(merged, timing_p75, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    arr = merged[['hit_rate_pct', 'avg_days_before']].values
    draws = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(arr), len(arr))
        s = arr[idx]
        draws.append(((s[:, 0] >= HIT_RATE_THRESHOLD) & (s[:, 1] >= timing_p75)).sum())
    draws = np.array(draws)
    return int(draws.mean()), int(np.quantile(draws, 0.025)), int(np.quantile(draws, 0.975))