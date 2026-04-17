import numpy as np
import pandas as pd

from .config import (ALPHA_QUANTILE, MIN_MARKET_VOLUME_USD, POSITION_CAP_USD,MAX_MARKET_VOLUME_SHARE, COST_BPS_GRID,)


def wallet_scope_audit(train, cold_test, warm_test):
    rows = []
    for group, frame in [
        ('Training wallets', train),
        ('Cold-test wallets (unseen)', cold_test),
        ('Warm-test wallets (seen-late)', warm_test),
    ]:
        rows.append({
            'group': group,
            'wallets': frame['wallet'].nunique(),
            'rows': len(frame),
            'date_span': f"{frame['resolution_date'].min().date()} to {frame['resolution_date'].max().date()}",
        })
    overlap = len(set(train['wallet']).intersection(set(cold_test['wallet'])))
    return pd.DataFrame(rows), overlap


def _build_filtered_slice(cold_test, final_probs, market_probs):
    slice_df = cold_test.copy()
    slice_df['model_prob'] = np.asarray(final_probs, dtype=float)
    slice_df['predicted_alpha'] = slice_df['model_prob'] - np.asarray(market_probs, dtype=float)

    alpha_threshold = slice_df['predicted_alpha'].quantile(ALPHA_QUANTILE)
    slice_df = slice_df.loc[slice_df['predicted_alpha'] >= alpha_threshold].copy()
    slice_df = slice_df[slice_df['market_volume'].fillna(0) >= MIN_MARKET_VOLUME_USD].copy()
    slice_df = slice_df[slice_df['net_usd'].fillna(0) > 0].copy()

    slice_df['effective_stake_usd'] = np.minimum.reduce([
        slice_df['net_usd'].fillna(0).to_numpy(),
        np.full(len(slice_df), POSITION_CAP_USD),
        slice_df['market_volume'].fillna(0).to_numpy() * MAX_MARKET_VOLUME_SHARE,
    ])
    slice_df = slice_df[slice_df['effective_stake_usd'] > 0].copy()
    return slice_df, alpha_threshold


def friction_analysis(cold_test, final_probs, market_probs):
    slice_df, _ = _build_filtered_slice(cold_test, final_probs, market_probs)

    price = np.clip(slice_df['market_implied_prob'].to_numpy(), 1e-6, 1 - 1e-6)
    outcome = slice_df['outcome_correct'].to_numpy().astype(float)
    stake = slice_df['effective_stake_usd'].to_numpy()
    shares = stake / price

    gross_edge_per_share = outcome - price
    gross_pnl_usd = shares * gross_edge_per_share

    rows = []
    for cost_bps in COST_BPS_GRID:
        cost_per_share = cost_bps / 10_000.0
        net_edge_per_share = gross_edge_per_share - cost_per_share
        net_pnl_usd = shares * net_edge_per_share
        net_roi = np.divide(net_pnl_usd, stake, out=np.full_like(net_pnl_usd, np.nan), where=stake > 0)

        rows.append({
            'cost_bps': cost_bps,
            'trades_kept': len(slice_df),
            'mean_market_prob': price.mean(),
            'mean_gross_edge_per_share': gross_edge_per_share.mean(),
            'mean_net_edge_per_share': net_edge_per_share.mean(),
            'gross_total_pnl_usd': gross_pnl_usd.sum(),
            'net_total_pnl_usd': net_pnl_usd.sum(),
            'mean_net_roi_on_stake': np.nanmean(net_roi),
            'median_effective_stake_usd': np.median(stake),
            'share_positive_net_pnl': (net_pnl_usd > 0).mean(),
            'min_market_volume_filter': MIN_MARKET_VOLUME_USD,
            'position_cap_usd': POSITION_CAP_USD,
            'max_volume_share': MAX_MARKET_VOLUME_SHARE,
        })

    coverage = pd.DataFrame([{
        'rows_after_volume_and_size_filters': len(slice_df),
        'median_effective_stake_usd': np.median(stake) if len(stake) else np.nan,
        'mean_effective_stake_usd': np.mean(stake) if len(stake) else np.nan,
    }])
    return pd.DataFrame(rows), coverage


def risk_metrics(cold_test, final_probs, market_probs):
    slice_df, _ = _build_filtered_slice(cold_test, final_probs, market_probs)

    price = np.clip(slice_df['market_implied_prob'].to_numpy(), 1e-6, 1 - 1e-6)
    outcome = slice_df['outcome_correct'].to_numpy().astype(float)
    stake = slice_df['effective_stake_usd'].to_numpy()
    shares = stake / price

    net_pnl = shares * (outcome - price)
    net_roi = np.where(stake > 0, net_pnl / stake, np.nan)

    mean_roi = float(np.nanmean(net_roi))
    std_roi = float(np.nanstd(net_roi))
    sharpe_pt = mean_roi / std_roi if std_roi > 0 else np.nan

    sorted_pnl = np.sort(net_pnl)
    var95_idx = int(np.floor(0.05 * len(sorted_pnl)))
    var_95_usd = float(-sorted_pnl[var95_idx]) if len(sorted_pnl) else np.nan
    cvar_95_usd = float(-sorted_pnl[:var95_idx].mean()) if var95_idx > 0 else np.nan

    slice_sorted = slice_df.copy()
    slice_sorted['net_pnl'] = net_pnl
    slice_sorted = slice_sorted.sort_values('resolution_date').reset_index(drop=True)
    cum_pnl = slice_sorted['net_pnl'].cumsum().to_numpy()
    running_max = np.maximum.accumulate(cum_pnl) if len(cum_pnl) else np.array([])
    drawdown = cum_pnl - running_max if len(cum_pnl) else np.array([])
    max_dd_usd = float(-drawdown.min()) if len(drawdown) else np.nan
    max_dd_pct = float(max_dd_usd / running_max.max() * 100) if len(running_max) and running_max.max() > 0 else 0.0

    summary = pd.DataFrame([{
        'trades_analysed': len(net_pnl),
        'total_gross_pnl_usd': float(net_pnl.sum()),
        'mean_per_trade_pnl_usd': float(np.mean(net_pnl)) if len(net_pnl) else np.nan,
        'std_per_trade_pnl_usd': float(np.std(net_pnl)) if len(net_pnl) else np.nan,
        'mean_per_trade_roi': mean_roi,
        'std_per_trade_roi': std_roi,
        'per_trade_sharpe': sharpe_pt,
        'var_95_usd': var_95_usd,
        'cvar_95_usd': cvar_95_usd,
        'max_drawdown_usd': max_dd_usd,
        'max_drawdown_pct': max_dd_pct,
    }])

    series = {
        'cum_pnl': cum_pnl,
        'running_max': running_max,
        'net_pnl': net_pnl,
        'var_95_usd': var_95_usd,
    }
    return summary, series


def aum_scaling_table(eval_months=7, top_alpha_trades=2179, gross_pnl_7m=133_529,
                     position_cap_base=500, team_cost_eur=4 * 120_000, usd_eur_rate=1.08):
    trades_per_month = top_alpha_trades / eval_months
    peak_capital_usd = trades_per_month * position_cap_base
    annual_factor = 12 / eval_months
    annual_pnl_base = gross_pnl_7m * annual_factor

    aum_tiers = [
        ('$1M',   1_000_000,    500,  1.00),
        ('$5M',   5_000_000,  2_500,  4.90),
        ('$10M', 10_000_000,  5_000,  9.00),
    ]

    rows = []
    for label, aum, cap, scale in aum_tiers:
        annual_pnl = annual_pnl_base * scale
        net_eur = annual_pnl / usd_eur_rate - team_cost_eur
        rows.append({
            'aum': label,
            'position_cap_usd': cap,
            'annual_pnl_usd': annual_pnl,
            'net_after_team_eur': net_eur,
            'justified': net_eur > 0,
        })

    header = pd.DataFrame([{
        'eval_months': eval_months,
        'top_alpha_trades': top_alpha_trades,
        'trades_per_month': trades_per_month,
        'peak_capital_deployed_usd': peak_capital_usd,
        'annual_team_cost_eur': team_cost_eur,
    }])
    return header, pd.DataFrame(rows)