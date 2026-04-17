from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import CONTEXT_ONLY_FEATURES

FIGURES_DIR = Path('figures')


def _ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_eda_diagnostics(fm, save=True):
    _ensure_figures_dir()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Cold-Start Trade Edge: Dataset Diagnostics', fontsize=16, fontweight='bold', y=1.01)

    plot_df = fm.copy()
    plot_df['price_decile'] = pd.qcut(plot_df['market_implied_prob'], 10, duplicates='drop')
    cal_df = (
        plot_df.groupby('price_decile', observed=True)
        .agg(
            implied_prob=('market_implied_prob', 'mean'),
            actual_hit_rate=('outcome_correct', 'mean'),
            mean_realized_edge=('realized_edge', 'mean'),
            n=('wallet', 'size'),
        )
        .reset_index(drop=True)
    )

    ax = axes[0, 0]
    ax.plot(cal_df['implied_prob'], cal_df['actual_hit_rate'], 'o-', color='crimson', linewidth=2, label='Observed hit rate')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Market baseline')
    ax.set_xlabel('Average market-implied probability')
    ax.set_ylabel('Observed hit rate')
    ax.set_title('Calibration by Market Price Decile', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(plot_df['realized_edge'], bins=60, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(plot_df['realized_edge'].mean(), color='crimson', linestyle='--', linewidth=2,
               label=f'Mean = {plot_df["realized_edge"].mean():+.3f}')
    ax.set_xlabel('Realized edge = outcome_correct - avg_price')
    ax.set_ylabel('Number of trades')
    ax.set_title('Realized-Edge Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    plot_df['hours_bucket'] = pd.qcut(plot_df['hours_before'].rank(method='first'), 8, duplicates='drop')
    hours_df = (
        plot_df.groupby('hours_bucket', observed=True)
        .agg(mean_hours=('hours_before', 'mean'), mean_edge=('realized_edge', 'mean'))
        .reset_index(drop=True)
    )
    ax.plot(hours_df['mean_hours'], hours_df['mean_edge'] * 100, 'o-', color='darkorange', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Average hours before resolution')
    ax.set_ylabel('Mean realized edge (pp)')
    ax.set_title('Timing and Realized Edge', fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    plot_df['usd_bucket'] = pd.qcut(plot_df['log_net_usd'].rank(method='first'), 8, duplicates='drop')
    usd_df = (
        plot_df.groupby('usd_bucket', observed=True)
        .agg(mean_log_usd=('log_net_usd', 'mean'), mean_edge=('realized_edge', 'mean'))
        .reset_index(drop=True)
    )
    ax.plot(usd_df['mean_log_usd'], usd_df['mean_edge'] * 100, 'o-', color='seagreen', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Average log position size')
    ax.set_ylabel('Mean realized edge (pp)')
    ax.set_title('Position Size and Realized Edge', fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / 'cold_start_diagnostics.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved → {out}')
    return fig


def plot_evaluation_grid(y_test, market_probs, logit_probs, baseline_context_probs,
                          context_probs, alpha_curve, selection_summary, robust_df,
                          save=True):
    _ensure_figures_dir()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Cold-Start Evaluation on Unseen Wallets', fontsize=16, fontweight='bold', y=1.01)

    model_specs = [
        (market_probs, 'Market price only', 'gray'),
        (logit_probs, 'Logistic (market + context)', 'steelblue'),
        (baseline_context_probs, 'Baseline XGB (market + context)', '#f4a261'),
        (context_probs, 'Tuned XGB (market + context)', 'crimson'),
    ]

    ax = axes[0, 0]
    for probs, label, color in model_specs:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=label, linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    base_precision = y_test.mean()
    ax.axhline(base_precision, color='black', linestyle='--', linewidth=1, label=f'Base rate ({base_precision:.3f})')
    for probs, label, color in model_specs:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax.plot(rec, prec, label=label, linewidth=2, color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    for probs, label, color in [
        (market_probs, 'Market price only', 'gray'),
        (context_probs, 'Tuned XGB (market + context)', 'crimson'),
    ]:
        tmp = pd.DataFrame({'prob': probs, 'y': np.asarray(y_test)})
        tmp['bucket'] = pd.qcut(tmp['prob'], 10, duplicates='drop')
        cal = tmp.groupby('bucket', observed=True).agg(pred=('prob', 'mean'), actual=('y', 'mean')).reset_index(drop=True)
        ax.plot(cal['pred'], cal['actual'], 'o-', linewidth=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Average predicted probability')
    ax.set_ylabel('Observed hit rate')
    ax.set_title('Calibration by Probability Decile', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    alpha_curve_plot = alpha_curve.copy()
    alpha_curve_plot['mean_realized_edge_pp'] = alpha_curve_plot['mean_realized_edge'] * 100
    ax.plot(alpha_curve_plot['Top fraction'], alpha_curve_plot['mean_realized_edge_pp'], 'o-', color='crimson', linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Selected fraction of trades')
    ax.set_ylabel('Mean realized edge (pp)')
    ax.set_title('Economic Value by Model-Alpha Slice', fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bar_df = selection_summary.copy()
    ax.barh(bar_df['selection_rule'], bar_df['mean_realized_edge'] * 100,
            color=['lightgray', 'crimson', 'steelblue', 'darkorange', 'mediumseagreen'][:len(bar_df)])
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Mean realized edge (pp)')
    ax.set_title('Economic Value of Ranking Rules', fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    ax = axes[1, 2]
    x = np.arange(len(robust_df))
    width = 0.35
    ax.bar(x - width/2, robust_df['market_pr_auc'], width=width, color='gray', label='Market PR-AUC')
    ax.bar(x + width/2, robust_df['model_pr_auc'], width=width, color='crimson', label='Model PR-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {seed}' for seed in robust_df['seed']])
    ax.set_ylabel('PR-AUC')
    ax.set_title('PR-AUC Across Unseen-Wallet Holdouts', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / 'cold_start_evaluation.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved → {out}')
    return fig


def plot_feature_importance(final_model, top_n=15, save=True):
    _ensure_figures_dir()
    importance = pd.Series(
        final_model.feature_importances_,
        index=CONTEXT_ONLY_FEATURES
    ).sort_values().tail(top_n)

    fig = plt.figure(figsize=(10, 7))
    plt.barh(importance.index, importance.values, color='crimson')
    plt.xlabel('Gain-based feature importance')
    plt.title('Main Model Feature Importance\n(XGBoost on market + context)', fontweight='bold')
    plt.grid(alpha=0.2, axis='x')
    plt.tight_layout()
    if save:
        out = FIGURES_DIR / 'cold_start_feature_importance.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved → {out}')
    return fig


def plot_risk_profile(risk_series, save=True):
    _ensure_figures_dir()
    cum_pnl = risk_series['cum_pnl']
    running_max = risk_series['running_max']
    net_pnl = risk_series['net_pnl']
    var_95_usd = risk_series['var_95_usd']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Risk Profile: Top-10% Alpha Slice', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(cum_pnl, color='#1A3A5C', linewidth=1.4, label='Cumulative P&L')
    ax.fill_between(range(len(cum_pnl)), running_max, cum_pnl,
                    where=(cum_pnl < running_max), color='#C0392B', alpha=0.25, label='Drawdown')
    ax.set_xlabel('Trade number (chronological order)')
    ax.set_ylabel('Cumulative net P&L ($)')
    ax.set_title('Cumulative P&L and Drawdowns')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(net_pnl, bins=60, color='#4A7FB5', edgecolor='white', linewidth=0.4)
    ax.axvline(-var_95_usd, color='#C0392B', linewidth=1.8, linestyle='--',
               label=f'VaR 95% = ${var_95_usd:.0f}')
    ax.axvline(float(np.mean(net_pnl)), color='#1E8449', linewidth=1.8, linestyle='-',
               label=f'Mean = ${float(np.mean(net_pnl)):.0f}')
    ax.set_xlabel('Per-trade net P&L ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Per-Trade P&L Distribution')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / 'risk_adjusted_profile.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved → {out}')
    return fig