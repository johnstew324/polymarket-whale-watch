import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss
)

from .config import (
    SEED, FULL_COLDSTART_FEATURES,
    HOLDOUT_MIN_TRADES, HOLDOUT_FRACTION, HOLDOUT_MIN_WALLETS,
)


def make_cold_start_split(fm, cutoff):
    train_pool = fm[fm['resolution_date'] <= cutoff].copy().reset_index(drop=True)
    late_pool = fm[fm['resolution_date'] > cutoff].copy().reset_index(drop=True)

    late_wallet_counts = late_pool['wallet'].value_counts()
    eligible = late_wallet_counts[late_wallet_counts >= HOLDOUT_MIN_TRADES].index.to_numpy()

    if len(eligible) == 0:
        raise ValueError(
            f'No late-period wallets have >= {HOLDOUT_MIN_TRADES} trades. '
            f'Check CUTOFF_QUANTILE ({len(late_pool)} rows in late pool, '
            f'{late_pool["wallet"].nunique()} unique wallets).'
        )

    rng = np.random.default_rng(SEED)
    n_holdout = max(HOLDOUT_MIN_WALLETS, int(len(eligible) * HOLDOUT_FRACTION))
    holdout_wallets = set(rng.choice(eligible, size=n_holdout, replace=False))

    train = train_pool[~train_pool['wallet'].isin(holdout_wallets)].copy().reset_index(drop=True)
    cold_test = late_pool[late_pool['wallet'].isin(holdout_wallets)].copy().reset_index(drop=True)
    warm_test = late_pool[~late_pool['wallet'].isin(holdout_wallets)].copy().reset_index(drop=True)

    assert set(train['wallet']).isdisjoint(set(cold_test['wallet']))

    feature_medians = train[FULL_COLDSTART_FEATURES].median(numeric_only=True)
    for frame in [train, cold_test, warm_test]:
        for col in FULL_COLDSTART_FEATURES:
            if col in frame.columns:
                frame[col] = frame[col].fillna(feature_medians.get(col, 0))

    print(f'Late-period cutoff: {cutoff}')
    print(f'Eligible late-period wallets for holdout: {len(eligible):,}')
    print(f'Cold-start holdout wallets sampled:      {len(holdout_wallets):,}')
    print(f'Train rows: {len(train):,}  |  Cold test: {len(cold_test):,}  |  Warm test: {len(warm_test):,}')

    return train, cold_test, warm_test, train_pool, late_pool


def build_matrix(frame, features):
    return frame[features].fillna(0)


def score_binary(y_true, probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=int)
    return {
        'roc_auc': roc_auc_score(y_true, probs),
        'pr_auc': average_precision_score(y_true, probs),
        'brier': brier_score_loss(y_true, probs),
        'logloss': log_loss(y_true, probs),
    }


def alpha_slice(frame, probs, top_frac=0.10):
    probs = np.asarray(probs, dtype=float)
    alpha = probs - frame['market_implied_prob'].to_numpy()
    threshold = np.quantile(alpha, 1 - top_frac)
    chosen = alpha >= threshold
    return {
        'n_rows': int(chosen.sum()),
        'mean_realized_edge': frame.loc[chosen, 'realized_edge'].mean(),
        'hit_rate': frame.loc[chosen, 'outcome_correct'].mean(),
        'mean_market_prob': frame.loc[chosen, 'market_implied_prob'].mean(),
    }