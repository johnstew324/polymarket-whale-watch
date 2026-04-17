import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss
)

from .config import (
    CONTEXT_ONLY_FEATURES, FULL_COLDSTART_FEATURES,
    HOLDOUT_MIN_TRADES, HOLDOUT_FRACTION, HOLDOUT_MIN_WALLETS,
)
from .split import build_matrix, score_binary, alpha_slice


def warm_vs_cold(cold_test, warm_test, context_probs, context_probs_warm,
                 market_probs, market_probs_warm):
    rows = []
    for name, frame, probs, market in [
        ('Cold start (unseen wallets)', cold_test, context_probs, market_probs),
        ('Warm start (seen late wallets)', warm_test, context_probs_warm, market_probs_warm),
    ]:
        model_scores = score_binary(frame['outcome_correct'].astype(int), probs)
        market_scores = score_binary(frame['outcome_correct'].astype(int), market)
        alpha_stats = alpha_slice(frame, probs, top_frac=0.10)
        rows.append({
            'split': name,
            'market_roc_auc': market_scores['roc_auc'],
            'model_roc_auc': model_scores['roc_auc'],
            'market_pr_auc': market_scores['pr_auc'],
            'model_pr_auc': model_scores['pr_auc'],
            'market_brier': market_scores['brier'],
            'model_brier': model_scores['brier'],
            'top10_realized_edge': alpha_stats['mean_realized_edge'],
            'top10_hit_rate': alpha_stats['hit_rate'],
        })
    return pd.DataFrame(rows)


def economic_ranking(cold_test, final_probs, div_probs, market_probs):
    test = cold_test
    final_alpha = final_probs - market_probs
    div_alpha = div_probs - market_probs

    selection_rows = [{
        'selection_rule': 'Overall cold-start test set',
        'n_rows': len(test),
        'mean_realized_edge': test['realized_edge'].mean(),
        'hit_rate': test['outcome_correct'].mean(),
        'mean_market_prob': test['market_implied_prob'].mean(),
    }]

    market_rank = test['market_implied_prob'].to_numpy()
    low_price_rank = -test['market_implied_prob'].to_numpy()

    for label, score in [
        ('Top 10% by divergence-only alpha', div_alpha),
        ('Top 10% by main model alpha', final_alpha),
        ('Top 10% highest market price', market_rank),
        ('Top 10% lowest market price', low_price_rank),
    ]:
        chosen = score >= np.quantile(score, 0.90)
        selection_rows.append({
            'selection_rule': label,
            'n_rows': int(chosen.sum()),
            'mean_realized_edge': test.loc[chosen, 'realized_edge'].mean(),
            'hit_rate': test.loc[chosen, 'outcome_correct'].mean(),
            'mean_market_prob': test.loc[chosen, 'market_implied_prob'].mean(),
        })

    alpha_curve_rows = []
    for top_frac in [0.50, 0.40, 0.30, 0.20, 0.10]:
        chosen = final_alpha >= np.quantile(final_alpha, 1 - top_frac)
        alpha_curve_rows.append({
            'Top fraction': f'Top {int(top_frac * 100)}%',
            'n_rows': int(chosen.sum()),
            'mean_realized_edge': test.loc[chosen, 'realized_edge'].mean(),
            'hit_rate': test.loc[chosen, 'outcome_correct'].mean(),
            'mean_market_prob': test.loc[chosen, 'market_implied_prob'].mean(),
        })

    return pd.DataFrame(selection_rows), pd.DataFrame(alpha_curve_rows)


def feature_importance(final_model, top_n=15):
    importance = pd.Series(
        final_model.feature_importances_,
        index=CONTEXT_ONLY_FEATURES
    ).sort_values().tail(top_n)
    return importance


def robustness(train_pool, late_pool, tuned_params, seeds=(11, 42, 99)):
    late_wallet_counts = late_pool['wallet'].value_counts()
    eligible = late_wallet_counts[late_wallet_counts >= HOLDOUT_MIN_TRADES].index.to_numpy()
    n_holdout = max(HOLDOUT_MIN_WALLETS, int(len(eligible) * HOLDOUT_FRACTION))

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample_wallets = set(rng.choice(eligible, size=n_holdout, replace=False))

        robust_train = train_pool[~train_pool['wallet'].isin(sample_wallets)].copy().reset_index(drop=True)
        robust_test = late_pool[late_pool['wallet'].isin(sample_wallets)].copy().reset_index(drop=True)

        medians = robust_train[FULL_COLDSTART_FEATURES].median(numeric_only=True)
        for frame in [robust_train, robust_test]:
            for col in FULL_COLDSTART_FEATURES:
                if col in frame.columns:
                    frame[col] = frame[col].fillna(medians.get(col, 0))

        X_fit = robust_train[CONTEXT_ONLY_FEATURES].fillna(0)
        y_fit = robust_train['outcome_correct'].astype(int)
        X_eval = robust_test[CONTEXT_ONLY_FEATURES].fillna(0)
        y_eval = robust_test['outcome_correct'].astype(int)

        robust_params = tuned_params.copy()
        robust_params['random_state'] = seed
        robust_model = xgb.XGBClassifier(**robust_params)
        robust_model.fit(X_fit, y_fit, verbose=False)
        robust_probs = robust_model.predict_proba(X_eval)[:, 1]
        robust_market = robust_test['market_implied_prob'].to_numpy()
        alpha_stats = alpha_slice(robust_test, robust_probs, top_frac=0.10)

        rows.append({
            'seed': seed,
            'cold_wallets': len(sample_wallets),
            'test_rows': len(robust_test),
            'market_roc_auc': score_binary(y_eval, robust_market)['roc_auc'],
            'model_roc_auc': score_binary(y_eval, robust_probs)['roc_auc'],
            'market_pr_auc': score_binary(y_eval, robust_market)['pr_auc'],
            'model_pr_auc': score_binary(y_eval, robust_probs)['pr_auc'],
            'market_brier': score_binary(y_eval, robust_market)['brier'],
            'model_brier': score_binary(y_eval, robust_probs)['brier'],
            'top10_realized_edge': alpha_stats['mean_realized_edge'],
            'top10_hit_rate': alpha_stats['hit_rate'],
        })

    robust_df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        'mean_market_roc_auc': robust_df['market_roc_auc'].mean(),
        'mean_model_roc_auc': robust_df['model_roc_auc'].mean(),
        'mean_market_pr_auc': robust_df['market_pr_auc'].mean(),
        'mean_model_pr_auc': robust_df['model_pr_auc'].mean(),
        'mean_market_brier': robust_df['market_brier'].mean(),
        'mean_model_brier': robust_df['model_brier'].mean(),
        'mean_top10_realized_edge': robust_df['top10_realized_edge'].mean(),
        'mean_top10_hit_rate': robust_df['top10_hit_rate'].mean(),
    }])
    return robust_df, summary


def platt_calibrate(final_model, X_train, y_train, X_test, y_test, final_probs):
    calibrated = CalibratedClassifierCV(final_model, cv='prefit', method='sigmoid')
    calibrated.fit(X_train, y_train)
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]

    report = pd.DataFrame([{
        'actual_positive_rate': y_test.mean(),
        'mean_pred_raw': final_probs.mean(),
        'mean_pred_calibrated': calibrated_probs.mean(),
        'roc_auc_raw': roc_auc_score(y_test, final_probs),
        'roc_auc_calibrated': roc_auc_score(y_test, calibrated_probs),
        'pr_auc_raw': average_precision_score(y_test, final_probs),
        'pr_auc_calibrated': average_precision_score(y_test, calibrated_probs),
    }])
    return calibrated_probs, report


def _metric_uplifts(y, market, model):
    return {
        'ROC-AUC uplift': roc_auc_score(y, model) - roc_auc_score(y, market),
        'PR-AUC uplift': average_precision_score(y, model) - average_precision_score(y, market),
        'Brier improvement': brier_score_loss(y, market) - brier_score_loss(y, model),
        'Log-loss improvement': log_loss(y, market) - log_loss(y, model),
    }


def bootstrap_uplift(cold_test, y_test, market_probs, final_probs, B=3000, seed=2026):
    infer_df = cold_test[['wallet']].copy().reset_index(drop=True)
    infer_df['y'] = np.asarray(y_test, dtype=int)
    infer_df['market'] = np.clip(np.asarray(market_probs, dtype=float), 1e-6, 1 - 1e-6)
    infer_df['model'] = np.clip(np.asarray(final_probs, dtype=float), 1e-6, 1 - 1e-6)

    observed = _metric_uplifts(
        infer_df['y'].to_numpy(),
        infer_df['market'].to_numpy(),
        infer_df['model'].to_numpy(),
    )

    wallet_ids = infer_df['wallet'].astype(str).unique()
    wallet_index = {
        w: infer_df.index[infer_df['wallet'].astype(str) == w].to_numpy()
        for w in wallet_ids
    }

    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(B):
        sampled = rng.choice(wallet_ids, size=len(wallet_ids), replace=True)
        idx = np.concatenate([wallet_index[w] for w in sampled])
        y_b = infer_df.loc[idx, 'y'].to_numpy()
        market_b = infer_df.loc[idx, 'market'].to_numpy()
        model_b = infer_df.loc[idx, 'model'].to_numpy()
        if np.unique(y_b).size < 2:
            continue
        boot_rows.append(_metric_uplifts(y_b, market_b, model_b))

    boot_df = pd.DataFrame(boot_rows)
    inference_rows = []
    for metric_name, observed_value in observed.items():
        lo, hi = boot_df[metric_name].quantile([0.025, 0.975])
        p = (1 + (boot_df[metric_name] <= 0).sum()) / (len(boot_df) + 1)
        inference_rows.append({
            'metric': metric_name,
            'observed_uplift': observed_value,
            'ci_95_low': lo,
            'ci_95_high': hi,
            'bootstrap_p_one_sided': p,
        })
    return pd.DataFrame(inference_rows)


def _safe_score_binary(y_true, probs):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        'roc_auc': np.nan,
        'pr_auc': np.nan,
        'brier': brier_score_loss(y_true, probs),
        'logloss': log_loss(y_true, probs),
    }
    if np.unique(y_true).size >= 2:
        out['roc_auc'] = roc_auc_score(y_true, probs)
        out['pr_auc'] = average_precision_score(y_true, probs)
    return out


def _weighted_mean_nonnull(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def walkforward(fm, tuned_params, min_live_rows=50):
    entry_cutoff = fm['entry_date'].quantile(0.70)
    eval_months = sorted(
        fm.loc[fm['entry_date'] > entry_cutoff, 'entry_date']
          .dt.to_period('M')
          .astype(str)
          .unique()
    )

    rows = []
    for month in eval_months:
        month_start = pd.Timestamp(f'{month}-01').tz_localize('UTC')
        month_end = month_start + pd.offsets.MonthBegin(1)

        hist = fm[fm['entry_date'] < month_start].copy().reset_index(drop=True)
        month_slice = fm[
            (fm['entry_date'] >= month_start) & (fm['entry_date'] < month_end)
        ].copy().reset_index(drop=True)

        if hist.empty or month_slice.empty:
            continue

        month_slice = month_slice[
            ~month_slice['wallet'].isin(set(hist['wallet']))
        ].copy().reset_index(drop=True)

        if len(month_slice) < min_live_rows:
            continue

        medians = hist[FULL_COLDSTART_FEATURES].median(numeric_only=True)
        for frame in [hist, month_slice]:
            for col in FULL_COLDSTART_FEATURES:
                if col in frame.columns:
                    frame[col] = frame[col].fillna(medians.get(col, 0))

        X_hist = build_matrix(hist, CONTEXT_ONLY_FEATURES)
        y_hist = hist['outcome_correct'].astype(int)
        X_month = build_matrix(month_slice, CONTEXT_ONLY_FEATURES)
        y_month = month_slice['outcome_correct'].astype(int)

        live_model = xgb.XGBClassifier(**tuned_params)
        live_model.fit(X_hist, y_hist, verbose=False)
        live_probs = live_model.predict_proba(X_month)[:, 1]
        live_market = month_slice['market_implied_prob'].to_numpy()

        model_scores = _safe_score_binary(y_month, live_probs)
        market_scores = _safe_score_binary(y_month, live_market)
        alpha_stats = alpha_slice(month_slice, live_probs, top_frac=0.10)

        rows.append({
            'entry_month': month,
            'rows': len(month_slice),
            'wallets': month_slice['wallet'].nunique(),
            'markets': month_slice['condition_id'].nunique(),
            'market_roc_auc': market_scores['roc_auc'],
            'model_roc_auc': model_scores['roc_auc'],
            'market_pr_auc': market_scores['pr_auc'],
            'model_pr_auc': model_scores['pr_auc'],
            'market_brier': market_scores['brier'],
            'model_brier': model_scores['brier'],
            'market_logloss': market_scores['logloss'],
            'model_logloss': model_scores['logloss'],
            'top10_realized_edge': alpha_stats['mean_realized_edge'],
            'top10_hit_rate': alpha_stats['hit_rate'],
        })

    walk_df = pd.DataFrame(rows)
    weights = walk_df['rows'].to_numpy() if len(walk_df) else np.array([])

    summary = pd.DataFrame([{
        'months_evaluated': len(walk_df),
        'rows_evaluated': int(walk_df['rows'].sum()) if len(walk_df) else 0,
        'weighted_market_roc_auc': _weighted_mean_nonnull(walk_df['market_roc_auc'], weights) if len(walk_df) else np.nan,
        'weighted_model_roc_auc': _weighted_mean_nonnull(walk_df['model_roc_auc'], weights) if len(walk_df) else np.nan,
        'weighted_market_pr_auc': _weighted_mean_nonnull(walk_df['market_pr_auc'], weights) if len(walk_df) else np.nan,
        'weighted_model_pr_auc': _weighted_mean_nonnull(walk_df['model_pr_auc'], weights) if len(walk_df) else np.nan,
        'weighted_market_brier': _weighted_mean_nonnull(walk_df['market_brier'], weights) if len(walk_df) else np.nan,
        'weighted_model_brier': _weighted_mean_nonnull(walk_df['model_brier'], weights) if len(walk_df) else np.nan,
        'weighted_market_logloss': _weighted_mean_nonnull(walk_df['market_logloss'], weights) if len(walk_df) else np.nan,
        'weighted_model_logloss': _weighted_mean_nonnull(walk_df['model_logloss'], weights) if len(walk_df) else np.nan,
        'weighted_top10_realized_edge': _weighted_mean_nonnull(walk_df['top10_realized_edge'], weights) if len(walk_df) else np.nan,
        'weighted_top10_hit_rate': _weighted_mean_nonnull(walk_df['top10_hit_rate'], weights) if len(walk_df) else np.nan,
    }])

    if len(walk_df):
        summary['roc_auc_uplift'] = summary['weighted_model_roc_auc'] - summary['weighted_market_roc_auc']
        summary['pr_auc_uplift'] = summary['weighted_model_pr_auc'] - summary['weighted_market_pr_auc']
        summary['brier_improvement'] = summary['weighted_market_brier'] - summary['weighted_model_brier']
        summary['logloss_improvement'] = summary['weighted_market_logloss'] - summary['weighted_model_logloss']

    return walk_df, summary