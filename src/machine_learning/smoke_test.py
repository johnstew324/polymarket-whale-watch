import numpy as np
import pandas as pd
from .split import make_cold_start_split
from .train import train_baselines, train_main_model, train_feature_ablation
from .evaluate import (
    warm_vs_cold, economic_ranking, robustness,
    bootstrap_uplift, walkforward, feature_importance,
)
from .config import FULL_COLDSTART_FEATURES, SEED

rng = np.random.default_rng(SEED)
n = 20000

fm = pd.DataFrame({col: rng.normal(size=n) for col in FULL_COLDSTART_FEATURES})
fm['wallet'] = [f'0x{i % 2000:040x}' for i in range(n)]
fm['condition_id'] = [f'0x{i % 500:064x}' for i in range(n)]
fm['question'] = 'fake'
fm['resolution_date'] = pd.date_range('2024-01-01', periods=n, freq='30min', tz='UTC')
fm['entry_date'] = fm['resolution_date'] - pd.Timedelta(days=1)
fm['bet_outcome'] = 'Yes'
fm['wallet_direction'] = 1
fm['market_volume'] = rng.lognormal(size=n)
fm['informed_label'] = (rng.random(n) > 0.95).astype(int)
fm['outcome_correct'] = (rng.random(n) > 0.5).astype(int)
fm['avg_price'] = rng.uniform(0.05, 0.95, size=n)
fm['market_implied_prob'] = fm['avg_price'].clip(1e-6, 1 - 1e-6)
fm['realized_edge'] = fm['outcome_correct'] - fm['market_implied_prob']

cutoff = fm['resolution_date'].quantile(0.70)
train, cold_test, warm_test, train_pool, late_pool = make_cold_start_split(fm, cutoff)

y_train = train['outcome_correct'].astype(int)
y_test = cold_test['outcome_correct'].astype(int)
market_probs = cold_test['market_implied_prob'].to_numpy()
market_probs_warm = warm_test['market_implied_prob'].to_numpy()

print('\n--- baselines ---')
baseline_df, logit_probs = train_baselines(train, cold_test, y_train, y_test, market_probs)
print(baseline_df.round(4))

print('\n--- main model (3 optuna trials) ---')
result = train_main_model(train, cold_test, warm_test, y_train, n_trials=3)
print(result['cv_comparison'].round(4))

print('\n--- feature ablation ---')
ablation_df = train_feature_ablation(train, cold_test, y_train, y_test, market_probs, result['tuned_params'])
print(ablation_df.round(4))

print('\n--- warm vs cold ---')
wc_df = warm_vs_cold(cold_test, warm_test, result['context_probs'], result['context_probs_warm'],
                     market_probs, market_probs_warm)
print(wc_df.round(4))

print('\n--- economic ranking ---')
selection_df, alpha_curve = economic_ranking(cold_test, result['context_probs'],
                                             result['div_probs'], market_probs)
print(selection_df.round(4))
print(alpha_curve.round(4))

print('\n--- feature importance (top 5) ---')
print(feature_importance(result['final_model'], top_n=5))

print('\n--- robustness (2 seeds) ---')
robust_df, robust_summary = robustness(train_pool, late_pool, result['tuned_params'], seeds=(11, 42))
print(robust_df.round(4))

print('\n--- bootstrap uplift (B=200) ---')
uplift_df = bootstrap_uplift(cold_test, y_test, market_probs, result['context_probs'], B=200)
print(uplift_df.round(4))

print('\n--- walkforward ---')
walk_df, walk_summary = walkforward(fm, result['tuned_params'], min_live_rows=10)
print(f'months evaluated: {len(walk_df)}')
if len(walk_df):
    print(walk_summary.round(4))

print('\nsmoke test passed')