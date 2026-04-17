from pathlib import Path

from .features.build import build_feature_matrix
from .split import make_cold_start_split
from .train import train_baselines, train_main_model, train_feature_ablation
from .evaluate import (
    warm_vs_cold, economic_ranking, robustness, bootstrap_uplift,
    walkforward, feature_importance,
)
from .deployment import (
    wallet_scope_audit, friction_analysis, risk_metrics, aum_scaling_table,
)
from .plots import (
    plot_eda_diagnostics, plot_evaluation_grid,
    plot_feature_importance, plot_risk_profile,
)

RESULTS_DIR = Path('results/ml')


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build
    fm, cutoff = build_feature_matrix()
    plot_eda_diagnostics(fm)

    # Split
    train, cold_test, warm_test, train_pool, late_pool = make_cold_start_split(fm, cutoff)
    y_train = train['outcome_correct'].astype(int)
    y_test = cold_test['outcome_correct'].astype(int)
    market_probs = cold_test['market_implied_prob'].to_numpy()
    market_probs_warm = warm_test['market_implied_prob'].to_numpy()

    scope_df, overlap = wallet_scope_audit(train, cold_test, warm_test)
    scope_df.to_csv(RESULTS_DIR / 'wallet_scope.csv', index=False)
    print(f'Wallet overlap: {overlap}')

    # Train
    baseline_df, logit_probs = train_baselines(train, cold_test, y_train, y_test, market_probs)
    baseline_df.to_csv(RESULTS_DIR / 'baselines.csv', index=False)

    result = train_main_model(train, cold_test, warm_test, y_train)
    result['cv_comparison'].to_csv(RESULTS_DIR / 'cv_comparison.csv', index=False)

    ablation_df = train_feature_ablation(
        train, cold_test, y_train, y_test, market_probs, result['tuned_params']
    )
    ablation_df.to_csv(RESULTS_DIR / 'feature_ablation.csv', index=False)

    # Evaluate
    wc_df = warm_vs_cold(
        cold_test, warm_test,
        result['context_probs'], result['context_probs_warm'],
        market_probs, market_probs_warm,
    )
    wc_df.to_csv(RESULTS_DIR / 'warm_vs_cold.csv', index=False)

    selection_df, alpha_curve = economic_ranking(
        cold_test, result['context_probs'], result['div_probs'], market_probs,
    )
    selection_df.to_csv(RESULTS_DIR / 'economic_ranking.csv', index=False)
    alpha_curve.to_csv(RESULTS_DIR / 'alpha_curve.csv', index=False)

    robust_df, robust_summary = robustness(train_pool, late_pool, result['tuned_params'])
    robust_df.to_csv(RESULTS_DIR / 'robustness.csv', index=False)
    robust_summary.to_csv(RESULTS_DIR / 'robustness_summary.csv', index=False)

    plot_evaluation_grid(
        y_test, market_probs, logit_probs,
        result['baseline_context_probs'], result['context_probs'],
        alpha_curve, selection_df, robust_df,
    )
    plot_feature_importance(result['final_model'])

    uplift_df = bootstrap_uplift(cold_test, y_test, market_probs, result['context_probs'])
    uplift_df.to_csv(RESULTS_DIR / 'bootstrap_uplift.csv', index=False)

    walk_df, walk_summary = walkforward(fm, result['tuned_params'])
    walk_df.to_csv(RESULTS_DIR / 'walkforward.csv', index=False)
    walk_summary.to_csv(RESULTS_DIR / 'walkforward_summary.csv', index=False)

    # Deployment
    friction_df, friction_coverage = friction_analysis(
        cold_test, result['context_probs'], market_probs,
    )
    friction_df.to_csv(RESULTS_DIR / 'friction.csv', index=False)
    friction_coverage.to_csv(RESULTS_DIR / 'friction_coverage.csv', index=False)

    risk_summary, risk_series = risk_metrics(
        cold_test, result['context_probs'], market_probs,
    )
    risk_summary.to_csv(RESULTS_DIR / 'risk_summary.csv', index=False)
    plot_risk_profile(risk_series)

    aum_header, aum_tiers = aum_scaling_table()
    aum_header.to_csv(RESULTS_DIR / 'aum_header.csv', index=False)
    aum_tiers.to_csv(RESULTS_DIR / 'aum_tiers.csv', index=False)

    print(f'\nAll outputs written to {RESULTS_DIR}/ and figures/')


if __name__ == '__main__':
    main()