import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    SEED, OPTUNA_TRIALS, XGB_CV_SPLITS, BASELINE_XGB_PARAMS,
    HISTORY_FEATURES, CONTEXT_ONLY_FEATURES, FULL_COLDSTART_FEATURES,
    DIVERGENCE_ONLY_FEATURES,
)
from .split import build_matrix, score_binary

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_baselines(train, cold_test, y_train, y_test, market_probs):
    X_train_context = build_matrix(train, CONTEXT_ONLY_FEATURES)
    X_test_context = build_matrix(cold_test, CONTEXT_ONLY_FEATURES)

    logit_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=500, random_state=SEED)),
    ])
    logit_pipe.fit(X_train_context, y_train)
    logit_probs = logit_pipe.predict_proba(X_test_context)[:, 1]

    baseline_metrics = pd.DataFrame([
        {'model': 'Market price only', **score_binary(y_test, market_probs)},
        {'model': 'Logistic (market + context)', **score_binary(y_test, logit_probs)},
    ])
    return baseline_metrics, logit_probs


def _evaluate_xgb_cv(params, X, y, groups, group_cv):
    roc_scores, pr_scores, brier_scores, logloss_scores = [], [], [], []
    for fit_idx, val_idx in group_cv.split(X, y, groups):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_fit, y_fit, verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        roc_scores.append(roc_auc_score(y_val, probs))
        pr_scores.append(average_precision_score(y_val, probs))
        brier_scores.append(brier_score_loss(y_val, probs))
        logloss_scores.append(log_loss(y_val, probs))
    return {
        'cv_roc_auc': float(np.mean(roc_scores)),
        'cv_pr_auc': float(np.mean(pr_scores)),
        'cv_brier': float(np.mean(brier_scores)),
        'cv_logloss': float(np.mean(logloss_scores)),
    }


def train_main_model(train, cold_test, warm_test, y_train, n_trials=OPTUNA_TRIALS):
    X_train_context = build_matrix(train, CONTEXT_ONLY_FEATURES)
    X_test_context = build_matrix(cold_test, CONTEXT_ONLY_FEATURES)
    X_warm_context = build_matrix(warm_test, CONTEXT_ONLY_FEATURES)
    X_train_full = build_matrix(train, FULL_COLDSTART_FEATURES)
    X_test_full = build_matrix(cold_test, FULL_COLDSTART_FEATURES)
    X_train_div = build_matrix(train, DIVERGENCE_ONLY_FEATURES)
    X_test_div = build_matrix(cold_test, DIVERGENCE_ONLY_FEATURES)

    train_groups = train['wallet'].astype(str).to_numpy()
    group_cv = GroupKFold(n_splits=XGB_CV_SPLITS)

    baseline_cv = _evaluate_xgb_cv(BASELINE_XGB_PARAMS, X_train_context, y_train, train_groups, group_cv)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 120, 360),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'subsample': trial.suggest_float('subsample', 0.65, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'eval_metric': 'logloss',
            'random_state': SEED,
            'n_jobs': -1,
        }
        scores = _evaluate_xgb_cv(params, X_train_context, y_train, train_groups, group_cv)
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        return 0.5 * (scores['cv_roc_auc'] + scores['cv_pr_auc']) - scores['cv_brier']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    tuned_params = study.best_params.copy()
    tuned_params.update({'eval_metric': 'logloss', 'random_state': SEED, 'n_jobs': -1})
    tuned_cv = study.best_trial.user_attrs.copy()

    # Fit the three variants on all training data
    baseline_context_model = xgb.XGBClassifier(**BASELINE_XGB_PARAMS)
    baseline_context_model.fit(X_train_context, y_train, verbose=False)
    baseline_context_probs = baseline_context_model.predict_proba(X_test_context)[:, 1]

    context_model = xgb.XGBClassifier(**tuned_params)
    context_model.fit(X_train_context, y_train, verbose=False)
    context_probs = context_model.predict_proba(X_test_context)[:, 1]
    context_probs_warm = context_model.predict_proba(X_warm_context)[:, 1]

    full_model = xgb.XGBClassifier(**tuned_params)
    full_model.fit(X_train_full, y_train, verbose=False)
    full_probs = full_model.predict_proba(X_test_full)[:, 1]

    div_model = xgb.XGBClassifier(**tuned_params)
    div_model.fit(X_train_div, y_train, verbose=False)
    div_probs = div_model.predict_proba(X_test_div)[:, 1]

    cv_comparison = pd.DataFrame([
        {'model': 'Baseline XGBoost (market + context)', **baseline_cv},
        {'model': 'Tuned XGBoost (market + context)', **tuned_cv},
    ])

    print('Primary model retained: Tuned XGBoost (market + context)')
    print('Best tuned params:', study.best_params)

    return {
        'final_model': context_model,
        'tuned_params': tuned_params,
        'study': study,
        'cv_comparison': cv_comparison,
        'baseline_context_probs': baseline_context_probs,
        'context_probs': context_probs,
        'context_probs_warm': context_probs_warm,
        'full_probs': full_probs,
        'div_probs': div_probs,
    }


def train_feature_ablation(train, cold_test, y_train, y_test, market_probs, tuned_params):
    history_plus_price = HISTORY_FEATURES + ['avg_price']
    context_plus_price = [f for f in CONTEXT_ONLY_FEATURES if f != 'avg_price'] + ['avg_price']

    rows = []
    for label, features in [
        ('Market price only', ['avg_price']),
        ('History only', history_plus_price),
        ('Context only', context_plus_price),
        ('Full cold-start set', FULL_COLDSTART_FEATURES),
    ]:
        if label == 'Market price only':
            probs = market_probs
        else:
            model = xgb.XGBClassifier(**tuned_params)
            model.fit(train[features].fillna(0), y_train, verbose=False)
            probs = model.predict_proba(cold_test[features].fillna(0))[:, 1]
        scores = score_binary(y_test, probs)
        rows.append({
            'model': label,
            'n_features': len(features),
            **scores,
        })
    return pd.DataFrame(rows)