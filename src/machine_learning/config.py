from pathlib import Path

SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────
DB_PATH = Path('data/analytical/polymarket.ddb')
FIN_PATH = Path('data/processed/financial_signals.csv')  # restore when CSV is back

# ── Market filter ──────────────────────────────────────────────────────────
GEO_TAGS = {'geopolitics', 'global-politics', 'politics'}

# ── Wallet-edge thresholds ─────────────────────────────────────────────────
MIN_BETS_FOR_EDGE = 30      # min resolved bets before a wallet gets an edge label
MIN_PRIOR_BETS = 5          # min history before a row is kept for modelling
CUTOFF_QUANTILE = 0.70      # resolution-date quantile used to split train/test

# ── Cold-start holdout ─────────────────────────────────────────────────────
HOLDOUT_MIN_TRADES = 3      # min late-period trades before a wallet is holdout-eligible
HOLDOUT_FRACTION = 0.20     # share of eligible late wallets sampled into cold_test
HOLDOUT_MIN_WALLETS = 500   # floor on cold_test wallet count

# ── XGBoost ────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 20
XGB_CV_SPLITS = 4

BASELINE_XGB_PARAMS = {
    'n_estimators': 250,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'logloss',
    'random_state': SEED,
    'n_jobs': -1,
}

# ── Deployment / frictions ─────────────────────────────────────────────────
ALPHA_QUANTILE = 0.90            # top-10% slice threshold
MIN_MARKET_VOLUME_USD = 10_000.0 # exclude markets below this liquidity
POSITION_CAP_USD = 500.0         # hard per-ticket cap
MAX_MARKET_VOLUME_SHARE = 0.005  # max share of market volume per trade
COST_BPS_GRID = [0, 25, 50, 100, 150]

# ── Final feature-matrix columns ───────────────────────────────────────────
FM_COLS = [
    'wallet', 'condition_id', 'question', 'resolution_date', 'entry_date',
    'net_usd', 'log_net_usd', 'avg_price', 'num_trades', 'hours_before', 'hit_rate_pct',
    'log_market_volume', 'z_score_roll', 'raw_edge_roll', 'n_prior_bets',
    'ts_weekly_score', 'ts_weekly_post_count', 'ts_weekly_volatility',
    'ts_weekly_divergence', 'ts_market_score', 'ts_market_direction',
    'ts_market_divergence', 'ts_market_has_coverage', 'ts_has_coverage',
    'divergence_score',
     'BZ=F_7d', 'GC=F_7d', 'LMT_7d', 'RTX_7d', '^VIX_7d',  # restore when financial_signals.csv is back
    'ft_sentiment_score_z', 'ft_sentiment_direction', 'ft_sentiment_volatility',
    'ft_post_count', 'ft_has_coverage', 'ft_divergence', 'informed_label',
    'outcome_correct', 'bet_outcome', 'wallet_direction', 'market_volume'
]

# Columns dropped from modelling (identifiers, labels, leakage)
DROP_COLS = [
    'outcome_correct', 'bet_outcome', 'wallet_direction',
    'wallet', 'condition_id', 'question', 'resolution_date', 'entry_date',
    'market_volume',
]

# ── Feature families ───────────────────────────────────────────────────────
HISTORY_FEATURES = ['hit_rate_pct', 'z_score_roll', 'raw_edge_roll', 'n_prior_bets']

PRICE_PROXY_FEATURES = ['avg_price', 'hours_before', 'log_hours_before', 'num_trades']

FULL_COLDSTART_FEATURES = [
    'net_usd', 'log_net_usd', 'avg_price', 'num_trades', 'hours_before', 'hit_rate_pct',
    'log_market_volume', 'z_score_roll', 'raw_edge_roll', 'n_prior_bets',
    'ts_weekly_score', 'ts_weekly_post_count', 'ts_weekly_volatility',
    'ts_weekly_divergence', 'ts_market_score', 'ts_market_direction',
    'ts_market_divergence', 'ts_market_has_coverage', 'ts_has_coverage',
    'divergence_score',
 'BZ=F_7d', 'GC=F_7d', 'LMT_7d', 'RTX_7d', '^VIX_7d',  # restore when financial_signals.csv is back
    'ft_sentiment_score_z', 'ft_sentiment_direction', 'ft_sentiment_volatility',
    'ft_post_count', 'ft_has_coverage', 'ft_divergence',
]

CONTEXT_ONLY_FEATURES = [f for f in FULL_COLDSTART_FEATURES if f not in HISTORY_FEATURES]

DIVERGENCE_ONLY_FEATURES = [f for f in CONTEXT_ONLY_FEATURES if f not in PRICE_PROXY_FEATURES]