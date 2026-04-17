from pathlib import Path
 
DB_PATH = Path('data/analytical/polymarket.ddb')
FIG_DIR = Path('figures')
SMART_WALLETS_CSV = Path('data/analytical/smart_wallets.csv')
 
# Polymarket infrastructure contracts — not traders
EXCLUDED_CONTRACTS = {
    '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',  # CTF Exchange
    '0xc5d563a36ae78145c45a50134d48a1215220f80a',  # Neg Risk CTF Exchange
}
 
# Thresholds (primary specification used throughout the report)
MIN_MARKETS = 10          # min resolved markets for hit-rate ranking
HIT_RATE_THRESHOLD = 65   # percent
VOLUME_QUANTILE = 0.75    # whale-quadrant volume threshold
EARLY_QUANTILE = 0.75     # early-accurate timing threshold
COLD_START_CUTOFF = '2025-09-01'
 
# Bootstrap
BOOTSTRAP_ITER = 3000
BOOTSTRAP_SEED = 42
 
# Shared palette (matches fignavy/figblue/figred/figgreen in report)
PALETTE = {
    'fignavy':  '#1a3a5f',
    'figblue':  '#2e6fb5',
    'figlblue': '#8ab4d8',
    'figred':   '#c0392b',
    'figgreen': '#2e7d52',
    'figgrey':  '#6b7280',
    'figmgrey': '#b8bcc4',
}
 