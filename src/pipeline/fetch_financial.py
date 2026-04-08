# fetches daily financial data (yfinance) for each geopolitical market
# maps markets to relevant tickers based on tags, computes % change
# in a 30-day window before resolution, saves to data/raw/financial_signals.json
# DuckDB loading is handled by load_db.py

import json
import time
from pathlib import Path
from datetime import timedelta

import duckdb
import pandas as pd
import yfinance as yf

DB  = Path("data/analytical/polymarket.ddb")
OUT = Path("data/raw/financial_signals.json")

WINDOW_DAYS = 30  # days before resolution to measure price change


# ticker mapping
# each tag maps to a list of (ticker, category) tuples
# categories: defence, oil, gold, vix


TAG_TICKER_MAP = {
    # Russia / Ukraine
    "russia":                    [("LMT", "defence"), ("RTX", "defence"), ("BZ=F", "oil"), ("^VIX", "vix")],
    "ukraine":                   [("LMT", "defence"), ("RTX", "defence"), ("BZ=F", "oil"), ("^VIX", "vix")],
    "putin":                     [("LMT", "defence"), ("RTX", "defence"), ("BZ=F", "oil")],
    "ukraine-map":               [("LMT", "defence"), ("RTX", "defence"), ("BZ=F", "oil")],
    "ukraine-peace-deal":        [("LMT", "defence"), ("^VIX", "vix")],

    # Iran / Middle East
    "iran":                      [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix"), ("LMT", "defence")],
    "us-iran":                   [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix"), ("LMT", "defence")],
    "trump-iran":                [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "khamenei":                  [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "iranian-leadership-regime": [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "israel-x-iran":             [("BZ=F", "oil"), ("GC=F", "gold"), ("LMT", "defence")],
    "nuclear":                   [("GC=F", "gold"), ("^VIX", "vix"), ("LMT", "defence")],
    "nuclear-weapons":           [("GC=F", "gold"), ("^VIX", "vix")],

    # Israel / Gaza / Lebanon / Hezbollah
    "israel":                    [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix"), ("LMT", "defence")],
    "gaza":                      [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "hamas":                     [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "hezbollah":                 [("BZ=F", "oil"), ("GC=F", "gold"), ("LMT", "defence")],
    "lebanon":                   [("BZ=F", "oil"), ("GC=F", "gold"), ("^VIX", "vix")],
    "daily-strikes":             [("BZ=F", "oil"), ("^VIX", "vix")],

    # China / Taiwan
    "china":                     [("LMT", "defence"), ("RTX", "defence"), ("BZ=F", "oil"), ("^VIX", "vix")],
    "taiwan":                    [("LMT", "defence"), ("RTX", "defence"), ("^VIX", "vix")],

    # Houthis / shipping / Hormuz / Suez
    "houthis":                   [("BZ=F", "oil"), ("^VIX", "vix")],
    "houthi":                    [("BZ=F", "oil"), ("^VIX", "vix")],
    "strait-of-hormuz":          [("BZ=F", "oil"), ("^VIX", "vix")],
    "suez-canal":                [("BZ=F", "oil")],
    "oil":                       [("BZ=F", "oil"), ("GC=F", "gold")],

    # Yemen
    "yemen":                     [("BZ=F", "oil"), ("^VIX", "vix"), ("LMT", "defence")],

    # India / Pakistan
    "india":                     [("^VIX", "vix"), ("GC=F", "gold")],
    "india-pakistan":            [("^VIX", "vix"), ("GC=F", "gold")],
    "pakistan":                  [("^VIX", "vix"), ("GC=F", "gold")],

    # North Korea
    "north-korea":               [("GC=F", "gold"), ("^VIX", "vix"), ("LMT", "defence")],

    # Venezuela / Latin America
    "venezuela":                 [("BZ=F", "oil"), ("^VIX", "vix")],
    "cartel":                    [("^VIX", "vix")],
    "mexico":                    [("^VIX", "vix")],

    # catch-alls
    "geopolitics":               [("^VIX", "vix"), ("GC=F", "gold")],
    "military-action":           [("LMT", "defence"), ("RTX", "defence"), ("^VIX", "vix")],
    "breaking-news":             [("^VIX", "vix"), ("GC=F", "gold")],
}

# specific tags processed first so they take priority over catch-alls
TAG_PRIORITY = [
    "iran", "us-iran", "trump-iran", "khamenei", "iranian-leadership-regime",
    "israel-x-iran", "nuclear", "nuclear-weapons",
    "russia", "ukraine", "putin", "ukraine-map", "ukraine-peace-deal",
    "israel", "gaza", "hamas", "hezbollah", "lebanon", "daily-strikes",
    "china", "taiwan",
    "houthis", "houthi", "strait-of-hormuz", "suez-canal", "oil",
    "yemen", "india-pakistan", "india", "pakistan", "north-korea",
    "venezuela", "cartel", "mexico",
    "military-action", "breaking-news",
    "geopolitics",  # catch-all last
]

def get_tickers_for_market(tags: list) -> list:
    """map a market's tags to (ticker, category) pairs, deduplicated."""
    seen_tickers = set()
    result = []

    ordered_tags = [t for t in TAG_PRIORITY if t in tags]
    ordered_tags += [t for t in tags if t not in TAG_PRIORITY]

    for tag in ordered_tags:
        if tag in TAG_TICKER_MAP:
            for ticker, category in TAG_TICKER_MAP[tag]:
                if ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    result.append((ticker, category))

    return result


def fetch_price_window(ticker: str, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """fetch daily OHLCV for a ticker over the given window."""
    try:
        data = yf.download(
            ticker,
            start=window_start.strftime("%Y-%m-%d"),
            end=(window_end + timedelta(days=1)).strftime("%Y-%m-%d"),  # end is exclusive in yfinance
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        return data
    except Exception as e:
        print(f"  warning: yfinance error for {ticker}: {e}")
        return pd.DataFrame()


def compute_signal(data: pd.DataFrame) -> dict:
    """compute pct_change and direction from price data. returns None if insufficient data."""
    if data.empty or len(data) < 2:
        return None

    # flatten multi-level columns that yfinance sometimes returns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].dropna()
    if len(close) < 2:
        return None

    price_start = float(close.iloc[0])
    price_end   = float(close.iloc[-1])

    if price_start == 0:
        return None

    pct_change = (price_end - price_start) / price_start * 100

    # direction: +1 = rising (market pricing in escalation/risk)
    #            -1 = falling (calm / de-escalation priced in)
    #             0 = flat (< 1% move, no meaningful signal)
    if pct_change > 1.0:
        direction = 1
    elif pct_change < -1.0:
        direction = -1
    else:
        direction = 0

    return {
        "pct_change_30d":     round(pct_change, 4),
        "avg_price":          round(float(close.mean()), 4),
        "price_start":        round(price_start, 4),
        "price_end":          round(price_end, 4),
        "direction":          direction,
        "data_coverage_days": len(close),
    }


def main():
    # load markets from DuckDB - resolved Yes/No markets that have trades
    con = duckdb.connect(str(DB))
    markets_df = con.execute("""
        SELECT
            m.conditionId  AS condition_id,
            m.closedTime   AS closed_time,
            m.tags
        FROM markets m
        INNER JOIN (SELECT DISTINCT condition_id FROM trades) t
            ON t.condition_id = m.conditionId
        WHERE m.resolvedOutcome IN ('Yes', 'No')
          AND m.closedTime IS NOT NULL
          AND m.volume > 1000
        ORDER BY m.closedTime
    """).df()
    con.close()

    print(f"processing {len(markets_df)} resolved markets with trades...")

    # cache downloaded price data - many markets share overlapping windows
    # key: (ticker, window_start_date, window_end_date)
    price_cache = {}

    records = []
    skipped_no_tickers = 0
    skipped_no_data    = 0

    for i, row in markets_df.iterrows():
        condition_id = row["condition_id"]
        closed_time  = pd.Timestamp(row["closed_time"])
        tags         = list(row["tags"]) if row["tags"] is not None else []  # ndarray -> list

        window_end   = closed_time.normalize()
        window_start = window_end - timedelta(days=WINDOW_DAYS)

        tickers = get_tickers_for_market(tags)
        if not tickers:
            skipped_no_tickers += 1
            continue

        if i % 100 == 0:
            print(f"  [{i}/{len(markets_df)}] {condition_id[:12]}... tags={tags[:3]}")

        for ticker, category in tickers:
            cache_key = (ticker, window_start.date(), window_end.date())

            if cache_key not in price_cache:
                data = fetch_price_window(ticker, window_start, window_end)
                price_cache[cache_key] = data
                time.sleep(0.1)  # light rate limiting - yfinance is unofficial
            else:
                data = price_cache[cache_key]

            signal = compute_signal(data)
            if signal is None:
                skipped_no_data += 1
                continue

            records.append({
                "condition_id": condition_id,
                "ticker":       ticker,
                "category":     category,
                "window_start": window_start.isoformat(),
                "window_end":   window_end.isoformat(),
                **signal,
            })

    print(f"\nfetched {len(records):,} signals")
    print(f"skipped {skipped_no_tickers} markets with no ticker mapping")
    print(f"skipped {skipped_no_data} (ticker, market) pairs with no price data")

    OUT.write_text(json.dumps(records, indent=2))
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()