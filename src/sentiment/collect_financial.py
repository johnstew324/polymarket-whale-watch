# collects financial market signals for each Polymarket market
# maps NER keywords -> relevant tickers via KEYWORD_TO_TICKERS in pipeline_config.py
# pulls daily OHLCV + computes abnormal returns (z-score vs 20-day rolling baseline)
# output:
# financial_signals_weekly.csv - weekly aggregated, aligns with sentiment week_start
#
# weekly schema:
# condition_id, ticker, week_start,
# weekly_return,        (compounded weekly price return)
# mean_z_score,         (mean daily z_score across week)
# max_abs_z_score,      (max |z_score| — captures biggest single day move)
# trading_days,         (days of data in this week)
# vix_close,            (mean VIX close that week — global fear baseline)
# vix_z_score           (mean VIX z_score that week — was volatility itself abnormal)

from pathlib import Path

import pandas as pd
import duckdb
import yfinance as yf
from src.sentiment.ner_keywords import extract_keywords
from src.sentiment.pipeline_config import KEYWORD_TO_TICKERS

DB = Path("data/analytical/polymarket.ddb")
OUT_DIR    = Path("data/processed")
OUT_WEEKLY = OUT_DIR / "financial" / "financial_signals_weekly.csv"

# rolling window for abnormal return baseline (trading days)
ROLLING_WINDOW = 20

# context days before market start to pull for rolling baseline
CONTEXT_DAYS = 30


def get_tickers_for_market(question):
    keywords = extract_keywords(question)
    tickers = []
    seen = set()
    for kw in keywords:
        for ticker in KEYWORD_TO_TICKERS.get(kw, []):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def fetch_ticker_data(ticker, start, end):
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"  warning: could not fetch {ticker} — {e}")
        return None


def compute_abnormal_returns(df):
    df = df.sort_values("date").copy()
    df["pct_change"]      = df["close"].pct_change()
    df["rolling_mean"]    = df["pct_change"].rolling(ROLLING_WINDOW, min_periods=5).mean()
    df["rolling_std"]     = df["pct_change"].rolling(ROLLING_WINDOW, min_periods=5).std()
    df["abnormal_return"] = df["pct_change"] - df["rolling_mean"]
    df["z_score"]         = df["abnormal_return"] / df["rolling_std"].clip(lower=1e-8)
    return df


def to_weekly(daily_df):
    df = daily_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True)
    elif df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")

    df["week_start"] = df["date"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.tz_localize("UTC")
    )

    weekly = (
        df.groupby(["condition_id", "ticker", "week_start"])
        .agg(
            weekly_return   =("pct_change",  lambda x: (1 + x).prod() - 1),
            mean_z_score    =("z_score",     "mean"),
            max_abs_z_score =("z_score",     lambda x: x.abs().max()),
            trading_days    =("z_score",     "count"),
        )
        .reset_index()
    )

    return weekly


print("Loading markets from DuckDB...")
con = duckdb.connect(str(DB), read_only=True)
markets = con.execute("""
    SELECT m.conditionId, m.question, m.startDate, m.endDate
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (
        SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId
    )
    ORDER BY m.startDate
""").fetchdf()
con.close()
print(f"  {len(markets)} markets loaded")


#  build ticker -> date range map (fetch each ticker once) 
print("\nBuilding ticker -> date range map...")

ticker_date_ranges = {}

for market in markets.itertuples():
    tickers = get_tickers_for_market(market.question)
    if not tickers:
        continue

    start = pd.Timestamp(market.startDate, tz="UTC") if pd.notna(market.startDate) else None
    end   = pd.Timestamp(market.endDate,   tz="UTC") if pd.notna(market.endDate)   else None
    if start is None or end is None:
        continue

    fetch_start = start - pd.Timedelta(days=CONTEXT_DAYS)

    for ticker in tickers:
        if ticker not in ticker_date_ranges:
            ticker_date_ranges[ticker] = [fetch_start, end]
        else:
            ticker_date_ranges[ticker][0] = min(ticker_date_ranges[ticker][0], fetch_start)
            ticker_date_ranges[ticker][1] = max(ticker_date_ranges[ticker][1], end)

print(f"  {len(ticker_date_ranges)} unique tickers to fetch")


# fetch VIX once across full market span 
print("\nFetching VIX global baseline...")

all_starts  = pd.to_datetime(markets["startDate"], utc=True)
all_ends    = pd.to_datetime(markets["endDate"],   utc=True)
vix_start   = all_starts.min() - pd.Timedelta(days=CONTEXT_DAYS)
vix_end     = all_ends.max()

vix_raw = fetch_ticker_data("^VIX", vix_start.date(), vix_end.date())
if vix_raw is not None:
    vix_raw = compute_abnormal_returns(vix_raw)
    vix_raw["date"] = pd.to_datetime(vix_raw["date"], utc=True)
    vix_raw["week_start"] = vix_raw["date"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.tz_localize("UTC")
    )
    vix_weekly = (
        vix_raw.groupby("week_start")
        .agg(
            vix_close  =("close",   "mean"),
            vix_z_score=("z_score", "mean"),
        )
        .reset_index()
    )
    vix_weekly["week_start_str"] = vix_weekly["week_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  VIX fetched: {len(vix_weekly)} weeks")
else:
    vix_weekly = None
    print("  warning: VIX fetch failed — vix_close and vix_z_score will be null")


#  fetch all keyword tickers 
print("\nFetching ticker data from yfinance...")

ticker_cache = {}

for i, (ticker, (start, end)) in enumerate(ticker_date_ranges.items(), 1):
    print(f"  [{i}/{len(ticker_date_ranges)}] {ticker}  ({start.date()} -> {end.date()})")
    df = fetch_ticker_data(ticker, start.date(), end.date())
    if df is not None:
        df = compute_abnormal_returns(df)
        ticker_cache[ticker] = df

print(f"\n  {len(ticker_cache)}/{len(ticker_date_ranges)} tickers fetched")
failed = set(ticker_date_ranges) - set(ticker_cache)
if failed:
    print(f"  failed tickers: {sorted(failed)}")


#  build per-market weekly rows 
print("\nBuilding per-market financial signals...")

weekly_rows = []
empty_count = 0

for market in markets.itertuples():
    cid      = market.conditionId
    question = market.question
    start    = pd.Timestamp(market.startDate, tz="UTC") if pd.notna(market.startDate) else None
    end      = pd.Timestamp(market.endDate,   tz="UTC") if pd.notna(market.endDate)   else None

    if start is None or end is None:
        empty_count += 1
        continue

    tickers = get_tickers_for_market(question)
    if not tickers:
        empty_count += 1
        continue

    market_daily = []
    for ticker in tickers:
        if ticker not in ticker_cache:
            continue

        df = ticker_cache[ticker].copy()

        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")

        in_window = df[
            (df["date"] >= start) &
            (df["date"] <= end)
        ].copy()

        if in_window.empty:
            continue

        in_window["condition_id"] = cid
        market_daily.append(in_window[[
            "condition_id", "ticker", "date",
            "open", "high", "low", "close", "volume",
            "pct_change", "rolling_mean", "rolling_std",
            "abnormal_return", "z_score"
        ]])

    if not market_daily:
        empty_count += 1
        continue

    market_df = pd.concat(market_daily, ignore_index=True)
    weekly_df = to_weekly(market_df)
    weekly_rows.append(weekly_df)






print(f"\nMarkets with financial signals: {len(markets) - empty_count}")
print(f"Markets with no signals:        {empty_count}")
#  write output 
(OUT_DIR / "financial").mkdir(parents=True, exist_ok=True)

if weekly_rows:
    weekly_result = pd.concat(weekly_rows, ignore_index=True)
    weekly_result["week_start"] = weekly_result["week_start"].dt.strftime("%Y-%m-%d %H:%M:%S")

    for col in ["weekly_return", "mean_z_score", "max_abs_z_score"]:
        weekly_result[col] = weekly_result[col].round(6)

    # join VIX as global columns
    if vix_weekly is not None:
        weekly_result = weekly_result.merge(
            vix_weekly[["week_start_str", "vix_close", "vix_z_score"]],
            left_on="week_start",
            right_on="week_start_str",
            how="left"
        ).drop(columns=["week_start_str"])
        weekly_result["vix_close"]   = weekly_result["vix_close"].round(4)
        weekly_result["vix_z_score"] = weekly_result["vix_z_score"].round(6)

    weekly_result.to_csv(OUT_WEEKLY, index=False)
    print(f"Weekly rows written: {len(weekly_result):,}  ->  {OUT_WEEKLY}")