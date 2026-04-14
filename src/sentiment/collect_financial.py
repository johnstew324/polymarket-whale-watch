# collects financial market signals for each Polymarket market
# maps NER keywords -> relevant tickers via TOPIC_TO_TICKERS in pipeline_config.py
# pulls daily OHLCV + computes abnormal returns (z-score vs 20-day rolling baseline)
# output:
# financial_signals_weekly.csv - weekly aggregated, aligns with sentiment week_start
#
# daily output is commented out for now
# for per-trade feature engineering (7-day windows around specific trade timestamps)
#
# weekly schema:
# condition_id, ticker, week_start,
# weekly_return,        (close Friday / close Monday - 1)
# mean_z_score,         (mean daily z_score across week)
# max_abs_z_score,      (max |z_score|  captures biggest single day move)
# trading_days          (days of data in this week)


from pathlib import Path

import pandas as pd
import duckdb
import yfinance as yf
from src.sentiment.ner_keywords import extract_keywords
from src.sentiment.pipeline_config import TOPIC_TO_TICKERS

DB       = Path("data/analytical/polymarket.ddb")
OUT_DIR    = Path("data/processed")
# OUT_DAILY  = OUT_DIR / "financial_signals_daily.csv"  
OUT_WEEKLY = OUT_DIR / "financial_signals_weekly.csv"

# rolling window for abnormal return baseline (trading days)
ROLLING_WINDOW = 20

# context days before market start to pull for rolling baseline
CONTEXT_DAYS = 30


def get_tickers_for_market(question):
    keywords = extract_keywords(question)
    tickers = []
    seen = set()
    for kw in keywords:
        for ticker in TOPIC_TO_TICKERS.get(kw, []):
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

        # flatten multi-level columns if yfinance returns them
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
    # ensure date is datetime
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
            weekly_return   =("pct_change",  lambda x: (1 + x).prod() - 1),  # compound weekly return
            mean_z_score    =("z_score",     "mean"),
            max_abs_z_score =("z_score",     lambda x: x.abs().max()),
            trading_days    =("z_score",     "count"),
        )
        .reset_index()
    )

    return weekly


#  load markets 
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


#  build ticker  date range map (fetch each ticker once)
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


# fetch all tickers
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


#  build per-market daily rows 
print("\nBuilding per-market financial signals...")

daily_rows  = []  # 
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

        # ensure tz-aware for comparison
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")

        # market window only  not the context days used for baseline
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

    # combine tickers for this market
    market_df = pd.concat(market_daily, ignore_index=True)
    # daily_rows.append(market_df)

    # aggregate to weekly
    weekly_df = to_weekly(market_df)
    weekly_rows.append(weekly_df)


print(f"\nMarkets with financial signals: {len(markets) - empty_count}")
print(f"Markets with no signals: {empty_count}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# daily output 
# if daily_rows:
#     daily_result = pd.concat(daily_rows, ignore_index=True)
#     daily_result["date"] = daily_result["date"].dt.strftime("%Y-%m-%d")
#     for col in ["open", "high", "low", "close", "pct_change",
#                 "rolling_mean", "rolling_std", "abnormal_return", "z_score"]:
#         daily_result[col] = daily_result[col].round(6)
#     daily_result.to_csv(OUT_DAILY, index=False)
#     print(f"Daily  rows written: {len(daily_result):,} ->   {OUT_DAILY}")

if weekly_rows:
    weekly_result = pd.concat(weekly_rows, ignore_index=True)
    weekly_result["week_start"] = weekly_result["week_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["weekly_return", "mean_z_score", "max_abs_z_score"]:
        weekly_result[col] = weekly_result[col].round(6)
    weekly_result.to_csv(OUT_WEEKLY, index=False)
    print(f"Weekly rows written: {len(weekly_result):,}  ->  {OUT_WEEKLY}")
