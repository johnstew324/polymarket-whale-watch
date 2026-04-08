"""
sentiment_timeseries.py

Scores all geopolitical posts with FinBERT and plots the rolling sentiment
time series. Extracted from TRUTH_SOCIAL_NLP.ipynb (cells 21, 23).

FinBERT loading and scoring is intentionally imported from
truth_social_collector.py — no duplication of model code.

Public API
----------
    score_all_posts(geo_df)                        -> pd.DataFrame
    plot_sentiment_timeseries(scored_df, output_path)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import FinBERT utilities from the existing collector — no duplication
from truth_social_collector import load_finbert, score_post

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SCORED_POSTS_PATH = "data/processed/truth_social_scored_posts.csv"

# Polymarket resolution dates to mark on the time series plot.
# Keys are short display labels; values are UTC timestamps.
RESOLUTION_DATES = {
    "ru_escalation\n2024-06-15":  pd.Timestamp("2024-06-15", tz="UTC"),
    "ru_ceasefire\n2024-07-01":   pd.Timestamp("2024-07-01", tz="UTC"),
    "ru_sanctions\n2025-08-05":   pd.Timestamp("2025-08-05", tz="UTC"),
    "ceasefire push\n2025-02-28": pd.Timestamp("2025-02-28", tz="UTC"),
    "peace deal\n2025-03-31":     pd.Timestamp("2025-03-31", tz="UTC"),
}


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────────────

def score_all_posts(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FinBERT over every post in geo_df and return the DataFrame with
    four new columns: net_score, positive, negative, neutral.

    Also saves the scored DataFrame to SCORED_POSTS_PATH so that
    inspect_market_posts() in truth_social_collector.py can load it
    without re-running FinBERT.

    Parameters
    ----------
    geo_df : filtered geopolitical posts (770 rows, 'text' column required)

    Returns
    -------
    geo_df with FinBERT score columns appended
    """
    tokenizer, model = load_finbert()

    print(f"Scoring {len(geo_df)} geopolitical posts with FinBERT...")
    scores = []
    for i, row in geo_df.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(geo_df)}")
        scores.append(score_post(row["text"], tokenizer, model))

    scored_df = pd.concat(
        [geo_df.reset_index(drop=True), pd.DataFrame(scores)],
        axis=1,
    )
    print("  Scoring complete.")

    os.makedirs(os.path.dirname(SCORED_POSTS_PATH), exist_ok=True)
    scored_df.to_csv(SCORED_POSTS_PATH, index=False)
    print(f"  Saved scored posts -> {SCORED_POSTS_PATH}")

    return scored_df


def plot_sentiment_timeseries(scored_df: pd.DataFrame, output_path: str) -> None:
    """
    Plot a 4-week rolling average of FinBERT net_score over time and mark
    each Polymarket resolution date as a vertical line.

    Positive net_score = constructive/bullish framing in FinBERT's sense.
    Negative = conflict/bearish framing. Green fill = positive periods,
    red fill = negative periods.

    Parameters
    ----------
    scored_df   : DataFrame with 'created_at' (UTC datetime) and 'net_score'
    output_path : path to save the PNG
    """
    df_sorted = scored_df.sort_values("created_at").copy()
    df_sorted = df_sorted.set_index("created_at")

    # Resample to weekly mean, then 4-week rolling average
    rolling_sentiment = (
        df_sorted["net_score"]
        .resample("W")
        .mean()
        .rolling(4)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        rolling_sentiment.index,
        rolling_sentiment.values,
        color="#1f4e79", linewidth=2.5,
        label="4-week rolling avg sentiment",
    )
    ax.fill_between(rolling_sentiment.index, rolling_sentiment.values, 0,
                    alpha=0.15, color="#1f4e79")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Shade positive and negative regions
    vals = rolling_sentiment.values
    ax.fill_between(rolling_sentiment.index, vals, 0,
                    where=(vals > 0), alpha=0.2, color="green",
                    label="Positive sentiment")
    ax.fill_between(rolling_sentiment.index, vals, 0,
                    where=(vals < 0), alpha=0.2, color="red",
                    label="Negative sentiment")

    # Mark Polymarket resolution dates
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.3
    for label, date in RESOLUTION_DATES.items():
        ax.axvline(date, color="darkorange", linewidth=1.2, linestyle=":", alpha=0.8)
        ax.text(
            date, y_top * 0.85, label,
            fontsize=7.5, color="darkorange",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="darkorange", alpha=0.7),
        )

    ax.set_title(
        "Trump Truth Social — Geopolitical Sentiment Over Time\n"
        "4-week rolling FinBERT net score  |  Orange lines = Polymarket resolution dates",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("FinBERT Net Score\n(+1 = positive, −1 = negative)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
