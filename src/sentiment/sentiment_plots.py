# -*- coding: utf-8 -*-
# visualises FinBERT sentiment output across ProQuest article corpora
# produces three plots per topic: article volume, sentiment distribution, sentiment timeline
#
# requires finbert_scorer.py to have been run first
#
# input:  data/processed/proquest/{topic}_articles_scored.csv
# output: data/processed/plots/{topic}_01_article_volume.png
#         data/processed/plots/{topic}_02_sentiment_histogram.png
#         data/processed/plots/{topic}_03_sentiment_timeline.png
#
# usage:
#   python -m src.sentiment.sentiment_plots                       # all topics
#   python -m src.sentiment.sentiment_plots --topic iran_israel   # one topic


import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import warnings
from pathlib import Path

from src.sentiment.pipeline_config import TOPICS

warnings.filterwarnings("ignore")

PQ_DIR    = Path("data/processed/proquest")
PLOTS_DIR = Path("data/processed/plots")


def load_scored(topic):
    path = PQ_DIR / f"{topic}_articles_scored.csv"

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "net_score"])

    if len(df) < 10:
        print(f"  [{topic}] only {len(df)} articles after cleaning — skipping")
        return None

    return df


def plot_article_volume(topic, df, out_dir):
    df_indexed = df.set_index("date")
    monthly = df_indexed.resample("ME").size()

    zero_months = monthly[monthly == 0]
    if not zero_months.empty:
        print(f"  WARNING — months with 0 articles:")
        for m in zero_months.index:
            print(f"    {m.strftime('%b %Y')}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.8)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(f"Article Volume by Month — {topic.replace('_', ' ').title()}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Article Count")
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{topic}_01_article_volume.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


def plot_sentiment_histogram(topic, df, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["net_score"], bins=40, ax=axs[0], color="steelblue", alpha=0.7)
    axs[0].axvline(df["net_score"].mean(), color="blue", linestyle="--",
                   linewidth=1.5, label=f"Mean = {df['net_score'].mean():.3f}")
    axs[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    axs[0].set_title(f"Net Score Distribution — {topic.replace('_', ' ').title()}")
    axs[0].set_xlabel("Net Score (positive − negative)")
    axs[0].set_ylabel("Article Count")
    axs[0].legend()

    sns.histplot(df["positive"], bins=40, ax=axs[1], color="green", alpha=0.5, label="Positive")
    sns.histplot(df["negative"], bins=40, ax=axs[1], color="red",   alpha=0.5, label="Negative")
    sns.histplot(df["neutral"],  bins=40, ax=axs[1], color="grey",  alpha=0.5, label="Neutral")
    axs[1].set_title(f"FinBERT Probabilities — {topic.replace('_', ' ').title()}")
    axs[1].set_xlabel("Probability (0–1)")
    axs[1].set_ylabel("Article Count")
    axs[1].legend()

    plt.tight_layout()
    out_path = out_dir / f"{topic}_02_sentiment_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


def plot_sentiment_timeline(topic, df, out_dir):
    df_indexed = df.set_index("date")
    weekly = df_indexed.resample("W")["net_score"].mean().dropna()
    colours = ["green" if v > 0 else "red" for v in weekly.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(weekly.index, weekly.values, color=colours, width=5, alpha=0.8)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(f"Weekly Average FinBERT Sentiment — {topic.replace('_', ' ').title()}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean Net Score (positive − negative)")
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{topic}_03_sentiment_timeline.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


def run_topic(topic, out_dir):
    df = load_scored(topic)
    if df is None:
        return

    plot_article_volume(topic, df, out_dir)
    plot_sentiment_histogram(topic, df, out_dir)
    plot_sentiment_timeline(topic, df, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Sentiment visualisation plots for ProQuest corpora")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Single topic to plot. Omit to run all topics."
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.topic:
        topic = args.topic.lower()
        if topic not in TOPICS:
            print(f"Unknown topic '{topic}'. Available: {TOPICS}")
            return
        run_topic(topic, PLOTS_DIR)
    else:
        print(f"Running sentiment plots for all {len(TOPICS)} topics...")
        for topic in TOPICS:
            run_topic(topic, PLOTS_DIR)

    print(f"\nAll done. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()