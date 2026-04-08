"""
run_nlp.py

Main entry point for the Truth Social NLP pipeline.
Replaces TRUTH_SOCIAL_NLP.ipynb — produces identical outputs.

Orchestrates all modules in sequence:
  1. Load & filter posts        (truth_social_collector.py)
  2. Raw frequency word cloud   (wordcloud_plots.py)
  3. TF-IDF word cloud          (wordcloud_plots.py)
  4. Market-type word clouds    (wordcloud_plots.py)
  5. LDA training               (lda_analysis.py)
  6. LDA topic word clouds      (lda_analysis.py)
  7. FinBERT score all posts    (sentiment_timeseries.py)
  8. Sentiment time series plot (sentiment_timeseries.py)

Usage
-----
    # Using local archive (recommended — faster, no download)
    python sentiment/run_nlp.py --local truth_social_archive.csv

    # Fetch live from stilesdata.com
    python sentiment/run_nlp.py

    # Custom output directories
    python sentiment/run_nlp.py --local truth_social_archive.csv \
        --figures-dir reports/figures --output-dir data/processed

Run from the PROJECT ROOT (the folder containing RUNME.txt), not from
inside sentiment/.

Outputs
-------
    figures/wc_raw_all_posts.png
    figures/wc_tfidf_all_posts.png
    figures/wc_by_market_type.png
    figures/lda_topic_wordclouds.png
    figures/sentiment_timeseries.png
    data/processed/truth_social_scored_posts.csv
"""

import argparse
import os
import re
import sys

import pandas as pd

# ── Make sentiment/ importable when running from the project root ─────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from truth_social_collector import load_truth_social
from wordcloud_plots import (
    plot_raw_wordcloud,
    plot_tfidf_wordcloud,
    plot_market_type_wordclouds,
)
from lda_analysis import train_lda, plot_lda_wordclouds
from sentiment_timeseries import score_all_posts, plot_sentiment_timeseries

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_URL = "https://stilesdata.com/trump-truth-social-archive/truth_archive.csv"

GEO_KEYWORDS = [
    "Russia", "Ukraine", "Kyiv", "Kremlin", "Putin",
    "Zelensky", "NATO", "ceasefire", "sanctions", "Moscow",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_and_filter_posts(filepath_or_url: str) -> pd.DataFrame:
    """
    Load the Truth Social archive and filter to geopolitical posts.

    Uses load_truth_social() from truth_social_collector.py for cleaning
    (fix_encoding, clean_post, min length, RT removal), then applies the
    global geopolitical keyword filter to produce the ~770-post corpus
    used by all NLP modules.
    """
    df = load_truth_social(filepath_or_url)

    pattern = "|".join(GEO_KEYWORDS)
    geo_df = df[df["text"].str.contains(pattern, case=False, na=False)].copy()
    geo_df["month"] = geo_df["created_at"].dt.to_period("M")
    geo_df = geo_df.reset_index(drop=True)

    print(f"  Geopolitical posts: {len(geo_df)}")
    print(f"  Date range: {geo_df['created_at'].min().date()} "
          f"to {geo_df['created_at'].max().date()}")
    return geo_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(local_path: str | None = None,
         figures_dir: str = "figures",
         output_dir: str = "data/processed") -> None:
    """
    Run the full NLP pipeline and save all outputs.

    Parameters
    ----------
    local_path  : path to local Truth Social CSV; if None, fetches live URL
    figures_dir : directory for PNG outputs (created if missing)
    output_dir  : directory for CSV outputs (created if missing)
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(output_dir,  exist_ok=True)

    source = local_path or DATA_URL

    # ── Step 1: Load & filter ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Load and filter posts")
    print("=" * 60)
    geo_df = load_and_filter_posts(source)

    # ── Step 2: Raw word cloud ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Raw frequency word cloud")
    print("=" * 60)
    plot_raw_wordcloud(
        geo_df,
        os.path.join(figures_dir, "wc_raw_all_posts.png"),
    )

    # ── Step 3: TF-IDF word cloud ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — TF-IDF weighted word cloud")
    print("=" * 60)
    plot_tfidf_wordcloud(
        geo_df,
        os.path.join(figures_dir, "wc_tfidf_all_posts.png"),
    )

    # ── Step 4: Market-type word clouds ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Word clouds by market type (escalation vs benign)")
    print("=" * 60)
    plot_market_type_wordclouds(
        geo_df,
        os.path.join(figures_dir, "wc_by_market_type.png"),
    )

    # ── Step 5 & 6: LDA ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Train LDA model")
    print("=" * 60)
    lda_model, _ = train_lda(geo_df)

    print("\n" + "=" * 60)
    print("STEP 6 — LDA topic word clouds")
    print("=" * 60)
    plot_lda_wordclouds(
        lda_model,
        os.path.join(figures_dir, "lda_topic_wordclouds.png"),
    )

    # ── Step 7 & 8: FinBERT + time series ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Score all posts with FinBERT (may take a few minutes)")
    print("=" * 60)
    scored_df = score_all_posts(geo_df)

    print("\n" + "=" * 60)
    print("STEP 8 — Sentiment time series plot")
    print("=" * 60)
    plot_sentiment_timeseries(
        scored_df,
        os.path.join(figures_dir, "sentiment_timeseries.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL OUTPUTS COMPLETE")
    print("=" * 60)
    outputs = [
        os.path.join(figures_dir, "wc_raw_all_posts.png"),
        os.path.join(figures_dir, "wc_tfidf_all_posts.png"),
        os.path.join(figures_dir, "wc_by_market_type.png"),
        os.path.join(figures_dir, "lda_topic_wordclouds.png"),
        os.path.join(figures_dir, "sentiment_timeseries.png"),
        os.path.join(output_dir,  "truth_social_scored_posts.csv"),
    ]
    for path in outputs:
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truth Social NLP pipeline — reproduces all Section 2 figures."
    )
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to local Truth Social CSV (omit to fetch live from stilesdata.com)",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures",
        help="Output directory for PNG figures (default: figures/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for CSV files (default: data/processed/)",
    )
    args = parser.parse_args()

    main(
        local_path=args.local,
        figures_dir=args.figures_dir,
        output_dir=args.output_dir,
    )
