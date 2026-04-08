"""
wordcloud_plots.py

Generates the three word cloud figures for Section 2 of the report.
Extracted from TRUTH_SOCIAL_NLP.ipynb (cells 9, 11, 13).

Public API
----------
    plot_raw_wordcloud(geo_df, output_path)
    plot_tfidf_wordcloud(geo_df, output_path)
    plot_market_type_wordclouds(geo_df, output_path)

Each function expects geo_df to be the filtered geopolitical posts DataFrame
produced by load_and_filter_posts() in run_nlp.py (770 posts, 'text' column,
'created_at' column with UTC timezone).
"""

import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from stopwords import ALL_STOPS
from truth_social_keywords import TRUTH_SOCIAL_MARKET_CONFIG

LOOKBACK_HOURS = 336  # 14 days — matches truth_social_collector.py


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, output_path: str) -> None:
    """Create parent directories if needed, save figure, close it."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _make_wordcloud(text: str, colormap: str, **kwargs) -> WordCloud:
    """Build a WordCloud from a text string with shared defaults."""
    return WordCloud(
        background_color="white",
        stopwords=ALL_STOPS,
        collocations=False,
        colormap=colormap,
        **kwargs,
    ).generate(text)


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────────────

def plot_raw_wordcloud(geo_df: pd.DataFrame, output_path: str) -> None:
    """
    Raw frequency word cloud across all geopolitical posts.

    Word size is proportional to raw occurrence count across the 770-post
    corpus, after removing ALL_STOPS (anchor terms, boilerplate, English stops).
    """
    all_text = " ".join(geo_df["text"].tolist())

    wc = _make_wordcloud(all_text, colormap="RdYlBu", width=1400, height=700, max_words=150)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Trump Truth Social — Geopolitical Posts\n"
        "Raw Word Frequency (all Ukraine/Russia posts)",
        fontsize=16, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    _save(fig, output_path)


def plot_tfidf_wordcloud(geo_df: pd.DataFrame, output_path: str) -> None:
    """
    TF-IDF weighted word cloud across all geopolitical posts.

    TF-IDF down-weights words that appear frequently across all posts and
    up-weights words that are distinctive to specific posts — better at
    surfacing event-specific signals than raw frequency.

    Vectoriser settings:
      - max_features=500, ngram_range=(1, 2), min_df=3
    """
    tfidf = TfidfVectorizer(
        stop_words=list(ALL_STOPS),
        max_features=500,
        ngram_range=(1, 2),
        min_df=3,
    )
    tfidf_matrix = tfidf.fit_transform(geo_df["text"])
    tfidf_scores = dict(zip(
        tfidf.get_feature_names_out(),
        tfidf_matrix.mean(axis=0).A1,
    ))

    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        colormap="coolwarm",
        max_words=150,
        collocations=False,
    ).generate_from_frequencies(tfidf_scores)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Trump Truth Social — Geopolitical Posts\n"
        "TF-IDF Weighted Word Cloud (distinctive terms per post)",
        fontsize=16, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    _save(fig, output_path)


def plot_market_type_wordclouds(geo_df: pd.DataFrame, output_path: str) -> None:
    """
    Side-by-side word clouds split by market type: escalation vs benign.

    For each market in TRUTH_SOCIAL_MARKET_CONFIG, posts are filtered to the
    14-day pre-resolution window and matched against anchor terms. Posts are
    then pooled into two buckets (escalation / benign) to show whether Trump's
    language differs between conflict-escalation and ceasefire/diplomatic periods.
    """
    escalation_posts, benign_posts = [], []

    for market_id, cfg in TRUTH_SOCIAL_MARKET_CONFIG.items():
        res_date     = cfg["resolution_date"]
        window_start = res_date - pd.Timedelta(hours=LOOKBACK_HOURS)
        anchor_pat   = "|".join(re.escape(t) for t in cfg["anchor_terms"])

        mask = (
            (geo_df["created_at"] >= window_start)
            & (geo_df["created_at"] <= res_date)
            & geo_df["text"].str.contains(anchor_pat, case=False, na=False)
        )
        matched = geo_df[mask]["text"].tolist()

        if cfg["market_type"] == "escalation":
            escalation_posts.extend(matched)
        else:
            benign_posts.extend(matched)

    print(f"  Escalation posts: {len(escalation_posts)}")
    print(f"  Benign/ceasefire posts: {len(benign_posts)}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    panels = [
        (escalation_posts, "Escalation Markets\n(strikes, sanctions, military aid)", "Reds"),
        (benign_posts,     "Benign / Ceasefire Markets\n(peace talks, NATO, diplomatic)", "Blues"),
    ]

    for ax, (texts, title, cmap) in zip(axes, panels):
        combined = " ".join(texts) if texts else "no data"
        wc = _make_wordcloud(combined, colormap=cmap, width=800, height=600, max_words=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Market Type: {title}", fontsize=14, fontweight="bold", pad=15)

    plt.suptitle(
        "Trump's Language by Polymarket Event Type",
        fontsize=17, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_path)
