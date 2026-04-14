# -*- coding: utf-8 -*-
# linguistic analysis of ProQuest article corpora
# produces word clouds, TF-IDF top terms, and LDA topic model plots per topic
#
# can run immediately after parse_proquest.py — does not require finbert scoring
# if scored CSV exists, also produces sentiment-split word clouds (positive vs negative)
#
# input:  data/processed/proquest/{topic}_articles.csv
#         data/processed/proquest/{topic}_articles_scored.csv  (optional, for split word clouds)
# output: data/processed/plots/{topic}_04_wordcloud_all.png
#         data/processed/plots/{topic}_05_wordcloud_positive.png  (if scored CSV available)
#         data/processed/plots/{topic}_06_wordcloud_negative.png  (if scored CSV available)
#         data/processed/plots/{topic}_07_tfidf_top20.png
#         data/processed/plots/{topic}_08_lda_topics.png
#
# usage:
#   python -m src.sentiment.text_features                       # all topics
#   python -m src.sentiment.text_features --topic iran_israel   # one topic

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from wordcloud import WordCloud, STOPWORDS

from src.sentiment.geo_vocab_mapping import STOPWORDS as GEO_STOPWORDS

warnings.filterwarnings("ignore")


# ----------------------------
# z_score utility
# matches gmscfeatureanalysis.py exactly
# ----------------------------
def z_score(series):
    return (series - series.mean()) / series.std(ddof=0)


# ----------------------------
# Config
# ----------------------------
PQ_DIR      = Path("data/processed/proquest")
PLOTS_DIR   = Path("data/processed/plots")

LDA_N_TOPICS   = 5
TFIDF_FEATURES = 5000
LDA_FEATURES   = 3000
MIN_ARTICLES   = 20     # skip topic if corpus too small

TOPICS = [
    "ceasefire_russia_ukraine",
    "china_taiwan",
    "congress_iran",
    "damascus_israel",
    "gaza_israel",
    "gaza_usa",
    "hamas_israel",
    "hezbollah_israel",
    "hezbollah_nasrallah",
    "houthi_israel",
    "india_pakistan",
    "iran_israel",
    "iran_khamenei",
    "iran_trump",
    "iran_usa",
    "iraq_israel",
    "israel_lebanon",
    "israel_saudi",
    "israel_syria",
    "israel_yemen",
    "kim_jong_un",
    "kupiansk_russia",
    "mbs",
    "merz_trump",
    "moscow_ukraine",
    "netanyahu_unga",
    "north_korea",
    "north_korea_south_korea",
    "pokrovsk_russia",
    "putin_trump",
    "putin_trump_zelenskyy",
    "putin_zelenskyy",
    "russia_siversk",
    "russia_sudzha",
    "russia_syria",
    "russia_ukraine",
    "south_korea_trump",
    "syria",
    "syria_usa",
    "trump_ukraine",
    "trump_unga",
    "trump_zelenskyy",
    "usa_yemen",
    "xi_jinping",
    "yoon",
    "zelenskyy",
    "al_sharaa",
]


# ----------------------------
# Build combined stopword set
# merges geo_vocab_mapping.STOPWORDS + wordcloud.STOPWORDS + sklearn English
# used by all three text analysis methods
# ----------------------------
def build_stopwords():
    # sklearn English stopwords as a set
    sklearn_stops = set(sklearn_text.ENGLISH_STOP_WORDS)

    # wordcloud STOPWORDS is already a set of strings
    wc_stops = {w.lower() for w in STOPWORDS}

    # geo_vocab_mapping.STOPWORDS — already lowercase
    geo_stops = {w.lower() for w in GEO_STOPWORDS}

    combined = sklearn_stops | wc_stops | geo_stops

    # additional noise terms common in FT articles that aren't in any stopword list
    extra = {
        "said", "say", "says", "also", "would", "could", "mr", "mrs",
        "ms", "according", "told", "added", "include", "including",
        "reuters", "ft", "financial", "times", "like", "one", "two",
        "three", "four", "five", "ten", "100", "per", "cent", "000",
        "bn", "tn", "th", "nd", "rd", "st",
    }
    combined = combined | extra

    # return both a set (for wordcloud) and a sorted list (for sklearn)
    return combined, sorted(combined)


# ----------------------------
# Text cleaning for LDA / TF-IDF
# not used for word clouds — raw text is more natural there
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\b\w{1,2}\b", " ", text)       # remove 1-2 char tokens
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Load raw articles CSV
# ----------------------------
def load_articles(topic):
    path = PQ_DIR / f"{topic}_articles.csv"
    if not path.exists():
        print(f"  [{topic}] no articles CSV found — run parse_proquest.py first, skipping")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["text", "date"])
    df["text"] = df["text"].astype(str)

    if len(df) < MIN_ARTICLES:
        print(f"  [{topic}] only {len(df)} articles — too small to analyse, skipping")
        return None

    return df


# ----------------------------
# Attempt to merge with scored CSV for sentiment-split word clouds
# returns df with net_score column if available, otherwise None
# ----------------------------
def try_merge_scores(topic, df):
    scored_path = PQ_DIR / f"{topic}_articles_scored.csv"
    if not scored_path.exists():
        return None

    scored = pd.read_csv(scored_path, usecols=["proquest_id", "net_score"])
    merged = df.merge(scored, on="proquest_id", how="inner")

    if merged.empty:
        return None

    return merged


# ----------------------------
# Section 1: Print corpus summary
# ----------------------------
def print_summary(topic, df, scored_df):
    print(f"\n{'='*50}")
    print(f"  {topic}")
    print(f"{'='*50}")
    print(f"  Articles:        {len(df):,}")
    print(f"  Date range:      {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Scored CSV:      {'yes' if scored_df is not None else 'no — word clouds will be all-articles only'}")

    if scored_df is not None:
        pos_pct = (scored_df["net_score"] > 0).mean() * 100
        neg_pct = (scored_df["net_score"] < 0).mean() * 100
        print(f"  Mean net_score:  {scored_df['net_score'].mean():.3f}")
        print(f"  % Positive (>0): {pos_pct:.1f}%")
        print(f"  % Negative (<0): {neg_pct:.1f}%")

    # estimated vocabulary size
    all_text = " ".join(df["text"].tolist())
    vocab_est = len(set(all_text.lower().split()))
    print(f"  Vocab estimate:  {vocab_est:,} unique tokens (before cleaning)")


# ----------------------------
# Section 2: Word clouds
# implements WordCloud + STOPWORDS from parser.ipynb imports
# ----------------------------
def plot_wordclouds(topic, df, scored_df, combined_stops, out_dir):

    def make_wordcloud(texts, colormap, title, out_path):
        joined = " ".join(texts)
        if len(joined.strip()) < 50:
            print(f"  skipping word cloud — not enough text")
            return

        wc = WordCloud(
            stopwords=combined_stops,
            background_color="white",
            width=1200,
            height=600,
            colormap=colormap,
            max_words=150,
            collocations=False,     # avoid duplicate bigrams
        ).generate(joined)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  saved: {out_path.name}")

    topic_title = topic.replace("_", " ").title()

    # 4 — all articles
    make_wordcloud(
        texts    = df["text"].tolist(),
        colormap = "viridis",
        title    = f"Word Cloud — All Articles: {topic_title}",
        out_path = out_dir / f"{topic}_04_wordcloud_all.png",
    )

    # 5 & 6 — sentiment split, only if scored CSV available
    if scored_df is not None:
        pos_threshold = scored_df["net_score"].quantile(0.75)
        neg_threshold = scored_df["net_score"].quantile(0.25)

        pos_texts = scored_df[scored_df["net_score"] > pos_threshold]["text"].tolist()
        neg_texts = scored_df[scored_df["net_score"] < neg_threshold]["text"].tolist()

        if pos_texts:
            make_wordcloud(
                texts    = pos_texts,
                colormap = "Greens",
                title    = f"Word Cloud — Most Positive Articles (top quartile): {topic_title}",
                out_path = out_dir / f"{topic}_05_wordcloud_positive.png",
            )
        else:
            print(f"  [{topic}] no positive articles for sentiment word cloud")

        if neg_texts:
            make_wordcloud(
                texts    = neg_texts,
                colormap = "Reds",
                title    = f"Word Cloud — Most Negative Articles (bottom quartile): {topic_title}",
                out_path = out_dir / f"{topic}_06_wordcloud_negative.png",
            )
        else:
            print(f"  [{topic}] no negative articles for sentiment word cloud")
    else:
        print(f"  [{topic}] skipping sentiment word clouds — no scored CSV available")


# ----------------------------
# Section 3: TF-IDF top 20 terms
# implements TfidfVectorizer from parser.ipynb imports
# horizontal barh matches gmscfeatureanalysis.py bar style
# ----------------------------
def plot_tfidf(topic, df, stops_list, out_dir):
    cleaned = df["text"].apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(
        max_features = TFIDF_FEATURES,
        ngram_range  = (1, 2),       # bigrams: "missile strike", "peace talks"
        min_df       = 3,             # must appear in at least 3 articles
        stop_words   = stops_list,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError as e:
        print(f"  [{topic}] TF-IDF failed: {e} — skipping")
        return

    # mean TF-IDF score per term across all articles
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    terms       = vectorizer.get_feature_names_out()

    top_idx   = mean_scores.argsort()[-20:][::-1]
    top_terms = [terms[i] for i in top_idx]
    top_vals  = [mean_scores[i] for i in top_idx]

    # print to console — matches lecturer always-print habit
    print(f"\n  TF-IDF top 20 terms — {topic}:")
    for term, val in zip(top_terms, top_vals):
        print(f"    {term:<30} {val:.4f}")

    # horizontal bar chart — matches gmscfeatureanalysis.py barh style
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(top_terms))

    ax.barh(y_pos, top_vals[::-1], align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_terms[::-1])
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title(f"TF-IDF Top 20 Terms — {topic.replace('_', ' ').title()}")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{topic}_07_tfidf_top20.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


# ----------------------------
# Section 4: LDA topic modelling
# implements CountVectorizer + LatentDirichletAllocation from parser.ipynb imports
# always prints model summary — matches gmscfeatureanalysis.py habit
# ----------------------------
def plot_lda(topic, df, stops_list, out_dir):
    cleaned = df["text"].apply(clean_text).tolist()

    vectorizer = CountVectorizer(
        max_features = LDA_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 3,
        stop_words   = stops_list,
    )

    try:
        count_matrix = vectorizer.fit_transform(cleaned)
    except ValueError as e:
        print(f"  [{topic}] LDA vectorizer failed: {e} — skipping")
        return

    lda = LatentDirichletAllocation(
        n_components    = LDA_N_TOPICS,
        random_state    = 42,
        max_iter        = 20,
        learning_method = "batch",
    )

    lda.fit(count_matrix)

    terms = vectorizer.get_feature_names_out()

    # assign each article its dominant topic
    topic_assignments = lda.transform(count_matrix).argmax(axis=1)

    # print topic summary — matches gmscfeatureanalysis.py always-print-model-summaries
    print(f"\n  === LDA Topics: {topic} ===")
    top_words_per_topic = []
    for i, component in enumerate(lda.components_):
        top_idx   = component.argsort()[-10:][::-1]
        top_words = [terms[j] for j in top_idx]
        n_articles = int((topic_assignments == i).sum())
        top_words_per_topic.append(top_words)
        print(f"  Topic {i+1} (n={n_articles:,}): {', '.join(top_words)}")

    # topic distribution bar chart
    topic_counts = [(topic_assignments == i).sum() for i in range(LDA_N_TOPICS)]
    topic_labels = [
        f"T{i+1}: {', '.join(top_words_per_topic[i][:3])}"
        for i in range(LDA_N_TOPICS)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(LDA_N_TOPICS)

    ax.barh(list(y_pos), topic_counts[::-1], align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(topic_labels[::-1])
    ax.set_xlabel("Number of Articles")
    ax.set_title(f"LDA Topic Distribution — {topic.replace('_', ' ').title()}")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{topic}_08_lda_topics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


# ----------------------------
# Run one topic
# ----------------------------
def run_topic(topic, out_dir, combined_stops, stops_list):
    df = load_articles(topic)
    if df is None:
        return

    scored_df = try_merge_scores(topic, df)

    print_summary(topic, df, scored_df)
    plot_wordclouds(topic, df, scored_df, combined_stops, out_dir)
    plot_tfidf(topic, df, stops_list, out_dir)
    plot_lda(topic, df, stops_list, out_dir)


# ----------------------------
# Entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Linguistic analysis plots for ProQuest article corpora"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Single topic to analyse. Omit to run all topics."
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_stops, stops_list = build_stopwords()

    if args.topic:
        topic = args.topic.lower()
        if topic not in TOPICS:
            print(f"Unknown topic '{topic}'. Available: {TOPICS}")
            return
        run_topic(topic, PLOTS_DIR, combined_stops, stops_list)
    else:
        print(f"Running text features for all {len(TOPICS)} topics...")
        for topic in TOPICS:
            run_topic(topic, PLOTS_DIR, combined_stops, stops_list)

    print(f"\nAll done. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
