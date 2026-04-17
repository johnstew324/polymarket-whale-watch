# -*- coding: utf-8 -*-
# global linguistic and sentiment analysis of Trump's Truth Social archive
# treats the entire archive as a single corpus — no per-topic slicing
#
# produces 8 plots covering post volume, VADER sentiment, word clouds,
# TF-IDF top terms, and LDA topic model
#
# input:  data/raw/truth_social_archive.csv
# output: data/processed/plots/truth_social/ts_01_post_volume.png
#         data/processed/plots/truth_social/ts_02_sentiment_histogram.png
#         data/processed/plots/truth_social/ts_03_sentiment_timeline.png
#         data/processed/plots/truth_social/ts_04_wordcloud_all.png
#         data/processed/plots/truth_social/ts_05_wordcloud_positive.png
#         data/processed/plots/truth_social/ts_06_wordcloud_negative.png
#         data/processed/plots/truth_social/ts_07_tfidf_top20.png
#         data/processed/plots/truth_social/ts_08_lda_topics.png
#
# usage:
#   python -m src.sentiment.truth_social_plots


import html
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

from src.sentiment.pipeline_config import STOPWORDS as GEO_STOPWORDS

warnings.filterwarnings("ignore")

ARCHIVE   = Path("data/raw/truth_social_archive.csv")
PLOTS_DIR = Path("data/processed/plots/truth_social")

LDA_N_TOPICS   = 7
TFIDF_FEATURES = 5000
LDA_FEATURES   = 3000
MIN_CHARS      = 30

def build_stopwords():
    sklearn_stops = set(sklearn_text.ENGLISH_STOP_WORDS)
    wc_stops      = {w.lower() for w in STOPWORDS}
    geo_stops     = {w.lower() for w in GEO_STOPWORDS}

    trump_noise = {
        "trump", "donald", "realdonaldtrump", "president", "djt",
        "rt", "http", "https", "www", "com",
        "great", "incredible", "tremendous", "fantastic", "beautiful",
        "wonderful", "very", "really", "truly", "always", "never",
        "make", "made", "many", "much", "big", "largest", "greatest",
        "going", "get", "got", "let", "just", "like", "know", "said",
        "say", "says", "also", "would", "could", "should", "will",
        "done", "doing", "told", "added", "want", "need", "come",
        "came", "back", "good", "bad", "new", "old", "long", "hard",
        "right", "left", "way", "time", "day", "week", "year", "years",
        "people", "country", "america", "american", "americans",
        "nation", "national", "world", "united", "states", "state",
        "government", "administration", "white", "house", "congress",
        "republican", "democrat", "democrats", "party",
        "re", "election", "endorsement", "endorse", "district",
        "congressional", "congressman", "representative", "senator",
        "candidate", "vote", "voted", "voting", "voters",
        "job", "jobs", "work", "working", "worked", "fighting",
        "tirelessly", "honour", "honor", "proud",
        "000", "100", "per", "cent", "bn", "tn",
        "one", "two", "three", "four", "five", "ten",
        "â", "don", "ve", "ll", "didn", "doesn", "isn",
        "Aren", "Wasn", "Weren", "Hasn", "Haven", "Won", "S", "T",
        "York", "Hunt", "Presidential", "Smith", "CNN",
        "Unselect", "Thank", "Total", "Complete", "Congratulations", "Happy",
        "Border", "Maga",
    }

    combined = sklearn_stops | wc_stops | geo_stops | {w.lower() for w in trump_noise}
    return combined, sorted(combined)


def clean_content(text: str) -> str:
    text = html.unescape(str(text))

    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = text.replace("â€˜", "'").replace("â€¦", "...").replace("Â", "")
    text = re.sub(r"â\S*", " ", text)

    text = re.sub(r"^RT @\w+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_bare_url_post(raw: str) -> bool:
    stripped = str(raw).strip()
    if not stripped:
        return True
    return bool(re.match(r"^https?://\S+$", stripped))


def clean_text_for_nlp(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_archive() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE, dtype={"id": str})

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])

    df = df[df["content"].notna()]
    df = df[df["content"].astype(str).str.strip() != ""]
    df = df[~df["content"].apply(is_bare_url_post)]

    df["content_clean"] = df["content"].apply(clean_content)
    df = df[df["content_clean"].str.len() >= MIN_CHARS]

    vader  = SentimentIntensityAnalyzer()
    scores = df["content_clean"].apply(lambda t: vader.polarity_scores(t))
    df["compound"]  = scores.apply(lambda s: s["compound"])
    df["vader_pos"] = scores.apply(lambda s: s["pos"])
    df["vader_neg"] = scores.apply(lambda s: s["neg"])
    df["vader_neu"] = scores.apply(lambda s: s["neu"])

    return df


def _save_barh(labels, values, xlabel, title, out_path, figsize=(10, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(labels))
    ax.barh(list(y_pos), values[::-1], align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_wordcloud(texts, colormap, title, out_path, combined_stops):
    joined = " ".join(texts)
    if len(joined.strip()) < 100:
        return

    wc = WordCloud(
        stopwords        = combined_stops,
        background_color = "white",
        width            = 1600,
        height           = 800,
        colormap         = colormap,
        max_words        = 200,
        collocations     = False,
    ).generate(joined)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_post_volume(df, out_dir):
    df_indexed = df.set_index("created_at")
    monthly    = df_indexed.resample("MS").size()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.8)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title("Truth Social — Post Volume by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Post Count")
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ts_01_post_volume.png", dpi=300)
    plt.close()


def plot_sentiment_histogram(df, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["compound"], bins=50, ax=axs[0], color="steelblue", alpha=0.7)
    axs[0].axvline(df["compound"].mean(), color="blue", linestyle="--",
                   linewidth=1.5, label=f"Mean = {df['compound'].mean():.3f}")
    axs[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    axs[0].set_title("VADER Compound Score Distribution")
    axs[0].set_xlabel("Compound Score  (negative ← 0 → positive)")
    axs[0].set_ylabel("Post Count")
    axs[0].legend()

    sns.histplot(df["vader_pos"], bins=40, ax=axs[1], color="green", alpha=0.5, label="Positive")
    sns.histplot(df["vader_neg"], bins=40, ax=axs[1], color="red",   alpha=0.5, label="Negative")
    sns.histplot(df["vader_neu"], bins=40, ax=axs[1], color="grey",  alpha=0.5, label="Neutral")
    axs[1].set_title("VADER Component Score Distributions")
    axs[1].set_xlabel("Component Proportion  (0 – 1)")
    axs[1].set_ylabel("Post Count")
    axs[1].legend()

    plt.suptitle("Truth Social — Sentiment Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "ts_02_sentiment_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_sentiment_timeline(df, out_dir):
    df_indexed = df.set_index("created_at")
    weekly     = df_indexed.resample("W")["compound"].mean().dropna()
    colours    = ["green" if v > 0 else "red" for v in weekly.values]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(weekly.index, weekly.values, color=colours, width=6, alpha=0.8)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title("Truth Social — Weekly Average VADER Sentiment")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean Compound Score  (negative ← 0 → positive)")
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ts_03_sentiment_timeline.png", dpi=300)
    plt.close()


def plot_wordclouds(df, combined_stops, out_dir):
    make_wordcloud(
        df["content_clean"].tolist(), "viridis",
        "Truth Social — Word Cloud: All Posts",
        out_dir / "ts_04_wordcloud_all.png",
        combined_stops,
    )

    pos_thresh = df["compound"].quantile(0.75)
    neg_thresh = df["compound"].quantile(0.25)

    make_wordcloud(
        df[df["compound"] > pos_thresh]["content_clean"].tolist(), "Greens",
        f"Truth Social — Word Cloud: Most Positive Posts (top quartile, compound > {pos_thresh:.2f})",
        out_dir / "ts_05_wordcloud_positive.png",
        combined_stops,
    )

    make_wordcloud(
        df[df["compound"] < neg_thresh]["content_clean"].tolist(), "Reds",
        f"Truth Social — Word Cloud: Most Negative Posts (bottom quartile, compound < {neg_thresh:.2f})",
        out_dir / "ts_06_wordcloud_negative.png",
        combined_stops,
    )


def plot_tfidf(df, stops_list, out_dir):
    cleaned = df["content_clean"].apply(clean_text_for_nlp).tolist()

    vectorizer = TfidfVectorizer(
        max_features = TFIDF_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 5,
        stop_words   = stops_list,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return

    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    terms       = vectorizer.get_feature_names_out()
    top_idx     = mean_scores.argsort()[-20:][::-1]

    _save_barh(
        labels   = [terms[i] for i in top_idx],
        values   = [mean_scores[i] for i in top_idx],
        xlabel   = "Mean TF-IDF Score",
        title    = "Truth Social — TF-IDF Top 20 Terms",
        out_path = out_dir / "ts_07_tfidf_top20.png",
    )


def plot_lda(df, stops_list, out_dir):
    cleaned = df["content_clean"].apply(clean_text_for_nlp).tolist()

    vectorizer = CountVectorizer(
        max_features = LDA_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 5,
        stop_words   = stops_list,
    )

    try:
        count_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return

    lda = LatentDirichletAllocation(
        n_components    = LDA_N_TOPICS,
        random_state    = 42,
        max_iter        = 30,
        learning_method = "batch",
    )
    lda.fit(count_matrix)

    terms             = vectorizer.get_feature_names_out()
    topic_assignments = lda.transform(count_matrix).argmax(axis=1)

    print("\n=== LDA Topics ===")
    top_words_per_topic = []
    for i, component in enumerate(lda.components_):
        top_idx   = component.argsort()[-10:][::-1]
        top_words = [terms[j] for j in top_idx]
        n_docs    = int((topic_assignments == i).sum())
        top_words_per_topic.append(top_words)
        print(f"  Topic {i+1} (n={n_docs:,}): {', '.join(top_words)}")

    _save_barh(
        labels   = [f"T{i+1}: {', '.join(top_words_per_topic[i][:3])}"
                    for i in range(LDA_N_TOPICS)],
        values   = [(topic_assignments == i).sum() for i in range(LDA_N_TOPICS)],
        xlabel   = "Number of Posts",
        title    = f"Truth Social — LDA Topic Distribution ({LDA_N_TOPICS} topics)",
        out_path = out_dir / "ts_08_lda_topics.png",
        figsize  = (10, 6),
    )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_stops, stops_list = build_stopwords()
    df = load_archive()

    plot_post_volume(df, PLOTS_DIR)
    plot_sentiment_histogram(df, PLOTS_DIR)
    plot_sentiment_timeline(df, PLOTS_DIR)
    plot_wordclouds(df, combined_stops, PLOTS_DIR)
    plot_tfidf(df, stops_list, PLOTS_DIR)
    plot_lda(df, stops_list, PLOTS_DIR)


if __name__ == "__main__":
    main()
