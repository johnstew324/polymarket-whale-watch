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
#   python -m src.sentiment.text_features --topic iran_trump    # one topic


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

from src.sentiment.pipeline_config import STOPWORDS as GEO_STOPWORDS
from src.sentiment.pipeline_config import TOPICS

warnings.filterwarnings("ignore")

PQ_DIR    = Path("data/processed/proquest")
PLOTS_DIR = Path("data/processed/plots")

LDA_N_TOPICS   = 5
TFIDF_FEATURES = 5000
LDA_FEATURES   = 3000
MIN_ARTICLES   = 20


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_stopwords():
    sklearn_stops = set(sklearn_text.ENGLISH_STOP_WORDS)
    wc_stops      = {w.lower() for w in STOPWORDS}
    geo_stops     = {w.lower() for w in GEO_STOPWORDS}

    extra = {
        "said", "say", "says", "also", "would", "could", "mr", "mrs",
        "ms", "according", "told", "added", "include", "including",
        "reuters", "ft", "financial", "times", "like",
        "one", "two", "three", "four", "five", "ten", "100", "per", "cent",
        "000", "bn", "tn",
        "th", "nd", "rd", "st",
        "s", "de", "t", "al", "don", "ve", "m", "ga", "og", "Â",
        "M", "V", "B", "P", "Min", "JSE",
        "years", "months", "days", "decade", "recent", "term",
        "today", "Sunday", "Thursday", "Friday", "Monday",
        "yesterday", "weeks", "early", "later", "hour",
        "far", "past", "big", "long", "good", "kind", "real", "high",
        "level", "close", "local", "central", "right", "left",
        "quite", "actually", "different", "important", "biggest", "largest",
        "main", "expected", "senior", "based", "North", "South",
        "willing", "Additional",
        "clear", "sign", "control", "public", "line", "position", "idea",
        "want", "way", "need", "work", "think", "mean", "took", "held",
        "came", "know", "look", "use", "used", "hit", "come", "going",
        "gonna", "talk", "really", "saying", "called", "backed", "near",
        "make", "got", "remain", "set", "continue", "warned", "help",
        "increase", "share", "following", "vote", "agree", "agreed",
        "meet", "face", "cut",
        "data", "food", "air", "social", "news", "point", "course", "base",
        "thing", "bit", "lot", "number", "sort", "plan", "member", "effort",
        "home", "ground", "hundreds", "book", "family", "project",
        "comment", "report", "agency", "media", "site", "meeting",
        "service", "issue", "case", "office", "scale", "sector",
        "systems", "partner", "executive", "cost", "order", "stock",
        "result", "law", "campaign", "step", "discussion", "News",
        "United", "diplomat", "hand", "anti",
        "60", "AI",
        "Xi", "IDF", "Arab", "Yeah", "House", "West", "London",
        "Vladimir", "Volodymyr", "Starmer", "George", "analyst",
        "British", "French", "Christopher", "Miller",
        "MUSIC", "PLAYING",
        "Rachman", "Yarvin", "Sajwani", "Gideon", "Marc", "Filippino",
        "Dubai", "bannon", "maga", "charisma", "glamour", "glamorre",
        "homeland", "didn", "man", "match", "labour",
        "brzezinski", "kissinger", "golf",
    }

    combined = sklearn_stops | wc_stops | geo_stops | {w.lower() for w in extra}
    return combined, sorted(combined)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_wordcloud(texts, colormap, title, out_path, combined_stops):
    joined = " ".join(texts)
    if len(joined.strip()) < 50:
        print(f"  skipping word cloud — not enough text")
        return

    wc = WordCloud(
        stopwords        = combined_stops,
        background_color = "white",
        width            = 1200,
        height           = 600,
        colormap         = colormap,
        max_words        = 150,
        collocations     = False,
    ).generate(joined)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  saved: {out_path.name}")


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
    print(f"  saved: {out_path.name}")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_articles(topic):
    path = PQ_DIR / f"{topic}_articles.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["text", "date"])
    df["text"] = df["text"].astype(str)

    market_pattern = (
        r"\b("
        r"ftse|omx|comp|dax|cac|nikkei|nasdaq|dow|stoxx|sensex|bovespa|"
        r"closing|mid\s+change|change\s+mid|latest\s+previous|index\s+latest|"
        r"previous\s+close|opening|bid|ask|spread|yield|"
        r"eur(?!\w)|gbp|jpy|chf|aud|nzd|sek|nok|dkk|pln|huf|czk|"
        r"peso|lira|rupee|won|ringgit|baht|dirham|riyal|"
        r"brent|wti|nymex|comex|"
        r"closing\s+closing|ftse\s+ftse"
        r")\b"
    )
    df = df[~df["text"].str.contains(market_pattern, regex=True, na=False)]
    df = df[~df["text"].str.contains(r"(\d+\.?\d*\s+){4,}", regex=True, na=False)]

    df["digit_ratio"] = df["text"].apply(
        lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
    )
    df = df[df["digit_ratio"] < 0.15].drop(columns=["digit_ratio"])

    if len(df) < MIN_ARTICLES:
        print(f"  [{topic}] only {len(df)} articles — too small to analyse, skipping")
        return None

    return df


def try_merge_scores(topic, df):
    scored_path = PQ_DIR / f"{topic}_articles_scored.csv"
    if not scored_path.exists():
        return None

    scored = pd.read_csv(scored_path, usecols=["proquest_id", "net_score"])
    merged = df.merge(scored, on="proquest_id", how="inner")
    return merged if not merged.empty else None


# ---------------------------------------------------------------------------
# plot functions
# ---------------------------------------------------------------------------

def plot_wordclouds(topic, df, scored_df, combined_stops, out_dir):
    topic_title = topic.replace("_", " ").title()

    make_wordcloud(df["text"].tolist(), "viridis",
        f"Word Cloud — All Articles: {topic_title}",
        out_dir / f"{topic}_04_wordcloud_all.png", combined_stops)

    if scored_df is not None:
        pos_threshold = scored_df["net_score"].quantile(0.75)
        neg_threshold = scored_df["net_score"].quantile(0.25)
        pos_texts = scored_df[scored_df["net_score"] > pos_threshold]["text"].tolist()
        neg_texts = scored_df[scored_df["net_score"] < neg_threshold]["text"].tolist()

        if pos_texts:
            make_wordcloud(pos_texts, "Greens",
                f"Word Cloud — Most Positive Articles (top quartile): {topic_title}",
                out_dir / f"{topic}_05_wordcloud_positive.png", combined_stops)
        else:
            print(f"  [{topic}] no positive articles for sentiment word cloud")

        if neg_texts:
            make_wordcloud(neg_texts, "Reds",
                f"Word Cloud — Most Negative Articles (bottom quartile): {topic_title}",
                out_dir / f"{topic}_06_wordcloud_negative.png", combined_stops)
        else:
            print(f"  [{topic}] no negative articles for sentiment word cloud")
    else:
        print(f"  [{topic}] skipping sentiment word clouds — no scored CSV available")


def plot_tfidf(topic, cleaned, out_dir):
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 3,
        stop_words   = stops_list_global,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError as e:
        print(f"  [{topic}] TF-IDF failed: {e} — skipping")
        return

    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    terms       = vectorizer.get_feature_names_out()
    top_idx     = mean_scores.argsort()[-20:][::-1]

    _save_barh(
        labels   = [terms[i] for i in top_idx],
        values   = [mean_scores[i] for i in top_idx],
        xlabel   = "Mean TF-IDF Score",
        title    = f"TF-IDF Top 20 Terms — {topic.replace('_', ' ').title()}",
        out_path = out_dir / f"{topic}_07_tfidf_top20.png",
    )


def plot_lda(topic, cleaned, out_dir):
    vectorizer = CountVectorizer(
        max_features = LDA_FEATURES,
        ngram_range  = (1, 2),
        min_df       = 3,
        stop_words   = stops_list_global,
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
    terms             = vectorizer.get_feature_names_out()
    topic_assignments = lda.transform(count_matrix).argmax(axis=1)

    print(f"\n  === LDA Topics: {topic} ===")
    top_words_per_topic = []
    for i, component in enumerate(lda.components_):
        top_idx   = component.argsort()[-10:][::-1]
        top_words = [terms[j] for j in top_idx]
        n_docs    = int((topic_assignments == i).sum())
        top_words_per_topic.append(top_words)
        print(f"  Topic {i+1} (n={n_docs:,}): {', '.join(top_words)}")

    _save_barh(
        labels   = [f"T{i+1}: {', '.join(top_words_per_topic[i][:3])}" for i in range(LDA_N_TOPICS)],
        values   = [(topic_assignments == i).sum() for i in range(LDA_N_TOPICS)],
        xlabel   = "Number of Articles",
        title    = f"LDA Topic Distribution — {topic.replace('_', ' ').title()}",
        out_path = out_dir / f"{topic}_08_lda_topics.png",
        figsize  = (10, 6),
    )


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

stops_list_global = []


def run_topic(topic, out_dir, combined_stops):
    df = load_articles(topic)
    if df is None:
        return

    scored_df = try_merge_scores(topic, df)
    plot_wordclouds(topic, df, scored_df, combined_stops, out_dir)

    cleaned = df["text"].apply(clean_text).tolist()
    plot_tfidf(topic, cleaned, out_dir)
    plot_lda(topic, cleaned, out_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global stops_list_global

    parser = argparse.ArgumentParser(
        description="Linguistic analysis plots for ProQuest article corpora"
    )
    parser.add_argument("--topic", type=str, default=None,
        help="Single topic to analyse. Omit to run all topics.")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_stops, stops_list_global = build_stopwords()

    if args.topic:
        topic = args.topic.lower()
        if topic not in TOPICS:
            print(f"Unknown topic '{topic}'. Available: {TOPICS}")
            return
        run_topic(topic, PLOTS_DIR, combined_stops)
    else:
        print(f"Running text features for all {len(TOPICS)} topics...")
        for topic in TOPICS:
            run_topic(topic, PLOTS_DIR, combined_stops)

    print(f"\nAll done. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()