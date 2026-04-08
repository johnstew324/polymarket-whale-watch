"""
lda_analysis.py

LDA topic modelling on the Truth Social geopolitical post corpus.
Extracted from TRUTH_SOCIAL_NLP.ipynb (cells 15, 17, 19).

Public API
----------
    preprocess_for_lda(text, stopwords) -> list[str]
    train_lda(geo_df, num_topics)       -> tuple[LdaModel, Dictionary]
    plot_lda_wordclouds(lda_model, output_path)

Typical usage (via run_nlp.py):
    lda_model, _ = train_lda(geo_df)
    plot_lda_wordclouds(lda_model, "figures/lda_topic_wordclouds.png")
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from gensim.utils import simple_preprocess
from wordcloud import WordCloud

from stopwords import ALL_STOPS

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

NUM_TOPICS = 5

# Labels reflect the actual LDA output — Topics 1-3 and 5 are dominated by
# domestic Trump political rhetoric; Topic 4 is the core geopolitical signal.
# (Labels corrected after inspecting top words from the trained model.)
TOPIC_LABELS = [
    "Topic 1: Domestic Politics (Trump vs Biden / FBI)",
    "Topic 2: Republican Political Rhetoric",
    "Topic 3: Senate Endorsements & Border Policy",
    "Topic 4: War, Peace & Ceasefire Talks",
    "Topic 5: America's Global Role",
]

TOPIC_CMAPS = ["Reds", "Blues", "Greens", "Purples", "Oranges"]


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────────────

def preprocess_for_lda(text: str, stopwords: set = ALL_STOPS) -> list:
    """
    Tokenise a single post for LDA: lowercase, remove accents, filter stops
    and short tokens (len <= 2).

    Parameters
    ----------
    text      : raw post string
    stopwords : set of words to exclude (defaults to ALL_STOPS)

    Returns
    -------
    list of clean tokens
    """
    tokens = simple_preprocess(text, deacc=True)
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def train_lda(
    geo_df: pd.DataFrame,
    num_topics: int = NUM_TOPICS,
) -> tuple:
    """
    Preprocess all posts, build a Gensim corpus, and train an LDA model.

    Dictionary filtering:
      - no_below=5  : ignore tokens appearing in fewer than 5 posts
      - no_above=0.6: ignore tokens appearing in more than 60% of posts

    LDA settings:
      - passes=15, alpha="auto", eta="auto", random_state=42

    Parameters
    ----------
    geo_df     : filtered geopolitical posts DataFrame (must have 'text' column)
    num_topics : number of LDA topics (default 5)

    Returns
    -------
    (lda_model, dictionary)
    """
    print("Preprocessing posts for LDA...")
    processed = [preprocess_for_lda(t) for t in geo_df["text"]]
    processed = [p for p in processed if len(p) >= 5]  # drop very short posts

    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    corpus = [dictionary.doc2bow(p) for p in processed]

    print(f"  Posts for LDA : {len(processed)}")
    print(f"  Vocabulary    : {len(dictionary)} tokens")

    print(f"\nTraining LDA ({num_topics} topics, 15 passes)...")
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15,
        alpha="auto",
        eta="auto",
    )

    print("\nTop words per topic:")
    for i in range(num_topics):
        top_words = [w for w, _ in lda_model.show_topic(i, topn=10)]
        label = TOPIC_LABELS[i] if i < len(TOPIC_LABELS) else f"Topic {i + 1}"
        print(f"  {label}: {', '.join(top_words)}")

    return lda_model, dictionary


def plot_lda_wordclouds(lda_model, output_path: str) -> None:
    """
    Generate one word cloud per LDA topic and save as a single figure.

    Word size is proportional to the word's probability weight within each
    topic (not raw frequency). Topic colourmaps are defined in TOPIC_CMAPS.

    Parameters
    ----------
    lda_model   : trained Gensim LdaModel
    output_path : path to save the PNG (e.g. "figures/lda_topic_wordclouds.png")
    """
    num_topics = lda_model.num_topics
    fig, axes = plt.subplots(1, num_topics, figsize=(22, 5))

    for i, ax in enumerate(axes):
        word_freq = dict(lda_model.show_topic(i, topn=50))
        cmap  = TOPIC_CMAPS[i] if i < len(TOPIC_CMAPS) else "viridis"
        label = TOPIC_LABELS[i] if i < len(TOPIC_LABELS) else f"Topic {i + 1}"

        wc = WordCloud(
            width=500, height=400,
            background_color="white",
            colormap=cmap,
            collocations=False,
        ).generate_from_frequencies(word_freq)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)

    plt.suptitle(
        "LDA Topic Modelling — Trump Truth Social Geopolitical Posts",
        fontsize=15, fontweight="bold", y=1.05,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
