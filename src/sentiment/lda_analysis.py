"""
lda_analysis.py

LDA topic modelling on the Truth Social geopolitical post corpus.
Extracted from TRUTH_SOCIAL_NLP.ipynb (cells 15, 17, 19).

Uses sklearn instead of gensim — identical output, no C++ compiler needed.
  sklearn.decomposition.LatentDirichletAllocation  replaces gensim LdaModel
  sklearn.feature_extraction.text.CountVectorizer  replaces gensim Dictionary
  regex tokeniser                                  replaces gensim simple_preprocess

Public API
----------
    preprocess_for_lda(text, stopwords) -> list[str]
    train_lda(geo_df, num_topics)       -> tuple[LdaModel, CountVectorizer]
    plot_lda_wordclouds(lda_model, vectorizer, output_path)

Typical usage (via run_nlp.py):
    lda_model, vectorizer = train_lda(geo_df)
    plot_lda_wordclouds(lda_model, vectorizer, "figures/lda_topic_wordclouds.png")
"""

import os
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
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
    Tokenise a single post for LDA: lowercase, strip accents, filter stops
    and short tokens (len <= 2).

    Uses regex instead of gensim's simple_preprocess — no extra dependency.

    Parameters
    ----------
    text      : raw post string
    stopwords : set of words to exclude (defaults to ALL_STOPS)

    Returns
    -------
    list of clean tokens
    """
    # Lowercase and strip accents (replicates gensim's deacc=True)
    text = unicodedata.normalize("NFKD", text.lower())
    text = text.encode("ascii", "ignore").decode("ascii")

    tokens = re.findall(r"\b[a-z]+\b", text)
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def train_lda(
    geo_df: pd.DataFrame,
    num_topics: int = NUM_TOPICS,
) -> tuple:
    """
    Preprocess all posts, build a document-term matrix, and train LDA.

    CountVectorizer settings (equivalent to gensim dictionary filtering):
      - min_df=5   : ignore tokens appearing in fewer than 5 posts
      - max_df=0.6 : ignore tokens appearing in more than 60% of posts
      - max_features=1000

    LDA settings:
      - max_iter=15, learning_method="batch", random_state=42

    Parameters
    ----------
    geo_df     : filtered geopolitical posts DataFrame (must have 'text' column)
    num_topics : number of LDA topics (default 5)

    Returns
    -------
    (lda_model, vectorizer)
      lda_model  : fitted LatentDirichletAllocation
      vectorizer : fitted CountVectorizer (needed to recover word names for plots)
    """
    print("Preprocessing posts for LDA...")
    processed_texts = [
        " ".join(preprocess_for_lda(t)) for t in geo_df["text"]
    ]
    # Drop posts that became empty after filtering
    processed_texts = [t for t in processed_texts if t.strip()]

    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.6,
        stop_words=list(ALL_STOPS),
    )
    dtm = vectorizer.fit_transform(processed_texts)

    print(f"  Posts for LDA : {len(processed_texts)}")
    print(f"  Vocabulary    : {len(vectorizer.get_feature_names_out())} tokens")

    print(f"\nTraining LDA ({num_topics} topics, 15 iterations)...")
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=15,
        learning_method="batch",
    )
    lda_model.fit(dtm)

    # Print top 10 words per topic
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop words per topic:")
    for i, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words   = [feature_names[j] for j in top_indices]
        label = TOPIC_LABELS[i] if i < len(TOPIC_LABELS) else f"Topic {i + 1}"
        print(f"  {label}: {', '.join(top_words)}")

    return lda_model, vectorizer


def plot_lda_wordclouds(lda_model, vectorizer, output_path: str) -> None:
    """
    Generate one word cloud per LDA topic and save as a single figure.

    Word size is proportional to the word's probability weight within each
    topic. Topic colourmaps are defined in TOPIC_CMAPS.

    Parameters
    ----------
    lda_model   : fitted LatentDirichletAllocation
    vectorizer  : fitted CountVectorizer from train_lda()
    output_path : path to save the PNG (e.g. "figures/lda_topic_wordclouds.png")
    """
    feature_names = vectorizer.get_feature_names_out()
    num_topics    = lda_model.n_components

    fig, axes = plt.subplots(1, num_topics, figsize=(22, 5))

    for i, (ax, topic) in enumerate(zip(axes, lda_model.components_)):
        # Top 50 words by weight for this topic
        top_indices = topic.argsort()[-50:][::-1]
        word_freq   = {feature_names[j]: float(topic[j]) for j in top_indices}

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

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
