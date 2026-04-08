"""
stopwords.py

Defines ALL_STOPS — the shared stopword set used by wordcloud_plots.py
and lda_analysis.py. Centralised here so both modules stay in sync.

Stopword layers:
  1. BASE_STOPS   — standard English (NLTK)
  2. CUSTOM_STOPS — geo anchors, Trump boilerplate, punctuation artefacts
  3. WC_STOPS     — WordCloud's built-in list
"""

from nltk.corpus import stopwords as nltk_stopwords
from wordcloud import STOPWORDS

# ── 1. Standard English ───────────────────────────────────────────────────────
BASE_STOPS = set(nltk_stopwords.words("english"))

# ── 2. Corpus-specific stops ──────────────────────────────────────────────────
# These appear in almost every post so carry no discriminative signal.

# Geo anchor terms — filtered at the post level, so unneeded in word clouds
GEO_ANCHOR_STOPS = {
    "russia", "ukraine", "ukrainian", "russian",
    "putin", "zelensky", "zelenskyy",
    "kyiv", "nato", "moscow", "kremlin",
}

# Trump's own name / titles (in nearly every post)
TRUMP_BOILERPLATE = {
    "trump", "donald", "president", "djt", "realdonaldtrump",
}

# Generic political / social-media boilerplate
POLITICAL_BOILERPLATE = {
    "said", "says", "will", "one", "also", "get", "going",
    "know", "people", "country", "great", "america", "american",
    "us", "would", "like", "make", "new", "many", "rt",
}

# Punctuation / tokenisation artefacts
ARTEFACTS = {"s", "t", "re", "ve", "ll", "don", "amp"}

CUSTOM_STOPS = (
    GEO_ANCHOR_STOPS
    | TRUMP_BOILERPLATE
    | POLITICAL_BOILERPLATE
    | ARTEFACTS
)

# ── 3. WordCloud built-ins ────────────────────────────────────────────────────
WC_STOPS = set(STOPWORDS)

# ── Final combined set (imported by other modules) ────────────────────────────
ALL_STOPS: set = BASE_STOPS | CUSTOM_STOPS | WC_STOPS
