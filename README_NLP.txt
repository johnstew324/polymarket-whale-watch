================================================================================
FIN42110 — Truth Social NLP Pipeline
Group member: Adam
================================================================================

This file explains what each Python script does and how to run the full NLP
pipeline that was originally in TRUTH_SOCIAL_NLP.ipynb. Running run_nlp.py
(Step 5, when complete) will reproduce all figures and CSVs in one command.

All scripts live in the sentiment/ folder. Run everything from the PROJECT ROOT
(the folder containing this file), not from inside sentiment/.

--------------------------------------------------------------------------------
QUICK START (once all steps are implemented)
--------------------------------------------------------------------------------

  python sentiment/run_nlp.py --local truth_social_archive.csv

This single command produces:
  figures/wc_raw_all_posts.png
  figures/wc_tfidf_all_posts.png
  figures/wc_by_market_type.png
  figures/lda_topic_wordclouds.png
  figures/sentiment_timeseries.png
  data/processed/truth_social_scored_posts.csv

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------

Install required packages before running:

  pip install pandas numpy torch transformers wordcloud scikit-learn gensim nltk matplotlib pytrends

Also download the NLTK stopwords corpus (one-time):

  python -c "import nltk; nltk.download('stopwords')"

FinBERT (ProsusAI/finbert) is downloaded automatically from Hugging Face on
first run. Requires an internet connection the first time only (~450MB).

--------------------------------------------------------------------------------
FILE-BY-FILE GUIDE
--------------------------------------------------------------------------------

── EXISTING FILES (do not modify) ──────────────────────────────────────────────

sentiment/truth_social_collector.py
  The core FinBERT pipeline. Scores Trump posts per Polymarket market window
  and aggregates to market-level sentiment features.

  Run standalone:
    python sentiment/truth_social_collector.py --local truth_social_archive.csv

  Output: data/processed/truth_social_sentiment_features.csv

sentiment/truth_social_keywords.py
  Configuration file. Defines TRUTH_SOCIAL_MARKET_CONFIG — the 9 Russia-Ukraine
  markets with resolution dates, anchor terms, and market type (escalation/benign).
  Imported by other modules. Do not run directly.

sentiment/market_keywords.py
  Defines MARKET_TRENDS_KEYWORDS — Google Trends search keywords per market.
  Imported by trend_signal.py. Do not run directly.

sentiment/trend_signal.py
  Fetches Google Trends data per market via pytrends. Rate limited — sleeps
  10 seconds between API calls (Google enforces this).

  Not run directly; called from the pytrends notebook (Oisin).

── NEW FILES (NLP pipeline, replaces TRUTH_SOCIAL_NLP.ipynb) ───────────────────

sentiment/stopwords.py                                          [STEP 1 — DONE]
  Defines ALL_STOPS — the shared stopword set used across the NLP pipeline.
  Combines: NLTK English stopwords + custom corpus-specific stops (geo anchors,
  Trump boilerplate, punctuation artefacts) + WordCloud built-ins.

  Do not run directly. Imported by wordcloud_plots.py and lda_analysis.py.

sentiment/wordcloud_plots.py                                    [STEP 2 — DONE]
  Generates all three word cloud figures:
    - Raw frequency word cloud (all 770 geopolitical posts)
    - TF-IDF weighted word cloud (distinctive terms per post)
    - Side-by-side word clouds split by market type (escalation vs benign)

  Imports ALL_STOPS from stopwords.py and TRUTH_SOCIAL_MARKET_CONFIG from
  truth_social_keywords.py. Do not run directly. Called by run_nlp.py.

sentiment/lda_analysis.py                                       [STEP 3 — DONE]
  LDA topic modelling on the geopolitical post corpus:
    - preprocess_for_lda(): tokenises, removes ALL_STOPS, filters short tokens
    - train_lda(): builds Gensim dictionary + corpus, trains 5-topic LDA model
      (15 passes, alpha/eta auto, random_state=42), prints top words per topic
    - plot_lda_wordclouds(): saves one word cloud per topic as a single figure

  Note: 4 of 5 topics capture domestic Trump political rhetoric; Topic 4
  ("War, Peace & Ceasefire Talks") is the core geopolitical signal.
  Topic labels are defined in TOPIC_LABELS inside this file.

  Imports ALL_STOPS from stopwords.py. Do not run directly. Called by run_nlp.py.

sentiment/sentiment_timeseries.py                               [STEP 4 — DONE]
  Scores all 770 geopolitical posts with FinBERT and plots rolling sentiment:
    - score_all_posts(): runs FinBERT over every post, saves scored CSV,
      returns DataFrame with net_score/positive/negative/neutral columns.
      Imports load_finbert() and score_post() from truth_social_collector.py
      (no duplication of FinBERT code).
    - plot_sentiment_timeseries(): plots 4-week rolling average FinBERT net
      score with green/red fill regions and orange vertical lines marking
      each Polymarket resolution date.

  Do not run directly. Called by run_nlp.py.

sentiment/run_nlp.py                                            [STEP 5 — DONE]
  Main entry point. Orchestrates the full NLP pipeline end-to-end across
  8 steps: load posts → 3 word clouds → LDA train → LDA plot → FinBERT
  score → sentiment time series. Prints a summary table at the end
  confirming every output file was written successfully.

  Usage:
    python sentiment/run_nlp.py --local truth_social_archive.csv

  Optional flags:
    --local PATH         Path to local Truth Social CSV (faster, no download).
                         Omit to fetch live from stilesdata.com instead.
    --figures-dir PATH   Where to save PNGs (default: figures/)
    --output-dir  PATH   Where to save CSVs (default: data/processed/)

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------

  figures/wc_raw_all_posts.png       Raw frequency word cloud
  figures/wc_tfidf_all_posts.png     TF-IDF weighted word cloud
  figures/wc_by_market_type.png      Escalation vs benign word clouds
  figures/lda_topic_wordclouds.png   5-topic LDA word clouds
  figures/sentiment_timeseries.png   FinBERT sentiment over time

  data/processed/truth_social_scored_posts.csv
    All 770 geopolitical posts with FinBERT scores (net_score, positive,
    negative, neutral). Used by inspect_market_posts() for audit/debugging.

  data/processed/truth_social_sentiment_features.csv
    Per-market aggregated features (produced by truth_social_collector.py,
    not run_nlp.py). This is the ML feature input.

--------------------------------------------------------------------------------
NOTES FOR GROUP MEMBERS
--------------------------------------------------------------------------------

- Run from the PROJECT ROOT, not from inside sentiment/.
- The FinBERT scoring step (~770 posts) takes a few minutes on CPU.
- If you get an NLTK error, run:
    python -c "import nltk; nltk.download('stopwords')"
- truth_social_archive.csv must be present in the project root.
  It is the local Trump Truth Social post archive (~Oct 2025).

================================================================================
