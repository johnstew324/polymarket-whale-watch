# -*- coding: utf-8 -*-
"""
proquest_parser.py

Parses ProQuest raw .txt export files into a clean CSV for FinBERT scoring.
Supports multiple topics — just configure the TOPICS dict below.

Usage:
    python proquest_parser.py                  # runs all topics
    python proquest_parser.py --topic ukraine  # runs one topic only

Output:
    One CSV per topic, e.g. ft_ukraine_articles_full.csv
"""

import re
import glob
import os
import argparse
import pandas as pd
from dateutil import parser as dtparser

# -------------------------------------------------------
# CONFIGURE YOUR TOPICS HERE
# Key   = topic name (used in CLI and output filename)
# glob  = file pattern to match your ProQuest .txt files
# out   = output CSV filename
# -------------------------------------------------------
TOPICS = {
    # ---- Broad / catch-all tags ----
    "geopolitics": {
        "glob": "Geopolitics*.txt",
        "out":  "ft_geopolitics_articles_full.csv",
    },
    "politics": {
        "glob": "Politics*.txt",
        "out":  "ft_politics_articles_full.csv",
    },
    "world": {
        "glob": "World*.txt",
        "out":  "ft_world_articles_full.csv",
    },
    "middle-east": {
        "glob": "MiddleEast*.txt",
        "out":  "ft_middle_east_articles_full.csv",
    },
    "foreign-policy": {
        "glob": "ForeignPolicy*.txt",
        "out":  "ft_foreign_policy_articles_full.csv",
    },
    "world-affairs": {
        "glob": "WorldAffairs*.txt",
        "out":  "ft_world_affairs_articles_full.csv",
    },
    "world-news": {
        "glob": "WorldNews*.txt",
        "out":  "ft_world_news_articles_full.csv",
    },
    "global-politics": {
        "glob": "GlobalPolitics*.txt",
        "out":  "ft_global_politics_articles_full.csv",
    },
    "military-action": {
        "glob": "MilitaryAction*.txt",
        "out":  "ft_military_action_articles_full.csv",
    },
 
    # ---- Elections ----
    "elections": {
        "glob": "Elections*.txt",
        "out":  "ft_elections_articles_full.csv",
    },
    "global-elections": {
        "glob": "GlobalElections*.txt",
        "out":  "ft_global_elections_articles_full.csv",
    },
    "world-elections": {
        "glob": "WorldElections*.txt",
        "out":  "ft_world_elections_articles_full.csv",
    },
 
    # ---- People ----
    "trump": {
        "glob": "Trump*.txt",
        "out":  "ft_trump_articles_full.csv",
    },
    "trump-presidency": {
        "glob": "TrumpPresidency*.txt",
        "out":  "ft_trump_presidency_articles_full.csv",
    },
    "putin": {
        "glob": "Putin*.txt",
        "out":  "ft_putin_articles_full.csv",
    },
    "khamenei": {
        "glob": "Khamenei*.txt",
        "out":  "ft_khamenei_articles_full.csv",
    },
    "iranian-leadership-regime": {
        "glob": "IranianLeadership*.txt",
        "out":  "ft_iranian_leadership_articles_full.csv",
    },
 
    # ---- Israel / Gaza / Lebanon ----
    "israel": {
        "glob": "Israel*.txt",
        "out":  "ft_israel_articles_full.csv",
    },
    "gaza": {
        "glob": "Gaza*.txt",
        "out":  "ft_gaza_articles_full.csv",
    },
    "lebanon": {
        "glob": "Lebanon*.txt",
        "out":  "ft_lebanon_articles_full.csv",
    },
    "hezbollah": {
        "glob": "Hezbollah*.txt",
        "out":  "ft_hezbollah_articles_full.csv",
    },
    "hamas": {
        "glob": "Hamas*.txt",
        "out":  "ft_hamas_articles_full.csv",
    },
    "israel-x-iran": {
        "glob": "IsraelIran*.txt",
        "out":  "ft_israel_iran_articles_full.csv",
    },
    "daily-strikes": {
        "glob": "DailyStrikes*.txt",
        "out":  "ft_daily_strikes_articles_full.csv",
    },
 
    # ---- Iran ----
    "iran": {
        "glob": "Iran*.txt",
        "out":  "ft_iran_articles_full.csv",
    },
 
    # ---- Russia / Ukraine ----
    "ukraine": {
        "glob": "Ukraine*.txt",
        "out":  "ft_ukraine_articles_full.csv",
    },
    "russia": {
        "glob": "Russia*.txt",
        "out":  "ft_russia_articles_full.csv",
    },
    "ukraine-map": {
        "glob": "UkraineMap*.txt",
        "out":  "ft_ukraine_map_articles_full.csv",
    },
 
    # ---- Yemen / Oil / Houthis ----
    "yemen": {
        "glob": "Yemen*.txt",
        "out":  "ft_yemen_articles_full.csv",
    },
    "oil": {
        "glob": "Oil*.txt",
        "out":  "ft_oil_articles_full.csv",
    },
 
    # ---- Other regions ----
    "venezuela": {
        "glob": "Venezuela*.txt",
        "out":  "ft_venezuela_articles_full.csv",
    },
    "mention-markets": {
        "glob": "MentionMarkets*.txt",
        "out":  "ft_mention_markets_articles_full.csv",
    },
}

# -------------------------------------------------------
# PARSER
# -------------------------------------------------------

def extract_field(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    articles = re.split(r"\nTitle:\s*", text)
    rows = []

    for art in articles:
        if len(art.strip()) < 100:
            continue

        title       = extract_field(r"^(.*?)\n", art)
        author      = extract_field(r"Author:\s*(.*)", art)
        publication = extract_field(r"Publication title:\s*(.*)", art)
        section     = extract_field(r"Section:\s*(.*)", art)
        doc_type    = extract_field(r"Document type:\s*(.*)", art)
        doc_id      = extract_field(r"ProQuest document ID:\s*(.*)", art)
        first_page  = extract_field(r"First page:\s*(.*)", art)

        date_raw = extract_field(r"Publication date:\s*(.*)", art)
        try:
            date = dtparser.parse(date_raw).date() if date_raw else None
        except Exception:
            date = None

        fulltext_match = re.search(
            r"Full text:\s*(.*?)(?:\nSubject:|\nCopyright:|\nLast updated:|\Z)",
            art, re.S
        )
        full_text = fulltext_match.group(1).strip() if fulltext_match else None

        rows.append({
            "source_file":   os.path.basename(filepath),
            "date":          date,
            "title":         title,
            "author":        author,
            "publication":   publication,
            "section":       section,
            "first_page":    first_page,
            "document_type": doc_type,
            "proquest_id":   doc_id,
            "text":          full_text,
        })

    return rows


def run_topic(topic_name, config):
    pattern = config["glob"]
    out_path = config["out"]

    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[{topic_name}] No files found matching '{pattern}' — skipping.")
        return

    print(f"\n[{topic_name}] Found {len(files)} file(s): {files}")

    all_rows = []
    for filepath in files:
        print(f"  Parsing {filepath}...")
        all_rows.extend(parse_file(filepath))

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset=["proquest_id"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[{topic_name}] Saved {len(df):,} unique articles to '{out_path}'")
    return df


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ProQuest multi-topic parser")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help=f"Topic to parse. Options: {list(TOPICS.keys())}. Omit to run all."
    )
    args = parser.parse_args()

    if args.topic:
        topic = args.topic.lower()
        if topic not in TOPICS:
            print(f"Unknown topic '{topic}'. Available: {list(TOPICS.keys())}")
            return
        run_topic(topic, TOPICS[topic])
    else:
        print("Running all topics...")
        for topic_name, config in TOPICS.items():
            run_topic(topic_name, config)


if __name__ == "__main__":
    main()