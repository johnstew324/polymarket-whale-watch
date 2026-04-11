# parses ProQuest FT article exports into per-topic CSVs
# adapted from homework 3 parsing logic
#
# to add a new topic: add it to the TOPICS list and re-run

import re
import glob
import pandas as pd
from pathlib import Path
from dateutil import parser as dtparser

# add topics here as files are downloaded .
# common tags in the market data
TOPICS = [
    "gaza",
    "hamas",
    "hezbollah",
    "iran",
    "israel",
    "lebanon",
    "oil",
    "putin",
    "trump",
    "ukraine",
    "venezuela",
    "yemen",
    # add new topics here as files are downloaded
    # "syria", "china","north-korea", "south-korea","india-pakistan", "turkey","japan","poland",
    # "uk","saudi-arabia","france","nato","qatar","taiwan","zelensky", "khamenei","netanyahu","houthis","thailand-cambodia","tariffs", "trade-war", "us-iran", "venezuela",
]

IN_DIR  = Path("data/raw/proquest")
OUT_DIR = Path("data/processed/proquest")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def parse_topic(topic):
    files = sorted(glob.glob(str(IN_DIR / topic / f"{topic}*.txt")))

    if not files:
        print(f"  [{topic}] no files found — skipping")
        return 0

    out_path = OUT_DIR / f"{topic}_articles.csv"
    all_rows = []

    for file in files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # split on Title: field — consistent across all ProQuest exports
        articles = re.split(r"\nTitle:\s*", text)

        for art in articles:
            if len(art.strip()) < 100:
                continue

            title       = extract(r"^(.*?)\n", art)
            author      = extract(r"Author:\s*(.*)", art)
            publication = extract(r"Publication title:\s*(.*)", art)
            doc_type    = extract(r"Document type:\s*(.*)", art)
            doc_id      = extract(r"ProQuest document ID:\s*(.*)", art)

            # date
            date_raw = extract(r"Publication date:\s*(.*)", art)
            try:
                date = dtparser.parse(date_raw).date() if date_raw else None
            except Exception:
                date = None
            # reject dates before 2022 - these are historical references in article text
            if date and date.year < 2022:
                date = None

            # full text — extract until next major field or end of article
            fulltext_match = re.search(
                r"Full text:\s*(.*?)(?:\nSubject:|\nCopyright:|\nLast updated:|\Z)",
                art,
                re.S
            )
            full_text = fulltext_match.group(1).strip() if fulltext_match else None

            all_rows.append({
                "source_file": file,
                "date":        date,
                "title":       title,
                "author":      author,
                "publication": publication,
                "document_type": doc_type,
                "proquest_id": doc_id,
                "text":        full_text,
                "topic":       topic,
            })

    df = pd.DataFrame(all_rows).dropna(subset=["text", "date"])

    # remove duplicate articles that appear across multiple files
    if "proquest_id" in df.columns:
        df = df.drop_duplicates(subset=["proquest_id"])

    df.to_csv(out_path, index=False, encoding="utf-8")
    return len(df)


print(f"Parsing {len(TOPICS)} topics...")

total = 0
for topic in TOPICS:
    n = parse_topic(topic)
    if n > 0:
        df = pd.read_csv(OUT_DIR / f"{topic}_articles.csv")
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        date_range = f"{dates.min().date()} → {dates.max().date()}" if len(dates) else "no dates"
        print(f"[{topic}] {n:,} articles  ({date_range})")
        total += n

print(f"\nDone. {total:,} total articles across {len(TOPICS)} topics")
print(f"Output: {OUT_DIR}")