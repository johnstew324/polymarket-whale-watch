# pretty messy AI- claude script so need to clean up etc but the idea is to help reduce proquest searchs 
# maybe delete when finished !


# generates targeted ProQuest FT search queries from NER keyword analysis
# of all resolved Polymarket geopolitical markets with trade data
#
# methodology:
#   - extracts NER keywords from each market question using ner_keywords.py
#   - groups markets by their top-3 keyword combination
#   - computes date range per group (earliest market open → latest close)
#   - adds 30-day context window, caps to Jan 2023 → Oct 2025
#   - aggregates child combo market counts into 2-keyword parents
#     e.g. "House of Councillors AND Japan AND JCP" (2 markets) +
#          "House of Councillors AND Japan AND Komeito" (2 markets) +
#          "House of Councillors AND Japan" (2 markets)
#          = "House of Councillors AND Japan" shows 6 total markets
#
# filtering rules applied before checklist output:
#   1. searches covering fewer than MIN_MARKETS markets are skipped
#   2. single-keyword searches for broad terms are skipped
#   3. 3-keyword searches are collapsed to their 2-keyword parent if parent exists
#
# outputs:
#   data/proquest_checklist.md filtered actionable download checklist
#   data/proquest_search_list.txt full unfiltered list for methodology reference
#

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import duckdb
import pandas as pd
from collections import defaultdict
from ner_keywords import extract_keywords


DATE_FLOOR   = pd.Timestamp("2023-01-01", tz="UTC")
DATE_CAP     = pd.Timestamp("2025-10-31", tz="UTC")
CONTEXT_DAYS = 30

# skip searches covering fewer than this many markets (after aggregation)
MIN_MARKETS = 3

# single-keyword searches for these broad terms are skipped —
# they return too many FT articles and are covered by specific multi-keyword searches
SKIP_IF_SINGLE = {
    "russia", "iran", "israel", "ukraine", "usa", "china",
    "trump", "nato", "eu", "uk", "gaza",
}

CHECKLIST_OUT = Path("data/proquest_checklist.md")
FULL_LIST_OUT = Path("data/proquest_search_list.txt")

con = duckdb.connect("data/analytical/polymarket.ddb", read_only=True)
markets = con.execute("""
    SELECT question, startDate, endDate
    FROM markets
    WHERE resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (SELECT 1 FROM trades t WHERE t.condition_id = conditionId)
""").fetchdf()
con.close()

markets["startDate"] = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
markets["endDate"]   = pd.to_datetime(markets["endDate"],   utc=True, errors="coerce")

# build keyword combinations 
combo_data = defaultdict(lambda: {"count": 0, "start": None, "end": None})

for _, row in markets.iterrows():
    kws = extract_keywords(row["question"])
    if not kws:
        continue
    combo = tuple(sorted(kws[:3]))
    combo_data[combo]["count"] += 1

    if pd.notna(row["startDate"]):
        if combo_data[combo]["start"] is None or row["startDate"] < combo_data[combo]["start"]:
            combo_data[combo]["start"] = row["startDate"]

    if pd.notna(row["endDate"]):
        if combo_data[combo]["end"] is None or row["endDate"] > combo_data[combo]["end"]:
            combo_data[combo]["end"] = row["endDate"]



# 3-keyword combos contribute their market count to all their 2-keyword subsets
# so "House of Councillors AND Japan AND JCP" (2) adds to "House of Councillors AND Japan"
parent_extra = defaultdict(int)
for combo, data in combo_data.items():
    if len(combo) == 3:
        subsets = [
            (combo[0], combo[1]),
            (combo[0], combo[2]),
            (combo[1], combo[2]),
        ]
        for s in subsets:
            parent_extra[s] += data["count"]

# build aggregated counts  for 2-keyword combos add child contributions
aggregated_counts = {}
for combo, data in combo_data.items():
    if len(combo) == 2:
        aggregated_counts[combo] = data["count"] + parent_extra.get(combo, 0)
    else:
        aggregated_counts[combo] = data["count"]


def get_dates(data):
    start = max(data["start"] - pd.Timedelta(days=CONTEXT_DAYS), DATE_FLOOR) \
            if data["start"] is not None and pd.notna(data["start"]) else DATE_FLOOR
    end   = min(data["end"], DATE_CAP) \
            if data["end"] is not None and pd.notna(data["end"]) else DATE_CAP
    return start, end


def to_folder(combo):
    return "_".join(kw.lower().replace(" ", "_") for kw in combo)


# sort by aggregated count
sorted_combos = sorted(combo_data.items(), key=lambda x: -aggregated_counts[x[0]])

#  write full unfiltered list
full_lines = [f"{'Markets':>7}  {'Date Range':<25}  Query", "-" * 90]
for combo, data in sorted_combos:
    query = " AND ".join(f'"{kw}"' for kw in combo)
    start, end = get_dates(data)
    count = aggregated_counts[combo]
    full_lines.append(
        f"{count:>7}  {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d'):<12}  {query}"
    )
full_lines.append(f"\nTotal combinations: {len(combo_data)} | Total markets: {len(markets)}")
FULL_LIST_OUT.parent.mkdir(parents=True, exist_ok=True)
FULL_LIST_OUT.write_text("\n".join(full_lines))
print(f"Full list saved to {FULL_LIST_OUT}")

#  apply filters for checklist

# 2keyword combos present after MIN_MARKETS filter — used for 3-keyword collapse check
two_keyword_combos = {
    combo for combo, data in combo_data.items()
    if len(combo) == 2 and aggregated_counts[combo] >= MIN_MARKETS
}

filtered = []
for combo, data in sorted_combos:
    count = aggregated_counts[combo]

    # rule 1: skip below minimum market threshold
    if count < MIN_MARKETS:
        continue

    # rule 2: skip single-keyword searches for broad terms
    if len(combo) == 1 and combo[0].lower() in SKIP_IF_SINGLE:
        continue

    # rule 3:collapse 3-keyword searches to 2-keyword parent if parent exists
    if len(combo) == 3:
        subsets = [
            (combo[0], combo[1]),
            (combo[0], combo[2]),
            (combo[1], combo[2]),
        ]
        if any(s in two_keyword_combos for s in subsets):
            continue

    start, end = get_dates(data)
    filtered.append((combo, count, data, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))


checklist_lines = [
    "# ProQuest FT Article Download Checklist",
    "",
    "Generated from NER keyword analysis of 1,801 resolved Polymarket geopolitical markets.",
    "Market counts for 2-keyword searches include markets from 3-keyword child combinations.",
    "",
    "",
]

for combo, count, data, start_str, end_str in filtered:
    query  = " AND ".join(f'"{kw}"' for kw in combo)
    folder = to_folder(combo)
    checklist_lines.append(
        f"- [ ] `{folder}` — {query} | {count} markets | {start_str} → {end_str}"
    )

checklist_lines.append("")
checklist_lines.append(f"*{len(filtered)} searches | {len(markets)} total markets*")

CHECKLIST_OUT.write_text("\n".join(checklist_lines))
print(f"Checklist saved to {CHECKLIST_OUT} ({len(filtered)} searches)")