# inspect_coverage.py
# diagnostic script — shows sentiment and financial signal coverage across markets
# run after load_sentiment.py to see what's covered and what's missing
#
# usage: python -m src.pipeline.inspect_coverage

import duckdb
import pandas as pd
from pathlib import Path

DB = Path("data/analytical/polymarket.ddb")
con = duckdb.connect(str(DB), read_only=True)

print(" Market Coverage Report \n")

# total resolved markets with trades
total = con.execute("""
    SELECT COUNT(DISTINCT m.conditionId)
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId)
""").fetchone()[0]

print(f"Total resolved markets with trades: {total:,}\n")

#  sentiment coverage 
print("Sentiment:")
sent_covered = con.execute("""
    SELECT COUNT(DISTINCT condition_id) FROM sentiment
""").fetchone()[0]

ts_covered = con.execute("""
    SELECT COUNT(DISTINCT condition_id) FROM sentiment WHERE source = 'truth_social'
""").fetchone()[0]

ft_covered = con.execute("""
    SELECT COUNT(DISTINCT condition_id) FROM sentiment WHERE source = 'ft_proquest'
""").fetchone()[0]

print(f"  any sentiment:   {sent_covered:,} / {total:,}  ({sent_covered/total*100:.1f}%)")
print(f"  truth_social:    {ts_covered:,} / {total:,}  ({ts_covered/total*100:.1f}%)")
print(f"  ft_proquest:     {ft_covered:,} / {total:,}  ({ft_covered/total*100:.1f}%)")

no_sentiment = total - sent_covered
print(f"  no sentiment:    {no_sentiment:,} / {total:,}  ({no_sentiment/total*100:.1f}%)\n")

#  financial coverage 
print("Financial:")
fin_covered = con.execute("""
    SELECT COUNT(DISTINCT condition_id) FROM financial
""").fetchone()[0]

no_financial = total - fin_covered
print(f"  any tickers:     {fin_covered:,} / {total:,}  ({fin_covered/total*100:.1f}%)")
print(f"  no tickers:      {no_financial:,} / {total:,}  ({no_financial/total*100:.1f}%)\n")

#  ticker breakdown 
print("Ticker breakdown (markets covered per ticker):")
tickers = con.execute("""
    SELECT ticker, COUNT(DISTINCT condition_id) as markets
    FROM financial
    GROUP BY ticker
    ORDER BY markets DESC
""").fetchdf()
for _, row in tickers.iterrows():
    print(f"  {row['ticker']:<15} {row['markets']:>4} markets")

#  sentiment source breakdown 
print("\nSentiment rows per source:")
sources = con.execute("""
    SELECT source, COUNT(*) as rows, COUNT(DISTINCT condition_id) as markets
    FROM sentiment
    GROUP BY source
    ORDER BY rows DESC
""").fetchdf()
for _, row in sources.iterrows():
    print(f"  {row['source']:<20} {row['rows']:>6,} rows  {row['markets']:>4} markets")

#  top uncovered markets (no sentiment AND no financial) 
print("\nTop 20 uncovered markets by volume (no sentiment AND no financial):")
uncovered = con.execute("""
    SELECT m.question, m.volume, m.resolvedOutcome
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId)
    AND NOT EXISTS (SELECT 1 FROM sentiment s WHERE s.condition_id = m.conditionId)
    AND NOT EXISTS (SELECT 1 FROM financial f  WHERE f.condition_id = m.conditionId)
    ORDER BY m.volume DESC
    LIMIT 20
""").fetchdf()

if uncovered.empty:
    print("  all markets covered!")
else:
    for _, row in uncovered.iterrows():
        print(f"  ${row['volume']:>10,.0f}  [{row['resolvedOutcome']}]  {row['question'][:75]}")

con.close()