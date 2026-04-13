import duckdb
from src.sentiment.ner_keywords import extract_keywords
from src.sentiment.geo_vocab_mapping import KNOWN_NAMES, STOPWORDS

con = duckdb.connect("data/analytical/polymarket.ddb", read_only=True)
questions = con.execute("""
    SELECT m.conditionId, m.question 
    FROM markets m
    WHERE m.resolvedOutcome IN ('Yes', 'No')
    AND EXISTS (
        SELECT 1 FROM trades t WHERE t.condition_id = m.conditionId
    )
""").fetchall()

empty = []
single = []
keyword_counts = []

for cid, q in questions:
    kws = extract_keywords(q)
    keyword_counts.append(len(kws))
    if not kws:
        empty.append(q)
    elif len(kws) == 1:
        single.append((q, kws))

print(f"Total markets:        {len(questions)}")
print(f"Empty keyword lists:  {len(empty)}")
print(f"Single keyword only:  {len(single)}")
print(f"Avg keywords/market:  {sum(keyword_counts)/len(keyword_counts):.2f}")

print(f"\nEmpty markets ({len(empty)})")
for q in empty:
    print(f"  {q}")

print(f"\nSample of single-keyword markets (first 20)")
for q, kws in single[:20]:
    print(f"  {kws} ← {q}")


from collections import Counter
all_keywords = []
for cid, q in questions:
    all_keywords.extend(extract_keywords(q))

freq = Counter(all_keywords).most_common()
print(f"\n All {len(freq)} unique keywords")
for kw, count in freq:
    print(f"  {count:>4}  {kw}")