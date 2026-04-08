# Extracts trades for our target markets from the full Polymarket trade history snapshot.
#
# The snapshot (orderFilled_complete.csv.xz) contains every trade ever on Polymarket -  
# 150M+ rows (30GB) - from 2023 to October 2025 - cant get more recent data due to subgraph cutoff.
#
# reason for cutoff was Polymarket migrated to a new exchange contract around (Oct 2025).
# the new contract subgraph (polymarket-orderbook-resync/prod) exists 
# but it does not expose maker/taker wallet addresses therefore making it useless for wallet-level analysis.
# The original subgraph (orderbook-subgraph/0.0.1) stopped indexing at that point.
#
# also the polymarket data api only allows for 3100 trades per market - not enough for analysis snapshot more wallet data.
#
#
# downloaded https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz 
# mainted by (maintained by warproxxx/poly_data, credits to @PendulumFlow)

import json
import subprocess # this is used to stream the xz file without decompressing it all at once ( )
import csv
import lzma 
from pathlib import Path

IN = Path("data/raw/orderFilled_complete.csv.xz") # the full trade history snapshot downloaded 
OUT = Path("data/raw/trades_raw.csv")

# each trade row has makerAssetId and takerAssetId:
# one will be a token ID, the other will be 0 (USDC)
token_to_condition = json.loads(Path("data/raw/token_to_condition.json").read_text()) # load the token->condition mapping

print(f"Filtering {len(token_to_condition):,} token IDs from {IN}:")


def get_match(token_id):
    match = token_to_condition.get(token_id)
    return (match["condition_id"], match["outcome"]) if match else (None, None)


matched = 0
checked = 0


#for mac 
# we use xzcat to stream the decompressed file line by line - so we dont have to decompress the whole 30GB file at once 
# which is too large to download and caused crashes when I tried to decompress it locally.

with subprocess.Popen(["xzcat", str(IN)], stdout=subprocess.PIPE, bufsize=1024*1024) as proc:
    reader = csv.reader(line.decode() for line in proc.stdout)
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(next(reader) + ["condition_id", "outcomes"])  # header
        for row in reader:
            checked += 1
            if checked % 5_000_000 == 0:
                print(f"  checked {checked:,} rows, matched {matched:,}")
                # check if either makerAssetId or takerAssetId is in our token list
            cid, outcome = get_match(row[2]) 
            if not cid:
                cid, outcome = get_match(row[5])
                
            if cid:
                writer.writerow(row + [cid, outcome])
                matched += 1

#for windows 
# lzma.open streams the .xz file line by line - same behaviour as xzcat
# without decompressing the whole 30GB at once
#with lzma.open(IN, mode="rt", encoding="utf-8") as f:
#    reader = csv.reader(f)
#    with open(OUT, "w", newline="") as out_f:
#        writer = csv.writer(out_f)
#        writer.writerow(next(reader) + ["condition_id", "outcomes"])  # header
#        for row in reader:
#            checked += 1
#            if checked % 5_000_000 == 0:
#                print(f"  checked {checked:,} rows, matched {matched:,}")
#            cid, outcome = get_match(row[2])
#            if not cid:
#                cid, outcome = get_match(row[5])
#            if cid:
#                writer.writerow(row + [cid, outcome])
#                matched += 1


print(f"\nDone: {matched:,} matching rows saved to {OUT}")