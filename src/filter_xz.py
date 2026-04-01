# Extracts trades for our target markets from the full Polymarket trade history snapshot.

# The snapshot (orderFilled_complete.csv.xz) contains every trade ever on Polymarket -  150M+ rows 
# downloaded https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz 
# mainted by (maintained by warproxxx/poly_data, credits to @PendulumFlow)

import json
import subprocess # this is used to stream the xz file without decompressing it all at once ( )
import csv
from pathlib import Path

IN = Path("orderFilled_complete.csv.xz") # the full trade history snapshot downloaded 
OUT = Path("data/raw/trades_raw.csv")

# each trade row has makerAssetId and takerAssetId:
# one will be a token ID, the other will be 0 (USDC)
token_to_condition = json.loads(Path("data/raw/token_to_condition.json").read_text()) # load the token->condition mapping

print(f"Filtering {len(token_to_condition):,} token IDs from {IN}:")

matched = 0
checked = 0


# we use xzcat to stream the decompressed file line by line - so we dont have to decompress the whole 30GB file at once 
# which is too large to download and caused crashes when I tried to decompress it locally.

with subprocess.Popen(["xzcat", str(IN)], stdout=subprocess.PIPE, bufsize=1024*1024) as proc:
    reader = csv.reader(line.decode() for line in proc.stdout)
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(next(reader) + ["condition_id"])  # header
        for row in reader:
            checked += 1
            if checked % 5_000_000 == 0:
                print(f"  checked {checked:,} rows, matched {matched:,}")
            # check if either makerAssetId or takerAssetId is in our token list
            cid = token_to_condition.get(row[2]) or token_to_condition.get(row[5])
            if cid:
                writer.writerow(row + [cid])
                matched += 1

print(f"\nDone: {matched:,} matching rows saved to {OUT}")