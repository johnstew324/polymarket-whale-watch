# Normalises raw on-chain trade data into wallet-centric rows.
# Each raw fill becomes two rows - one for maker, one for taker.
# This lets us query by wallet directly without checking both maker/taker columns.
import csv
from pathlib import Path

IN  = Path("data/raw/trades_raw.csv")
OUT = Path("data/processed/trades_clean.csv")

HEADER = ["timestamp", "wallet", "side", "outcomes", "usd_amount", "token_amount", "price", "condition_id", "transactionHash"]

skipped = 0
written = 0

with open(IN) as f_in, open(OUT, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=HEADER)
    writer.writeheader()

    for row in reader:
        # makerAssetId == "0" means the maker paid USDC, so they are buying the outcome token
        # makerAssetId != "0" means the maker paid the outcome token, so they are selling
        maker_is_buyer = row["makerAssetId"] == "0"

        if maker_is_buyer:
            usd_amount   = float(row["makerAmountFilled"]) / 1e6
            token_amount = float(row["takerAmountFilled"]) / 1e6
        else:
            usd_amount   = float(row["takerAmountFilled"]) / 1e6
            token_amount = float(row["makerAmountFilled"]) / 1e6

        # skip dust trades where token amount rounds to zero
        if token_amount == 0:
            skipped += 1
            continue

        price    = usd_amount / token_amount
        outcomes = row["outcomes"]
        cid      = row["condition_id"]
        tx       = row["transactionHash"]
        ts       = row["timestamp"]

        # maker row
        writer.writerow({
            "timestamp":    ts,
            "wallet":       row["maker"],
            "side":         "BUY" if maker_is_buyer else "SELL",
            "outcomes":     outcomes,
            "usd_amount":   round(usd_amount, 6),
            "token_amount": round(token_amount, 6),
            "price":        round(price, 6),
            "condition_id": cid,
            "transactionHash": tx
        })

        # taker row - opposite side of the same fill
        writer.writerow({
            "timestamp":    ts,
            "wallet":       row["taker"],
            "side":         "SELL" if maker_is_buyer else "BUY",
            "outcomes":     outcomes,
            "usd_amount":   round(usd_amount, 6),
            "token_amount": round(token_amount, 6),
            "price":        round(price, 6),
            "condition_id": cid,
            "transactionHash": tx
        })

        written += 2

print(f"Done: {written:,} rows written, {skipped:,} skipped (zero token amount)")