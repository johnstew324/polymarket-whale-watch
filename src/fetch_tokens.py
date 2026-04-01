# maps each market's condition ID to its YES/NO token IDs using the Polymarket CLOB API
import httpx
import json
import time
from pathlib import Path

BASE = "https://clob.polymarket.com"

# load markets and filter to high volume ones (>= $1k) 1`1`
markets = json.loads(Path("data/raw/markets.json").read_text())

high_vol = []
for m in markets:
    if (m.get("volume") or 0) >= 1000:
        high_vol.append(m)

print(f"Fetching tokens for {len(high_vol)} markets (volume >= $1k):")


token_to_condition = {}
for i, m in enumerate(high_vol):
    cid = m["conditionId"]
    try:
        r = httpx.get(f"{BASE}/markets/{cid}")
        r.raise_for_status()

        # each market has 2 tokens (YES/NO), we map both to the same condition ID
    
        for token in r.json().get("tokens", []):
            token_to_condition[str(token["token_id"])] = {
                "condition_id": cid,
                "outcome": token.get("outcome")  # "Yes" or "No"
            }

    # catch and log any errors
    except Exception as e:
        print(f"ERROR {cid[:20]}: {e}")

    # print progress every 100 markets 
    if i % 100 == 0:
        print(f"{i}/{len(high_vol)} - {len(token_to_condition)} tokens")
    time.sleep(0.1) # avoid rate limits 

# Output: data/raw/token_to_condition.json  {token_id: condition_id}
Path("data/raw/token_to_condition.json").write_text(json.dumps(token_to_condition, indent=2))
print(f"\nDone: {len(token_to_condition)} token IDs saved")