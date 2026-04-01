import httpx # httpx better than requests for async and modern features - if we need it later, we can switch to async easily
import json
from pathlib import Path 

BASE = "https://gamma-api.polymarket.com"
OUT = Path("../data/raw/markets.json") 


# paginates through the API - max limit 100 per request (aroudn 16 pages for current dataset)
def fetch_all_geopolitical_events():
    all_events = []
    offset = 0 
    limit = 100 # API limit is 100 ( as per documentation )
    
    while True: # while loop because we dont know how many events
        r = httpx.get(f"{BASE}/events", params={ "limit": limit, 
                                                "closed": "true", 
                                                "tag_slug": "geopolitics",
                                                "offset": offset }) # filters: closed markets only, tagged with "geopolitics"
        r.raise_for_status()  # crash loudly if error
        batch = r.json() # API returns a list of events

        if not batch: 
            break
            
        all_events.extend(batch) # add the batch to our full list of events
        print(f"offset {offset}: got {len(batch)} events, total: {len(all_events)}")
        
        if len(batch) < limit: # last page criterion ( if we got less than the limit)
            break
            
        offset += limit
    
    return all_events

# extract relevant market info from the events 
def extract_markets(events):
    markets = [] # flatten into a list of market recordsz
    for event in events: 
        for market in event.get("markets", []):
            # derive resolved outcome from prices
            try:
                prices = json.loads(market.get("outcomePrices", "[0,0]"))
                outcomes = json.loads(market.get("outcomes", '["Yes","No"]'))
                resolved = outcomes[prices.index(max(prices))] if prices else None # resolvedOutcome is derived from outcomePrices the outcome with price=1 is the one that resolved True. 
            except Exception:
                resolved = None

            markets.append({
                "conditionId": market["conditionId"],
                "question": market["question"],
                "eventTitle": event.get("title"),
                "endDate": event.get("endDate") or market.get("endDate"),
                "closedTime": event.get("closedTime"),
                "volume": market.get("volumeNum"),
                "outcomes": market.get("outcomes"),
                "outcomePrices": market.get("outcomePrices"),
                "tags": [t["slug"] for t in event.get("tags", [])], # included other event tags to see if theres possible insider trading for certain domains 
                "resolvedOutcome": resolved, 
            })
    return markets


events = fetch_all_geopolitical_events()
markets = extract_markets(events)

OUT.write_text(json.dumps(markets, indent=2))
print(f"\nDone: {len(markets)} markets saved to {OUT}")