import httpx, json
r = httpx.get("https://gamma-api.polymarket.com/events", params={
    "limit": 1, "closed": "true", "tag_slug": "geopolitics"
})
print(json.dumps(r.json()[0], indent=2))