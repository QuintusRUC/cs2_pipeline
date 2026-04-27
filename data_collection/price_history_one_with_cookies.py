import requests
from steam_auth import make_steam_session
market_hash_name = "AK-47 | Legion of Anubis (Field-Tested)"

# Paste YOUR cookie values here (keep them private)
session = make_steam_session()

URL = "https://steamcommunity.com/market/pricehistory/"
PARAMS = {
    "appid": 730,
    "market_hash_name": market_hash_name,
    "currency": 3,  # EUR; if it fails we can switch to 1 for USD
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://steamcommunity.com/market/listings/730/" + requests.utils.quote(market_hash_name),
}

print("Fetching price history for:", market_hash_name)

r = session.get(URL, params=PARAMS, headers=headers)

print("Status:", r.status_code)
print("First 80 chars of response:", r.text[:80])

# If it works, the response is a dict with keys like 'success' and 'prices'
data = r.json()

print("Success:", data.get("success"))
prices = data.get("prices", [])
print("Number of price rows:", len(prices))

print("\nFirst 3 rows:")
for row in prices[:3]:
    print(row)
