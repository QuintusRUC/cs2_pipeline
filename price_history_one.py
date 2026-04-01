import requests

market_hash_name = "AK-47 | Legion of Anubis (Field-Tested)"

URL = "https://steamcommunity.com/market/pricehistory/"
PARAMS = {
    "appid": 730,
    "market_hash_name": market_hash_name,
    "currency": 3,
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": f"https://steamcommunity.com/market/listings/730/{market_hash_name}"
}

print("Fetching price history for:")
print(market_hash_name)

r = requests.get(URL, params=PARAMS, headers=headers)

print("Status:", r.status_code)
print("\nFirst 300 chars of response text:")
print(r.text[:300])
