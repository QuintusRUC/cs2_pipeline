import requests

market_hash_name = "AK-47 | Legion of Anubis (Field-Tested)"

URL = "https://steamcommunity.com/market/pricehistory/"
PARAMS = {
    "appid": 730,
    "market_hash_name": market_hash_name,
    "currency": 3
}

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, params=PARAMS, headers=headers)

print("Status:", r.status_code)
print("Raw response text:", r.text[:200])
