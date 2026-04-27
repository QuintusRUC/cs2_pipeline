import requests

item_nameid = "176185973"  # the number you just extracted

URL = "https://steamcommunity.com/market/pricehistory/"

# IMPORTANT:
# Instead of market_hash_name, we pass item_nameid.
# This is the internal identifier Steam uses.
PARAMS = {
    "appid": 730,
    "item_nameid": item_nameid,
    "currency": 3,  # 3 is commonly EUR; we can change later if needed
}

headers = {"User-Agent": "Mozilla/5.0"}

print("Fetching price history for item_nameid:", item_nameid)

r = requests.get(URL, params=PARAMS, headers=headers)
print("Status:", r.status_code)

data = r.json()

print("Success:", data.get("success"))
prices = data.get("prices", [])

print("Number of rows in 'prices':", len(prices))

print("\nFirst 5 price rows:")
for row in prices[:5]:
    print(row)
