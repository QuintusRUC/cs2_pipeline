import requests
import re

market_hash_name = "AK-47 | Legion of Anubis (Field-Tested)"

# This is the listing page in your browser
listing_url = "https://steamcommunity.com/market/listings/730/" + requests.utils.quote(market_hash_name)

headers = {"User-Agent": "Mozilla/5.0"}

print("Fetching listing page:")
print(listing_url)

r = requests.get(listing_url, headers=headers)
print("Status:", r.status_code)

html = r.text

# We search the HTML for a pattern where Steam calls:
# Market_LoadOrderSpread( <some_number> )
m = re.search(r"Market_LoadOrderSpread\(\s*(\d+)\s*\)", html)

if not m:
    print("\nCould not find item_nameid in the HTML.")
    print("First 300 chars of page:")
    print(html[:300])
else:
    item_nameid = m.group(1)
    print("\nFound item_nameid:", item_nameid)
