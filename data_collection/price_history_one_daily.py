import requests
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil import parser
import re
from steam_auth import make_steam_session

market_hash_name = "AK-47 | Legion of Anubis (Field-Tested)"

session = make_steam_session()

URL = "https://steamcommunity.com/market/pricehistory/"
PARAMS = {
    "appid": 730,
    "market_hash_name": market_hash_name,
    "currency": 3,
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://steamcommunity.com/market/listings/730/" + requests.utils.quote(market_hash_name),
}

def normalize_steam_datetime(dt_str: str) -> str:
    """
    Fixes weird Steam datetime strings like:
      'Aug 07 2020 01: +0'  -> 'Aug 07 2020 01:00 +0'
    so dateutil can parse them.
    """
    s = str(dt_str).strip()

    # Case: '... HH: +0' (hour with colon but no minutes)
    # Insert '00' minutes.
    s = re.sub(r"(\b\d{1,2}):\s*\+0\b", r"\1:00 +0", s)

    # Sometimes timezone appears as just '+0' without space issues
    s = re.sub(r"\s*\+0\b", " +0", s)

    return s

def parse_steam_int(int_str: str) -> int:
    return int(str(int_str).strip().replace(",", ""))


r = session.get(URL, params=PARAMS, headers=headers)
data = r.json()
prices = data.get("prices", [])

print("Fetched raw rows:", len(prices))

# 2) Aggregate by day
daily_prices = defaultdict(list)
daily_volume = defaultdict(int)

bad_rows = 0
first_error = None

for row in prices:
    dt_str, price_val, vol_str = row[0], row[1], row[2]

    try:
        dt_fixed = normalize_steam_datetime(dt_str)
        dt = parser.parse(dt_fixed, fuzzy=True)
        day = dt.strftime("%Y-%m-%d")

        # price_val is already numeric in your case
        price = float(price_val)
        vol = parse_steam_int(vol_str)

        daily_prices[day].append(price)
        daily_volume[day] += vol

    except Exception as e:
        bad_rows += 1
        if first_error is None:
            first_error = (row, str(e))

print("Bad rows skipped:", bad_rows)
if first_error:
    print("\nExample of first skipped row + error:")
    print("Row:", first_error[0])
    print("Error:", first_error[1])

def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    if n % 2 == 1:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])

daily_rows = []
for day in daily_prices:
    med = median(daily_prices[day])
    vol = daily_volume[day]
    daily_rows.append((day, med, vol))

daily_rows.sort(key=lambda x: x[0])

print("\nDaily rows total:", len(daily_rows))
print("Last 3 daily rows:")
for row in daily_rows[-3:]:
    print(row)

# 3) Keep last 30 days
cutoff_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
last_30 = [row for row in daily_rows if row[0] >= cutoff_date]

print("Daily rows (last 30 days):", len(last_30))

# 4) Save to CSV
output_file = "price_history_one_daily.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["hash_name", "date", "median_price", "volume", "currency"])
    for day, med, vol in last_30:
        writer.writerow([market_hash_name, day, med, vol, PARAMS["currency"]])

print("Saved:", output_file)
