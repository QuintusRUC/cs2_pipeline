import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
import csv
import time
import random
from collections import defaultdict
from dateutil import parser
import re
import os
from steam_auth import make_steam_session

# ---------------- Config ----------------
CURRENCY = 3  # EUR
APPID = 730

INPUT_SKINS_FILE = "skins.csv"
OUTPUT_FILE = "market_daily_2023-09-01_to_2025-09-01.csv"

START_DATE = "2023-09-01"
END_DATE   = "2025-09-01"

URL = "https://steamcommunity.com/market/pricehistory/"

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}

# Slow down a lot to avoid 429
SLEEP_BETWEEN_ITEMS = 6.0

# Collect in batches so Steam doesn't hard-throttle you.
# Example: first run 0..49, next run 50..99, etc.
BATCH_START = 150
BATCH_END = 186  # change this each run

# ---------------- Helpers ----------------
def normalize_steam_datetime(dt_str: str) -> str:
    s = str(dt_str).strip()
    s = re.sub(r"(\b\d{1,2}):\s*\+0\b", r"\1:00 +0", s)
    s = re.sub(r"\s*\+0\b", " +0", s)
    return s

def parse_steam_int(int_str: str) -> int:
    return int(str(int_str).strip().replace(",", ""))

def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    if n % 2 == 1:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])

def get_pricehistory_json(session: requests.Session, market_hash_name: str, max_retries=5):
    params = {
        "appid": APPID,
        "market_hash_name": market_hash_name,
        "currency": CURRENCY,
    }

    headers = dict(BASE_HEADERS)
    headers["Referer"] = "https://steamcommunity.com/market/listings/730/" + requests.utils.quote(market_hash_name)

    wait = 2.0
    for attempt in range(1, max_retries + 1):
        r = session.get(URL, params=params, headers=headers)

        if r.status_code in (429, 400):
            sleep_s = wait + random.uniform(0, 1.5)
            print(f"  ⚠️  HTTP {r.status_code} for '{market_hash_name}'. Sleeping {sleep_s:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)
            wait *= 2
            continue

        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"

        try:
            data = r.json()
        except Exception:
            return None, "non-JSON response"

        if not data.get("success"):
            sleep_s = wait + random.uniform(0, 1.0)
            print(f"  ⚠️  success=False. Sleeping {sleep_s:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)
            wait *= 2
            continue

        return data, None

    return None, "gave up after retries"

def daily_aggregate_in_range(pricehistory_json: dict):
    prices = pricehistory_json.get("prices", [])
    if not prices:
        return []

    daily_prices = defaultdict(list)
    daily_volume = defaultdict(int)

    for row in prices:
        dt_str, price_val, vol_str = row[0], row[1], row[2]

        dt_fixed = normalize_steam_datetime(dt_str)
        dt = parser.parse(dt_fixed, fuzzy=True)
        day = dt.strftime("%Y-%m-%d")

        if day < START_DATE or day > END_DATE:
            continue

        price = float(price_val)
        vol = parse_steam_int(vol_str)

        daily_prices[day].append(price)
        daily_volume[day] += vol

    daily_rows = []
    for day in daily_prices:
        daily_rows.append((day, median(daily_prices[day]), daily_volume[day]))

    daily_rows.sort(key=lambda x: x[0])
    return daily_rows

# ---------------- Main ----------------
# Load hash_names
hash_names = []
with open(INPUT_SKINS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hash_names.append(row["hash_name"])
hash_names = list(dict.fromkeys(hash_names))

print(f"Loaded {len(hash_names)} unique hash_name values from {INPUT_SKINS_FILE}")

# Batch slice
batch = hash_names[BATCH_START:BATCH_END]
print(f"Running batch: {BATCH_START}..{BATCH_END-1}  (items: {len(batch)})")

session = make_steam_session()

total_rows_written = 0
errors = 0

# ✅ APPEND MODE + only write header once
file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    if not file_exists:
        writer.writerow(["hash_name", "date", "median_price", "volume", "currency"])

    for i, hn in enumerate(batch, start=1):
        print(f"\n[{i}/{len(batch)}] Fetching: {hn}")

        data, err = get_pricehistory_json(session, hn)
        if err:
            print("  ERROR:", err)
            errors += 1
        else:
            daily_rows = daily_aggregate_in_range(data)
            print("  Days returned:", len(daily_rows))

            for day, med, vol in daily_rows:
                writer.writerow([hn, day, med, vol, CURRENCY])
                total_rows_written += 1

        time.sleep(SLEEP_BETWEEN_ITEMS + random.uniform(0, 1.0))

print(f"\nDone! Appended {total_rows_written} rows to {OUTPUT_FILE}")
print(f"Errors: {errors}")
print("Next run: set BATCH_START/BATCH_END to the next slice (e.g., 50..100).")