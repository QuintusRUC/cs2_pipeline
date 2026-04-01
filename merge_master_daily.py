import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv

# ✅ Update these to your 2-year window files
MARKET_FILE = "market_daily_2023-09-01_to_2025-09-01.csv"
FEATURES_FILE = "item_features.csv"
OUTPUT_FILE = "master_daily_2023-09-01_to_2025-09-01.csv"

# 1) Load item features into a dict: hash_name -> feature row
features = {}

with open(FEATURES_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hn = row["hash_name"]
        features[hn] = row

print("Loaded item features:", len(features))

# 2) Read market daily rows and write merged output
missing_features = 0
rows_written = 0

with open(MARKET_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:
    market_reader = csv.DictReader(f_in)

    # Define output columns (market + features)
    fieldnames = [
        "hash_name", "date", "median_price", "volume", "currency",
        "weapon", "skin_name", "wear", "is_stattrak", "rarity",
        "sell_listings", "sell_price_text",
    ]

    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for mrow in market_reader:
        hn = mrow["hash_name"]
        frow = features.get(hn)

        if frow is None:
            missing_features += 1
            out = {
                "hash_name": hn,
                "date": mrow.get("date", ""),
                "median_price": mrow.get("median_price", ""),
                "volume": mrow.get("volume", ""),
                "currency": mrow.get("currency", ""),
                "weapon": "",
                "skin_name": "",
                "wear": "",
                "is_stattrak": "",
                "rarity": "",
                "sell_listings": "",
                "sell_price_text": "",
            }
        else:
            out = {
                "hash_name": hn,
                "date": mrow.get("date", ""),
                "median_price": mrow.get("median_price", ""),
                "volume": mrow.get("volume", ""),
                "currency": mrow.get("currency", ""),
                "weapon": frow.get("weapon", ""),
                "skin_name": frow.get("skin_name", ""),
                "wear": frow.get("wear", ""),
                "is_stattrak": frow.get("is_stattrak", ""),
                "rarity": frow.get("rarity", ""),
                "sell_listings": frow.get("sell_listings", ""),
                "sell_price_text": frow.get("sell_price_text", ""),
            }

        writer.writerow(out)
        rows_written += 1

print("Wrote:", OUTPUT_FILE)
print("Rows written:", rows_written)
print("Market rows missing features:", missing_features)