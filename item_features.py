import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import time
import re
import requests
from steam_auth import make_steam_session

INPUT_FILE = "skins.csv"
OUTPUT_FILE = "item_features.csv"
APPID = 730

HEADERS = {"User-Agent": "Mozilla/5.0"}

RARITY_CANDIDATES = [
    "Consumer Grade",
    "Industrial Grade",
    "Mil-Spec Grade",
    "Restricted",
    "Classified",
    "Covert",
    "Contraband",
]

def parse_hash_name(hash_name: str):
    """
    Example:
      'StatTrak™ AK-47 | Legion of Anubis (Field-Tested)'
    """
    is_stattrak = hash_name.startswith("StatTrak™")
    name = hash_name.replace("StatTrak™ ", "") if is_stattrak else hash_name

    wear = None
    m = re.search(r"\(([^)]+)\)$", name)
    if m:
        wear = m.group(1)
        base = name[:m.start()].strip()
    else:
        base = name

    weapon = None
    skin = None
    if " | " in base:
        weapon, skin = base.split(" | ", 1)
        weapon = weapon.strip()
        skin = skin.strip()
    else:
        weapon = base.strip()

    return is_stattrak, weapon, skin, wear

def extract_rarity_from_item(item: dict):
    desc = item.get("asset_description", {}) or {}
    tags = desc.get("tags", []) or []
    for t in tags:
        name = t.get("name") or t.get("localized_tag_name") or ""
        if name in RARITY_CANDIDATES:
            return name

    ttype = desc.get("type", "") or ""
    for cand in RARITY_CANDIDATES:
        if cand in ttype:
            return cand

    return None

def fetch_search_snapshot(session: requests.Session, hash_name: str):
    """
    Returns (sell_listings, sell_price_text, rarity) or (None, None, None) if not found.
    Retries on 429 with backoff.
    """
    url = "https://steamcommunity.com/market/search/render/"
    params = {
        "query": hash_name,
        "start": 0,
        "count": 50,
        "search_descriptions": 0,
        "sort_column": "popular",
        "sort_dir": "desc",
        "appid": APPID,
        "norender": 1
    }

    wait = 2.0
    for attempt in range(1, 6):
        r = session.get(url, params=params, headers=HEADERS)

        if r.status_code == 429:
            sleep_s = wait + (attempt * 0.3)
            print(f"429 rate limit. Sleeping {sleep_s:.1f}s (attempt {attempt}/5)")
            time.sleep(sleep_s)
            wait *= 2
            continue

        if r.status_code != 200:
            return None, None, None

        try:
            data = r.json()
        except Exception:
            return None, None, None

        results = data.get("results", [])

        # exact match preferred
        for item in results:
            if item.get("hash_name") == hash_name:
                return item.get("sell_listings"), item.get("sell_price_text"), extract_rarity_from_item(item)

        # fallback: if only one result, use it
        if len(results) == 1:
            item = results[0]
            return item.get("sell_listings"), item.get("sell_price_text"), extract_rarity_from_item(item)

        return None, None, None

    return None, None, None

def load_unique_hash_names():
    hash_names = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hash_names.append(row["hash_name"])
    return list(dict.fromkeys(hash_names))

def main():
    hash_names = load_unique_hash_names()
    print(f"Loaded {len(hash_names)} unique hash_name values from {INPUT_FILE}")

    session = make_steam_session()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hash_name",
            "weapon",
            "skin_name",
            "wear",
            "is_stattrak",
            "rarity",
            "sell_listings",
            "sell_price_text",
        ])

        for i, hn in enumerate(hash_names, start=1):
            print(f"[{i}/{len(hash_names)}] {hn}")

            is_st, weapon, skin, wear = parse_hash_name(hn)
            sell_listings, sell_price_text, rarity = fetch_search_snapshot(session, hn)

            writer.writerow([
                hn,
                weapon,
                skin,
                wear,
                is_st,
                rarity,
                sell_listings,
                sell_price_text,
            ])

            # polite delay
            time.sleep(1.5)

    print("Wrote:", OUTPUT_FILE)

if __name__ == "__main__":
    main()