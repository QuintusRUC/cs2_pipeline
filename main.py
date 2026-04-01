import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
import csv
import time
import random

URL = "https://steamcommunity.com/market/search/render/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

APPID = 730
OUTPUT_FILE = "skins.csv"

# ✅ Your 20 base skins (replace with your current list if different)
base_skins = [
    "AK-47 | Redline",
    "AK-47 | Asiimov",
    "M4A1-S | Hyper Beast",
    "M4A1-S | Printstream",
    "M4A4 | The Emperor",
    "M4A4 | Neo-Noir",
    "AUG | Syd Mead",
    "FAMAS | Roll Cage",
    "AWP | Asiimov",
    "AWP | Hyper Beast",
    "AWP | Redline",
    "SSG 08 | Dragonfire",
    "Desert Eagle | Printstream",
    "Desert Eagle | Code Red",
    "Glock-18 | Water Elemental",
    "USP-S | Kill Confirmed",
    "P250 | Asiimov",
    "Five-SeveN | Monkey Business",
    "MAC-10 | Neon Rider",
    "UMP-45 | Primal Saber",
]

# Exclude pattern-dependent finishes
PATTERN_EXCLUDE = ["Case Hardened", "Doppler", "Marble Fade", "Fade", "Crimson Web"]

# Exclude knives/gloves (simple keyword-based filter)
KNIFE_GLOVE_EXCLUDE = [
    "Gloves",
    "Knife",
    "Karambit",
    "Bayonet",
    "M9 Bayonet",
    "Butterfly Knife",
    "Flip Knife",
    "Gut Knife",
    "Huntsman Knife",
    "Bowie Knife",
    "Falchion Knife",
    "Shadow Daggers",
    "Stiletto Knife",
    "Talon Knife",
    "Ursus Knife",
    "Navaja Knife",
    "Nomad Knife",
    "Paracord Knife",
    "Survival Knife",
    "Skeleton Knife",
]

def should_exclude(hash_name: str) -> bool:
    for s in PATTERN_EXCLUDE:
        if s in hash_name:
            return True
    for s in KNIFE_GLOVE_EXCLUDE:
        if s in hash_name:
            return True
    return False

def steam_search_json(session: requests.Session, query: str, count: int = 50, max_retries: int = 8):
    """
    Calls Steam search/render with retries + exponential backoff on 429.
    Returns JSON dict on success, or None on failure.
    """
    params = {
        "query": query,
        "start": 0,
        "count": count,
        "search_descriptions": 0,
        "sort_column": "popular",
        "sort_dir": "desc",
        "appid": APPID,
        "norender": 1
    }

    wait = 2.0
    for attempt in range(1, max_retries + 1):
        r = session.get(URL, params=params, headers=HEADERS)

        if r.status_code == 429:
            sleep_s = wait + random.uniform(0, 1.0)
            print(f"⚠️  429 rate limit for '{query}'. Sleeping {sleep_s:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)
            wait *= 2
            continue

        if r.status_code != 200:
            print(f"⚠️  Bad status {r.status_code} for '{query}'. Skipping.")
            return None

        try:
            return r.json()
        except Exception:
            print(f"⚠️  Response not JSON for '{query}'. Skipping.")
            return None

    print(f"❌ Gave up after retries for '{query}'.")
    return None

def main():
    session = requests.Session()

    all_rows = []  # (base_query, hash_name)

    for base in base_skins:
        print(f"\nSearching for: {base}")

        data = steam_search_json(session, base, count=50)
        if not data:
            print("  Skipped (no data).")
            continue

        results = data.get("results", [])
        if not results:
            print("  No results.")
            continue

        # Extract hash_names
        hash_names = []
        for item in results:
            hn = item.get("hash_name")
            if hn and not should_exclude(hn):
                hash_names.append(hn)

        # Deduplicate, keep order
        seen = set()
        unique = []
        for hn in hash_names:
            if hn not in seen:
                seen.add(hn)
                unique.append(hn)

        print("  Found:", len(unique))
        for hn in unique:
            print("   -", hn)
            all_rows.append((base, hn))

        # be polite between base skins
        time.sleep(3.0)

    # Write skins.csv
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["base_query", "hash_name"])
        for base, hn in all_rows:
            writer.writerow([base, hn])

    print(f"\nDone! Saved {len(all_rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()