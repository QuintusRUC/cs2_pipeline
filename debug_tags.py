import requests
from bs4 import BeautifulSoup
from steam_auth import make_steam_session

hash_name = "AK-47 | Legion of Anubis (Field-Tested)"
url = "https://steamcommunity.com/market/listings/730/" + requests.utils.quote(hash_name)

session = make_steam_session()

r = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("Status:", r.status_code)
print("URL:", url)

soup = BeautifulSoup(r.text, "lxml")

tags = [t.get_text(strip=True) for t in soup.select(".app_tag")]
print("\nFound", len(tags), "tags using .app_tag:")
for t in tags:
    print("-", t)

tag_block = soup.select_one(".app_tags")
print("\nHas .app_tags block:", tag_block is not None)
if tag_block:
    print("Block text preview:")
    print(tag_block.get_text(" | ", strip=True)[:600])
