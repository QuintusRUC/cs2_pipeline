import json
import requests

def make_steam_session(secrets_path: str = "secrets.json") -> requests.Session:
    """
    Creates a requests.Session() with Steam cookies loaded from secrets.json.

    secrets.json must look like:
    {
      "steamLoginSecure": "...",
      "sessionid": "..."
    }
    """
    with open(secrets_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)

    for k in ("steamLoginSecure", "sessionid"):
        if k not in cookies or not cookies[k]:
            raise ValueError(f"Missing cookie '{k}' in {secrets_path}")

    s = requests.Session()
    s.cookies.update({
        "steamLoginSecure": cookies["steamLoginSecure"],
        "sessionid": cookies["sessionid"],
    })
    return s
