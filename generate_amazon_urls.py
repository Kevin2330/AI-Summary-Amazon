# save as generate_amazon_urls.py
import csv
import random
import time
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BEST_SELLERS_CATEGORIES = [
    "https://www.amazon.com/Best-Sellers/zgbs/electronics",
    "https://www.amazon.com/Best-Sellers/zgbs/books",
    "https://www.amazon.com/Best-Sellers/zgbs/home-garden",
    "https://www.amazon.com/Best-Sellers/zgbs/tools-home-improvement",
    "https://www.amazon.com/Best-Sellers/zgbs/sports-outdoors",
]

def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

def fetch_links_from_category(session, url, limit=50):
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in soup.select("a.a-link-normal"):
        href = a.get("href")
        if not href:
            continue
        # match product pages: /dp/ASIN or /gp/product/ASIN
        if re.search(r"/(dp|gp/product)/[A-Z0-9]{10}", href):
            full = urljoin(url, href.split("?")[0])
            links.append(full)
    # dedupe
    links = list(dict.fromkeys(links))
    return links[:limit]

def main():
    session = get_session()
    all_links = []
    for cat in BEST_SELLERS_CATEGORIES:
        print(f"Fetching category: {cat}")
        links = fetch_links_from_category(session, cat, limit=100)
        print(f"  found {len(links)} links")
        all_links.extend(links)
        time.sleep(2)  # polite pause

    # shuffle & pick 200
    random.shuffle(all_links)
    selected = all_links[:200]

    # Write CSV
    out_file = "amazon_urls_200.csv"
    with open(out_file, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url"])
        for u in selected:
            w.writerow([u])

    print(f"Wrote {len(selected)} URLs to {out_file}")

if __name__ == "__main__":
    main()
