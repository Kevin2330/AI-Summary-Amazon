#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amazon_customers_say_ultimate_v2.py

Hardened scraper for Amazon "Customers say" AI summary.

Strategy (no Selenium by default):
  PDP (DOM -> embedded JSON -> text-based) -> Reviews page (DOM -> embedded JSON -> text-based).
Adds:
  • Stronger browser-like headers and EN-US cookies.
  • Additional JSON extractors (data-a-state, application/json, P.register(...) blobs).
  • Text-based fallback looking for "Customers say" and AI-label strings anywhere in HTML.

Usage:
  python amazon_customers_say_ultimate_v2.py "https://www.amazon.com/dp/B00JQQBPMG"
  python amazon_customers_say_ultimate_v2.py -i urls.txt -o out.csv --format csv
  python amazon_customers_say_ultimate_v2.py -i urls.txt -o out.jsonl --selenium
"""
import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Iterable, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Optional Selenium (only if --selenium)
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.chrome.options import Options  # type: ignore
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    SELENIUM_AVAILABLE = True
except Exception:
    pass

ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})|/gp/product/([A-Z0-9]{10})|/product-reviews/([A-Z0-9]{10})", re.I)
P_REGISTER_JSON_RE = re.compile(r'P\.register\(\s*["\']cr-[^"\']+["\']\s*,\s*function\s*\(\)\s*\{\s*return\s*(\{.*?\})\s*;\s*\}\s*\)\s*;', re.S)
CUSTOMERS_SAY_PHRASE_RE = re.compile(r"\bcustomers say\b", re.I)
AI_LABEL_PHRASE_RE = re.compile(r"\bAI Generated from the text of customer reviews\b", re.I)

@dataclass
class CustomersSayRecord:
    url: str
    asin: Optional[str]
    ai_summary: Optional[str]
    ai_label: Optional[str]
    source: Optional[str]
    found_widget: bool
    error: Optional[str] = None

def debug(msg: str, enabled: bool):
    if enabled:
        print(f"[debug] {msg}", file=sys.stderr)

def extract_asin_from_url(url: str) -> Optional[str]:
    m = ASIN_RE.search(url)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.upper()
    return None

def get_session() -> requests.Session:
    s = requests.Session()
    # Browser-like headers
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
        "DNT": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    # EN-US cookies help avoid localization variants
    s.cookies.set("i18n-prefs", "USD", domain=".amazon.com")
    s.cookies.set("lc-main", "en_US", domain=".amazon.com")
    return s

def fetch_html(url: str, use_selenium: bool = False, wait_sec: float = 4.0, debug_on: bool=False) -> str:
    if use_selenium:
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available. Install with: pip install selenium webdriver-manager")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1280,2200")
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        try:
            driver.get(url)
            time.sleep(wait_sec)
            html = driver.page_source
            debug(f"Selenium fetched {len(html)} chars from {url}", debug_on)
            return html
        finally:
            driver.quit()
    else:
        session = get_session()
        last_err = None
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code == 200 and resp.text:
                    debug(f"requests fetched {len(resp.text)} chars from {url}", debug_on)
                    return resp.text
                last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def extract_asin_from_dom(soup: BeautifulSoup) -> Optional[str]:
    cont = soup.select_one("#cr-product-insights-cards")
    if cont and cont.has_attr("data-asin"):
        return cont["data-asin"]
    meta_asin = soup.select_one("input#ASIN[value]")
    if meta_asin:
        return meta_asin.get("value")
    return None

def _parse_dom_for_ai_summary(soup: BeautifulSoup) -> Dict[str, Any]:
    widget = soup.select_one('div[data-hook="cr-insights-widget"], #cr-product-insights-cards')
    found_widget = bool(widget)

    ai_summary = None
    summary_container = soup.select_one('[data-hook="cr-insights-widget-summary"]')
    if summary_container:
        ai_summary = " ".join(summary_container.stripped_strings)

    ai_label = None
    ai_label_el = soup.select_one('[data-hook="cr-insights-ai-generated-text"]')
    if ai_label_el:
        ai_label = ai_label_el.get("aria-label") or ai_label_el.get_text(strip=True) or None

    asin = extract_asin_from_dom(soup)
    return {"asin": asin, "ai_summary": ai_summary, "ai_label": ai_label, "found_widget": found_widget}

def _parse_embedded_json_for_ai_summary(html: str, debug_on: bool=False) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    blobs: List[str] = []

    # application/json and data-a-state blobs
    for s in soup.find_all("script"):
        if s.get("type") == "application/json" and s.string:
            if ("insight" in s.string.lower()) or ("cr-" in s.string.lower()) or ("customers say" in s.string.lower()):
                blobs.append(s.string)
        if s.has_attr("data-a-state"):
            txt = s.text or ""
            if txt.strip():
                blobs.append(txt)

    # P.register("cr-...") style blobs
    for m in P_REGISTER_JSON_RE.finditer(html):
        blobs.append(m.group(1))

    ai_summary = None
    ai_label = None

    def safe_json_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def walk(x):
        nonlocal ai_summary, ai_label
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str):
                    val = v.strip()
                    lv = val.lower()
                    # Heuristic phrases
                    if (CUSTOMERS_SAY_PHRASE_RE.search(val) and len(val) > 30) and not ai_summary:
                        ai_summary = val
                    if (AI_LABEL_PHRASE_RE.search(val)) and not ai_label:
                        ai_label = val
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    for raw in blobs:
        obj = safe_json_load(raw)
        if obj is not None:
            walk(obj)
            if ai_summary and ai_label:
                break

    return {"ai_summary": ai_summary, "ai_label": ai_label, "found_widget": bool(ai_summary or ai_label)}

def _parse_text_fallback(html: str) -> Dict[str, Any]:
    """
    Last-resort text search for phrases. Not perfect, but often recovers the summary.
    """
    ai_summary = None
    ai_label = None

    # Try to find a "Customers say ..." paragraph (look for the phrase and capture the next ~400 chars).
    m = re.search(r"(Customers say[^.<]{0,200}(?:\.[^.<]{0,200}){0,3})", html, re.I | re.S)
    if m:
        snippet = BeautifulSoup(m.group(1), "lxml").get_text(" ", strip=True)
        if 20 <= len(snippet) <= 600:
            ai_summary = snippet

    # AI label exact phrase
    if AI_LABEL_PHRASE_RE.search(html):
        ai_label = "AI Generated from the text of customer reviews"

    return {"ai_summary": ai_summary, "ai_label": ai_label, "found_widget": bool(ai_summary or ai_label)}

def build_reviews_url(asin: str, original_url: str) -> str:
    p = urlparse(original_url)
    host = p.netloc or "www.amazon.com"
    scheme = p.scheme or "https"
    # Force English
    return f"{scheme}://{host}/product-reviews/{asin}/?reviewerType=all_reviews&language=en_US&filterByStar=all_stars"

def parse_customers_say(html: str, debug_on: bool=False) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    dom = _parse_dom_for_ai_summary(soup)
    if dom.get("ai_summary") or dom.get("ai_label") or dom.get("found_widget"):
        dom["source"] = "pdp_dom"
        return dom

    js = _parse_embedded_json_for_ai_summary(html, debug_on=debug_on)
    if js.get("ai_summary") or js.get("ai_label"):
        js["asin"] = dom.get("asin")
        js["found_widget"] = True
        js["source"] = "pdp_json"
        return js

    tx = _parse_text_fallback(html)
    if tx.get("ai_summary") or tx.get("ai_label"):
        tx["asin"] = dom.get("asin")
        tx["source"] = "pdp_text"
        return tx

    dom["source"] = "pdp_dom"
    return dom

def parse_customers_say_on_reviews(html: str, debug_on: bool=False) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    dom = _parse_dom_for_ai_summary(soup)
    if dom.get("ai_summary") or dom.get("ai_label") or dom.get("found_widget"):
        dom["source"] = "reviews_dom"
        return dom

    js = _parse_embedded_json_for_ai_summary(html, debug_on=debug_on)
    if js.get("ai_summary") or js.get("ai_label"):
        js["asin"] = dom.get("asin")
        js["found_widget"] = True
        js["source"] = "reviews_json"
        return js

    tx = _parse_text_fallback(html)
    if tx.get("ai_summary") or tx.get("ai_label"):
        tx["asin"] = dom.get("asin")
        tx["source"] = "reviews_text"
        return tx

    dom["source"] = "reviews_dom"
    return dom

@dataclass
class Result:
    url: str
    asin: Optional[str]
    ai_summary: Optional[str]
    ai_label: Optional[str]
    source: Optional[str]
    found_widget: bool
    error: Optional[str] = None

def scrape_one(url: str, use_selenium: bool = False, wait_sec: float = 4.0, debug_on: bool=False) -> Result:
    try:
        html = fetch_html(url, use_selenium=use_selenium, wait_sec=wait_sec, debug_on=debug_on)
        parsed = parse_customers_say(html, debug_on=debug_on)

        asin = parsed.get("asin") or extract_asin_from_url(url)

        if not (parsed.get("ai_summary") or parsed.get("ai_label")) and asin:
            reviews_url = build_reviews_url(asin, url)
            if debug_on:
                debug(f"Falling back to reviews page: {reviews_url}", True)
            html2 = fetch_html(reviews_url, use_selenium=use_selenium, wait_sec=wait_sec, debug_on=debug_on)
            parsed2 = parse_customers_say_on_reviews(html2, debug_on=debug_on)
            if parsed2.get("ai_summary") or parsed2.get("ai_label"):
                parsed = parsed2

        return Result(
            url=url,
            asin=asin or parsed.get("asin"),
            ai_summary=parsed.get("ai_summary"),
            ai_label=parsed.get("ai_label"),
            source=parsed.get("source"),
            found_widget=bool(parsed.get("found_widget")),
            error=None
        )
    except Exception as e:
        return Result(url=url, asin=None, ai_summary=None, ai_label=None, source=None, found_widget=False, error=str(e))

def iter_urls(args) -> Iterable[str]:
    if not sys.stdin.isatty():
        for line in sys.stdin:
            u = line.strip()
            if u: yield u
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u: yield u
    for u in args.urls or []:
        yield u

def write_jsonl(records: Iterable[Result], outpath: Optional[str] = None):
    def _write(fh):
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    if outpath:
        with open(outpath, "w", encoding="utf-8") as f:
            _write(f)
    else:
        _write(sys.stdout)

def write_csv(records: Iterable[Result], outpath: Optional[str] = None):
    fieldnames = ["url", "asin", "ai_summary", "ai_label", "source", "found_widget", "error"]
    def _write(fh):
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(asdict(rec))
    if outpath:
        with open(outpath, "w", encoding="utf-8", newline="") as f:
            _write(f)
    else:
        _write(sys.stdout)

def main():
    p = argparse.ArgumentParser(description="Scrape Amazon 'Customers say' AI summary from product pages.")
    p.add_argument("urls", nargs="*", help="Amazon product page URLs")
    p.add_argument("-i", "--infile", help="Path to a text file containing URLs (one per line)")
    p.add_argument("-o", "--out", help="Output file path (defaults to stdout)")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format (default: jsonl)")
    p.add_argument("--delay", type=float, default=2.5, help="Delay in seconds between requests (default: 2.5)")
    p.add_argument("--selenium", action="store_true", help="Use Selenium to fetch rendered HTML (slower, more robust)")
    p.add_argument("--wait", type=float, default=4.0, help="Wait seconds for Selenium page load (default: 4.0)")
    p.add_argument("--debug", action="store_true", help="Print debug info to stderr")
    args = p.parse_args()

    urls = list(dict.fromkeys(iter_urls(args)))
    if not urls:
        print("No URLs provided. Use positionals, -i file, or pipe via stdin.", file=sys.stderr)
        sys.exit(2)

    if args.selenium and not SELENIUM_AVAILABLE:
        print("Selenium requested but not available. Install with: pip install selenium webdriver-manager", file=sys.stderr)
        sys.exit(2)

    results: List[Result] = []
    for idx, u in enumerate(urls, 1):
        results.append(scrape_one(u, use_selenium=args.selenium, wait_sec=args.wait, debug_on=args.debug))
        if idx < len(urls):
            time.sleep(max(0.0, args.delay))

    if args.format == "jsonl":
        write_jsonl(results, args.out)
    else:
        write_csv(results, args.out)

if __name__ == "__main__":
    main()
