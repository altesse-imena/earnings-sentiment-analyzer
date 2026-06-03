"""
Fetches earnings call transcripts from SEC EDGAR full-text search API.
No API key required. Caches results to avoid redundant calls.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22earnings+call%22+%22{ticker}%22&forms=8-K&dateRange=custom&startdt={start}&enddt={end}"
EDGAR_FULL_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22+%22earnings+call%22&forms=8-K&dateRange=custom&startdt={start}&enddt={end}"
HEADERS = {"User-Agent": "earnings-sentiment-analyzer research@example.com"}

CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/raw/cache"))
TRANSCRIPT_DIR = Path("data/raw/transcripts")


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve a ticker symbol to an SEC CIK number."""
    cache_key = f"cik_{ticker.upper()}"
    cached = _load_cache(cache_key)
    if cached:
        logger.debug(f"CIK cache hit for {ticker}")
        return cached.get("cik")

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(tickers_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        tickers_data = resp.json()
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _save_cache(cache_key, {"cik": cik})
                logger.info(f"Resolved {ticker} → CIK {cik}")
                return cik
    except Exception as e:
        logger.error(f"Failed to resolve CIK for {ticker}: {e}")
    return None


def get_company_name_for_ticker(ticker: str) -> str | None:
    """Resolve a ticker symbol to the official SEC company name for full-text search."""
    cache_key = f"company_name_{ticker.upper()}"
    cached = _load_cache(cache_key)
    if cached:
        return cached.get("name")

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(tickers_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                name = entry.get("title", "")
                _save_cache(cache_key, {"name": name})
                logger.info(f"Resolved {ticker} → company name '{name}'")
                return name
    except Exception as e:
        logger.error(f"Failed to resolve company name for {ticker}: {e}")
    return None


def search_8k_filings(ticker: str, start_year: int, end_year: int) -> list[dict]:
    """Search SEC EDGAR for 8-K filings using the company's submissions endpoint.

    Uses the CIK-scoped submissions API (data.sec.gov/submissions/CIK{cik}.json)
    which returns only that company's own filings — reliable regardless of whether
    the company name appears in other companies' documents.
    """
    cache_key = f"filings_{ticker}_{start_year}_{end_year}"
    cached = _load_cache(cache_key)
    if cached:
        logger.info(f"Filings cache hit for {ticker} ({start_year}-{end_year})")
        return cached

    results = []

    cik = get_cik_for_ticker(ticker)
    if not cik:
        logger.warning(f"Could not resolve CIK for {ticker}; skipping EDGAR search")
        _save_cache(cache_key, results)
        return results

    company_name = get_company_name_for_ticker(ticker) or ticker

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        logger.info(f"Fetching submissions for '{company_name}' ({ticker}, CIK {cik})")
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])
        primary_doc = recent.get("primaryDocument", [])

        for i in range(len(forms)):
            if forms[i] != "8-K":
                continue
            filing_year = int(dates[i][:4]) if dates[i] else 0
            if not (start_year <= filing_year <= end_year):
                continue

            acc = accessions[i]
            cik_int = int(cik)
            results.append({
                "ticker": ticker,
                "accession_no": acc,
                "file_date": dates[i],
                "entity_name": company_name,
                "form_type": forms[i],
                "primary_doc": primary_doc[i] if i < len(primary_doc) else "",
                "file_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/",
            })

        time.sleep(0.3)
    except Exception as e:
        logger.error(f"Submissions fetch failed for {ticker}: {e}")

    _save_cache(cache_key, results)
    logger.info(f"Found {len(results)} 8-K filings for {ticker} ({start_year}-{end_year})")
    return results


def fetch_transcript_text(accession_no: str, ticker: str, primary_doc: str = "") -> str | None:
    """Fetch the full text of an 8-K filing from EDGAR.

    Tries the primary_doc path directly first; falls back to parsing the filing
    index page to find the best document link.
    """
    cache_key = f"transcript_{accession_no.replace('-', '_')}"
    cached = _load_cache(cache_key)
    if cached:
        return cached.get("text")

    acc_clean = accession_no.replace("-", "")
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return None

    cik_int = int(cik)
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"
    from bs4 import BeautifulSoup

    def _extract_text(url: str) -> str | None:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.debug(f"Could not fetch {url}: {e}")
            return None

    # 1. Try primary document directly if provided
    if primary_doc:
        text = _extract_text(f"{base_url}/{primary_doc}")
        if text and len(text) > 500:
            _save_cache(cache_key, {"text": text})
            return text

    # 2. Parse the filing index to find the best exhibit document
    index_url = f"{base_url}/{accession_no}-index.htm"
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        doc_link = None

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                desc = cells[1].get_text(strip=True).lower()
                if any(k in desc for k in ["transcript", "exhibit 99", "ex-99", "ex99"]):
                    a = cells[2].find("a")
                    if a:
                        doc_link = "https://www.sec.gov" + a["href"]
                        break

        if not doc_link:
            first_table = soup.find("table", {"class": "tableFile"})
            if first_table:
                a = first_table.find("a")
                if a:
                    doc_link = "https://www.sec.gov" + a["href"]

        if doc_link:
            time.sleep(0.3)
            text = _extract_text(doc_link)
            if text and len(text) > 500:
                _save_cache(cache_key, {"text": text})
                return text

    except Exception as e:
        logger.error(f"Failed to fetch transcript {accession_no}: {e}")
    return None


def run_ingestion(tickers: list[str], years: list[int]):
    """Main ingestion entry point."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    start_year = min(years)
    end_year = max(years)

    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        filings = search_8k_filings(ticker, start_year, end_year)

        ticker_dir = TRANSCRIPT_DIR / ticker.upper()
        ticker_dir.mkdir(exist_ok=True)

        saved = 0
        for filing in filings:
            acc = filing["accession_no"]
            if not acc:
                continue
            out_path = ticker_dir / f"{filing['file_date']}_{acc.replace('-', '_')}.txt"
            if out_path.exists():
                logger.debug(f"Already saved: {out_path.name}")
                continue

            text = fetch_transcript_text(acc, ticker, primary_doc=filing.get("primary_doc", ""))
            if text and len(text) > 500:
                out_path.write_text(text)
                saved += 1
                logger.info(f"Saved transcript: {out_path.name}")
            time.sleep(0.5)

        logger.info(f"{ticker}: saved {saved} new transcripts (total filings found: {len(filings)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch earnings transcripts from SEC EDGAR")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols e.g. AAPL MSFT")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to fetch e.g. 2023 2024")
    args = parser.parse_args()

    logger.info(f"Starting ingestion for tickers={args.tickers}, years={args.years}")
    run_ingestion(args.tickers, args.years)
    logger.info("Ingestion complete")
