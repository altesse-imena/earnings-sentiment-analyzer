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

    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=8-K&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        # Extract CIK from the response URL redirect or content
        company_url = f"https://data.sec.gov/submissions/CIK"
        # Use the company tickers JSON endpoint instead
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp2 = requests.get(tickers_url, headers=HEADERS, timeout=15)
        resp2.raise_for_status()
        tickers_data = resp2.json()
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _save_cache(cache_key, {"cik": cik})
                logger.info(f"Resolved {ticker} → CIK {cik}")
                return cik
    except Exception as e:
        logger.error(f"Failed to resolve CIK for {ticker}: {e}")
    return None


def search_8k_filings(ticker: str, start_year: int, end_year: int) -> list[dict]:
    """Search SEC EDGAR for 8-K filings containing earnings call transcripts."""
    cache_key = f"filings_{ticker}_{start_year}_{end_year}"
    cached = _load_cache(cache_key)
    if cached:
        logger.info(f"Filings cache hit for {ticker} ({start_year}-{end_year})")
        return cached

    results = []
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    # EDGAR full-text search for 8-K filings mentioning earnings calls
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
        f"+%22earnings+call%22&forms=8-K"
        f"&dateRange=custom&startdt={start}&enddt={end}"
    )

    try:
        logger.info(f"Searching EDGAR for {ticker} 8-K filings ({start_year}-{end_year})")
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            results.append({
                "ticker": ticker,
                "accession_no": src.get("accession_no", ""),
                "file_date": src.get("file_date", ""),
                "entity_name": src.get("entity_name", ""),
                "form_type": src.get("form_type", ""),
                "file_url": f"https://www.sec.gov/Archives/edgar/data/{src.get('file_num', '')}/{src.get('accession_no', '').replace('-', '')}/",
            })
        time.sleep(0.5)  # Be polite to SEC servers
    except Exception as e:
        logger.error(f"EDGAR search failed for {ticker}: {e}")

    _save_cache(cache_key, results)
    logger.info(f"Found {len(results)} 8-K filings for {ticker}")
    return results


def fetch_transcript_text(accession_no: str, ticker: str) -> str | None:
    """Fetch the full text of an 8-K filing from EDGAR."""
    cache_key = f"transcript_{accession_no.replace('-', '_')}"
    cached = _load_cache(cache_key)
    if cached:
        return cached.get("text")

    acc_clean = accession_no.replace("-", "")
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return None

    # Try fetching the index page for this filing
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession_no}-index.htm"
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        # Parse to find the actual document URL
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        doc_link = None
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                desc = cells[1].get_text(strip=True).lower()
                if any(k in desc for k in ["transcript", "exhibit 99", "ex-99", "ex99"]):
                    link = cells[2].find("a")
                    if link:
                        doc_link = "https://www.sec.gov" + link["href"]
                        break

        if not doc_link:
            # Fall back to first document link
            first_link = soup.find("table", {"class": "tableFile"})
            if first_link:
                a = first_link.find("a")
                if a:
                    doc_link = "https://www.sec.gov" + a["href"]

        if doc_link:
            time.sleep(0.3)
            doc_resp = requests.get(doc_link, headers=HEADERS, timeout=30)
            doc_resp.raise_for_status()
            soup2 = BeautifulSoup(doc_resp.text, "html.parser")
            text = soup2.get_text(separator="\n", strip=True)
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

            text = fetch_transcript_text(acc, ticker)
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
