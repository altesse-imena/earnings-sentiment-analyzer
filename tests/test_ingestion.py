"""Unit tests for ingestion module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.price_fetcher import compute_price_change, fetch_price_window


class TestComputePriceChange:
    def _make_df(self, dates, closes):
        df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))
        return df

    def test_positive_direction(self):
        df = self._make_df(
            ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
            [100.0, 105.0, 108.0, 112.0],
        )
        result = compute_price_change(df, "2024-01-31")
        assert result["direction"] == 1
        assert result["pct_change_48h"] > 0
        assert result["close_event"] == 105.0

    def test_negative_direction(self):
        df = self._make_df(
            ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
            [100.0, 105.0, 98.0, 94.0],
        )
        result = compute_price_change(df, "2024-01-31")
        assert result["direction"] == 0
        assert result["pct_change_48h"] < 0

    def test_empty_dataframe_returns_empty_dict(self):
        result = compute_price_change(pd.DataFrame(), "2024-01-31")
        assert result == {}

    def test_none_returns_empty_dict(self):
        result = compute_price_change(None, "2024-01-31")
        assert result == {}

    def test_no_post_event_days_returns_empty(self):
        df = self._make_df(["2024-01-30", "2024-01-31"], [100.0, 105.0])
        result = compute_price_change(df, "2024-01-31")
        assert result == {}

    def test_pct_change_calculation(self):
        df = self._make_df(
            ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"],
            [100.0, 100.0, 100.0, 110.0],
        )
        result = compute_price_change(df, "2024-01-31")
        assert abs(result["pct_change_48h"] - 10.0) < 0.01


class TestCikResolution:
    @patch("src.ingestion.edgar_fetcher.requests.get")
    def test_cik_uses_cache_on_second_call(self, mock_get):
        from src.ingestion.edgar_fetcher import _save_cache, get_cik_for_ticker
        _save_cache("cik_TESTX", {"cik": "0000123456"})
        result = get_cik_for_ticker("TESTX")
        assert result == "0000123456"
        mock_get.assert_not_called()


class TestEdgarSearch:
    @patch("src.ingestion.edgar_fetcher.requests.get")
    def test_search_returns_cached_results(self, mock_get):
        from src.ingestion.edgar_fetcher import _save_cache, search_8k_filings
        fake = [{"ticker": "FAKE", "accession_no": "0001234-24-001", "file_date": "2024-02-01"}]
        _save_cache("filings_FAKE_2024_2024", fake)
        result = search_8k_filings("FAKE", 2024, 2024)
        assert result == fake
        mock_get.assert_not_called()

    @patch("src.ingestion.edgar_fetcher.requests.get")
    def test_search_handles_api_error(self, mock_get):
        from src.ingestion.edgar_fetcher import search_8k_filings
        mock_get.side_effect = Exception("Network error")
        result = search_8k_filings("ERRX", 2023, 2023)
        assert isinstance(result, list)
