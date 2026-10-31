"""Unit and integration tests for SEC EDGAR ingestion (backend/ingest/edgar.py).

Verifies:
- Safe handling of missing SEC_USER_AGENT configuration
- Ticker to CIK resolution (positive, negative, and malformed cases)
- Submissions metadata retrieval, 10-K detection, and array validation
- Document downloading and database persistence into Filing table
- Disk caching behavior for SEC requests
- Rate limiter timing enforcement
- Transient failure retry logic (HTTP 429 with numeric seconds & HTTP-date Retry-After)
- Response validation converting malformed/incomplete structures to ValueErrors
- Live SEC EDGAR integration test fetching and persisting a real 10-K filing
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from sqlmodel import Session, create_engine, select

from backend.ingest.edgar import (
    SECRateLimiter,
    fetch_and_store_filing,
    fetch_sec_url,
    fetch_submissions,
    get_user_agent,
    parse_retry_after,
    resolve_ticker,
)
from backend.models import Filing


MOCK_TICKERS_DATA = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

MOCK_SUBMISSIONS_DATA = {
    "cik": "0001045810",
    "sic": "3674",
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001045810-24-000020", "0001045810-23-000010"],
            "filingDate": ["2024-02-21", "2023-08-25"],
            "reportDate": ["2024-01-28", "2023-07-30"],
            "form": ["10-K", "10-Q"],
            "primaryDocument": ["nvda-20240128.htm", "nvda-20230730.htm"],
        }
    },
}

MOCK_FILING_TEXT = "<html><body><h1>Item 1. Business</h1><p>NVIDIA designs GPUs.</p></body></html>"


def test_missing_sec_user_agent_raises_error(monkeypatch):
    """Missing SEC_USER_AGENT must raise a RuntimeError without exposing tokens."""
    monkeypatch.setenv("SEC_USER_AGENT", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_user_agent()
    assert "SEC_USER_AGENT is not configured" in str(exc_info.value)


def test_parse_retry_after_numeric_seconds():
    """parse_retry_after supports positive finite numbers and rejects invalid, negative, and non-finite values."""
    assert parse_retry_after("60") == 60.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after("0") is None
    assert parse_retry_after("-10") is None
    assert parse_retry_after("inf") is None
    assert parse_retry_after("-inf") is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("NaN") is None
    assert parse_retry_after("Infinity") is None
    assert parse_retry_after("-Infinity") is None
    assert parse_retry_after("invalid") is None
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None


def test_parse_retry_after_http_date():
    """parse_retry_after correctly calculates delay for future HTTP-date and rejects past dates."""
    # Future date: 120 seconds ahead
    future_dt = datetime.now(timezone.utc) + timedelta(seconds=120)
    future_str = format_datetime(future_dt)
    delay = parse_retry_after(future_str)
    assert delay is not None
    assert 115.0 <= delay <= 125.0

    # Past date
    past_dt = datetime.now(timezone.utc) - timedelta(seconds=60)
    past_str = format_datetime(past_dt)
    assert parse_retry_after(past_str) is None


def test_http_retry_with_http_date_header(tmp_path, monkeypatch):
    """HTTP 429 response with HTTP-date Retry-After delays appropriately and retries."""
    mock_sleep = MagicMock()
    monkeypatch.setattr("backend.ingest.edgar.time.sleep", mock_sleep)

    mock_client = MagicMock()
    future_seconds = 120
    future_dt = datetime.now(timezone.utc) + timedelta(seconds=future_seconds)
    date_str = format_datetime(future_dt)

    request = httpx.Request("GET", "https://www.sec.gov/test-date")
    resp_429 = httpx.Response(429, request=request, headers={"retry-after": date_str})
    resp_200 = httpx.Response(200, request=request, text="Success after date delay")
    mock_client.get.side_effect = [resp_429, resp_200]

    mock_limiter = MagicMock()

    content = fetch_sec_url(
        "https://www.sec.gov/test-date",
        user_agent="TestAgent test@example.com",
        client=mock_client,
        cache_dir=tmp_path,
        rate_limiter=mock_limiter,
        max_retries=2,
    )
    assert content == "Success after date delay"
    assert mock_client.get.call_count == 2
    assert mock_sleep.call_count == 1
    sleep_arg = mock_sleep.call_args[0][0]
    # Assert that the actual HTTP-date-derived delay is passed to sleep
    assert 115.0 <= sleep_arg <= 125.0


def test_resolve_ticker_success(tmp_path):
    """Ticker is resolved to zero-padded CIK and company name."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(MOCK_TICKERS_DATA)
    mock_client.get.return_value = mock_response

    result = resolve_ticker("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path)
    assert result["ticker"] == "NVDA"
    assert result["cik"] == "0001045810"
    assert result["cik_int"] == 1045810
    assert result["company_name"] == "NVIDIA CORP"


def test_resolve_ticker_unknown(tmp_path):
    """Unknown ticker raises a descriptive ValueError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(MOCK_TICKERS_DATA)
    mock_client.get.return_value = mock_response

    with pytest.raises(ValueError) as exc_info:
        resolve_ticker("UNKNOWN_TICKER", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path)
    assert "Ticker 'UNKNOWN_TICKER' not found" in str(exc_info.value)


def test_malformed_company_tickers_json(tmp_path):
    """Malformed company_tickers responses raise clear ValueErrors."""
    mock_client = MagicMock()

    # Case 1: Non-JSON response
    mock_client.get.return_value = MagicMock(status_code=200, text="Not a JSON document")
    with pytest.raises(ValueError) as exc_info1:
        resolve_ticker("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "c1")
    assert "Malformed SEC company tickers response" in str(exc_info1.value)

    # Case 2: JSON is a string or primitive instead of dict/list
    mock_client.get.return_value = MagicMock(status_code=200, text=json.dumps("just a string"))
    with pytest.raises(ValueError) as exc_info2:
        resolve_ticker("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "c2")
    assert "expected JSON object or array" in str(exc_info2.value)

    # Case 3: Entry missing cik_str
    bad_entries = {"0": {"ticker": "NVDA", "title": "NVIDIA CORP"}}
    mock_client.get.return_value = MagicMock(status_code=200, text=json.dumps(bad_entries))
    with pytest.raises(ValueError) as exc_info3:
        resolve_ticker("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "c3")
    assert "missing 'cik_str'" in str(exc_info3.value)

    # Case 4: Entry with non-numeric cik_str
    bad_cik = {"0": {"ticker": "NVDA", "cik_str": "not_an_int", "title": "NVIDIA CORP"}}
    mock_client.get.return_value = MagicMock(status_code=200, text=json.dumps(bad_cik))
    with pytest.raises(ValueError) as exc_info4:
        resolve_ticker("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "c4")
    assert "Malformed CIK" in str(exc_info4.value)


def test_malformed_submissions_json(tmp_path):
    """Malformed submissions responses raise clear ValueErrors."""
    mock_client = MagicMock()

    # Case 1: Non-dict submissions
    mock_client.get.return_value = MagicMock(status_code=200, text=json.dumps([1, 2, 3]))
    with pytest.raises(ValueError) as exc_info1:
        fetch_submissions("0001045810", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "s1")
    assert "expected JSON object" in str(exc_info1.value)

    # Case 2: Submissions missing 'filings'
    tickers_resp = MagicMock(status_code=200, text=json.dumps(MOCK_TICKERS_DATA))
    missing_filings = {"cik": "0001045810", "name": "NVIDIA CORP"}
    mock_client.get.side_effect = [tickers_resp, MagicMock(status_code=200, text=json.dumps(missing_filings))]
    with pytest.raises(ValueError) as exc_info2:
        fetch_and_store_filing("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "s2")
    assert "missing or invalid 'filings' object" in str(exc_info2.value)


def test_incomplete_or_mismatched_filing_arrays(tmp_path):
    """Missing or truncated parallel arrays in recent filings raise clear ValueErrors."""
    mock_client = MagicMock()

    # Case 1: Missing accessionNumber list
    tickers_resp = MagicMock(status_code=200, text=json.dumps(MOCK_TICKERS_DATA))
    incomplete_subs = {
        "cik": "0001045810",
        "sic": "3674",
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-K"],
                # missing accessionNumber
                "filingDate": ["2024-02-21"],
                "primaryDocument": ["nvda-20240128.htm"],
            }
        },
    }
    mock_client.get.side_effect = [tickers_resp, MagicMock(status_code=200, text=json.dumps(incomplete_subs))]
    with pytest.raises(ValueError) as exc_info1:
        fetch_and_store_filing("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "a1")
    assert "required field 'accessionNumber' is missing" in str(exc_info1.value)

    # Case 2: primaryDocument array shorter than form array
    truncated_subs = {
        "cik": "0001045810",
        "sic": "3674",
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K"],
                "accessionNumber": ["acc-1", "acc-2"],
                "filingDate": ["2024-01-01", "2024-02-21"],
                "primaryDocument": ["doc-1"],  # only 1 item, 10-K is at index 1
            }
        },
    }
    mock_client.get.side_effect = [tickers_resp, MagicMock(status_code=200, text=json.dumps(truncated_subs))]
    with pytest.raises(ValueError) as exc_info2:
        fetch_and_store_filing("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "a2")
    assert "array length (1) is shorter than 10-K index (1)" in str(exc_info2.value)

    # Case 3: empty primaryDocument string
    empty_doc_subs = {
        "cik": "0001045810",
        "sic": "3674",
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["acc-1"],
                "filingDate": ["2024-02-21"],
                "primaryDocument": ["   "],  # whitespace only
            }
        },
    }
    mock_client.get.side_effect = [tickers_resp, MagicMock(status_code=200, text=json.dumps(empty_doc_subs))]
    with pytest.raises(ValueError) as exc_info3:
        fetch_and_store_filing("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path / "a3")
    assert "empty or invalid 'primaryDocument' entry" in str(exc_info3.value)


def test_empty_filing_document_content(tmp_path):
    """Empty filing document content raises a descriptive ValueError."""
    mock_client = MagicMock()
    tickers_resp = MagicMock(status_code=200, text=json.dumps(MOCK_TICKERS_DATA))
    subs_resp = MagicMock(status_code=200, text=json.dumps(MOCK_SUBMISSIONS_DATA))
    empty_doc_resp = MagicMock(status_code=200, text="   ")
    mock_client.get.side_effect = [tickers_resp, subs_resp, empty_doc_resp]

    with pytest.raises(ValueError) as exc_info:
        fetch_and_store_filing("NVDA", user_agent="TestAgent test@example.com", client=mock_client, cache_dir=tmp_path)
    assert "Received empty filing document from URL" in str(exc_info.value)


def test_missing_10k_filing_raises_error(tmp_path):
    """Submissions lacking Form 10-K raise ValueError."""
    mock_client = MagicMock()

    tickers_resp = MagicMock(status_code=200, text=json.dumps(MOCK_TICKERS_DATA))
    no_10k_submissions = {
        "cik": "0001045810",
        "sic": "3674",
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-23-000010"],
                "filingDate": ["2023-08-25"],
                "form": ["10-Q"],
                "primaryDocument": ["nvda-20230730.htm"],
            }
        },
    }
    subs_resp = MagicMock(status_code=200, text=json.dumps(no_10k_submissions))
    mock_client.get.side_effect = [tickers_resp, subs_resp]

    with pytest.raises(ValueError) as exc_info:
        fetch_and_store_filing(
            "NVDA",
            user_agent="TestAgent test@example.com",
            client=mock_client,
            cache_dir=tmp_path,
        )
    assert "No 10-K filing found" in str(exc_info.value)


def test_fetch_and_store_filing_mocked_success(tmp_path):
    """Verify end-to-end fetch and database persistence with mocked SEC endpoints."""
    db_file = tmp_path / "test_edgar.db"
    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Filing.metadata.create_all(test_engine)

    mock_client = MagicMock()
    tickers_resp = MagicMock(status_code=200, text=json.dumps(MOCK_TICKERS_DATA))
    subs_resp = MagicMock(status_code=200, text=json.dumps(MOCK_SUBMISSIONS_DATA))
    doc_resp = MagicMock(status_code=200, text=MOCK_FILING_TEXT)
    mock_client.get.side_effect = [tickers_resp, subs_resp, doc_resp]

    with Session(test_engine) as session:
        filing = fetch_and_store_filing(
            "NVDA",
            session=session,
            user_agent="TestAgent test@example.com",
            client=mock_client,
            cache_dir=tmp_path,
        )

        assert filing.id is not None
        assert filing.ticker == "NVDA"
        assert filing.cik == "0001045810"
        assert filing.company_name == "NVIDIA CORP"
        assert filing.sic_code == "3674"
        assert filing.filing_date == "2024-02-21"
        assert "nvda-20240128.htm" in filing.filing_url
        assert filing.raw_text == MOCK_FILING_TEXT

    with Session(test_engine) as query_session:
        statement = select(Filing).where(Filing.ticker == "NVDA")
        saved = query_session.exec(statement).first()
        assert saved is not None
        assert saved.ticker == "NVDA"
        assert saved.cik == "0001045810"
        assert saved.sic_code == "3674"


def test_edgar_disk_cache(tmp_path):
    """Subsequent requests for the same URL read from disk cache and bypass HTTP client."""
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=200, text="Cached SEC content")
    mock_client.get.return_value = mock_resp

    test_url = "https://www.sec.gov/sample-test-url"
    ua = "TestAgent test@example.com"

    # First fetch - cache miss
    text1 = fetch_sec_url(test_url, user_agent=ua, cache_dir=tmp_path, client=mock_client)
    assert text1 == "Cached SEC content"
    assert mock_client.get.call_count == 1

    # Second fetch - cache hit
    text2 = fetch_sec_url(test_url, user_agent=ua, cache_dir=tmp_path, client=mock_client)
    assert text2 == "Cached SEC content"
    assert mock_client.get.call_count == 1  # Not incremented


def test_rate_limiter_enforces_interval():
    """SECRateLimiter enforces minimum interval between calls."""
    limiter = SECRateLimiter(max_requests_per_second=20.0)  # 50ms interval
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045


def test_http_retry_on_429_with_retry_after(tmp_path):
    """HTTP 429 retries with backoff respecting numeric Retry-After header."""
    mock_client = MagicMock()

    request = httpx.Request("GET", "https://www.sec.gov/test")
    resp_429 = httpx.Response(429, request=request, headers={"retry-after": "0.02"})
    resp_200 = httpx.Response(200, request=request, text="Success after rate limit")
    mock_client.get.side_effect = [resp_429, resp_200]

    content = fetch_sec_url(
        "https://www.sec.gov/test-rate-limit",
        user_agent="TestAgent test@example.com",
        client=mock_client,
        cache_dir=tmp_path,
        max_retries=2,
    )

    assert content == "Success after rate limit"
    assert mock_client.get.call_count == 2


def test_real_sec_integration(tmp_path):
    """Real SEC EDGAR integration test fetching and persisting an actual Form 10-K.

    Skipped safely when SEC_USER_AGENT is not set or if network is unreachable in the environment.
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        pytest.skip("SEC_USER_AGENT not configured in environment")

    db_file = tmp_path / "real_sec_test.db"
    real_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Filing.metadata.create_all(real_engine)

    try:
        with Session(real_engine) as session:
            filing = fetch_and_store_filing("NVDA", session=session, cache_dir=tmp_path)
            assert filing.ticker == "NVDA"
            assert filing.cik == "0001045810"
            assert filing.company_name
            assert filing.sic_code
            assert filing.filing_url.startswith("https://www.sec.gov/Archives/edgar/data/")
            assert filing.filing_date
            assert len(filing.raw_text) > 1000

            statement = select(Filing).where(Filing.ticker == "NVDA")
            retrieved = session.exec(statement).first()
            assert retrieved is not None
            assert retrieved.cik == "0001045810"
            assert len(retrieved.raw_text) > 1000
    except (httpx.ConnectError, RuntimeError) as err:
        pytest.skip(f"SEC network connection not reachable in current execution environment: {err}")
