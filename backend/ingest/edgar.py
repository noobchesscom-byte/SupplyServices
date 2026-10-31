"""SEC EDGAR ingestion module for SupplyGraph.

Fetches 10-K filings using SEC submissions and Archives APIs:
- Resolves ticker to CIK via SEC company_tickers.json
- Retrieves company metadata (name, SIC code) and filing history
- Downloads the latest 10-K filing document
- Caches every SEC response on disk to respect SEC guidelines
- Enforces an 8 request/second rate limit
- Retries transient failures with exponential backoff and Retry-After (seconds and HTTP-date)
- Validates SEC response shapes and converts malformed data into clear ValueErrors
- Persists filing metadata and raw text via SQLModel
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import httpx
from sqlmodel import Session, select

from backend.db import engine
from backend.models import Filing

logger = logging.getLogger(__name__)

# Repository root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "sec"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"


class SECRateLimiter:
    """Thread-safe rate limiter enforcing a maximum request rate per second."""

    def __init__(self, max_requests_per_second: float = 8.0):
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the minimum interval between requests has passed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.monotonic()


_GLOBAL_RATE_LIMITER = SECRateLimiter(max_requests_per_second=8.0)


def get_user_agent() -> str:
    """Retrieve SEC_USER_AGENT safely from environment.

    Raises RuntimeError with a clear message if not configured.
    Never prints or logs the value.
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not configured. Please set SEC_USER_AGENT in the repository-root .env file."
        )
    return ua


def parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header which can be numeric seconds or an HTTP-date.

    Returns positive delay in seconds, or None if invalid, zero, or negative.
    """
    if not header_value:
        return None
    val = header_value.strip()

    # 1. Try numeric seconds (integer or float)
    try:
        seconds = float(val)
        if math.isfinite(seconds) and seconds > 0:
            return seconds
        return None
    except ValueError:
        pass

    # 2. Try HTTP-date format (e.g. 'Wed, 21 Oct 2026 07:28:00 GMT')
    try:
        dt = parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delay = (dt - now).total_seconds()
        if delay > 0:
            return delay
        return None
    except Exception:
        return None


def get_cache_key(url: str) -> str:
    """Derive disk cache key from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _read_cache(cache_file: Path) -> Optional[str]:
    """Read cached HTTP response text from disk."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("content")
    except Exception as exc:
        logger.warning("Failed to read SEC cache at %s: %s", cache_file, exc)
        return None


def _write_cache(cache_file: Path, url: str, content: str) -> None:
    """Write HTTP response text atomically to disk cache."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".tmp")
        payload = {
            "url": url,
            "cached_at": time.time(),
            "content": content,
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        temp_file.replace(cache_file)
    except Exception as exc:
        logger.warning("Failed to write SEC cache at %s: %s", cache_file, exc)


def fetch_sec_url(
    url: str,
    user_agent: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
    rate_limiter: Optional[SECRateLimiter] = None,
    max_retries: int = 4,
    base_backoff: float = 0.5,
) -> str:
    """Fetch URL from SEC EDGAR with disk caching, rate limiting, and retries.

    Parameters:
        url: SEC URL to fetch.
        user_agent: Custom User-Agent (defaults to get_user_agent()).
        cache_dir: Directory for cached responses.
        client: Optional httpx.Client (e.g. for testing).
        rate_limiter: SECRateLimiter instance (defaults to 8 req/s).
        max_retries: Number of retries on transient errors (429, 5xx).
        base_backoff: Base exponential backoff delay in seconds.

    Returns:
        Response body text.
    """
    target_cache = cache_dir or DEFAULT_CACHE_DIR
    cache_key = get_cache_key(url)
    cache_file = target_cache / f"{cache_key}.json"

    # 1. Check disk cache
    cached_content = _read_cache(cache_file)
    if cached_content is not None:
        return cached_content

    # 2. Prepare request
    resolved_ua = user_agent or get_user_agent()
    headers = {
        "User-Agent": resolved_ua,
        "Accept-Encoding": "gzip, deflate",
    }
    limiter = rate_limiter or _GLOBAL_RATE_LIMITER

    own_client = False
    http_client = client
    if http_client is None:
        http_client = httpx.Client(timeout=30.0)
        own_client = True

    try:
        for attempt in range(max_retries + 1):
            try:
                limiter.wait()
                response = http_client.get(url, headers=headers, follow_redirects=True)

                if response.status_code == 200:
                    content = response.text
                    _write_cache(cache_file, url, content)
                    return content

                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt >= max_retries:
                        response.raise_for_status()

                    retry_header = response.headers.get("retry-after") or response.headers.get("Retry-After")
                    sleep_time = parse_retry_after(retry_header)
                    if sleep_time is None:
                        # Fallback to exponential backoff with jitter
                        sleep_time = (base_backoff * (2**attempt)) + random.uniform(0.1, 0.4)

                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()

            except (httpx.ConnectError, httpx.TimeoutException) as req_err:
                if attempt >= max_retries:
                    raise RuntimeError(f"Network error communicating with SEC EDGAR: {req_err}") from req_err
                time.sleep((base_backoff * (2**attempt)) + random.uniform(0.1, 0.4))

        raise RuntimeError(f"Failed to fetch SEC URL after {max_retries + 1} attempts: {url}")
    finally:
        if own_client:
            http_client.close()


def resolve_ticker(
    ticker: str,
    user_agent: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Resolve a stock ticker to its zero-padded CIK and company title via SEC tickers list.

    Returns:
        Dict with keys: 'ticker', 'cik' (10 digits), 'cik_int', 'company_name'.
    """
    normalized_ticker = ticker.strip().upper()
    content = fetch_sec_url(
        SEC_TICKERS_URL,
        user_agent=user_agent,
        cache_dir=cache_dir,
        client=client,
    )
    try:
        tickers_data = json.loads(content)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed SEC company tickers response: {err}") from err

    if not isinstance(tickers_data, (dict, list)):
        raise ValueError(
            f"Malformed SEC company tickers response: expected JSON object or array, got {type(tickers_data).__name__}"
        )

    entries = tickers_data.values() if isinstance(tickers_data, dict) else tickers_data
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ticker_val = entry.get("ticker")
        if isinstance(ticker_val, str) and ticker_val.strip().upper() == normalized_ticker:
            cik_raw = entry.get("cik_str")
            if cik_raw is None:
                raise ValueError(f"Malformed entry for ticker '{normalized_ticker}': missing 'cik_str'")
            try:
                cik_int = int(cik_raw)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Malformed CIK '{cik_raw}' for ticker '{normalized_ticker}': {err}") from err

            company_title = entry.get("title")
            title_clean = company_title.strip() if isinstance(company_title, str) else normalized_ticker

            return {
                "ticker": normalized_ticker,
                "cik": str(cik_int).zfill(10),
                "cik_int": cik_int,
                "company_name": title_clean,
            }

    raise ValueError(f"Ticker '{normalized_ticker}' not found in SEC company directory.")


def fetch_submissions(
    cik: str,
    user_agent: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Retrieve company submissions metadata given a 10-digit CIK."""
    padded_cik = str(int(cik)).zfill(10)
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=padded_cik)
    content = fetch_sec_url(
        url,
        user_agent=user_agent,
        cache_dir=cache_dir,
        client=client,
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed SEC submissions response for CIK {padded_cik}: {err}") from err

    if not isinstance(data, dict):
        raise ValueError(
            f"Malformed SEC submissions response for CIK {padded_cik}: expected JSON object, got {type(data).__name__}"
        )
    return data


def fetch_and_store_filing(
    ticker: str,
    session: Optional[Session] = None,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
    user_agent: Optional[str] = None,
) -> Filing:
    """Fetch the latest 10-K filing for a ticker and persist it in the database.

    Steps:
    1. Resolve ticker to CIK.
    2. Retrieve submissions and locate latest Form 10-K with validated parallel arrays.
    3. Download the filing document.
    4. Store in SQLModel Filing table (insert or update existing).

    Returns:
        Persisted Filing instance.
    """
    normalized_ticker = ticker.strip().upper()

    # 1. Resolve ticker
    resolved = resolve_ticker(
        normalized_ticker,
        user_agent=user_agent,
        cache_dir=cache_dir,
        client=client,
    )
    cik = resolved["cik"]
    cik_int = resolved["cik_int"]
    company_name = resolved["company_name"]

    # 2. Get submissions and validate structure
    submissions = fetch_submissions(
        cik,
        user_agent=user_agent,
        cache_dir=cache_dir,
        client=client,
    )
    sic_raw = submissions.get("sic")
    sic_code = str(sic_raw).strip() if sic_raw is not None else None
    company_name = submissions.get("name") or company_name

    filings_obj = submissions.get("filings")
    if not isinstance(filings_obj, dict):
        raise ValueError(f"Malformed submissions JSON for CIK {cik}: missing or invalid 'filings' object")

    recent = filings_obj.get("recent")
    if not isinstance(recent, dict):
        raise ValueError(f"Malformed submissions JSON for CIK {cik}: missing or invalid 'recent' object")

    forms = recent.get("form")
    if not isinstance(forms, list):
        raise ValueError(f"Malformed submissions JSON for CIK {cik}: 'form' field is missing or not a list")

    if "10-K" not in forms:
        raise ValueError(f"No 10-K filing found for ticker '{normalized_ticker}' (CIK {cik}).")

    idx = forms.index("10-K")

    # Validate parallel arrays in recent
    for field_name in ("accessionNumber", "filingDate", "primaryDocument"):
        field_list = recent.get(field_name)
        if not isinstance(field_list, list):
            raise ValueError(
                f"Malformed submissions JSON for CIK {cik}: required field '{field_name}' is missing or not a list"
            )
        if idx >= len(field_list):
            raise ValueError(
                f"Malformed submissions JSON for CIK {cik}: '{field_name}' array length ({len(field_list)}) is shorter than 10-K index ({idx})"
            )
        val = field_list[idx]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"Malformed submissions JSON for CIK {cik}: empty or invalid '{field_name}' entry at index {idx}"
            )

    accession_number = recent["accessionNumber"][idx].strip()
    filing_date = recent["filingDate"][idx].strip()
    primary_doc = recent["primaryDocument"][idx].strip()

    # 3. Construct document URL and download document
    accession_no_hyphens = accession_number.replace("-", "")
    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_hyphens}/{primary_doc}"

    raw_text = fetch_sec_url(
        filing_url,
        user_agent=user_agent,
        cache_dir=cache_dir,
        client=client,
    )
    if not raw_text or not raw_text.strip():
        raise ValueError(f"Received empty filing document from URL: {filing_url}")

    # 4. Persist to database
    def _persist(db_session: Session) -> Filing:
        statement = select(Filing).where(
            Filing.ticker == normalized_ticker,
            Filing.filing_url == filing_url,
        )
        existing = db_session.exec(statement).first()
        if existing:
            existing.cik = cik
            existing.company_name = company_name
            existing.sic_code = sic_code
            existing.filing_date = filing_date
            existing.raw_text = raw_text
            db_session.add(existing)
            db_session.commit()
            db_session.refresh(existing)
            return existing

        new_filing = Filing(
            ticker=normalized_ticker,
            cik=cik,
            company_name=company_name,
            sic_code=sic_code,
            filing_url=filing_url,
            filing_date=filing_date,
            raw_text=raw_text,
        )
        db_session.add(new_filing)
        db_session.commit()
        db_session.refresh(new_filing)
        return new_filing

    if session is not None:
        return _persist(session)
    with Session(engine) as db_session:
        return _persist(db_session)
