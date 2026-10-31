"""Tests for the SupplyGraph backend scaffold, /health endpoint, and database initialization."""

from pathlib import Path
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, select

from backend.api.main import app
from backend.db import get_session, init_db
from backend.models import Filing


def test_health_endpoint():
    """Verify /health returns HTTP 200 with status ok."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("status") == "ok"
        assert payload.get("service") == "SupplyGraph Backend"


def test_database_initialization_and_model_crud(tmp_path: Path):
    """Verify SQLModel database table creation and basic record persistence."""
    test_db_file = tmp_path / "test_scaffold.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_file}",
        connect_args={"check_same_thread": False},
    )

    # Initialize tables
    init_db(engine_override=test_engine)

    # Insert test filing
    with Session(test_engine) as session:
        filing = Filing(
            ticker="NVDA",
            cik="0001045810",
            company_name="NVIDIA CORP",
            sic_code="3674",
            filing_url="https://www.sec.gov/Archives/edgar/data/1045810/sample.htm",
            filing_date="2024-02-21",
            raw_text="Sample Item 1 and Item 2 content",
        )
        session.add(filing)
        session.commit()
        session.refresh(filing)
        assert filing.id is not None

    # Query back
    with Session(test_engine) as session:
        statement = select(Filing).where(Filing.ticker == "NVDA")
        retrieved = session.exec(statement).first()
        assert retrieved is not None
        assert retrieved.ticker == "NVDA"
        assert retrieved.cik == "0001045810"
        assert retrieved.company_name == "NVIDIA CORP"
        assert retrieved.sic_code == "3674"
        assert retrieved.filing_date == "2024-02-21"


def test_get_session_provider():
    """Verify get_session yields an active SQLModel session."""
    session_gen = get_session()
    session = next(session_gen)
    assert isinstance(session, Session)
    # Complete generator
    try:
        next(session_gen)
    except StopIteration:
        pass
