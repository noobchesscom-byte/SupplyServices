"""SQLModel data models for SupplyGraph backend."""

from typing import Optional
from sqlmodel import Field, SQLModel


class Filing(SQLModel, table=True):
    """Filing table storing retrieved SEC 10-K metadata and content."""

    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    cik: str = Field(index=True)
    company_name: Optional[str] = None
    sic_code: Optional[str] = None
    filing_url: str
    filing_date: Optional[str] = None
    raw_text: Optional[str] = None
