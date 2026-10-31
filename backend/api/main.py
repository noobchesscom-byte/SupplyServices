"""FastAPI application entry point for SupplyGraph."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from backend.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database tables on application startup."""
    init_db()
    yield


app = FastAPI(
    title="SupplyGraph API",
    description="Real-world event exposure analysis backed by SEC 10-K filings.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint confirming service availability."""
    return {
        "status": "ok",
        "service": "SupplyGraph Backend",
    }
