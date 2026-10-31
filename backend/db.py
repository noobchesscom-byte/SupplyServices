"""Database engine and session management for SupplyGraph using SQLModel & SQLite."""

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

# Ensure environment is loaded from root .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_DB_FILE = Path(__file__).resolve().parent / "supplygraph.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_FILE}")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db(engine_override=None) -> None:
    """Create all SQLModel database tables."""
    target_engine = engine_override or engine
    SQLModel.metadata.create_all(target_engine)


def get_session() -> Generator[Session, None, None]:
    """Dependency provider for database sessions."""
    with Session(engine) as session:
        yield session
