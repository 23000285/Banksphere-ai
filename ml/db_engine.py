"""Database engine and session factory."""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ml.db_models import Base

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine.

    If no URL is given, uses sqlite:///data/banksphere.db relative to project root.
    Ensures the data/ directory exists for SQLite databases.
    """
    if database_url is None:
        database_url = f"sqlite:///{_DATA_DIR / 'banksphere.db'}"

    if database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return create_engine(database_url, echo=False)


def init_db(database_url: str | None = None):
    """Create all tables and return the engine."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    """Create a new session from the given engine."""
    factory = sessionmaker(bind=engine)
    return factory()
