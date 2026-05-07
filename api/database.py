"""
database.py — SQLAlchemy models + session factory
Supporte PostgreSQL (prod / Docker) et SQLite (dev sans Docker).
La variable DATABASE_URL détermine le moteur utilisé.
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from contextlib import contextmanager

# ── Connexion ────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/cv_intelligence.db"   # fallback local sans Docker
)

# SQLite ne supporte pas "postgresql+..." — on ne touche pas l'URL
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Base ─────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Modèle Candidate ─────────────────────────────────────────────────
class Candidate(Base):
    __tablename__ = "candidates"

    candidate_id      = Column(String(64),  primary_key=True)
    received_at       = Column(DateTime,    default=datetime.utcnow, index=True)
    source_filename   = Column(String(255), nullable=True)

    # Identité
    name              = Column(String(255), nullable=True)
    email             = Column(String(255), nullable=True)
    phone             = Column(String(64),  nullable=True)
    gender            = Column(String(16),  nullable=True)
    age               = Column(Integer,     nullable=True)

    # Features CV
    sector            = Column(String(128), nullable=True)
    target_role       = Column(String(255), nullable=True)
    years_experience  = Column(Float,       nullable=True)
    education_level   = Column(Float,       nullable=True)

    # ML output
    score             = Column(Float,       nullable=True, index=True)
    decision          = Column(String(32),  nullable=True, index=True)
    threshold_used    = Column(Float,       nullable=True)
    priority_rank     = Column(Integer,     nullable=True)
    eliminated_reason = Column(String(512), nullable=True)
    shap_json         = Column(Text,        nullable=True)

    # Workflow RH
    status            = Column(String(32),  default="inbox", index=True)

    __table_args__ = (
        Index("ix_candidates_decision_score", "decision", "score"),
    )


# ── Helpers session ──────────────────────────────────────────────────
@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Modèle Job ───────────────────────────────────────────────────────
class Job(Base):
    __tablename__ = "jobs"

    job_id       = Column(String(64),  primary_key=True, default=lambda: uuid.uuid4().hex)
    title        = Column(String(255), nullable=False)
    department   = Column(String(128), nullable=True)
    location     = Column(String(255), nullable=True)
    description  = Column(Text,        nullable=True)

    # Workflow
    status       = Column(String(32),  default="active", index=True)   # active | draft | closed | paused
    stage        = Column(String(32),  default="sourcing")              # sourcing | review | interview | final_interview | closed
    priority     = Column(String(32),  default="normal")                # normal | high | strategic

    # Denormalized counters (mis à jour par l'UI ou cron)
    applicants_count = Column(Integer, default=0)
    avg_score        = Column(Float,   nullable=True)

    created_at   = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at   = Column(DateTime, nullable=True)


def init_db():
    """Crée les tables si elles n'existent pas encore."""
    Base.metadata.create_all(bind=engine)
