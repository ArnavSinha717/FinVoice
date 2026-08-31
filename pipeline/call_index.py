"""SQLite index over processed call records.

/api/stats, /api/calls and /api/review-queue each globbed data/processed and fully
parsed every *_record.json on every request. A record is large — full transcript,
per-word timestamps, per-segment emotions — so at a few thousand calls a single
dashboard load means a few thousand multi-megabyte JSON parses.

The JSON files stay the source of truth; this is a derived index holding only the
summary columns those three endpoints actually read. It is written when a record is
written and can be rebuilt from disk at any time, so a lost or stale index is a
performance problem rather than a correctness one. Every reader falls back to
scanning the directory if the index is unavailable.
"""

import json
import sqlite3
import threading
from pathlib import Path
from loguru import logger

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id                TEXT PRIMARY KEY,
    audio_file             TEXT,
    duration_seconds       REAL,
    language               TEXT,
    call_type              TEXT,
    num_speakers           INTEGER,
    overall_risk_level     TEXT,
    compliance_score       INTEGER,
    requires_human_review  INTEGER,
    review_priority        INTEGER,
    call_summary           TEXT,
    pii_count              INTEGER,
    tamper_risk            TEXT,
    review_reasons         TEXT,   -- JSON array
    degradations           TEXT,   -- JSON array of component names
    mtime                  REAL,
    source_path            TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_review   ON calls(requires_human_review, review_priority);
CREATE INDEX IF NOT EXISTS idx_calls_mtime    ON calls(mtime DESC);
CREATE INDEX IF NOT EXISTS idx_calls_risk     ON calls(overall_risk_level);
"""

_COLUMNS = [
    "call_id", "audio_file", "duration_seconds", "language", "call_type",
    "num_speakers", "overall_risk_level", "compliance_score",
    "requires_human_review", "review_priority", "call_summary", "pii_count",
    "tamper_risk", "review_reasons", "degradations", "mtime", "source_path",
]


def index_path(results_dir: str = "data/processed") -> Path:
    return Path(results_dir) / "index.sqlite3"


def _connect(results_dir: str = "data/processed") -> sqlite3.Connection:
    p = index_path(results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _row_from_record(data: dict, path: Path) -> tuple:
    return (
        data.get("call_id"),
        data.get("audio_file"),
        data.get("duration_seconds") or 0.0,
        data.get("detected_language") or data.get("language") or "en",
        data.get("call_type"),
        data.get("num_speakers") or 0,
        data.get("overall_risk_level") or "low",
        data.get("compliance_score") or 0,
        1 if data.get("requires_human_review") else 0,
        data.get("review_priority") if data.get("review_priority") is not None else 5,
        (data.get("call_summary") or "")[:500],
        data.get("pii_count") or 0,
        data.get("tamper_risk") or "none",
        json.dumps(data.get("review_reasons") or []),
        json.dumps([d.get("component") for d in (data.get("degradations") or [])]),
        path.stat().st_mtime if path.exists() else 0.0,
        str(path),
    )


def upsert(data: dict, path: str | Path, results_dir: str = "data/processed") -> None:
    """Index one record. Never raises — a broken index must not fail a pipeline run."""
    try:
        with _LOCK, _connect(results_dir) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO calls ({','.join(_COLUMNS)}) "
                f"VALUES ({','.join('?' * len(_COLUMNS))})",
                _row_from_record(data, Path(path)),
            )
    except Exception as e:
        logger.warning(f"Call index upsert failed (non-fatal): {e}")


def rebuild(results_dir: str = "data/processed") -> int:
    """Rebuild the whole index from disk. Returns the number of records indexed."""
    files = sorted(Path(results_dir).glob("*_record.json"))
    rows = []
    for f in files:
        try:
            rows.append(_row_from_record(json.loads(f.read_text()), f))
        except Exception as e:
            logger.warning(f"Skipping {f.name} during index rebuild: {e}")
    with _LOCK, _connect(results_dir) as conn:
        conn.execute("DELETE FROM calls")
        conn.executemany(
            f"INSERT OR REPLACE INTO calls ({','.join(_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_COLUMNS))})", rows)
    logger.info(f"Call index rebuilt: {len(rows)} record(s)")
    return len(rows)


def _is_fresh(results_dir: str) -> bool:
    """Cheap staleness check: same number of records on disk as in the index."""
    try:
        n_disk = sum(1 for _ in Path(results_dir).glob("*_record.json"))
        with _connect(results_dir) as conn:
            n_idx = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        return n_disk == n_idx
    except Exception:
        return False


def ensure_fresh(results_dir: str = "data/processed") -> bool:
    """Rebuild if the index has drifted from disk. Returns True if usable."""
    try:
        if not _is_fresh(results_dir):
            rebuild(results_dir)
        return True
    except Exception as e:
        logger.warning(f"Call index unavailable, falling back to directory scan: {e}")
        return False


def query(sql: str, params: tuple = (), results_dir: str = "data/processed") -> list[sqlite3.Row]:
    with _connect(results_dir) as conn:
        return conn.execute(sql, params).fetchall()
