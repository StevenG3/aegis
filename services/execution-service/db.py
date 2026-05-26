from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/trading-agent-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    new_path = data_dir / "trading.sqlite"
    old_path = data_dir / "phase1.sqlite"
    if not new_path.exists() and old_path.exists():
        old_path.replace(new_path)
    return new_path


def connect() -> sqlite3.Connection:
    path = db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists executions (
            execution_id text primary key,
            payload_json text not null,
            result_json text not null,
            created_at text not null
        )
        """
    )
    conn.commit()
