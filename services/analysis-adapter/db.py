from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/aegis-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "analysis_adapter.sqlite"


def connect() -> sqlite3.Connection:
    path = db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists analysis_jobs (
            job_id text primary key,
            actor text not null,
            symbol text not null,
            asset_type text not null,
            requested_at text not null,
            finished_at text,
            status text not null,
            scorecard_id text,
            error text,
            raw_response_json text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_analysis_jobs_actor "
        "on analysis_jobs(actor, requested_at desc)"
    )
    conn.execute(
        """
        create table if not exists llm_calls (
            call_id text primary key,
            job_id text,
            task text not null,
            provider text not null,
            model text not null,
            latency_ms integer not null,
            input_tokens integer not null,
            output_tokens integer not null,
            total_tokens integer not null,
            est_cost_usd text not null,
            success integer not null,
            error text,
            created_at text not null
        )
        """
    )
    conn.execute(
        "create index if not exists idx_llm_calls_created_at "
        "on llm_calls(created_at desc)"
    )
    conn.execute(
        "create index if not exists idx_llm_calls_job_id "
        "on llm_calls(job_id, created_at desc)"
    )
    conn.commit()
