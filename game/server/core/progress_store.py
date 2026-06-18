from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from server.config import GAME_ROOT


DB_PATH = GAME_ROOT / "progress" / "local.sqlite"
PLAYER_ID = "local"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            create table if not exists players (
                id text primary key,
                created_at text default current_timestamp,
                updated_at text default current_timestamp
            );
            create table if not exists mission_progress (
                player_id text not null,
                mission_id text not null,
                status text not null,
                attempts integer not null default 0,
                best_score real,
                updated_at text default current_timestamp,
                primary key (player_id, mission_id)
            );
            create table if not exists unlocked_modules (
                player_id text not null,
                module_id text not null,
                unlocked_at text default current_timestamp,
                primary key (player_id, module_id)
            );
            create table if not exists unlocked_artifacts (
                player_id text not null,
                artifact_id text not null,
                unlocked_at text default current_timestamp,
                primary key (player_id, artifact_id)
            );
            create table if not exists notebook_entries (
                player_id text not null,
                concept_id text not null,
                status text not null default 'learned',
                unlocked_at text default current_timestamp,
                primary key (player_id, concept_id)
            );
            """
        )
        conn.execute(
            "insert or ignore into players(id) values (?)",
            (PLAYER_ID,),
        )


def get_progress() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        missions = conn.execute(
            "select mission_id, status, attempts, best_score from mission_progress where player_id = ?",
            (PLAYER_ID,),
        ).fetchall()
        modules = conn.execute(
            "select module_id from unlocked_modules where player_id = ? order by module_id",
            (PLAYER_ID,),
        ).fetchall()
        artifacts = conn.execute(
            "select artifact_id from unlocked_artifacts where player_id = ? order by artifact_id",
            (PLAYER_ID,),
        ).fetchall()
        notebook = conn.execute(
            "select concept_id, status from notebook_entries where player_id = ? order by concept_id",
            (PLAYER_ID,),
        ).fetchall()

    completed = [row["mission_id"] for row in missions if row["status"] == "completed"]
    return {
        "player_id": PLAYER_ID,
        "completed_missions": completed,
        "mission_progress": {
            row["mission_id"]: {
                "status": row["status"],
                "attempts": row["attempts"],
                "best_score": row["best_score"],
            }
            for row in missions
        },
        "unlocked_modules": [row["module_id"] for row in modules],
        "unlocked_artifacts": [row["artifact_id"] for row in artifacts],
        "notebook": [{"concept": row["concept_id"], "status": row["status"]} for row in notebook],
    }


def record_attempt(mission_id: str, passed: bool, score: float | None = None) -> None:
    init_db()
    status = "completed" if passed else "attempted"
    with _connect() as conn:
        existing = conn.execute(
            "select attempts, best_score, status from mission_progress where player_id = ? and mission_id = ?",
            (PLAYER_ID, mission_id),
        ).fetchone()
        if existing:
            attempts = int(existing["attempts"]) + 1
            best_score = score if existing["best_score"] is None else max(existing["best_score"], score or 0)
            final_status = "completed" if passed or existing["status"] == "completed" else status
            conn.execute(
                """
                update mission_progress
                set status = ?, attempts = ?, best_score = ?, updated_at = current_timestamp
                where player_id = ? and mission_id = ?
                """,
                (final_status, attempts, best_score, PLAYER_ID, mission_id),
            )
        else:
            conn.execute(
                """
                insert into mission_progress(player_id, mission_id, status, attempts, best_score)
                values (?, ?, ?, 1, ?)
                """,
                (PLAYER_ID, mission_id, status, score),
            )


def unlock_modules(module_ids: list[str]) -> None:
    init_db()
    with _connect() as conn:
        conn.executemany(
            "insert or ignore into unlocked_modules(player_id, module_id) values (?, ?)",
            [(PLAYER_ID, module_id) for module_id in module_ids],
        )


def unlock_artifacts(artifact_ids: list[str]) -> None:
    init_db()
    with _connect() as conn:
        conn.executemany(
            "insert or ignore into unlocked_artifacts(player_id, artifact_id) values (?, ?)",
            [(PLAYER_ID, artifact_id) for artifact_id in artifact_ids],
        )


def unlock_concept(concept_id: str | None) -> None:
    if not concept_id:
        return
    init_db()
    with _connect() as conn:
        conn.execute(
            "insert or ignore into notebook_entries(player_id, concept_id, status) values (?, ?, 'learned')",
            (PLAYER_ID, concept_id),
        )


def reset_progress() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("delete from mission_progress where player_id = ?", (PLAYER_ID,))
        conn.execute("delete from unlocked_modules where player_id = ?", (PLAYER_ID,))
        conn.execute("delete from unlocked_artifacts where player_id = ?", (PLAYER_ID,))
        conn.execute("delete from notebook_entries where player_id = ?", (PLAYER_ID,))
