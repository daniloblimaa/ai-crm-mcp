import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path("crm.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_transaction() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                email       TEXT    NOT NULL UNIQUE,
                description TEXT    NOT NULL,
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    logger.info("Database initialized at %s", DB_PATH)


def insert_user(name: str, email: str, description: str) -> int:
    with db_transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, email, description) VALUES (?, ?, ?)",
            (name, email, description),
        )
        user_id = cursor.lastrowid
    logger.info("Inserted user id=%d email=%s", user_id, email)
    return user_id


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, name, email, description FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_users_by_ids(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT id, name, email, description FROM users WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}
    finally:
        conn.close()


def list_all_users() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, email, description, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def email_exists(email: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        return row is not None
    finally:
        conn.close()