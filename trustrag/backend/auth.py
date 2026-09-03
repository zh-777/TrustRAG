"""Local account/session + per-user chat persistence for TrustRAG.

Uses SQLite + PBKDF2 password hashing from the Python standard library.
No third-party auth provider is required. Guest mode remains available.
LLM API keys are never stored in the database.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("TRUSTRAG_DB_PATH", os.path.join(ROOT_DIR, "data", "users.db"))
SESSION_DAYS = 7
PBKDF2_ITERATIONS = 310_000
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_auth_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                source_mode TEXT NOT NULL DEFAULT 'dataset',
                source_name TEXT NOT NULL DEFAULT '',
                document_text TEXT NOT NULL DEFAULT '',
                row_index INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT 'gemini',
                model TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT '',
                top_k INTEGER NOT NULL DEFAULT 4,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chats_user_updated
            ON chats(user_id, updated_at DESC);
            """
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except Exception:
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _public_user(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"], "created_at": row["created_at"]}


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ChatSaveRequest(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=120)
    messages: list[dict] = Field(default_factory=list)
    source_mode: str = Field(default="dataset", max_length=32)
    source_name: str = Field(default="", max_length=255)
    document_text: str = Field(default="", max_length=500_000)
    row_index: int = Field(default=0, ge=0)
    provider: str = Field(default="gemini", max_length=40)
    model: str = Field(default="", max_length=160)
    base_url: str = Field(default="", max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


class ChatRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


def _validate_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return email


def _new_session(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(40)
    now = _now()
    expires = now + timedelta(days=SESSION_DAYS)
    conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
    conn.execute(
        "INSERT INTO sessions(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (_hash_token(token), user_id, expires.isoformat(), now.isoformat()),
    )
    return token


def register_user(req: RegisterRequest) -> dict:
    init_auth_db()
    email = _validate_email(req.email)
    name = req.name.strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Name must contain at least 2 characters.")
    with _connect() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users(name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, _hash_password(req.password), _now().isoformat()),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
        user_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        token = _new_session(conn, user_id)
    return {"token": token, "user": _public_user(row)}


def login_user(req: LoginRequest) -> dict:
    init_auth_db()
    email = _validate_email(req.email)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone()
        if row is None or not _verify_password(req.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = _new_session(conn, row["id"])
    return {"token": token, "user": _public_user(row)}


def get_user_from_token(token: str) -> Optional[dict]:
    if not token:
        return None
    init_auth_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT u.*, s.expires_at
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (_hash_token(token),),
        ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) <= _now():
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
            return None
        return _public_user(row)


def require_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization.split(" ", 1)[1].strip()
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")
    return user


def logout_token(token: str) -> None:
    if not token:
        return
    init_auth_db()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


def _chat_summary(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "source_name": row["source_name"],
        "source_mode": row["source_mode"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _chat_full(row: sqlite3.Row) -> dict:
    data = _chat_summary(row)
    try:
        messages = json.loads(row["messages_json"] or "[]")
    except json.JSONDecodeError:
        messages = []
    data.update({
        "messages": messages,
        "document_text": row["document_text"],
        "row_index": row["row_index"],
        "provider": row["provider"],
        "model": row["model"],
        "base_url": row["base_url"],
        "top_k": row["top_k"],
    })
    return data


def list_user_chats(user_id: int) -> list[dict]:
    init_auth_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100", (user_id,)
        ).fetchall()
    return [_chat_summary(row) for row in rows]


def create_user_chat(user_id: int, req: ChatSaveRequest) -> dict:
    init_auth_db()
    chat_id = uuid.uuid4().hex
    now = _now().isoformat()
    title = req.title.strip() or "New chat"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chats(
                id,user_id,title,messages_json,source_mode,source_name,document_text,row_index,
                provider,model,base_url,top_k,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chat_id, user_id, title, json.dumps(req.messages, ensure_ascii=False), req.source_mode,
                req.source_name, req.document_text, req.row_index, req.provider, req.model,
                req.base_url, req.top_k, now, now,
            ),
        )
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return _chat_full(row)


def get_user_chat(user_id: int, chat_id: str) -> dict:
    init_auth_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return _chat_full(row)


def update_user_chat(user_id: int, chat_id: str, req: ChatSaveRequest) -> dict:
    init_auth_db()
    title = req.title.strip() or "New chat"
    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Chat not found.")
        conn.execute(
            """
            UPDATE chats SET title=?, messages_json=?, source_mode=?, source_name=?, document_text=?,
                row_index=?, provider=?, model=?, base_url=?, top_k=?, updated_at=?
            WHERE id=? AND user_id=?
            """,
            (
                title, json.dumps(req.messages, ensure_ascii=False), req.source_mode, req.source_name,
                req.document_text, req.row_index, req.provider, req.model, req.base_url, req.top_k,
                _now().isoformat(), chat_id, user_id,
            ),
        )
        row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    return _chat_full(row)


def rename_user_chat(user_id: int, chat_id: str, title: str) -> dict:
    init_auth_db()
    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Chat title cannot be empty.")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?",
            (title, _now().isoformat(), chat_id, user_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat not found.")
        row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
    return _chat_summary(row)


def delete_user_chat(user_id: int, chat_id: str) -> None:
    init_auth_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat not found.")
