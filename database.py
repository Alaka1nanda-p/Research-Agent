"""
database.py — SQLite models and initialization for Research Agent
"""
import sqlite3
import json
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DATABASE_PATH = os.getenv("DATABASE_PATH", "research_agent.db")


def get_db():
    """Return a database connection with row_factory set."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Papers / References table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id    TEXT UNIQUE NOT NULL,
            title       TEXT NOT NULL,
            authors     TEXT,
            year        INTEGER,
            abstract    TEXT,
            summary     TEXT,
            url         TEXT,
            venue       TEXT,
            source      TEXT,
            tags        TEXT DEFAULT '[]',
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # LLM response cache (keyed by sha256 of prompt)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key   TEXT UNIQUE NOT NULL,
            response    TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Token usage tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            call_type       TEXT NOT NULL,
            tokens_in       INTEGER DEFAULT 0,
            tokens_out      INTEGER DEFAULT 0,
            model           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # Search history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            keywords    TEXT,
            result_ids  TEXT DEFAULT '[]',
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Generated reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT,
            query           TEXT,
            citation_style  TEXT,
            content         TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


# ── Paper CRUD ──────────────────────────────────────────────────────────────

def save_paper(paper: dict) -> int:
    """Insert or update a paper record. Returns the row id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO papers (paper_id, title, authors, year, abstract, summary,
                            url, venue, source, tags)
        VALUES (:paper_id, :title, :authors, :year, :abstract, :summary,
                :url, :venue, :source, :tags)
        ON CONFLICT(paper_id) DO UPDATE SET
            summary = COALESCE(excluded.summary, papers.summary),
            tags    = excluded.tags
    """, {
        "paper_id": paper.get("paper_id", ""),
        "title":    paper.get("title", ""),
        "authors":  json.dumps(paper.get("authors", [])),
        "year":     paper.get("year"),
        "abstract": paper.get("abstract", ""),
        "summary":  paper.get("summary", ""),
        "url":      paper.get("url", ""),
        "venue":    paper.get("venue", ""),
        "source":   paper.get("source", ""),
        "tags":     json.dumps(paper.get("tags", [])),
    })
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_papers():
    """Return all saved papers as a list of dicts."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
    conn.close()
    return [_row_to_paper(r) for r in rows]


def get_paper_by_id(paper_id: str):
    """Return a single paper dict by paper_id string."""
    conn = get_db()
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    conn.close()
    return _row_to_paper(row) if row else None


def search_papers(query: str):
    """Full-text search across title/abstract/tags."""
    q = f"%{query}%"
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM papers
        WHERE title LIKE ? OR abstract LIKE ? OR tags LIKE ?
        ORDER BY created_at DESC
    """, (q, q, q)).fetchall()
    conn.close()
    return [_row_to_paper(r) for r in rows]


def update_paper_summary(paper_id: str, summary: str):
    conn = get_db()
    conn.execute("UPDATE papers SET summary = ? WHERE paper_id = ?", (summary, paper_id))
    conn.commit()
    conn.close()


def update_paper_tags(paper_id: str, tags: list):
    conn = get_db()
    conn.execute("UPDATE papers SET tags = ? WHERE paper_id = ?",
                 (json.dumps(tags), paper_id))
    conn.commit()
    conn.close()


def delete_paper(paper_id: str):
    conn = get_db()
    conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
    conn.commit()
    conn.close()


def _row_to_paper(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    for field in ("authors", "tags"):
        try:
            d[field] = json.loads(d.get(field) or "[]")
        except (json.JSONDecodeError, TypeError):
            d[field] = []
    return d


# ── LLM Cache ───────────────────────────────────────────────────────────────

def cache_key_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_response(key: str):
    conn = get_db()
    row = conn.execute(
        "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["response"] if row else None


def set_cached_response(key: str, response: str):
    conn = get_db()
    conn.execute("""
        INSERT INTO llm_cache (cache_key, response)
        VALUES (?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET response = excluded.response
    """, (key, response))
    conn.commit()
    conn.close()


# ── Token Usage ──────────────────────────────────────────────────────────────

def record_token_usage(session_id: str, call_type: str,
                       tokens_in: int, tokens_out: int, model: str = ""):
    conn = get_db()
    conn.execute("""
        INSERT INTO token_usage (session_id, call_type, tokens_in, tokens_out, model)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, call_type, tokens_in, tokens_out, model))
    conn.commit()
    conn.close()


def get_session_token_total(session_id: str) -> dict:
    conn = get_db()
    row = conn.execute("""
        SELECT COALESCE(SUM(tokens_in), 0)  AS total_in,
               COALESCE(SUM(tokens_out), 0) AS total_out,
               COUNT(*) AS calls
        FROM token_usage WHERE session_id = ?
    """, (session_id,)).fetchone()
    conn.close()
    return {"tokens_in": row["total_in"], "tokens_out": row["total_out"],
            "calls": row["calls"]}


def get_daily_token_total() -> int:
    conn = get_db()
    row = conn.execute("""
        SELECT COALESCE(SUM(tokens_in + tokens_out), 0) AS total
        FROM token_usage
        WHERE date(created_at) = date('now')
    """).fetchone()
    conn.close()
    return row["total"]


# ── Search History ───────────────────────────────────────────────────────────

def save_search(query: str, keywords: list, result_ids: list):
    conn = get_db()
    conn.execute("""
        INSERT INTO search_history (query, keywords, result_ids)
        VALUES (?, ?, ?)
    """, (query, json.dumps(keywords), json.dumps(result_ids)))
    conn.commit()
    conn.close()


# ── Reports ─────────────────────────────────────────────────────────────────

def save_report(title: str, query: str, citation_style: str, content: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (title, query, citation_style, content)
        VALUES (?, ?, ?, ?)
    """, (title, query, citation_style, content))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_reports():
    conn = get_db()
    rows = conn.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_report(report_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
