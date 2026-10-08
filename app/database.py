"""
database.py — Persistent storage layer (SQLite)
--------------------------------------------------
PREMIUM UPGRADE: replaces the old in-memory `stats` dict (which reset
every time the server restarted) with a real SQLite database so that
history, stats, feedback and API keys all survive restarts.

No external DB server needed — everything lives in models/truthline.db
"""

import os
import sqlite3
import secrets
import string
from datetime import datetime, timedelta
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "models", "truthline.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                text_preview TEXT,
                full_text TEXT,
                label TEXT,
                confidence REAL,
                model_used TEXT,
                sensationalism REAL,
                readability REAL,
                credibility TEXT,
                source TEXT,
                feedback TEXT,
                created_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT UNIQUE,
                label TEXT,
                tier TEXT DEFAULT 'premium',
                requests_used INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        conn.commit()
        # Safe migration: add user_id column if it doesn't exist yet
        try:
            c.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def create_user(username, email, password_hash):
    with get_conn() as conn:
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (username.strip(), email.strip().lower(), password_hash, datetime.utcnow().isoformat()))
            return c.lastrowid, None
        except sqlite3.IntegrityError as e:
            err_msg = str(e).lower()
            if "username" in err_msg:
                return None, "Username already exists."
            elif "email" in err_msg:
                return None, "Email already registered."
            return None, "User registration failed."


def get_user_by_identifier(identifier):
    """identifier can be username or email"""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)
        """, (identifier.strip(), identifier.strip()))
        row = c.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, email, created_at FROM users WHERE id=?", (user_id,))
        row = c.fetchone()
        return dict(row) if row else None


def add_history(entry):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO history
            (user_id, text_preview, full_text, label, confidence, model_used,
             sensationalism, readability, credibility, source, feedback, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            entry.get("user_id"), entry.get("text_preview"), entry.get("full_text"),
            entry.get("label"), entry.get("confidence"), entry.get("model_used"),
            entry.get("sensationalism"), entry.get("readability"), entry.get("credibility"),
            entry.get("source"), None, datetime.utcnow().isoformat()
        ))
        return c.lastrowid


def set_feedback(entry_id, feedback):
    """feedback: 'correct' or 'incorrect'"""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE history SET feedback=? WHERE id=?", (feedback, entry_id))
        return c.rowcount > 0


def get_history(limit=50, search=None):
    with get_conn() as conn:
        c = conn.cursor()
        if search:
            c.execute("""SELECT * FROM history WHERE text_preview LIKE ?
                         ORDER BY id DESC LIMIT ?""", (f"%{search}%", limit))
        else:
            c.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


def get_user_history(user_id, limit=50, search=None):
    """Return history entries belonging to a specific user."""
    with get_conn() as conn:
        c = conn.cursor()
        if search:
            c.execute("""
                SELECT * FROM history
                WHERE user_id=? AND text_preview LIKE ?
                ORDER BY id DESC LIMIT ?
            """, (user_id, f"%{search}%", limit))
        else:
            c.execute("""
                SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?
            """, (user_id, limit))
        return [dict(r) for r in c.fetchall()]


def get_user_stats(user_id):
    """Return per-user statistics."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) AS n FROM history WHERE user_id=?", (user_id,))
        total = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE user_id=? AND label='Real'", (user_id,))
        real = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE user_id=? AND label='Fake'", (user_id,))
        fake = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE user_id=? AND feedback='correct'", (user_id,))
        correct = c.fetchone()["n"]
        c.execute("SELECT AVG(confidence) AS a FROM history WHERE user_id=?", (user_id,))
        avg_conf = c.fetchone()["a"] or 0

        # Activity: checks per day for the last 7 days
        c.execute("""
            SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS cnt
            FROM history
            WHERE user_id=? AND created_at >= date('now', '-7 days')
            GROUP BY day
            ORDER BY day ASC
        """, (user_id,))
        activity = [{"day": r["day"], "count": r["cnt"]} for r in c.fetchall()]

        return {
            "total_checked": total,
            "real_count": real,
            "fake_count": fake,
            "feedback_correct": correct,
            "avg_confidence": round(avg_conf, 2),
            "activity": activity,
        }


def get_trending(limit=6):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT text_preview, full_text FROM history ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()

    stop_words = {
        "the", "a", "an", "in", "on", "of", "to", "for", "is", "was", "are",
        "and", "or", "this", "that", "with", "from", "at", "by", "as", "it",
        "be", "has", "have", "had", "not", "but", "what", "all", "were", "when",
        "can", "said", "new", "news", "will", "after", "who", "they", "been",
        "he", "she", "we", "about", "more", "into", "over", "their", "there"
    }
    import re
    from collections import Counter
    words = []
    for r in rows:
        txt = (r["full_text"] or r["text_preview"] or "").lower()
        clean = re.findall(r"[a-z]{3,}", txt)
        words.extend([w for w in clean if w not in stop_words])

    top = Counter(words).most_common(limit)
    if not top:
        return [
            {"keyword": "Space Exploration", "count": 14},
            {"keyword": "AI Verification", "count": 11},
            {"keyword": "Climate Science", "count": 8},
            {"keyword": "Global Semiconductor", "count": 6},
            {"keyword": "Medical Technology", "count": 5},
        ]
    return [{"keyword": k.title(), "count": v} for k, v in top]


def get_stats():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) AS n FROM history")
        total = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE label='Real'")
        real = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE label='Fake'")
        fake = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE feedback='correct'")
        correct = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n FROM history WHERE feedback='incorrect'")
        incorrect = c.fetchone()["n"]
        c.execute("SELECT AVG(confidence) AS a FROM history")
        avg_conf = c.fetchone()["a"] or 0
        return {
            "total_checked": total, "real_count": real, "fake_count": fake,
            "feedback_correct": correct, "feedback_incorrect": incorrect,
            "avg_confidence": round(avg_conf, 2),
        }


def _generate_key():
    return "tl_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))


def create_api_key(label="default"):
    key = _generate_key()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO api_keys (api_key, label, created_at) VALUES (?,?,?)",
                  (key, label, datetime.utcnow().isoformat()))
        return key


def validate_api_key(key):
    if not key:
        return None
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM api_keys WHERE api_key=?", (key,))
        row = c.fetchone()
        if row:
            c.execute("UPDATE api_keys SET requests_used = requests_used + 1 WHERE api_key=?", (key,))
            return dict(row)
        return None


def list_api_keys():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, api_key, label, tier, requests_used, created_at FROM api_keys ORDER BY id DESC")
        return [dict(r) for r in c.fetchall()]
