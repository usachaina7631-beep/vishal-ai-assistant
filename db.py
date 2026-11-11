"""
SQLite data layer (local file, or Turso when configured).

Backend selection: if BOTH a Turso database URL and auth token are
supplied (Config.TURSO_DATABASE_URL / TURSO_AUTH_TOKEN), the bot talks
to Turso over libsql - a permanent database that survives redeploys on
ephemeral hosts like Render free. If either value is missing, it falls
back to the plain local SQLite file, exactly as before. All SQL stays
SQLite dialect, so the two backends are interchangeable.

Isolation rule: EVERY row is keyed by telegram user id and/or business
code. A customer bound to business A can never read business B's FAQs,
leads or appointments - all queries filter on the binding.

Tables:
  users        - one row per telegram user (plan, language, binding)
  businesses   - one row per client business (owner, name, code, about)
  faqs         - per-business Q&A used by Customer Support mode
  leads        - lead-generation form results
  appointments - appointment requests + status
  chat_memory  - short rolling AI memory per user+mode
  usage        - per-user, per-day AI call counter (rate limiting)
"""
import sqlite3
import threading
from datetime import date, datetime

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    language    TEXT DEFAULT 'en',
    plan        TEXT DEFAULT 'free',
    business_code TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS businesses (
    code        TEXT PRIMARY KEY,
    owner_id    INTEGER NOT NULL,
    name        TEXT NOT NULL,
    category    TEXT DEFAULT '',
    about       TEXT DEFAULT '',
    phone       TEXT DEFAULT '',
    timing      TEXT DEFAULT '',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS faqs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_code TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_code TEXT,
    user_id     INTEGER,
    name        TEXT,
    phone       TEXT,
    need        TEXT,
    budget      TEXT,
    status      TEXT DEFAULT 'new',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS appointments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_code TEXT,
    user_id     INTEGER,
    name        TEXT,
    phone       TEXT,
    date        TEXT,
    time        TEXT,
    purpose     TEXT,
    status      TEXT DEFAULT 'pending',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chat_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS usage (
    user_id     INTEGER NOT NULL,
    day         TEXT NOT NULL,
    ai_calls    INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_code TEXT NOT NULL,
    name        TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS doc_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    business_code TEXT NOT NULL,
    doc_name    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_business ON doc_chunks(business_code);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks(doc_id);
"""

# Plan-wise document limits for /adddoc (Pro v3 RAG).
DOC_LIMITS = {"free": 1, "lite": 3, "standard": 10, "premium": 25}


class _RowCursor:
    """Wraps a libsql cursor so rows come back as plain dicts.

    libsql (0.1.x) has no row_factory and returns plain tuples, while
    the rest of this codebase expects sqlite3.Row-style access
    (row["col"] and dict(row)). Converting via cursor.description at
    fetch time gives both, without touching any query call site.
    """

    def __init__(self, cursor):
        self._cursor = cursor
        desc = cursor.description or []
        self._cols = [d[0] for d in desc]

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    def _wrap(self, row):
        if row is None:
            return None
        return dict(zip(self._cols, row))

    def fetchone(self):
        return self._wrap(self._cursor.fetchone())

    def fetchall(self):
        return [self._wrap(r) for r in self._cursor.fetchall()]


class _LibsqlConnection:
    """Minimal sqlite3-shaped wrapper over a libsql connection."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        return _RowCursor(self._conn.execute(sql, params))

    def executescript(self, script):
        return self._conn.executescript(script)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def _open_connection(path, turso_url="", turso_token=""):
    """Return (connection, backend_name).

    Turso is used only when BOTH the URL and token are set; any other
    combination falls back to the local SQLite file at `path`.
    """
    if turso_url and turso_token:
        try:
            import libsql
        except ImportError as exc:
            raise RuntimeError(
                "TURSO_DATABASE_URL/TURSO_AUTH_TOKEN are set but the "
                "'libsql' package is not installed - run: pip install -r "
                "requirements.txt") from exc
        return _LibsqlConnection(libsql.connect(turso_url, auth_token=turso_token)), "turso"
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


class DB:
    def __init__(self, path="vishal_ai.db", turso_url="", turso_token=""):
        self.conn, self.backend = _open_connection(path, turso_url, turso_token)
        with _lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---- users ----
    def upsert_user(self, user_id, username="", first_name="", language="en"):
        with _lock:
            self.conn.execute(
                """INSERT INTO users (user_id, username, first_name, language)
                   VALUES (?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     username=excluded.username, first_name=excluded.first_name,
                     last_seen=CURRENT_TIMESTAMP""",
                (user_id, username, first_name, language))
            self.conn.commit()

    def get_user(self, user_id):
        row = self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def bind_business(self, user_id, code):
        with _lock:
            self.conn.execute("UPDATE users SET business_code=? WHERE user_id=?", (code, user_id))
            self.conn.commit()

    def set_language(self, user_id, lang):
        with _lock:
            self.conn.execute("UPDATE users SET language=? WHERE user_id=?", (lang, user_id))
            self.conn.commit()

    def set_plan(self, user_id, plan):
        with _lock:
            self.conn.execute("UPDATE users SET plan=? WHERE user_id=?", (plan, user_id))
            self.conn.commit()

    def count_users(self):
        return self.conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]

    # ---- businesses ----
    def create_business(self, code, owner_id, name, category="", about="", phone="", timing=""):
        with _lock:
            self.conn.execute(
                "INSERT INTO businesses (code, owner_id, name, category, about, phone, timing)"
                " VALUES (?,?,?,?,?,?,?)",
                (code, owner_id, name, category, about, phone, timing))
            self.conn.commit()
        return code

    def get_business(self, code):
        if not code:
            return None
        row = self.conn.execute("SELECT * FROM businesses WHERE code=?", (code,)).fetchone()
        return dict(row) if row else None

    def get_business_by_owner(self, owner_id):
        row = self.conn.execute(
            "SELECT * FROM businesses WHERE owner_id=? ORDER BY created_at LIMIT 1",
            (owner_id,)).fetchone()
        return dict(row) if row else None

    def business_count(self):
        row = self.conn.execute("SELECT COUNT(*) c FROM businesses").fetchone()
        return row["c"]

    def next_business_seq(self):
        return self.business_count() + 1

    # ---- FAQs ----
    def add_faq(self, code, question, answer):
        with _lock:
            cur = self.conn.execute(
                "INSERT INTO faqs (business_code, question, answer) VALUES (?,?,?)",
                (code, question, answer))
            self.conn.commit()
            return cur.lastrowid

    def list_faqs(self, code):
        rows = self.conn.execute(
            "SELECT * FROM faqs WHERE business_code=? ORDER BY id", (code,)).fetchall()
        return [dict(r) for r in rows]

    def delete_faq(self, code, faq_id):
        with _lock:
            self.conn.execute("DELETE FROM faqs WHERE id=? AND business_code=?", (faq_id, code))
            self.conn.commit()

    def faq_context(self, code, limit=20):
        faqs = self.list_faqs(code)[:limit]
        return "\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in faqs)

    # ---- leads ----
    def add_lead(self, business_code, user_id, name, phone, need, budget):
        with _lock:
            cur = self.conn.execute(
                "INSERT INTO leads (business_code, user_id, name, phone, need, budget)"
                " VALUES (?,?,?,?,?,?)",
                (business_code, user_id, name, phone, need, budget))
            self.conn.commit()
            return cur.lastrowid

    def list_leads(self, business_code=None, limit=50):
        if business_code:
            rows = self.conn.execute(
                "SELECT * FROM leads WHERE business_code=? ORDER BY id DESC LIMIT ?",
                (business_code, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM leads ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def count_leads(self, business_code=None):
        if business_code:
            return self.conn.execute(
                "SELECT COUNT(*) c FROM leads WHERE business_code=?",
                (business_code,)).fetchone()["c"]
        return self.conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]

    # ---- appointments ----
    def add_appointment(self, business_code, user_id, name, phone, date_, time_, purpose):
        with _lock:
            cur = self.conn.execute(
                "INSERT INTO appointments (business_code, user_id, name, phone, date, time, purpose)"
                " VALUES (?,?,?,?,?,?,?)",
                (business_code, user_id, name, phone, date_, time_, purpose))
            self.conn.commit()
            return cur.lastrowid

    def list_appointments(self, business_code=None, limit=50):
        if business_code:
            rows = self.conn.execute(
                "SELECT * FROM appointments WHERE business_code=? ORDER BY id DESC LIMIT ?",
                (business_code, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM appointments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def set_appointment_status(self, appt_id, status):
        with _lock:
            self.conn.execute("UPDATE appointments SET status=? WHERE id=?", (status, appt_id))
            self.conn.commit()

    def get_appointment(self, appt_id):
        row = self.conn.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
        return dict(row) if row else None

    def count_appointments(self, business_code=None):
        if business_code:
            return self.conn.execute(
                "SELECT COUNT(*) c FROM appointments WHERE business_code=?",
                (business_code,)).fetchone()["c"]
        return self.conn.execute("SELECT COUNT(*) c FROM appointments").fetchone()["c"]

    # ---- chat memory ----
    def add_memory(self, user_id, mode, role, content):
        with _lock:
            self.conn.execute(
                "INSERT INTO chat_memory (user_id, mode, role, content) VALUES (?,?,?,?)",
                (user_id, mode, role, content))
            self.conn.commit()

    def get_memory(self, user_id, mode, turns=6):
        rows = self.conn.execute(
            "SELECT role, content FROM chat_memory WHERE user_id=? AND mode=?"
            " ORDER BY id DESC LIMIT ?",
            (user_id, mode, turns * 2)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_memory(self, user_id):
        with _lock:
            self.conn.execute("DELETE FROM chat_memory WHERE user_id=?", (user_id,))
            self.conn.commit()

    # ---- usage / rate limiting ----
    def usage_today(self, user_id):
        row = self.conn.execute(
            "SELECT ai_calls FROM usage WHERE user_id=? AND day=?",
            (user_id, date.today().isoformat())).fetchone()
        return row["ai_calls"] if row else 0

    def bump_usage(self, user_id):
        with _lock:
            self.conn.execute(
                """INSERT INTO usage (user_id, day, ai_calls) VALUES (?,?,1)
                   ON CONFLICT(user_id, day) DO UPDATE SET ai_calls = ai_calls + 1""",
                (user_id, date.today().isoformat()))
            self.conn.commit()
        return self.usage_today(user_id)

    def total_ai_calls_today(self):
        row = self.conn.execute(
            "SELECT COALESCE(SUM(ai_calls),0) s FROM usage WHERE day=?",
            (date.today().isoformat(),)).fetchone()
        return row["s"]

    # ---- documents & chunks (Pro v3 RAG) ----
    # All queries go through self.conn (the SQLite/Turso adapter), in the
    # same ?-param + dict-row style as the rest of this file, so they
    # behave identically on the local file and on Turso/libsql.
    def add_document(self, business_code, name, chunk_count=0):
        with _lock:
            cur = self.conn.execute(
                "INSERT INTO documents (business_code, name, chunk_count)"
                " VALUES (?,?,?)", (business_code, name, chunk_count))
            self.conn.commit()
            return cur.lastrowid

    def list_documents(self, business_code):
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE business_code=? ORDER BY id",
            (business_code,)).fetchall()
        return [dict(r) for r in rows]

    def count_documents(self, business_code):
        return self.conn.execute(
            "SELECT COUNT(*) c FROM documents WHERE business_code=?",
            (business_code,)).fetchone()["c"]

    def get_document(self, business_code, doc_id):
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id=? AND business_code=?",
            (doc_id, business_code)).fetchone()
        return dict(row) if row else None

    def delete_document(self, business_code, doc_id):
        """Delete a document AND all its chunks. Scoped to the owning
        business - another business's doc_id deletes nothing.
        Returns True if a document row was removed.

        Adapted for the Turso adapter: existence is checked with a
        SELECT first instead of cursor.rowcount, which the libsql
        row-cursor wrapper does not expose."""
        with _lock:
            doc = self.conn.execute(
                "SELECT id FROM documents WHERE id=? AND business_code=?",
                (doc_id, business_code)).fetchone()
            if not doc:
                return False
            self.conn.execute(
                "DELETE FROM documents WHERE id=? AND business_code=?",
                (doc_id, business_code))
            self.conn.execute(
                "DELETE FROM doc_chunks WHERE doc_id=? AND business_code=?",
                (doc_id, business_code))
            self.conn.commit()
            return True

    def doc_limit_for_plan(self, plan):
        return DOC_LIMITS.get((plan or "free").lower(), DOC_LIMITS["free"])

    def can_add_document(self, business_code, plan):
        return self.count_documents(business_code) < self.doc_limit_for_plan(plan)

    def store_chunks(self, doc_id, business_code, chunks, vectors=None):
        """chunks: list[str]; vectors: optional parallel list of
        embedding lists (None entries allowed - keyword-searchable)."""
        import json as _json
        doc = self.conn.execute(
            "SELECT name FROM documents WHERE id=?", (doc_id,)).fetchone()
        doc_name = doc["name"] if doc else ""
        with _lock:
            for i, text in enumerate(chunks):
                vec = vectors[i] if vectors and i < len(vectors) else None
                self.conn.execute(
                    "INSERT INTO doc_chunks (doc_id, business_code, doc_name,"
                    " chunk_index, text, embedding) VALUES (?,?,?,?,?,?)",
                    (doc_id, business_code, doc_name, i, text,
                     _json.dumps(vec) if vec is not None else None))
            self.conn.execute(
                "UPDATE documents SET chunk_count=? WHERE id=?",
                (len(chunks), doc_id))
            self.conn.commit()

    def list_chunks(self, business_code, doc_id=None):
        import json as _json
        if doc_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM doc_chunks WHERE business_code=? AND doc_id=?"
                " ORDER BY chunk_index", (business_code, doc_id)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM doc_chunks WHERE business_code=?"
                " ORDER BY doc_id, chunk_index", (business_code,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            raw = d.get("embedding")
            d["embedding"] = _json.loads(raw) if raw else None
            out.append(d)
        return out

    def count_chunks(self, business_code):
        return self.conn.execute(
            "SELECT COUNT(*) c FROM doc_chunks WHERE business_code=?",
            (business_code,)).fetchone()["c"]
