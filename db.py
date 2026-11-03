"""
SQLite data layer.

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
"""


class DB:
    def __init__(self, path="vishal_ai.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
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
