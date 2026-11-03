"""Pure helper functions (no network, no DB) - unit-tested in tests/."""
import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
PHONE_IN = re.compile(r"^(?:\+?91[\s-]?)?([6-9]\d{9})$")


def detect_language(text: str) -> str:
    """'hi' if the text contains Devanagari script, else 'en'."""
    return "hi" if DEVANAGARI.search(text or "") else "en"


def normalize_phone_in(raw: str):
    """Return a normalized +91XXXXXXXXXX Indian mobile, or None if invalid."""
    if not raw:
        return None
    m = PHONE_IN.match(raw.strip())
    return "+91" + m.group(1) if m else None


def mask_phone(phone: str) -> str:
    """+919876543210 -> +91*****3210 (safe for logs/screens)."""
    if not phone or len(phone) < 6:
        return "*****"
    return phone[:3] + "*****" + phone[-4:]


def valid_date_ddmmyyyy(text: str) -> bool:
    """True for a real calendar date typed as DD-MM-YYYY."""
    from datetime import datetime
    try:
        datetime.strptime((text or "").strip(), "%d-%m-%Y")
        return True
    except (ValueError, TypeError):
        return False


def valid_time_hhmm(text: str) -> bool:
    """True for HH:MM in 24-hour format."""
    m = re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", (text or "").strip())
    return bool(m)


def business_code(name: str, seq: int) -> str:
    """'Sharma Coaching' + 7 -> 'BIZ-SHARMA-0007' style shareable code."""
    first_word = (name or "").strip().split()[0] if (name or "").strip() else ""
    slug = re.sub(r"[^A-Za-z]", "", first_word).upper()[:8] or "BIZ"
    return f"BIZ-{slug}-{seq:04d}"


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def telegram_safe(text: str, limit: int = 3900) -> str:
    """Telegram messages cap at 4096 chars; stay safely under."""
    return truncate(text, limit)
