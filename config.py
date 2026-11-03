"""
Vishal AI Assistant - configuration.

Secrets are read ONLY from environment variables (or a local .env file
for Termux/local testing). Nothing secret is ever hard-coded, logged,
or committed. On Render/Northflank, set these in the dashboard instead
of using a .env file.
"""
import os


def _load_dotenv(path=".env"):
    """Minimal .env loader (no external dependency). Existing env wins."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


class Config:
    # --- Telegram (required) ---
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    # Your own Telegram numeric user id (get it from @userinfobot).
    # Only this id can use /admin. Comma-separate for more than one.
    ADMIN_IDS = {
        int(x) for x in os.environ.get("ADMIN_IDS", "").replace(" ", "").split(",")
        if x.isdigit()
    }

    # --- AI providers (at least ONE key required) ---
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    # Model routing: fast/cheap model for high-volume chat & support,
    # quality model for creator/research work. Model names change over
    # time - verify the current names at console.groq.com/docs/models
    # and override via env if a name is retired.
    GROQ_MODEL_FAST = os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
    GROQ_MODEL_QUALITY = os.environ.get("GROQ_MODEL_QUALITY", "openai/gpt-oss-120b")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # --- Limits (protect the shared free quota) ---
    FREE_DAILY_LIMIT = int(os.environ.get("FREE_DAILY_LIMIT", "30"))   # AI replies / user / day
    MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "2000"))
    MEMORY_TURNS = int(os.environ.get("MEMORY_TURNS", "6"))            # chat memory per user

    # --- Optional: Make.com webhook for lead -> Google Sheets/email ---
    # Leave empty to disable. This integration is OPTIONAL and is only
    # "working" once you test it with your own Make scenario (see docs).
    MAKE_WEBHOOK_URL = os.environ.get("MAKE_WEBHOOK_URL", "")

    DB_PATH = os.environ.get("DB_PATH", "vishal_ai.db")
    PORT = int(os.environ.get("PORT", "8080"))  # health endpoint for free hosts


def validate():
    """Return a list of human-readable problems (empty list = ready)."""
    problems = []
    if not Config.BOT_TOKEN:
        problems.append("TELEGRAM_BOT_TOKEN is not set.")
    if not Config.GROQ_API_KEY and not Config.GEMINI_API_KEY:
        problems.append("Set at least one AI key: GROQ_API_KEY and/or GEMINI_API_KEY.")
    if not Config.ADMIN_IDS:
        problems.append("ADMIN_IDS is not set - /admin will stay locked for everyone.")
    return problems
