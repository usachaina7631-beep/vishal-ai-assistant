"""
Vishal AI Assistant - Telegram bot (long-polling, no framework needed).

Pro features
------------
* One bot, many client businesses: /setbusiness creates a business and a
  share link  t.me/<bot>?start=BIZ-XXXX-0001 . Customers opening that
  link are bound to that business for Support / Leads / Appointments.
* 7 AI modes (chat, translator, youtube, business, support, study,
  research) with Groq -> Gemini automatic fallback and smart model
  routing (fast model for volume, quality model for creation).
  /chat opens the free-form AI Chat mode (quality model).
  Optional: /deepseek runs AI Chat on the DeepSeek engine instead -
  only when the admin has set DEEPSEEK_API_KEY (see README).
* Utilities: Photo Enhance (/enhance - Pillow only, fully local, no
  paid APIs) and legal Book Search (/book <name> - Google Books
  metadata/preview links + a Project Gutenberg link; never pirated
  copies). Plus a static Daily Class Schedule (/schedule, IST) that
  marks the class running now and the next one, and a GK Quiz (/quiz):
  10 text-based questions per session from a built-in original
  question bank, with live score and a result at the end.
* Guided forms for Lead Generation and Appointment Requests with Indian
  phone/date/time validation.
* Telegram-native Admin Dashboard (/admin): live stats, leads &
  appointments with Confirm/Reject buttons, CSV export inside chat,
  plan upgrades, broadcast to all users. Mobile-only management.
* Safety: per-user daily AI limit (plan-aware), input length cap,
  admin-only commands checked on every call, secrets only from env,
  user-data commands /mydata and /deletedata, full error handling -
  the polling loop never dies on a bad update.

Run:  python bot.py     (after setting env vars - see docs/02-SETUP.md)
"""
import csv
import io
import json
import logging
import random
import threading
import time
from datetime import datetime, time as dt_time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import requests

from ai_client import AIClient
from config import Config, validate
from db import DB
from prompts import MODE_LABELS, build_system_prompt
from utils import (business_code, detect_language, mask_phone,
                   normalize_phone_in, telegram_safe, truncate,
                   valid_date_ddmmyyyy, valid_time_hhmm)

try:  # Photo Enhance is Pillow-only; the bot must still run without it
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on deployment env
    PIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vishal-bot")

API = f"https://api.telegram.org/bot{Config.BOT_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{Config.BOT_TOKEN}"
db = DB(Config.DB_PATH)
ai = AIClient()

PLAN_LIMITS = {"free": Config.FREE_DAILY_LIMIT, "lite": 300,
               "standard": 1000, "premium": 5000}
# 'chat' (AI Chat) is here on purpose: free-form AI Chat answers use the
# quality model (Config.GROQ_MODEL_QUALITY) via ai_client's quality path.
QUALITY_MODES = {"youtube", "research", "business", "chat"}

# In-memory conversation state: user_id -> {"flow": ..., "step": ..., "data": {}}
states = {}
# Current AI mode per user (defaults to 'chat')
modes = {}
# AI Chat engine per user: absent = default Groq chain, 'deepseek' =
# DeepSeek engine (only set by /deepseek when the admin enabled it)
engines = {}
# Users who pressed /enhance and were asked to send a photo next
awaiting_photo = set()
# GK Quiz sessions: user_id -> {"order": [question index, ...],
# "pos": current question position, "score": sahi jawabon ki sankhya}
quiz_sessions = {}


# ----------------------------------------------------------------------
# Telegram API helpers
# ----------------------------------------------------------------------
def tg(method, **kwargs):
    try:
        r = requests.post(f"{API}/{method}", json=kwargs, timeout=30)
        data = r.json()
        if not data.get("ok"):
            log.warning("Telegram %s failed: %s", method, data.get("description"))
        return data
    except Exception as exc:
        log.warning("Telegram %s error: %s", method, type(exc).__name__)
        return {"ok": False}


def send(chat_id, text, keyboard=None, parse_mode=None):
    payload = {"chat_id": chat_id, "text": telegram_safe(text)}
    if keyboard:
        payload["reply_markup"] = keyboard
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return tg("sendMessage", **payload)


def send_document(chat_id, filename, content_bytes, caption=""):
    try:
        r = requests.post(
            f"{API}/sendDocument",
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (filename, content_bytes)}, timeout=60)
        return r.json()
    except Exception as exc:
        log.warning("sendDocument error: %s", type(exc).__name__)
        return {"ok": False}


def send_photo(chat_id, photo_bytes, caption=""):
    try:
        r = requests.post(
            f"{API}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("enhanced.jpg", photo_bytes, "image/jpeg")},
            timeout=60)
        return r.json()
    except Exception as exc:
        log.warning("sendPhoto error: %s", type(exc).__name__)
        return {"ok": False}


def typing(chat_id):
    tg("sendChatAction", chat_id=chat_id, action="typing")


def answer_callback(cb_id, text=""):
    tg("answerCallbackQuery", callback_query_id=cb_id, text=text)


def main_keyboard():
    rows = [
        [{"text": MODE_LABELS["chat"]}, {"text": MODE_LABELS["translator"]}],
        [{"text": MODE_LABELS["youtube"]}, {"text": MODE_LABELS["business"]}],
        [{"text": MODE_LABELS["support"]}, {"text": MODE_LABELS["study"]}],
        [{"text": MODE_LABELS["research"]}, {"text": "📝 Lead Form"}],
        [{"text": "📅 Appointment"}, {"text": "📸 Photo Enhance"}],
        [{"text": "📖 Book Search"}, {"text": "🗓 Class Schedule"}],
        [{"text": "📝 Quiz"}, {"text": "ℹ️ Help / मदद"}],
    ]
    return {"keyboard": rows, "resize_keyboard": True}


LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

WELCOME = (
    "🙏 Namaste! Main hoon *Vishal AI Assistant*.\n\n"
    "Main aapki madad kar sakta hoon:\n"
    "💬 AI Chat • 🔤 Hindi-English Translator\n"
    "🎬 YouTube Content • 💼 Business Help\n"
    "🛎️ Customer Support • 📚 Study Help\n"
    "🔎 Research • 📝 Leads • 📅 Appointments\n"
    "📸 Photo Enhance • 📖 Book Search • 🗓 Class Schedule\n"
    "📝 GK Quiz (roz practice ke liye)\n\n"
    "Neeche button chunein ya seedha apna sawal likhein.\n"
    "AI Chat ke liye /chat, photo saaf karne ke liye /enhance, "
    "book dhundhne ke liye /book <naam>, roz ki class timing ke "
    "liye /schedule aur GK quiz ke liye /quiz likhein.\n"
    "Choose a button below or just type your question."
)
HELP = (
    "ℹ️ *Help / मदद*\n\n"
    "• Koi bhi mode button dabakar chunein, phir apna sawal likhein.\n"
    "• /chat — AI Chat mode ON (koi bhi sawal poochho)  • /stopchat — AI Chat band\n"
    "• /deepseek — DeepSeek engine par AI Chat (admin ne enable kiya ho tabhi chalega)\n"
    "• /enhance — photo bhejo, main use saaf/enhance karke wapas bhej dunga\n"
    "• /book <book ka naam> — book search (sirf legal preview/info links milenge)\n"
    "• /schedule — roz ki class time-table (abhi chal rahi class aur agli class ke saath)\n"
    "• /quiz — GK Quiz: 10 sawal (Bihar GK, Polity, History, Geography, Science), jawab A/B/C/D ya 1-4 me dein  • /stopquiz — quiz band\n"
    "• /lead — enquiry form  • /book — appointment form\n"
    "• /setbusiness — business owners: apna business jodein aur apna "
    "customer link payein\n"
    "• /mybusiness — apna business, FAQs aur stats dekhein\n"
    "• /language — Hindi/English chunein\n"
    "• /mydata — apna saved data dekhein  • /deletedata — data delete karein\n"
    "• /admin — owner dashboard (sirf admin ke liye)"
)


# ----------------------------------------------------------------------
# AI call with limits, memory and fallback
# ----------------------------------------------------------------------
DEEPSEEK_OFF_MSG = (
    "🔒 DeepSeek abhi enable nahi hai.\n"
    "Admin ne abhi tak DeepSeek ki key nahi lagayi hai, isliye ye "
    "engine band hai.\n"
    "Tab tak /chat wala AI Chat bilkul chal raha hai — wahi use "
    "karein. / DeepSeek is not enabled yet - please use /chat."
)


def ask_ai(user_id, mode, user_text, engine=None):
    user = db.get_user(user_id) or {}
    plan = user.get("plan", "free")
    limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    if user_id not in Config.ADMIN_IDS and db.usage_today(user_id) >= limit:
        return ("⏳ Aaj ki free limit khatm ho gayi hai (limit reached for today). "
                "Kal phir try karein, ya owner se plan upgrade karwayein.")
    business = db.get_business(user.get("business_code")) if mode == "support" else None
    if mode == "support" and not business:
        return ("🛎️ Pehle kisi business se judein — business ka link kholein, "
                "ya owner se apna customer link mangein. / First open a business "
                "link so I know which business to support.")
    faq_text = db.faq_context(business["code"]) if business else ""
    messages = [{"role": "system",
                 "content": build_system_prompt(mode, business, faq_text)}]
    messages += db.get_memory(user_id, mode, Config.MEMORY_TURNS)
    messages.append({"role": "user", "content": user_text})
    if engine == "deepseek":
        # User explicitly picked the DeepSeek engine with /deepseek
        text, _provider = ai.chat_deepseek(messages)
    else:
        text, _provider = ai.chat(messages, quality=mode in QUALITY_MODES)
    if _provider:  # only count successful AI calls
        db.bump_usage(user_id)
        db.add_memory(user_id, mode, "user", user_text[:500])
        db.add_memory(user_id, mode, "assistant", text[:1000])
    return text


def notify_admins(text):
    for admin_id in Config.ADMIN_IDS:
        send(admin_id, text)


def push_make_webhook(event):
    """Optional Make.com hook (lead/appointment -> Sheets/email).
    Disabled unless MAKE_WEBHOOK_URL is set. Fire-and-forget."""
    if not Config.MAKE_WEBHOOK_URL:
        return
    def _post():
        try:
            requests.post(Config.MAKE_WEBHOOK_URL, json=event, timeout=15)
        except Exception as exc:
            log.warning("Make webhook failed: %s", type(exc).__name__)
    threading.Thread(target=_post, daemon=True).start()


# ----------------------------------------------------------------------
# Photo Enhance (Pillow only - no paid APIs, no external AI services)
# ----------------------------------------------------------------------
MAX_PHOTO_BYTES = 10 * 1024 * 1024   # reject bigger downloads (free-host RAM)
MAX_ENHANCED_DIM = 4096              # cap the 2x upscale's long edge

ENHANCE_HOWTO = (
    "📸 *Photo Enhance*\n\n"
    "Apni photo yahin bhej dein — main use enhance karke (2x bada, "
    "sharp, rang saaf) wapas bhej dunga.\n"
    "Bas photo attach karke bhej dein 👇\n"
    "Send your photo here and I will send back an enhanced copy."
)


def download_file(file_id):
    """Download a Telegram file by id; return bytes or None."""
    info = tg("getFile", file_id=file_id)
    path = info.get("result", {}).get("file_path")
    if not path:
        return None
    try:
        r = requests.get(f"{FILE_API}/{path}", timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        log.warning("File download failed: %s", type(exc).__name__)
        return None


def enhance_image_bytes(raw):
    """Enhance image bytes with Pillow: mild denoise, 2x upscale,
    sharpen, gentle contrast/colour lift. Return JPEG bytes, or None
    if the input is not a readable image (or Pillow is unavailable)."""
    if not PIL_AVAILABLE or not raw:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None
    try:
        img = img.filter(ImageFilter.MedianFilter(size=3))  # mild denoise
        w, h = img.size
        scale = min(2.0, MAX_ENHANCED_DIM / max(w, h))
        if scale > 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120,
                                                 threshold=2))
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.15)
        img = ImageEnhance.Brightness(img).enhance(1.02)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=90)
        data = out.getvalue()
        if len(data) > MAX_PHOTO_BYTES:  # stay under Telegram's photo cap
            w, h = img.size
            img = img.resize((round(w * 0.7), round(h * 0.7)), Image.LANCZOS)
            out = io.BytesIO()
            img.save(out, "JPEG", quality=85)
            data = out.getvalue()
        return data
    except Exception as exc:
        log.warning("Enhance failed: %s", type(exc).__name__)
        return None


def handle_photo(msg):
    """Enhance any photo (or image file) the user sends and return it."""
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {})
    user_id = user.get("id")
    if user_id:
        db.upsert_user(user_id, user.get("username", ""),
                       user.get("first_name", ""), "en")
        awaiting_photo.discard(user_id)
    if msg.get("photo"):
        largest = msg["photo"][-1]  # Telegram sends sizes small -> large
        file_id = largest.get("file_id")
        file_size = largest.get("file_size") or 0
    else:  # image sent as a document/file
        doc = msg.get("document") or {}
        file_id = doc.get("file_id")
        file_size = doc.get("file_size") or 0
    if not file_id:
        return send(chat_id, "⚠️ Photo samajh nahi aaya. Phir se bhejein.")
    if file_size > MAX_PHOTO_BYTES:
        return send(chat_id, "⚠️ Ye photo bahut badi hai (10 MB se zyada). "
                             "Thodi chhoti photo bhejein. / File too large - "
                             "please send a photo under 10 MB.")
    typing(chat_id)
    raw = download_file(file_id)
    if raw is None:
        return send(chat_id, "⚠️ Photo download nahi ho payi. "
                             "Phir se try karein. / Download failed, try again.")
    if len(raw) > MAX_PHOTO_BYTES:
        return send(chat_id, "⚠️ Ye photo bahut badi hai (10 MB se zyada). "
                             "Thodi chhoti photo bhejein. / File too large.")
    enhanced = enhance_image_bytes(raw)
    if enhanced is None:
        return send(chat_id, "⚠️ Ye file image nahi lagi. Kripya ek photo "
                             "(JPG/PNG) bhejein. / That file is not an image "
                             "- please send a photo.")
    send(chat_id, "✨ Photo enhance ho gayi hai, bhej raha hoon…")
    result = send_photo(chat_id, enhanced,
                        caption="📸✨ Enhanced photo taiyaar hai! "
                                "(2x upscale + sharp + rang saaf)\n"
                                "Aur photo ke liye /enhance dabayein.")
    if not result.get("ok"):
        send(chat_id, "⚠️ Enhanced photo bhejne me dikkat aayi. "
                      "Phir se try karein.")


# ----------------------------------------------------------------------
# Book Search (LEGAL only - Google Books metadata + preview/info links,
# plus a Project Gutenberg link for public-domain full texts. This bot
# never fetches, links or delivers pirated copies of books.)
# ----------------------------------------------------------------------
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"


def search_books(query, max_results=5):
    """Return a formatted Hinglish reply for a book search."""
    try:
        r = requests.get(GOOGLE_BOOKS_URL,
                         params={"q": query, "maxResults": max_results},
                         timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("Google Books failed: %s", type(exc).__name__)
        return ("⚠️ Book search abhi kaam nahi kar raha. "
                "Thodi der baad phir try karein.")
    gutenberg = ("https://www.gutenberg.org/ebooks/search/?query="
                 + quote_plus(query))
    items = data.get("items") or []
    if not items:
        return (f"📖 '{query}' naam ki koi book nahi mili.\n\n"
                " spelling check karke phir try karein.\n"
                f"📜 Public-domain (free, legal) books yahan bhi milegi:\n"
                f"{gutenberg}")
    lines = [f"📖 *'{query}' ke liye top books:*\n"]
    for item in items:
        info = item.get("volumeInfo", {})
        title = info.get("title") or "Unknown title"
        authors = ", ".join(info.get("authors") or []) or "Unknown author"
        block = f"📚 *{title}*\n✍️ {authors}"
        desc = truncate(info.get("description") or "", 200)
        if desc:
            block += f"\n{desc}"
        link = info.get("infoLink") or info.get("previewLink")
        if link:
            block += f"\n🔗 Preview / details: {link}"
        lines.append(block)
    lines.append("📜 Purani public-domain books ka *pura text* (free aur "
                 f"legal) Project Gutenberg par:\n{gutenberg}")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# Daily Class Schedule (static timetable - timings are IST / Asia/Kolkata
# because the host server may run in another timezone. Admin: class
# timings badalne ke liye sirf CLASS_SCHEDULE list edit karein.)
# ----------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")

# (start time IST, class name) - din ke order me rakhein.
CLASS_SCHEDULE = [
    (dt_time(7, 0), "Current Affairs"),
    (dt_time(8, 0), "Science"),
    (dt_time(9, 0), "Mix"),
    (dt_time(11, 0), "Hindi"),
    (dt_time(12, 0), "History Marathon"),
    (dt_time(13, 0), "Mix GK GS"),
    (dt_time(14, 0), "Reasoning"),
    (dt_time(15, 0), "Maths"),
    (dt_time(16, 0), "C.D.P"),
    (dt_time(17, 0), "Geography"),
    (dt_time(18, 0), "History"),
    (dt_time(19, 0), "English"),
    (dt_time(19, 30), "100 Question"),
    (dt_time(20, 30), "Polity"),
    (dt_time(21, 30), "Members Demands"),
    (dt_time(22, 30), "Mix"),
]


def _fmt_slot_time(t):
    """dt_time(7, 0) -> '7:00 AM'; dt_time(19, 30) -> '7:30 PM'."""
    return t.strftime("%I:%M %p").lstrip("0")


def current_and_next(now=None):
    """(current, next) CLASS_SCHEDULE entries for `now`.

    `now` may be any aware datetime (converted to IST); naive values are
    read as IST, and None means the current IST time. A class counts as
    running from its start time until the next class starts. The day
    wraps at midnight: pehli class se pehle, pichhle din ki aakhri class
    'current' aur aaj ki pehli class 'next' hoti hai.
    """
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)
    now_t = now.time()
    current = CLASS_SCHEDULE[-1]  # wrap: pichhle din ki aakhri class
    nxt = CLASS_SCHEDULE[0]
    for i, entry in enumerate(CLASS_SCHEDULE):
        if entry[0] <= now_t:
            current = entry
            nxt = CLASS_SCHEDULE[(i + 1) % len(CLASS_SCHEDULE)]
        else:
            break
    return current, nxt


def schedule_text(now=None):
    """Poora din ka class time-table; chal rahi aur agli class marked."""
    current, nxt = current_and_next(now)
    lines = ["🗓 *Roz ki Class Time-Table* (samay: IST)\n"]
    for entry in CLASS_SCHEDULE:
        line = f"{_fmt_slot_time(entry[0])} — {entry[1]}"
        if entry == current:
            line += "  ⬅️ *abhi chal rahi hai*"
        elif entry == nxt:
            line += "  ⏭ *agli class*"
        lines.append(line)
    lines.append("\n⏰ Class ke samay par join kar lein. Sawal poochhna "
                 "ho to /study mode ya /chat use karein!")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# GK Quiz (/quiz) - text flow: bot sawal ke 4 options (A-D) bhejta hai,
# user letter ya number me jawab deta hai. Telegram quiz-polls is code
# me fit nahi hote: poll ke jawab 'poll_answer' naam ke alag update
# type me aate hain, jise ye polling loop handle hi nahi karta - isliye
# bot ke baaki guided flows wala text pattern hi sahi baithta hai.
# Admin: sawal jodne/badlane ke liye sirf QUIZ_QUESTIONS list edit
# karein. Har entry: "q" (sawal), "options" (thik 4), "answer" (sahi
# option ka index, 0 = A) aur "why" (ek line ki vyakhya). Saare sawal
# original aur pathyapustak-star ke pakke tathyon par aadharit hain.
# ----------------------------------------------------------------------
QUIZ_LENGTH = 10
OPTION_LETTERS = "ABCD"

QUIZ_QUESTIONS = [
    # --- Bihar GK ---
    {"q": "Bihar ki rajdhani kaun si hai?",
     "options": ["Gaya", "Patna", "Darbhanga", "Bhagalpur"], "answer": 1,
     "why": "Patna ka prachin naam Pataliputra tha; ye Bihar ki rajdhani aur sabse bada shahar hai."},
    {"q": "Bihar Diwas har saal kab manaya jata hai?",
     "options": ["15 August", "26 January", "22 March", "2 October"], "answer": 2,
     "why": "22 March 1912 ko Bihar, Bangal se alag prant bana tha; isi din Bihar Diwas manaya jata hai."},
    {"q": "Prachin Nalanda Mahavihara ke avashesh aaj kis jile me sthit hain?",
     "options": ["Patna", "Nalanda", "Gaya", "Vaishali"], "answer": 1,
     "why": "Nalanda ke avashesh aaj ke Nalanda jile me hain; ye prachin kaal ka vishwa-prasiddh shiksha kendra tha."},
    {"q": "Mahatma Gandhi ne Champaran Satyagraha kis varsh shuru kiya tha?",
     "options": ["1915", "1919", "1920", "1917"], "answer": 3,
     "why": "1917 me Gandhi ji ne Champaran ke neel kisano ke liye apna pehla satyagraha kiya tha."},
    {"q": "Jharkhand rajya Bihar se alag hokar kab bana?",
     "options": ["15 November 2000", "1 November 2000", "15 August 2000", "26 January 2001"], "answer": 0,
     "why": "15 November 2000 ko Jharkhand, Bihar se alag hokar Bharat ka naya rajya bana."},
    {"q": "Ganga nadi par bana Mahatma Gandhi Setu kin do shahron ko jodta hai?",
     "options": ["Patna - Ara", "Patna - Hajipur", "Munger - Bhagalpur", "Patna - Danapur"], "answer": 1,
     "why": "Mahatma Gandhi Setu Ganga nadi par bana hai aur Patna ko Hajipur se jodta hai."},
    {"q": "Bodh Gaya me Gautam Buddha ko kis vriksh ke neeche gyaan ki prapti hui thi?",
     "options": ["Bargad", "Neem", "Aam", "Peepal (Bodhi vriksh)"], "answer": 3,
     "why": "Bodh Gaya me Bodhi vriksh (peepal) ke neeche Buddha ko gyaan ki prapti hui thi."},
    {"q": "Kshetrafal ke hisaab se Bihar ka sabse bada jila kaun sa hai?",
     "options": ["Paschim Champaran", "Patna", "Gaya", "Darbhanga"], "answer": 0,
     "why": "Paschim Champaran kshetrafal me Bihar ka sabse bada jila hai."},
    {"q": "'Bihar' naam 'Vihar' shabd se bana mana jata hai. 'Vihar' ka arth kya hai?",
     "options": ["Kila", "Bandargah", "Bauddh bhikshuon ka math", "Mela"], "answer": 2,
     "why": "Is kshetra me Bauddh vihar (math) bahut the, isi 'Vihar' shabd se 'Bihar' naam pada."},
    {"q": "Bharat ke pehle Rashtrapati Dr. Rajendra Prasad kis rajya se the?",
     "options": ["Uttar Pradesh", "Madhya Pradesh", "Bihar", "Bengal"], "answer": 2,
     "why": "Dr. Rajendra Prasad ka janm Bihar ke Siwan jile ke Ziradei gaon me hua tha."},
    # --- India GK ---
    {"q": "Bharat ka Rashtragan 'Jana Gana Mana' kisne likha tha?",
     "options": ["Bankim Chandra Chatterjee", "Rabindranath Tagore", "Sarojini Naidu", "Subhas Chandra Bose"], "answer": 1,
     "why": "'Jana Gana Mana' Gurudev Rabindranath Tagore ne likha tha; 1950 me ye rashtragan bana."},
    {"q": "Bharat ka Rashtriya Geet 'Vande Mataram' kis pustak se liya gaya hai?",
     "options": ["Gitanjali", "Godan", "Madhushala", "Anandamath"], "answer": 3,
     "why": "'Vande Mataram' Bankim Chandra Chatterjee ke upanyas 'Anandamath' (1882) se liya gaya hai."},
    {"q": "Bharat ka rashtriya pashu kaun sa hai?",
     "options": ["Asiatic Singh", "Bengal Tiger", "Bharatiya Haathi", "Genda"], "answer": 1,
     "why": "Bengal Tiger (bagh) Bharat ka rashtriya pashu hai."},
    {"q": "Bharat ka pehla kritrim upgrah (satellite) kaun sa tha?",
     "options": ["Bhaskar", "Rohini", "Aryabhata", "INSAT-1A"], "answer": 2,
     "why": "Aryabhata 19 April 1975 ko bheja gaya Bharat ka pehla upgrah tha."},
    {"q": "Bharat ka sabse lamba bandh (dam) kaun sa hai?",
     "options": ["Hirakud Bandh", "Bhakra Bandh", "Sardar Sarovar", "Tehri Bandh"], "answer": 0,
     "why": "Odisha me Mahanadi nadi par bana Hirakud Bharat ka sabse lamba bandh hai."},
    # --- Polity ---
    {"q": "Bharat ka Samvidhan kab lagu hua tha?",
     "options": ["15 August 1947", "26 November 1949", "26 January 1950", "2 October 1950"], "answer": 2,
     "why": "Samvidhan 26 November 1949 ko apnaya gaya aur 26 January 1950 ko pure desh me lagu hua."},
    {"q": "Bharat me matdaan (vote) karne ki nyuntam aayu kitni hai?",
     "options": ["21 varsh", "19 varsh", "16 varsh", "18 varsh"], "answer": 3,
     "why": "61ve Samvidhan Sanshodhan se matdaan ki aayu 21 se ghatakar 18 kar di gayi."},
    {"q": "Rajya Sabha ke sadasya ka karyakaal kitne varsh ka hota hai?",
     "options": ["5 varsh", "6 varsh", "4 varsh", "3 varsh"], "answer": 1,
     "why": "Rajya Sabha ek sthayi sabha hai; iske sadasya 6 varsh ke karyakaal ke liye chune jate hain."},
    {"q": "Samvidhan ka Anuchhed 370 kis vishay se sambandhit tha?",
     "options": ["Chunav Aayog", "Hindi bhasha", "Jammu-Kashmir ki vishesh sthiti", "Vitt Aayog"], "answer": 2,
     "why": "Anuchhed 370 Jammu-Kashmir ko vishesh darja deta tha; August 2019 me use prabhavheen kar diya gaya."},
    {"q": "Vartaman me Bharat me kul kitne Rajya hain?",
     "options": ["29", "27", "30", "28"], "answer": 3,
     "why": "Vartaman me Bharat me 28 rajya aur 8 kendra shasit pradesh hain."},
    # --- History ---
    {"q": "Jallianwala Bagh hatyakand kis varsh hua tha?",
     "options": ["1919", "1917", "1920", "1918"], "answer": 0,
     "why": "13 April 1919 ko Amritsar ke Jallianwala Bagh me General Dyer ne bheed par goli chalwayi thi."},
    {"q": "Dandi Yatra (Namak Satyagraha) kis varsh hui thi?",
     "options": ["1920", "1942", "1930", "1932"], "answer": 2,
     "why": "Gandhi ji ne 12 March 1930 ko Sabarmati Ashram se Dandi tak ki yatra shuru ki thi."},
    {"q": "Panipat ki pehli ladai kis varsh hui thi?",
     "options": ["1556", "1526", "1761", "1506"], "answer": 1,
     "why": "1526 me Babur ne Ibrahim Lodi ko hara kar Bharat me Mughal shasan ki neev rakhi."},
    {"q": "Samrat Ashok kis vansh ke shasak the?",
     "options": ["Gupta vansh", "Kushan vansh", "Mughal vansh", "Maurya vansh"], "answer": 3,
     "why": "Samrat Ashok Maurya vansh ke sabse prasiddh shasak the; unki rajdhani Pataliputra thi."},
    {"q": "'Jai Jawan Jai Kisan' ka nara kisne diya tha?",
     "options": ["Lal Bahadur Shastri", "Jawaharlal Nehru", "Indira Gandhi", "Sardar Patel"], "answer": 0,
     "why": "Pradhan Mantri Lal Bahadur Shastri ne 1965 me 'Jai Jawan Jai Kisan' ka nara diya tha."},
    # --- Geography ---
    {"q": "Bharat ki sabse lambi nadi kaun si hai?",
     "options": ["Godavari", "Ganga", "Yamuna", "Brahmaputra"], "answer": 1,
     "why": "Ganga lagbhag 2,525 km lambi hai aur Bharat ki sabse lambi nadi mani jati hai."},
    {"q": "Kshetrafal me Bharat ka sabse bada rajya kaun sa hai?",
     "options": ["Madhya Pradesh", "Maharashtra", "Uttar Pradesh", "Rajasthan"], "answer": 3,
     "why": "Rajasthan kshetrafal (lagbhag 3.42 lakh varg km) me Bharat ka sabse bada rajya hai."},
    {"q": "Bharat me sabse adhik varsha wala sthan kaun sa hai?",
     "options": ["Agumbe", "Mahabaleshwar", "Mawsynram", "Darjeeling"], "answer": 2,
     "why": "Meghalaya ka Mawsynram vishwa me sabse adhik ausat varsha wale sthanon me sabse upar hai."},
    {"q": "Bharatiya manak samay (IST) GMT se kitna aage hai?",
     "options": ["4 ghante 30 minute", "6 ghante", "5 ghante", "5 ghante 30 minute"], "answer": 3,
     "why": "Bharatiya manak samay (IST) GMT se 5 ghante 30 minute aage hai."},
    {"q": "Bharat ka sabse bada marusthal (registan) kaun sa hai?",
     "options": ["Thar", "Kutch", "Ladakh", "Bundelkhand"], "answer": 0,
     "why": "Thar ka marusthal adhiktar Rajasthan me hai; ye Bharat ka sabse bada registan hai."},
    # --- Science ---
    {"q": "Ek vayask (adult) manushya ke sharir me kul kitni haddiyan hoti hain?",
     "options": ["196", "216", "206", "226"], "answer": 2,
     "why": "Ek vayask manushya ke sharir me kul 206 haddiyan hoti hain."},
    {"q": "Vayu-mandal (hawa) me sabse zyada matra me kaun si gas pai jati hai?",
     "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "answer": 1,
     "why": "Hawa me lagbhag 78 pratishat Nitrogen aur lagbhag 21 pratishat Oxygen hoti hai."},
    {"q": "Lohe (Iron) ka rasayanik sanket (symbol) kya hai?",
     "options": ["Ir", "Fe", "In", "Io"], "answer": 1,
     "why": "Lohe ka sanket Fe hai, jo uske Latin naam 'Ferrum' se liya gaya hai."},
    {"q": "Sabse halki gas kaun si hai?",
     "options": ["Helium", "Oxygen", "Nitrogen", "Hydrogen"], "answer": 3,
     "why": "Hydrogen sabse halka tatva hai, isiliye ye sabse halki gas bhi hai."},
    {"q": "Prakash (roshni) ki gati kitni mani jati hai?",
     "options": ["300 km prati second", "3,000 km prati second", "3 lakh km prati second", "30,000 km prati second"], "answer": 2,
     "why": "Prakash ki gati lagbhag 3 lakh kilometer prati second hoti hai; ye sabse tez gati mani jati hai."},
]


def quiz_question_text(session):
    """Chalu sawal ka formatted text (progress aur score ke saath)."""
    q = QUIZ_QUESTIONS[session["order"][session["pos"]]]
    total = len(session["order"])
    lines = [f"📝 *GK Quiz* • Sawal {session['pos'] + 1}/{total} "
             f"• Score: {session['score']}\n",
             f"❓ {q['q']}\n"]
    for i, opt in enumerate(q["options"]):
        lines.append(f"{OPTION_LETTERS[i]}) {opt}")
    lines.append("\n👉 Jawab A, B, C ya D (ya 1-4) me likhein. "
                 "Band karne ke liye /stopquiz.")
    return "\n".join(lines)


def parse_quiz_answer(text):
    """'A'..'D' ya '1'..'4' -> option index 0..3; samajh na aaye to None."""
    t = (text or "").strip().upper().rstrip(".)")
    if len(t) == 1 and t in OPTION_LETTERS:
        return OPTION_LETTERS.index(t)
    if len(t) == 1 and t in "1234":
        return int(t) - 1
    return None


def quiz_result_text(score, total):
    """Quiz ke ant ka result message, score ke hisaab se hosla-afzai."""
    if score >= total:
        grade = "🏆 Perfect score! Aap to GK champion nikle!"
    elif score >= 8:
        grade = "🌟 Bahut badhiya! Taiyari mazboot hai, aise hi lage raho."
    elif score >= 5:
        grade = "👍 Achha prayas! Thodi aur practice aur score aur upar jayega."
    elif score >= 3:
        grade = ("🙂 Shuruaat ho gayi hai. Roz ek quiz khelo, "
                 "score khud badhega.")
    else:
        grade = ("💪 Koi baat nahi, haar mat mano! Galat jawabon ki "
                 "vyakhya padho aur /quiz se phir try karo.")
    return ("🏁 *Quiz Khatm!*\n\n"
            f"🎯 *Aapka Score: {score}/{total}*\n\n"
            f"{grade}\n\n"
            "🔁 Phir se khelne ke liye /quiz dabayein ya 📝 Quiz button.")


def start_quiz(user_id, chat_id):
    """Naya 10-sawalon ka quiz session shuru karo aur pehla sawal bhejo."""
    awaiting_photo.discard(user_id)
    total = min(QUIZ_LENGTH, len(QUIZ_QUESTIONS))
    order = random.sample(range(len(QUIZ_QUESTIONS)), total)
    quiz_sessions[user_id] = {"order": order, "pos": 0, "score": 0}
    intro = ("📝 *GK Quiz shuru!*\n"
             f"{total} sawal aayenge — Bihar GK, India GK, Polity, "
             "History, Geography aur Science ka mix.\n"
             "Har sahi jawab par +1 number. Taiyaar? Pehla sawal 👇\n\n")
    return send(chat_id, intro + quiz_question_text(quiz_sessions[user_id]),
                parse_mode="Markdown")


def handle_quiz_answer(user_id, chat_id, text):
    """Quiz chal raha ho to ek jawab process karein.

    True return hota hai jab message quiz ne khaya ho (sahi jawab,
    galat jawab, ya samajh-na-aane par naram re-prompt) - session
    teeno haalat me surakshit aage badhta ya wahi rukta hai.
    """
    session = quiz_sessions.get(user_id)
    if session is None:
        return False
    picked = parse_quiz_answer(text)
    if picked is None:
        send(chat_id, "⚠️ Ye jawab samajh nahi aaya. Kripya A, B, C ya D "
                      "(ya 1, 2, 3, 4) me jawab dein.\n"
                      "Quiz band karne ke liye /stopquiz likhein.")
        send(chat_id, quiz_question_text(session), parse_mode="Markdown")
        return True
    q = QUIZ_QUESTIONS[session["order"][session["pos"]]]
    correct = q["answer"]
    if picked == correct:
        session["score"] += 1
        feedback = f"✅ *Sahi jawab!* +1\n💡 {q['why']}"
    else:
        feedback = ("❌ *Galat jawab!*\n"
                    f"✔️ Sahi jawab: *{OPTION_LETTERS[correct]}) "
                    f"{q['options'][correct]}*\n💡 {q['why']}")
    send(chat_id, feedback, parse_mode="Markdown")
    session["pos"] += 1
    if session["pos"] >= len(session["order"]):
        score, total = session["score"], len(session["order"])
        quiz_sessions.pop(user_id, None)
        send(chat_id, quiz_result_text(score, total), parse_mode="Markdown")
    else:
        send(chat_id, quiz_question_text(session), parse_mode="Markdown")
    return True


# ----------------------------------------------------------------------
# Guided flows: business setup, FAQ, lead, appointment
# ----------------------------------------------------------------------
def start_flow(user_id, flow):
    quiz_sessions.pop(user_id, None)  # guided flow shuru = quiz khatm
    states[user_id] = {"flow": flow, "step": 0, "data": {}}


FLOWS = {
    "setbusiness": {
        "prompts": [
            "🏢 Business ka naam likhein: / Type your business name:",
            "📂 Category likhein (jaise Coaching, Kirana, Salon, Property):",
            "ℹ️ Business ke baare me 2-3 line (services, khaasiyat):",
            "📞 Business ka phone number (customers ke liye):",
            "🕐 Timing likhein (jaise Mon-Sat 9am-8pm):",
        ],
        "fields": ["name", "category", "about", "phone", "timing"],
    },
    "addfaq": {
        "prompts": [
            "❓ FAQ ka sawal likhein (jaise: Fees kitni hai?):",
            "✅ Uska sahi jawab likhein:",
        ],
        "fields": ["question", "answer"],
    },
    "lead": {
        "prompts": [
            "📝 Apna naam likhein: / Your name:",
            "📞 Apna 10-digit mobile number likhein:",
            "💬 Aapko kya chahiye? Ek line me likhein:",
            "💰 Budget range likhein (jaise 5000-10000, ya 'pata nahi'):",
        ],
        "fields": ["name", "phone", "need", "budget"],
    },
    "book": {
        "prompts": [
            "📅 Apna naam likhein: / Your name:",
            "📞 Apna 10-digit mobile number likhein:",
            "🗓️ Date likhein DD-MM-YYYY format me (jaise 15-11-2026):",
            "🕐 Time likhein HH:MM format me (jaise 17:30):",
            "🎯 Kis kaam ke liye milna hai? Ek line me likhein:",
        ],
        "fields": ["name", "phone", "date", "time", "purpose"],
    },
}


def validate_field(flow, field, value):
    """Return (ok, cleaned_value_or_error_message)."""
    if field == "phone":
        if flow == "setbusiness":
            return True, value.strip()
        phone = normalize_phone_in(value)
        return (True, phone) if phone else (
            False, "❌ Ye number sahi nahi laga. 10-digit ka Indian mobile "
                   "number phir se likhein:")
    if field == "date":
        return (True, value.strip()) if valid_date_ddmmyyyy(value) else (
            False, "❌ Date format galat hai. DD-MM-YYYY me likhein (jaise 15-11-2026):")
    if field == "time":
        return (True, value.strip()) if valid_time_hhmm(value) else (
            False, "❌ Time format galat hai. HH:MM me likhein (jaise 17:30):")
    if len(value.strip()) < 2:
        return False, "❌ Thoda sa detail me likhein:"
    return True, value.strip()


def finish_flow(user_id, chat_id, flow, data):
    user = db.get_user(user_id) or {}
    code = user.get("business_code")
    if flow == "setbusiness":
        biz_code = business_code(data["name"], db.next_business_seq())
        db.create_business(biz_code, user_id, data["name"], data["category"],
                           data["about"], data["phone"], data["timing"])
        db.bind_business(user_id, biz_code)
        me = tg("getMe")
        username = me.get("result", {}).get("username", "YourBot")
        send(chat_id,
             f"🎉 Business jud gaya!\n\n🏢 {data['name']}\n🔑 Code: {biz_code}\n"
             f"🔗 Apna customer link customers ko dein:\n"
             f"https://t.me/{username}?start={biz_code}\n\n"
             "Is link se aane wale customers ko Support, Lead aur Appointment "
             "aapke business ke naam par milega.\n"
             "Ab /addfaq se apne sawalon ke jawab jodein taaki Support mode "
             "sahi jawab de sake.")
        notify_admins(f"🏢 New business registered: {data['name']} ({biz_code})")
    elif flow == "addfaq":
        biz = db.get_business_by_owner(user_id)
        if not biz:
            return send(chat_id, "⚠️ Pehle /setbusiness karein.")
        db.add_faq(biz["code"], data["question"], data["answer"])
        send(chat_id, "✅ FAQ jud gaya. Support mode ab is jawab ka use karega.\n"
                      "Aur jodna ho to /addfaq phir se dabayein.")
    elif flow == "lead":
        lead_id = db.add_lead(code, user_id, data["name"], data["phone"],
                              data["need"], data["budget"])
        send(chat_id, "✅ Dhanyavaad! Aapki enquiry mil gayi hai. "
                      "Team jald hi sampark karegi. / Thank you! Your enquiry "
                      "is saved, the team will contact you soon.")
        biz = db.get_business(code)
        note = (f"🆕 New Lead #{lead_id}\n👤 {data['name']}\n"
                f"📞 {data['phone']}\n💬 {data['need']}\n💰 {data['budget']}"
                + (f"\n🏢 For: {biz['name']}" if biz else ""))
        if biz:
            send(biz["owner_id"], note)
        notify_admins(note)
        push_make_webhook({"type": "lead", "id": lead_id, **data,
                           "business_code": code})
    elif flow == "book":
        appt_id = db.add_appointment(code, user_id, data["name"], data["phone"],
                                     data["date"], data["time"], data["purpose"])
        send(chat_id, "✅ Appointment request mil gayi hai. Confirmation ka "
                      "intezaar karein. / Request saved - please wait for "
                      "confirmation.")
        biz = db.get_business(code)
        kb = {"inline_keyboard": [[
            {"text": "✅ Confirm", "callback_data": f"appt_ok:{appt_id}"},
            {"text": "❌ Reject", "callback_data": f"appt_no:{appt_id}"}]]}
        note = (f"📅 New Appointment #{appt_id}\n👤 {data['name']}\n"
                f"📞 {data['phone']}\n🗓️ {data['date']} {data['time']}\n"
                f"🎯 {data['purpose']}"
                + (f"\n🏢 For: {biz['name']}" if biz else ""))
        if biz:
            send(biz["owner_id"], note, keyboard=kb)
        for admin_id in Config.ADMIN_IDS:
            if not biz or admin_id != biz["owner_id"]:
                send(admin_id, note, keyboard=kb)
        push_make_webhook({"type": "appointment", "id": appt_id, **data,
                           "business_code": code})


def handle_flow_text(user_id, chat_id, text):
    st = states[user_id]
    flow = FLOWS[st["flow"]]
    field = flow["fields"][st["step"]]
    ok, cleaned = validate_field(st["flow"], field, text)
    if not ok:
        return send(chat_id, cleaned)
    st["data"][field] = cleaned
    st["step"] += 1
    if st["step"] < len(flow["prompts"]):
        return send(chat_id, flow["prompts"][st["step"]])
    data, flow_name = st["data"], st["flow"]
    states.pop(user_id, None)
    finish_flow(user_id, chat_id, flow_name, data)


# ----------------------------------------------------------------------
# Admin dashboard (Telegram-native)
# ----------------------------------------------------------------------
def is_admin(user_id):
    return user_id in Config.ADMIN_IDS


def admin_text():
    return ("🛠️ *Admin Dashboard*\n\n"
            f"👥 Total users: {db.count_users()}\n"
            f"🏢 Businesses: {db.business_count()}\n"
            f"📝 Total leads: {db.count_leads()}\n"
            f"📅 Total appointments: {db.count_appointments()}\n"
            f"🤖 AI calls today: {db.total_ai_calls_today()}")


def admin_keyboard():
    return {"inline_keyboard": [
        [{"text": "📝 Leads", "callback_data": "adm:leads"},
         {"text": "📅 Appointments", "callback_data": "adm:appts"}],
        [{"text": "📥 Export Leads CSV", "callback_data": "adm:export"},
         {"text": "📣 Broadcast", "callback_data": "adm:broadcast"}],
        [{"text": "🔄 Refresh", "callback_data": "adm:home"}]]}


def leads_text(business_code=None, limit=10):
    leads = db.list_leads(business_code, limit)
    if not leads:
        return "Abhi koi lead nahi hai."
    lines = ["📝 *Latest Leads*\n"]
    for ld in leads:
        lines.append(f"#{ld['id']} {ld['name']} • {mask_phone(ld['phone'])}\n"
                     f"💬 {ld['need']} • 💰 {ld['budget']} • {ld['status']}")
    return "\n\n".join(lines)


def appts_text(business_code=None, limit=10):
    appts = db.list_appointments(business_code, limit)
    if not appts:
        return "Abhi koi appointment nahi hai."
    lines = ["📅 *Latest Appointments*\n"]
    for a in appts:
        lines.append(f"#{a['id']} {a['name']} • {a['date']} {a['time']}\n"
                     f"🎯 {a['purpose']} • Status: {a['status']}")
    return "\n\n".join(lines)


def export_leads_csv(chat_id):
    leads = db.list_leads(None, 1000)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "business_code", "name", "phone", "need",
                     "budget", "status", "created_at"])
    for ld in leads:
        writer.writerow([ld["id"], ld["business_code"], ld["name"], ld["phone"],
                         ld["need"], ld["budget"], ld["status"], ld["created_at"]])
    send_document(chat_id, "leads.csv", buf.getvalue().encode("utf-8-sig"),
                  caption="📥 Leads export (owner/admin only)")


# ----------------------------------------------------------------------
# Update handling
# ----------------------------------------------------------------------
def handle_callback(cb):
    data = cb.get("data", "")
    user_id = cb["from"]["id"]
    chat_id = cb["message"]["chat"]["id"]
    answer_callback(cb["id"])
    if data.startswith("appt_"):
        appt_id = int(data.split(":")[1])
        appt = db.get_appointment(appt_id)
        if not appt:
            return
        owner_ok = is_admin(user_id)
        biz = db.get_business(appt.get("business_code"))
        if biz and biz["owner_id"] == user_id:
            owner_ok = True
        if not owner_ok:
            return send(chat_id, "⛔ Ye action sirf owner/admin ke liye hai.")
        status = "confirmed" if data.startswith("appt_ok") else "rejected"
        db.set_appointment_status(appt_id, status)
        send(chat_id, f"Appointment #{appt_id} → {status}")
        send(appt["user_id"],
             f"📅 Aapki appointment #{appt_id} ({appt['date']} {appt['time']}) "
             f"ka status: *{status}*", parse_mode="Markdown")
    elif data.startswith("adm:") and is_admin(user_id):
        action = data.split(":")[1]
        if action == "leads":
            send(chat_id, leads_text(), parse_mode="Markdown")
        elif action == "appts":
            send(chat_id, appts_text(), parse_mode="Markdown")
        elif action == "export":
            export_leads_csv(chat_id)
        elif action == "broadcast":
            start_flow(user_id, "broadcast")
            send(chat_id, "📣 Broadcast message likhein — ye SAB users ko jayega:")
        elif action == "home":
            send(chat_id, admin_text(), keyboard=admin_keyboard(),
                 parse_mode="Markdown")


def _is_image_document(msg):
    doc = msg.get("document") or {}
    return (doc.get("mime_type") or "").startswith("image/")


def handle_message(msg):
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {})
    user_id = user.get("id")
    if not user_id:
        return
    # Photo Enhance: photos / image files arrive without text - handle
    # them before the text-only path below drops them.
    if msg.get("photo") or _is_image_document(msg):
        return handle_photo(msg)
    if msg.get("document") and user_id in awaiting_photo:
        awaiting_photo.discard(user_id)
        return send(chat_id, "⚠️ Ye file image nahi hai. Kripya photo "
                             "(JPG/PNG) bhejein — /enhance se phir try karein.")
    text = (msg.get("text") or "").strip()
    if not text:
        return
    lang = detect_language(text)
    db.upsert_user(user_id, user.get("username", ""), user.get("first_name", ""), lang)

    # Active guided flow consumes the message first
    if user_id in states:
        flow_name = states[user_id]["flow"]
        if text in ("/cancel", "cancel"):
            states.pop(user_id, None)
            return send(chat_id, "❌ Cancel ho gaya.", keyboard=main_keyboard())
        if flow_name == "broadcast":
            states.pop(user_id, None)
            return do_broadcast(chat_id, text)
        return handle_flow_text(user_id, chat_id, text)

    # GK Quiz chal raha ho to /stopquiz (ya /cancel) yahin ruk jaye
    if text in ("/stopquiz", "/cancel") and user_id in quiz_sessions:
        quiz_sessions.pop(user_id, None)
        return send(chat_id, "🛑 Quiz band ho gaya. Jab chaaho /quiz se "
                             "phir shuru kar sakte ho.",
                    keyboard=main_keyboard())

    # Commands
    if text == "/start" or text.startswith("/start "):
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[1].startswith("BIZ-"):
            biz = db.get_business(parts[1])
            if biz:
                db.bind_business(user_id, biz["code"])
                return send(chat_id,
                            f"🙏 {biz['name']} me aapka swagat hai!\n"
                            f"{biz.get('about','')}\n\n" + WELCOME,
                            keyboard=main_keyboard(), parse_mode="Markdown")
            return send(chat_id, "⚠️ Ye business link sahi nahi hai.",
                        keyboard=main_keyboard())
        return send(chat_id, WELCOME, keyboard=main_keyboard(), parse_mode="Markdown")
    if text in ("/help", "ℹ️ Help / मदद"):
        return send(chat_id, HELP, keyboard=main_keyboard(), parse_mode="Markdown")
    if text == "/cancel" and user_id in awaiting_photo:
        awaiting_photo.discard(user_id)
        return send(chat_id, "❌ Cancel ho gaya.", keyboard=main_keyboard())
    if text in ("/enhance", "📸 Photo Enhance"):
        awaiting_photo.add(user_id)
        return send(chat_id, ENHANCE_HOWTO, parse_mode="Markdown")
    if text == "/chat":
        modes[user_id] = "chat"
        engines.pop(user_id, None)  # default engine = Groq chain
        quiz_sessions.pop(user_id, None)  # mode change = quiz khatm
        return send(chat_id, "💬 *AI Chat* mode ON hai!\n"
                             "Koi bhi sawal poochho — padhai, kaam, ideas, "
                             "kuch bhi.\nBand karne ke liye /stopchat likhein, "
                             "ya koi aur mode button dabayein.\n"
                             "🧠 DeepSeek engine ke liye /deepseek likhein "
                             "(admin ne enable kiya ho tabhi chalega).",
                    parse_mode="Markdown")
    if text == "/deepseek":
        if not ai.deepseek_available():
            return send(chat_id, DEEPSEEK_OFF_MSG)
        modes[user_id] = "chat"
        engines[user_id] = "deepseek"
        quiz_sessions.pop(user_id, None)  # mode change = quiz khatm
        return send(chat_id, "🧠 *DeepSeek AI Chat* ON hai!\n"
                             "Ab aapke sawalon ke jawab DeepSeek engine "
                             "dega.\nBand karne ke liye /stopchat, aur "
                             "default engine par wapas ke liye /chat likhein.",
                    parse_mode="Markdown")
    if text == "/stopchat":
        modes.pop(user_id, None)
        engines.pop(user_id, None)
        return send(chat_id, "💬 AI Chat mode band ho gaya. Koi aur mode "
                             "button chunein, ya /chat se phir shuru karein.",
                    keyboard=main_keyboard())
    if text in ("/schedule", "🗓 Class Schedule"):
        return send(chat_id, schedule_text(), parse_mode="Markdown")
    if text in ("/quiz", "📝 Quiz"):
        return start_quiz(user_id, chat_id)
    if text == "/stopquiz":
        return send(chat_id, "Abhi koi quiz nahi chal raha. Shuru karne "
                             "ke liye /quiz dabayein.")
    if text == "📖 Book Search":
        return send(chat_id, "📖 Book dhundhne ke liye is format me likhein:\n"
                             "/book <book ka naam>\nJaise: /book Ramayan\n"
                             "(Sirf legal preview / info links milenge.)")
    if text == "/admin":
        if not is_admin(user_id):
            return send(chat_id, "⛔ Admin access nahi hai.")
        return send(chat_id, admin_text(), keyboard=admin_keyboard(),
                    parse_mode="Markdown")
    if text == "/setbusiness":
        if db.get_business_by_owner(user_id):
            return send(chat_id, "🏢 Aapka business pehle se juda hai. "
                                 "Details ke liye /mybusiness dekhein.")
        start_flow(user_id, "setbusiness")
        return send(chat_id, FLOWS["setbusiness"]["prompts"][0])
    if text == "/addfaq":
        if not db.get_business_by_owner(user_id):
            return send(chat_id, "⚠️ Pehle /setbusiness karein.")
        start_flow(user_id, "addfaq")
        return send(chat_id, FLOWS["addfaq"]["prompts"][0])
    if text == "/mybusiness":
        biz = db.get_business_by_owner(user_id)
        if not biz:
            return send(chat_id, "⚠️ Abhi koi business nahi juda. /setbusiness karein.")
        me = tg("getMe")
        username = me.get("result", {}).get("username", "YourBot")
        return send(chat_id,
                    f"🏢 *{biz['name']}*\n🔑 {biz['code']}\n"
                    f"🔗 https://t.me/{username}?start={biz['code']}\n"
                    f"❓ FAQs: {len(db.list_faqs(biz['code']))}\n"
                    f"📝 Leads: {db.count_leads(biz['code'])}\n"
                    f"📅 Appointments: {db.count_appointments(biz['code'])}",
                    parse_mode="Markdown")
    if text in ("/lead", "📝 Lead Form"):
        start_flow(user_id, "lead")
        return send(chat_id, FLOWS["lead"]["prompts"][0])
    if text.startswith("/book "):
        # Book Search: "/book <naam>". Bare "/book" stays the appointment
        # form below - existing behaviour is unchanged.
        query = text[len("/book "):].strip()
        if not query:
            return send(chat_id, "📖 Book ka naam likhein: /book <naam>\n"
                                 "Jaise: /book Ramayan")
        typing(chat_id)
        return send(chat_id, search_books(query), parse_mode="Markdown")
    if text in ("/book", "📅 Appointment"):
        start_flow(user_id, "book")
        return send(chat_id, FLOWS["book"]["prompts"][0])
    if text == "/language":
        kb = {"inline_keyboard": [[
            {"text": "🇮🇳 Hindi", "callback_data": "lang:hi"},
            {"text": "🇬🇧 English", "callback_data": "lang:en"}]]}
        return send(chat_id, "Apni bhasha chunein / Choose your language:", keyboard=kb)
    if text.startswith("lang:"):
        return  # handled as callback normally; ignore as text
    if text == "/mydata":
        u = db.get_user(user_id) or {}
        return send(chat_id,
                    f"🗂️ *Your saved data*\nPlan: {u.get('plan')}\n"
                    f"Business: {u.get('business_code') or '-'}\n"
                    f"AI calls today: {db.usage_today(user_id)}\n"
                    f"Leads by you: {len([l for l in db.list_leads(None, 1000) if l['user_id'] == user_id])}",
                    parse_mode="Markdown")
    if text == "/deletedata":
        db.clear_memory(user_id)
        return send(chat_id, "🗑️ Aapki chat memory delete ho gayi. "
                             "(Leads/appointments business record ke liye "
                             "owner se sampark karein.)")
    if text.startswith("/setplan") and is_admin(user_id):
        parts = text.split()
        if len(parts) == 3 and parts[2] in PLAN_LIMITS:
            db.set_plan(int(parts[1]), parts[2])
            return send(chat_id, f"✅ User {parts[1]} ka plan → {parts[2]}")
        return send(chat_id, "Usage: /setplan <user_id> <free|lite|standard|premium>")
    if text.startswith("/"):
        return send(chat_id, "⚠️ Ye command samajh nahi aaya. /help dekhein.",
                    keyboard=main_keyboard())

    # Mode selection buttons
    if text in LABEL_TO_MODE:
        awaiting_photo.discard(user_id)
        engines.pop(user_id, None)  # mode change = back to default engine
        quiz_sessions.pop(user_id, None)  # mode change = quiz khatm
        modes[user_id] = LABEL_TO_MODE[text]
        return send(chat_id, f"{text} mode ON hai. Apna sawal likhein 👇")

    # GK Quiz chal raha hai to har text jawab ke roop me process hoga
    if user_id in quiz_sessions:
        return handle_quiz_answer(user_id, chat_id, text)

    # /enhance ke baad photo ki jagah text aaya: wait khatm, note bhej do
    if user_id in awaiting_photo:
        awaiting_photo.discard(user_id)
        return send(chat_id, "📸 Photo ki jagah text mila — koi baat nahi. "
                             "Jab chaaho photo bhej do, main seedha enhance "
                             "karke bhej dunga. / Just send a photo any time "
                             "and I will enhance it.")

    # Plain message -> AI in current mode
    if len(text) > Config.MAX_INPUT_CHARS:
        return send(chat_id, f"⚠️ Message bahut lamba hai (max {Config.MAX_INPUT_CHARS} "
                             "characters). Thoda chhota karke bhejein.")
    typing(chat_id)
    mode = modes.get(user_id, "chat")
    # DeepSeek engine applies to AI Chat only; other modes stay default
    engine = engines.get(user_id) if mode == "chat" else None
    reply = ask_ai(user_id, mode, text, engine=engine)
    # Telegram 4096-char cap: split long AI replies into chunks
    for i in range(0, len(reply), 3900):
        send(chat_id, reply[i:i + 3900])


def do_broadcast(admin_chat_id, text):
    rows = db.conn.execute("SELECT user_id FROM users").fetchall()
    sent_ok = 0
    for row in rows:
        result = send(row["user_id"], f"📣 {text}")
        if result.get("ok"):
            sent_ok += 1
        time.sleep(0.05)  # stay under Telegram broadcast rate limits
    send(admin_chat_id, f"✅ Broadcast complete: {sent_ok}/{len(rows)} users")


def handle_lang_callback(cb):
    data = cb.get("data", "")
    if data.startswith("lang:"):
        user_id = cb["from"]["id"]
        db.set_language(user_id, data.split(":")[1])
        answer_callback(cb["id"], "✅ Language saved")
        send(cb["message"]["chat"]["id"], "✅ Bhasha save ho gayi / Language saved.")
        return True
    return False


# ----------------------------------------------------------------------
# Health endpoint (free hosts need a web port) + polling loop
# ----------------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(
            {"status": "ok", "service": "Vishal AI Assistant"}).encode())

    def log_message(self, *args):
        pass


def start_health_server():
    try:
        HTTPServer(("0.0.0.0", Config.PORT), HealthHandler).serve_forever()
    except Exception as exc:
        log.warning("Health server failed: %s", type(exc).__name__)


def main():
    problems = validate()
    if problems:
        for p in problems:
            log.error("CONFIG: %s", p)
        if not Config.BOT_TOKEN or (not Config.GROQ_API_KEY and not Config.GEMINI_API_KEY):
            raise SystemExit("Fix the config problems above, then restart. "
                             "See docs/02-SETUP.md")
    threading.Thread(target=start_health_server, daemon=True).start()
    log.info("Vishal AI Assistant started (polling). Admin ids: %s",
             sorted(Config.ADMIN_IDS))
    tg("deleteWebhook")  # polling and webhook cannot run together
    offset = 0
    while True:
        try:
            r = requests.get(f"{API}/getUpdates",
                             params={"offset": offset, "timeout": 50},
                             timeout=60)
            data = r.json()
            if not data.get("ok"):
                time.sleep(3)
                continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                try:
                    if "callback_query" in update:
                        cb = update["callback_query"]
                        if not handle_lang_callback(cb):
                            handle_callback(cb)
                    elif "message" in update:
                        handle_message(update["message"])
                except Exception:
                    log.exception("Failed to handle update %s", update.get("update_id"))
        except Exception as exc:
            log.warning("Polling error: %s - retrying in 5s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
