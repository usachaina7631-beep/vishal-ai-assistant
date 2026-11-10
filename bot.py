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
* Group Quiz (/groupquiz): the same question bank as native Telegram
  quiz polls in a group - tap-to-answer, 30 seconds per question,
  automatic next question and a winners list at the end.
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
# Group Quiz sessions: chat_id -> session dict (see the Group Quiz
# section below - sawal, scores, timer generation, sab in-memory;
# bot restart par chalta quiz khatm ho jata hai, ye jaan-boojh kar hai)
group_sessions = {}
# poll_id -> chat_id: poll_answer updates ko sahi session tak le jaane
# ke liye (Telegram poll_answer me chat_id nahi aata, sirf poll_id)
group_poll_chats = {}
group_sessions_lock = threading.Lock()
_group_quiz_gen = 0  # har start/stop par badhta hai; purane timers invalid


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
    "• /groupquiz - Group me tap-wala quiz (30 sec/sawal)  • /stopgroupquiz — group quiz band\n"
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
# user letter ya number me jawab deta hai. Ye 1-to-1 practice ke liye
# text pattern hi sahi baithta hai; group me tap-wala asli quiz-poll
# anubhav alag /groupquiz feature deta hai (neeche Group Quiz section -
# uske 'poll_answer' updates ko polling loop ab handle karta hai).
# Admin: sawal jodne/badlane ke liye sirf QUIZ_QUESTIONS list edit
# karein. Har entry: "q" (sawal), "options" (thik 4), "answer" (sahi
# option ka index, 0 = A) aur "why" (ek line ki vyakhya). Saare sawal
# original aur pathyapustak-star ke pakke tathyon par aadharit hain.
# ----------------------------------------------------------------------
QUIZ_LENGTH = 10
OPTION_LETTERS = "ABCD"

QUIZ_QUESTIONS = [
    # --- Bihar GK ---
    {"q": "बिहार की राजधानी कौन सी है?",
     "options": ["गया", "पटना", "दरभंगा", "भागलपुर"], "answer": 1,
     "why": "पटना का प्राचीन नाम पाटलिपुत्र था; यह बिहार की राजधानी और सबसे बड़ा शहर है।"},
    {"q": "बिहार दिवस हर साल कब मनाया जाता है?",
     "options": ["15 अगस्त", "26 जनवरी", "22 मार्च", "2 अक्टूबर"], "answer": 2,
     "why": "22 मार्च 1912 को बिहार, बंगाल से अलग प्रांत बना था; इसी दिन बिहार दिवस मनाया जाता है।"},
    {"q": "प्राचीन नालंदा महाविहार के अवशेष आज किस ज़िले में स्थित हैं?",
     "options": ["पटना", "नालंदा", "गया", "वैशाली"], "answer": 1,
     "why": "नालंदा के अवशेष आज के नालंदा ज़िले में हैं; यह प्राचीन काल का विश्व-प्रसिद्ध शिक्षा केंद्र था।"},
    {"q": "महात्मा गांधी ने चंपारण सत्याग्रह किस वर्ष शुरू किया था?",
     "options": ["1915", "1919", "1920", "1917"], "answer": 3,
     "why": "1917 में गांधी जी ने चंपारण के नील किसानों के लिए अपना पहला सत्याग्रह किया था।"},
    {"q": "झारखंड राज्य बिहार से अलग होकर कब बना?",
     "options": ["15 नवंबर 2000", "1 नवंबर 2000", "15 अगस्त 2000", "26 जनवरी 2001"], "answer": 0,
     "why": "15 नवंबर 2000 को झारखंड, बिहार से अलग होकर भारत का नया राज्य बना।"},
    {"q": "गंगा नदी पर बना महात्मा गांधी सेतु किन दो शहरों को जोड़ता है?",
     "options": ["पटना - आरा", "पटना - हाजीपुर", "मुंगेर - भागलपुर", "पटना - दानापुर"], "answer": 1,
     "why": "महात्मा गांधी सेतु गंगा नदी पर बना है और पटना को हाजीपुर से जोड़ता है।"},
    {"q": "बोध गया में गौतम बुद्ध को किस वृक्ष के नीचे ज्ञान की प्राप्ति हुई थी?",
     "options": ["बरगद", "नीम", "आम", "पीपल (बोधि वृक्ष)"], "answer": 3,
     "why": "बोध गया में बोधि वृक्ष (पीपल) के नीचे बुद्ध को ज्ञान की प्राप्ति हुई थी।"},
    {"q": "क्षेत्रफल के हिसाब से बिहार का सबसे बड़ा ज़िला कौन सा है?",
     "options": ["पश्चिम चंपारण", "पटना", "गया", "दरभंगा"], "answer": 0,
     "why": "पश्चिम चंपारण क्षेत्रफल में बिहार का सबसे बड़ा ज़िला है।"},
    {"q": "'बिहार' नाम 'विहार' शब्द से बना माना जाता है। 'विहार' का अर्थ क्या है?",
     "options": ["किला", "बंदरगाह", "बौद्ध भिक्षुओं का मठ", "मेला"], "answer": 2,
     "why": "इस क्षेत्र में बौद्ध विहार (मठ) बहुत थे, इसी 'विहार' शब्द से 'बिहार' नाम पड़ा।"},
    {"q": "भारत के पहले राष्ट्रपति डॉ. राजेंद्र प्रसाद किस राज्य से थे?",
     "options": ["उत्तर प्रदेश", "मध्य प्रदेश", "बिहार", "बंगाल"], "answer": 2,
     "why": "डॉ. राजेंद्र प्रसाद का जन्म बिहार के सिवान ज़िले के ज़िरादेई गाँव में हुआ था।"},
    # --- India GK ---
    {"q": "भारत का राष्ट्रगान 'जन गण मन' किसने लिखा था?",
     "options": ["बंकिम चंद्र चटर्जी", "रबींद्रनाथ ठाकुर", "सरोजिनी नायडू", "सुभाष चंद्र बोस"], "answer": 1,
     "why": "'जन गण मन' गुरुदेव रबींद्रनाथ ठाकुर ने लिखा था; 1950 में यह राष्ट्रगान बना।"},
    {"q": "भारत का राष्ट्रीय गीत 'वंदे मातरम्' किस पुस्तक से लिया गया है?",
     "options": ["गीतांजलि", "गोदान", "मधुशाला", "आनंदमठ"], "answer": 3,
     "why": "'वंदे मातरम्' बंकिम चंद्र चटर्जी के उपन्यास 'आनंदमठ' (1882) से लिया गया है।"},
    {"q": "भारत का राष्ट्रीय पशु कौन सा है?",
     "options": ["एशियाई सिंह", "बंगाल टाइगर", "भारतीय हाथी", "गैंडा"], "answer": 1,
     "why": "बंगाल टाइगर (बाघ) भारत का राष्ट्रीय पशु है।"},
    {"q": "भारत का पहला कृत्रिम उपग्रह (satellite) कौन सा था?",
     "options": ["भास्कर", "रोहिणी", "आर्यभट्ट", "INSAT-1A"], "answer": 2,
     "why": "आर्यभट्ट 19 अप्रैल 1975 को भेजा गया भारत का पहला उपग्रह था।"},
    {"q": "भारत का सबसे लंबा बांध (dam) कौन सा है?",
     "options": ["हीराकुड बांध", "भाखड़ा बांध", "सरदार सरोवर", "टिहरी बांध"], "answer": 0,
     "why": "ओडिशा में महानदी नदी पर बना हीराकुड भारत का सबसे लंबा बांध है।"},
    # --- Polity ---
    {"q": "भारत का संविधान कब लागू हुआ था?",
     "options": ["15 अगस्त 1947", "26 नवंबर 1949", "26 जनवरी 1950", "2 अक्टूबर 1950"], "answer": 2,
     "why": "संविधान 26 नवंबर 1949 को अपनाया गया और 26 जनवरी 1950 को पूरे देश में लागू हुआ।"},
    {"q": "भारत में मतदान (vote) करने की न्यूनतम आयु कितनी है?",
     "options": ["21 वर्ष", "19 वर्ष", "16 वर्ष", "18 वर्ष"], "answer": 3,
     "why": "61वें संविधान संशोधन से मतदान की आयु 21 से घटाकर 18 कर दी गई।"},
    {"q": "राज्य सभा के सदस्य का कार्यकाल कितने वर्ष का होता है?",
     "options": ["5 वर्ष", "6 वर्ष", "4 वर्ष", "3 वर्ष"], "answer": 1,
     "why": "राज्य सभा एक स्थायी सभा है; इसके सदस्य 6 वर्ष के कार्यकाल के लिए चुने जाते हैं।"},
    {"q": "संविधान का अनुच्छेद 370 किस विषय से संबंधित था?",
     "options": ["चुनाव आयोग", "हिंदी भाषा", "जम्मू-कश्मीर की विशेष स्थिति", "वित्त आयोग"], "answer": 2,
     "why": "अनुच्छेद 370 जम्मू-कश्मीर को विशेष दर्जा देता था; अगस्त 2019 में उसे प्रभावहीन कर दिया गया।"},
    {"q": "वर्तमान में भारत में कुल कितने राज्य हैं?",
     "options": ["29", "27", "30", "28"], "answer": 3,
     "why": "वर्तमान में भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं।"},
    # --- History ---
    {"q": "जलियाँवाला बाग हत्याकांड किस वर्ष हुआ था?",
     "options": ["1919", "1917", "1920", "1918"], "answer": 0,
     "why": "13 अप्रैल 1919 को अमृतसर के जलियाँवाला बाग में जनरल डायर ने भीड़ पर गोली चलवाई थी।"},
    {"q": "दांडी यात्रा (नमक सत्याग्रह) किस वर्ष हुई थी?",
     "options": ["1920", "1942", "1930", "1932"], "answer": 2,
     "why": "गांधी जी ने 12 मार्च 1930 को साबरमती आश्रम से दांडी तक की यात्रा शुरू की थी।"},
    {"q": "पानीपत की पहली लड़ाई किस वर्ष हुई थी?",
     "options": ["1556", "1526", "1761", "1506"], "answer": 1,
     "why": "1526 में बाबर ने इब्राहिम लोदी को हराकर भारत में मुग़ल शासन की नींव रखी।"},
    {"q": "सम्राट अशोक किस वंश के शासक थे?",
     "options": ["गुप्त वंश", "कुषाण वंश", "मुग़ल वंश", "मौर्य वंश"], "answer": 3,
     "why": "सम्राट अशोक मौर्य वंश के सबसे प्रसिद्ध शासक थे; उनकी राजधानी पाटलिपुत्र थी।"},
    {"q": "'जय जवान जय किसान' का नारा किसने दिया था?",
     "options": ["लाल बहादुर शास्त्री", "जवाहरलाल नेहरू", "इंदिरा गांधी", "सरदार पटेल"], "answer": 0,
     "why": "प्रधानमंत्री लाल बहादुर शास्त्री ने 1965 में 'जय जवान जय किसान' का नारा दिया था।"},
    # --- Geography ---
    {"q": "भारत की सबसे लंबी नदी कौन सी है?",
     "options": ["गोदावरी", "गंगा", "यमुना", "ब्रह्मपुत्र"], "answer": 1,
     "why": "गंगा लगभग 2,525 किमी लंबी है और भारत की सबसे लंबी नदी मानी जाती है।"},
    {"q": "क्षेत्रफल में भारत का सबसे बड़ा राज्य कौन सा है?",
     "options": ["मध्य प्रदेश", "महाराष्ट्र", "उत्तर प्रदेश", "राजस्थान"], "answer": 3,
     "why": "राजस्थान क्षेत्रफल (लगभग 3.42 लाख वर्ग किमी) में भारत का सबसे बड़ा राज्य है।"},
    {"q": "भारत में सबसे अधिक वर्षा वाला स्थान कौन सा है?",
     "options": ["अगुम्बे", "महाबलेश्वर", "मासिनराम", "दार्जिलिंग"], "answer": 2,
     "why": "मेघालय का मासिनराम विश्व में सबसे अधिक औसत वर्षा वाले स्थानों में सबसे ऊपर है।"},
    {"q": "भारतीय मानक समय (IST) GMT से कितना आगे है?",
     "options": ["4 घंटे 30 मिनट", "6 घंटे", "5 घंटे", "5 घंटे 30 मिनट"], "answer": 3,
     "why": "भारतीय मानक समय (IST) GMT से 5 घंटे 30 मिनट आगे है।"},
    {"q": "भारत का सबसे बड़ा मरुस्थल (रेगिस्तान) कौन सा है?",
     "options": ["थार", "कच्छ", "लद्दाख", "बुंदेलखंड"], "answer": 0,
     "why": "थार का मरुस्थल अधिकतर राजस्थान में है; यह भारत का सबसे बड़ा रेगिस्तान है।"},
    # --- Science ---
    {"q": "एक वयस्क (adult) मनुष्य के शरीर में कुल कितनी हड्डियाँ होती हैं?",
     "options": ["196", "216", "206", "226"], "answer": 2,
     "why": "एक वयस्क मनुष्य के शरीर में कुल 206 हड्डियाँ होती हैं।"},
    {"q": "वायुमंडल (हवा) में सबसे ज़्यादा मात्रा में कौन सी गैस पाई जाती है?",
     "options": ["ऑक्सीजन", "नाइट्रोजन", "कार्बन डाइऑक्साइड", "हाइड्रोजन"], "answer": 1,
     "why": "हवा में लगभग 78 प्रतिशत नाइट्रोजन और लगभग 21 प्रतिशत ऑक्सीजन होती है।"},
    {"q": "लोहे (Iron) का रासायनिक संकेत (symbol) क्या है?",
     "options": ["Ir", "Fe", "In", "Io"], "answer": 1,
     "why": "लोहे का संकेत Fe है, जो उसके Latin नाम 'Ferrum' से लिया गया है।"},
    {"q": "सबसे हल्की गैस कौन सी है?",
     "options": ["हीलियम", "ऑक्सीजन", "नाइट्रोजन", "हाइड्रोजन"], "answer": 3,
     "why": "हाइड्रोजन सबसे हल्का तत्व है, इसीलिए यह सबसे हल्की गैस भी है।"},
    {"q": "प्रकाश (रोशनी) की गति कितनी मानी जाती है?",
     "options": ["300 किमी प्रति सेकंड", "3,000 किमी प्रति सेकंड", "3 लाख किमी प्रति सेकंड", "30,000 किमी प्रति सेकंड"], "answer": 2,
     "why": "प्रकाश की गति लगभग 3 लाख किलोमीटर प्रति सेकंड होती है; यह सबसे तेज़ गति मानी जाती है।"},
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
# Group Quiz (/groupquiz) - QUIJ-group style: bot group me ek-ek karke
# asli Telegram quiz-poll bhejta hai; har poll 30 second me apne aap
# band hota hai, agla sawal timer se apne aap aata hai, aur aakhir me
# score + samay ke saath winners ki list banti hai. Sawal wahi
# QUIZ_QUESTIONS bank se aate hain - kuch bhi haath se bharna nahi.
# Jawab 'poll_answer' updates se aate hain (handle_poll_answer).
# Sessions sirf memory me hain: bot restart = chalta quiz khatm.
# ----------------------------------------------------------------------
GROUP_QUIZ_SECONDS = 30        # har sawal ka open_period (seconds)
GROUP_QUIZ_DELAY = 33          # timer: open_period + 3s buffer
GROUP_QUIZ_DEFAULT = 10
GROUP_QUIZ_MIN = 3
GROUP_QUIZ_MAX = 35


def group_poll_payload(chat_id, q, pos, total):
    """sendPoll ke kwargs: ek sawal = ek quiz poll (pure function)."""
    return {
        "chat_id": chat_id,
        "question": f"Sawal {pos + 1}/{total}: {q['q']}",
        "options": list(q["options"]),
        "type": "quiz",
        "correct_option_id": q["answer"],
        "is_anonymous": False,
        "open_period": GROUP_QUIZ_SECONDS,
        "explanation": q["why"][:200],  # Telegram explanation cap ke andar
    }


def group_record_answer(session, user_id, name, option_ids, now=None):
    """Ek poll jawab ko session ke score me jodein (koi I/O nahi).

    Per user, per poll sirf PEHLA jawab ginti hota hai. Sahi jawab par
    correct +1 aur jawab dene me laga samay (seconds) judta hai; galat
    jawab na score deta hai na samay. Jawab record hua to True.
    """
    key = (session.get("poll_id"), user_id)
    if key in session["answered"]:
        return False
    session["answered"].add(key)
    entry = session["scores"].setdefault(
        user_id, {"name": name or "Player", "correct": 0, "time": 0.0})
    if name:
        entry["name"] = name
    if option_ids and option_ids[0] == session.get("correct"):
        answered_at = time.time() if now is None else now
        entry["correct"] += 1
        entry["time"] += max(0.0, answered_at
                             - session.get("posted_at", answered_at))
    return True


def group_leaderboard_text(scores, total):
    """Quiz ke ant ka winners message (pure function).

    Sort: zyada sahi jawab pehle; barabari ho to kul samay kam pehle.
    """
    if not scores:
        return ("🏁 Quiz khatm! Is baar kisi ne koi jawab nahi diya — "
                "kal phir try karte hain.")
    ranked = sorted(scores.values(),
                    key=lambda e: (-e["correct"], e["time"]))
    medals = ["🥇", "🥈", "🥉"]
    lines = ["🏁 *Quiz khatm!* 🏆 Winners:"]
    for i, entry in enumerate(ranked[:10]):
        rank = medals[i] if i < 3 else f"{i + 1}."
        lines.append(f"{rank} {entry['name']} – {entry['correct']}/{total} "
                     f"sahi ({entry['time']:.0f} sec)")
    return "\n".join(lines)


def _drop_group_session(chat_id):
    """Session + uske saare poll_id mappings hatayein.

    Sirf group_sessions_lock pakde hue bulayein.
    """
    group_sessions.pop(chat_id, None)
    stale = [pid for pid, cid in group_poll_chats.items() if cid == chat_id]
    for pid in stale:
        group_poll_chats.pop(pid, None)


def _post_group_question(chat_id, generation):
    """Session ka current sawal quiz-poll bana kar bhejein + timer lagayein."""
    with group_sessions_lock:
        session = group_sessions.get(chat_id)
        if (not session or not session.get("active")
                or session["generation"] != generation):
            return
        q = session["questions"][session["index"]]
        payload = group_poll_payload(chat_id, q, session["index"],
                                     len(session["questions"]))
    result = tg("sendPoll", **payload)
    poll = (result.get("result") or {}).get("poll") or {}
    with group_sessions_lock:
        session = group_sessions.get(chat_id)
        if (not session or not session.get("active")
                or session["generation"] != generation):
            return  # beech me quiz ruk/badla gaya - ye poll bekaar
        session["poll_id"] = poll.get("id")
        session["correct"] = q["answer"]
        session["posted_at"] = time.time()
        if session["poll_id"]:
            group_poll_chats[session["poll_id"]] = chat_id
    threading.Thread(target=_group_quiz_timer,
                     args=(chat_id, generation), daemon=True).start()


def _group_quiz_timer(chat_id, generation):
    """Sawal ke 33s baad: agla sawal bhejo, ya aakhri par winners.

    Generation mismatch = quiz ruk gaya ya naya shuru ho gaya; purana
    timer chup-chaap mar jata hai.
    """
    time.sleep(GROUP_QUIZ_DELAY)
    results_text = None
    with group_sessions_lock:
        session = group_sessions.get(chat_id)
        if (not session or not session.get("active")
                or session["generation"] != generation):
            return
        session["index"] += 1
        if session["index"] >= len(session["questions"]):
            results_text = group_leaderboard_text(
                session["scores"], len(session["questions"]))
            _drop_group_session(chat_id)
    if results_text is not None:
        send(chat_id, results_text, parse_mode="Markdown")
    else:
        _post_group_question(chat_id, generation)


def start_group_quiz(chat_id, starter_id, count=None):
    """/groupquiz: is chat me tap-wala group quiz shuru karein."""
    global _group_quiz_gen
    total = count if count else GROUP_QUIZ_DEFAULT
    total = max(GROUP_QUIZ_MIN,
                min(GROUP_QUIZ_MAX, total, len(QUIZ_QUESTIONS)))
    with group_sessions_lock:
        current = group_sessions.get(chat_id)
        if current and current.get("active"):
            busy = True
            generation = None
        else:
            busy = False
            _group_quiz_gen += 1
            generation = _group_quiz_gen
            group_sessions[chat_id] = {
                "questions": random.sample(QUIZ_QUESTIONS, total),
                "index": 0,
                "poll_id": None,
                "correct": None,
                "posted_at": 0.0,
                "scores": {},
                "answered": set(),
                "starter_id": starter_id,
                "generation": generation,
                "active": True,
            }
    if busy:
        return send(chat_id, "⏳ Is chat me pehle se ek quiz chal raha "
                             "hai. Use khatm hone do ya /stopgroupquiz bhejo.")
    send(chat_id,
         "🏆 *GK Group Quiz shuru!*\n"
         f"{total} sawal — har sawal ke liye sirf *30 second*. "
         "Option par tap karke jawab do.\n"
         "Sabse zyada sahi + sabse tez = winner! 🥇",
         parse_mode="Markdown")
    _post_group_question(chat_id, generation)


def stop_group_quiz(chat_id, user_id):
    """/stopgroupquiz: chalta quiz rokein - sirf starter ya admin."""
    global _group_quiz_gen
    with group_sessions_lock:
        session = group_sessions.get(chat_id)
        if not session or not session.get("active"):
            state = "none"
        elif user_id != session["starter_id"] and not is_admin(user_id):
            state = "denied"
        else:
            state = "stopped"
            _group_quiz_gen += 1  # pending timer invalid
            _drop_group_session(chat_id)
    if state == "none":
        return send(chat_id, "Abhi is chat me koi group quiz nahi chal "
                             "raha. Shuru karne ke liye /groupquiz dabayein.")
    if state == "denied":
        return send(chat_id, "⛔ Quiz sirf wahi rok sakta hai jisne use "
                             "shuru kiya tha, ya koi admin.")
    return send(chat_id, "⏹ Quiz rok diya gaya.")


def handle_poll_answer(pa):
    """Telegram 'poll_answer' update: group quiz ka tap-jawab score karein.

    Anjaan ya purane (expired) poll ke jawab chup-chaap ignore hote hain.
    """
    poll_id = pa.get("poll_id")
    user = pa.get("user") or {}
    user_id = user.get("id")
    if not poll_id or user_id is None:
        return
    with group_sessions_lock:
        chat_id = group_poll_chats.get(poll_id)
        if chat_id is None:
            return
        session = group_sessions.get(chat_id)
        if (not session or not session.get("active")
                or session.get("poll_id") != poll_id):
            return
        group_record_answer(session, user_id, user.get("first_name", ""),
                            pa.get("option_ids") or [])


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
    if text == "/groupquiz" or text.startswith("/groupquiz "):
        count = GROUP_QUIZ_DEFAULT
        parts = text.split()
        if len(parts) > 1:
            try:
                count = int(parts[1])
            except ValueError:
                count = GROUP_QUIZ_DEFAULT
        return start_group_quiz(chat_id, user_id, count)
    if text == "/stopgroupquiz":
        return stop_group_quiz(chat_id, user_id)
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
                             params={"offset": offset, "timeout": 50,
                                     "allowed_updates": json.dumps(
                                         ["message", "callback_query",
                                          "poll_answer"])},
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
                    elif "poll_answer" in update:
                        handle_poll_answer(update["poll_answer"])
                    elif "message" in update:
                        handle_message(update["message"])
                except Exception:
                    log.exception("Failed to handle update %s", update.get("update_id"))
        except Exception as exc:
            log.warning("Polling error: %s - retrying in 5s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
