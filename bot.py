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
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from ai_client import AIClient
from config import Config, validate
from db import DB
from prompts import MODE_LABELS, build_system_prompt
from utils import (business_code, detect_language, mask_phone,
                   normalize_phone_in, telegram_safe, valid_date_ddmmyyyy,
                   valid_time_hhmm)

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
QUALITY_MODES = {"youtube", "research", "business"}

# In-memory conversation state: user_id -> {"flow": ..., "step": ..., "data": {}}
states = {}
# Current AI mode per user (defaults to 'chat')
modes = {}


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
        [{"text": "📅 Appointment"}, {"text": "ℹ️ Help / मदद"}],
    ]
    return {"keyboard": rows, "resize_keyboard": True}


LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

WELCOME = (
    "🙏 Namaste! Main hoon *Vishal AI Assistant*.\n\n"
    "Main aapki madad kar sakta hoon:\n"
    "💬 AI Chat • 🔤 Hindi-English Translator\n"
    "🎬 YouTube Content • 💼 Business Help\n"
    "🛎️ Customer Support • 📚 Study Help\n"
    "🔎 Research • 📝 Leads • 📅 Appointments\n\n"
    "Neeche button chunein ya seedha apna sawal likhein.\n"
    "Choose a button below or just type your question."
)
HELP = (
    "ℹ️ *Help / मदद*\n\n"
    "• Koi bhi mode button dabakar chunein, phir apna sawal likhein.\n"
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
def ask_ai(user_id, mode, user_text):
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
# Guided flows: business setup, FAQ, lead, appointment
# ----------------------------------------------------------------------
def start_flow(user_id, flow):
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


def handle_message(msg):
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {})
    user_id = user.get("id")
    text = (msg.get("text") or "").strip()
    if not user_id or not text:
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
        modes[user_id] = LABEL_TO_MODE[text]
        return send(chat_id, f"{text} mode ON hai. Apna sawal likhein 👇")

    # Plain message -> AI in current mode
    if len(text) > Config.MAX_INPUT_CHARS:
        return send(chat_id, f"⚠️ Message bahut lamba hai (max {Config.MAX_INPUT_CHARS} "
                             "characters). Thoda chhota karke bhejein.")
    typing(chat_id)
    mode = modes.get(user_id, "chat")
    reply = ask_ai(user_id, mode, text)
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
# ----------------------------------------------------------------------
# Pro v2 web layer: Website chat UI + /api/chat on the SAME backend.
# Telegram aur Website ek hi AI (Groq -> Gemini fallback) use karte hain.
# API key sirf server env me rehti hai - browser ko kabhi nahi jati.
# ----------------------------------------------------------------------
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
WEB_DAILY_LIMIT = int(os.environ.get("WEB_DAILY_LIMIT", "30"))
_web_usage = {}  # "YYYY-MM-DD:<ip>" -> count (in-memory, resets on restart)


def _web_rate_ok(ip):
    import datetime
    key = f"{datetime.date.today().isoformat()}:{ip}"
    if _web_usage.get(key, 0) >= WEB_DAILY_LIMIT:
        return False
    _web_usage[key] = _web_usage.get(key, 0) + 1
    return True


def web_chat(mode, message):
    """Website ke liye AI jawab. Support mode website par band hai -
    woh sirf business ke Telegram link se chalta hai."""
    if mode not in ("chat", "translator", "youtube", "business", "study", "research"):
        mode = "chat"
    messages = [{"role": "system", "content": build_system_prompt(mode)},
                {"role": "user", "content": message}]
    text, _provider = ai.chat(messages, quality=mode in QUALITY_MODES)
    return text


class HealthHandler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            return self._json(200, {"status": "ok",
                                    "service": "Vishal AI Assistant Pro v2.1"})
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception as exc:
                log.warning("Web UI serve failed: %s", type(exc).__name__)
                return self._json(200, {"status": "ok",
                                        "service": "Vishal AI Assistant Pro v2.1"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json(400, {"reply": "⚠️ Request samajh nahi aayi."})
        message = str(payload.get("message") or "").strip()
        mode = str(payload.get("mode") or "chat")
        if not message:
            return self._json(400, {"reply": "⚠️ Pehle apna sawal likhein."})
        if len(message) > Config.MAX_INPUT_CHARS:
            return self._json(400, {"reply": f"⚠️ Message bahut lamba hai (max {Config.MAX_INPUT_CHARS} characters)."})
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        if not _web_rate_ok(ip):
            return self._json(429, {"reply": "⏳ Aaj ki free website limit (30/din) khatm ho gayi hai. Telegram bot par bina is limit ke mode use karein: https://t.me/VishalSahayakBot"})
        return self._json(200, {"reply": web_chat(mode, message)})

    def log_message(self, *args):
        pass


def start_health_server():
    try:
        ThreadingHTTPServer(("0.0.0.0", Config.PORT), HealthHandler).serve_forever()
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
