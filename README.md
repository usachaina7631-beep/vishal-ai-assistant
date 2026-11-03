# Vishal AI Assistant — All-in-One AI Personal Agent (Telegram)

Hindi + English Telegram bot for small businesses, coaching institutes, shop
owners and content creators. Built on a **₹0 stack**: Telegram Bot API +
Groq/Gemini free tiers + SQLite + free hosting. One bot serves many client
businesses, each with its own support FAQs, leads and appointments, plus a
Telegram-native admin dashboard — fully manageable from an Android phone.

## Features
💬 AI Chat · 🔤 Hindi–English Translator · 🎬 YouTube Creator pack ·
💼 Business Assistant · 🛎️ Per-business Customer Support ·
📝 Lead Generation form · 📅 Appointment Requests with owner confirm ·
📚 Study Assistant · 🔎 Research Assistant (on pasted text) ·
🛠️ Admin Dashboard (stats, leads, CSV export, broadcast, plan control)

## Status (honest)
Code complete. **14/14 local unit tests pass** (validation, business data
isolation, AI fallback, limits, memory). Live Telegram integration is
**pending owner setup** — it needs YOUR BotFather token and free AI keys,
which must never be shared in chat. Follow `docs/02-SETUP.md`, then verify
every box in `docs/05-TESTING-CHECKLIST.md` before calling it live.

## Quick start (local / Termux)
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your token + keys in the file, never in chat
python bot.py
python -m unittest discover tests -v   # run the test suite
```

## Docs
1. `docs/01-ARCHITECTURE.md` — architecture + what ₹0 can and cannot do
2. `docs/02-SETUP.md` — step-by-step setup (Android only)
3. `docs/03-TELEGRAM-INTEGRATION.md` — BotFather, deep links, QR, limits
4. `docs/04-DEPLOYMENT.md` — Northflank / Render / Termux + data caveat
5. `docs/05-TESTING-CHECKLIST.md` — tested vs to-be-tested, in writing
6. `docs/06-BUSINESS-MODEL-PRICING.md` — pricing, cost math, revenue ladder
7. `docs/07-MAKE-INTEGRATION.md` — optional Make → Sheets (untested until you run it)
8. `docs/08-SALES-SCRIPTS.md` — Hinglish pitch, WhatsApp message, objections

## Files
`bot.py` main bot · `ai_client.py` Groq→Gemini fallback + model routing ·
`db.py` SQLite (per-user/per-business isolation) · `prompts.py` mode prompts ·
`config.py` env-only secrets · `utils.py` validation helpers · `tests/` suite
