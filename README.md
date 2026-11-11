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
📸 Photo Enhance (local, Pillow) · 📖 Book Search (legal links only) ·
🧠 DeepSeek AI Chat engine (optional, `/deepseek`) ·
🗓 Daily Class Schedule (`/schedule`, IST, live class marked) ·
📝 GK Quiz (`/quiz`, 10 sawal per session, original question bank, score) ·
📄 Business Documents / RAG (`/adddoc`, `/mydocs`, `/removedoc`) ·
🌐 Website chat UI (`/` + `/api/chat`) ·
🛠️ Admin Dashboard (stats, leads, CSV export, broadcast, plan control)

## Restored: Web UI + RAG (Pro v2/v3)
Two features from the Pro builds are part of this codebase again:
- **Website:** the bot's HTTP server serves a web chat page at `/`
  (files in `web/`) and answers `POST /api/chat` — same AI backend as
  Telegram, 30 free chats/day per visitor (`WEB_DAILY_LIMIT`).
  Support mode stays Telegram-only. `/health` still returns the plain
  `{"status": "ok", "service": "Vishal AI Assistant"}` JSON.
- **RAG documents:** business owners run `/adddoc` and attach a PDF,
  TXT file or long pasted text (fees lists, rules, course details).
  The text is chunked, embedded with the existing Gemini key and
  stored per business (`documents`/`doc_chunks` tables — works on
  local SQLite and Turso alike). Support answers then cite the
  business's own documents (`(Source: <name>)`); document text is
  always treated as data, never as instructions. Without a Gemini
  key, retrieval falls back to keyword matching. PDF reading needs
  the optional `pypdf` package (in `requirements.txt`).

## Status (honest)
Code complete. **44/44 local unit tests pass** (validation, business data
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

## Optional: enable DeepSeek
AI Chat runs on the Groq→Gemini chain by default. To ALSO offer the
DeepSeek engine (users switch with `/deepseek` inside the bot):

1. Create an API key — either the **official DeepSeek API**
   (platform.deepseek.com — paid, pay-as-you-go) or an **OpenRouter**
   key (openrouter.ai — free tier available).
2. In the **Render** dashboard → your service → **Environment**, set
   `DEEPSEEK_API_KEY` to that key. For the OpenRouter route, also set
   `DEEPSEEK_API_BASE=https://openrouter.ai/api/v1` and
   `DEEPSEEK_MODEL` to a DeepSeek free model id.
3. **Redeploy.** Users can then type `/deepseek` for DeepSeek answers
   and `/chat` to return to the default engine. Without the key,
   `/deepseek` simply replies that DeepSeek is not enabled yet — the
   rest of the bot is unaffected. Never paste the key anywhere else.

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
