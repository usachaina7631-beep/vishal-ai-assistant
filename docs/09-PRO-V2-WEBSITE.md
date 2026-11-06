# Vishal AI — Pro v2 (Telegram + Website, ek backend)

## Screenshots wale plan se tulna (sach, bina hype)

| Baat | ChatGPT screenshot plan (Apps Script) | Hamara Pro v2 (pehle se live system ka upgrade) |
|---|---|---|
| Backend | Google Apps Script — execution ~6 min limit, roz ke quota, slow web-app, Telegram webhook ke liye kamzor | Python (Render par LIVE) — long-polling Telegram, koi 6-min limit nahi |
| AI dimaag | Sirf Gemini | Groq (fast, free) → Gemini automatic fallback — ek down ho to dusra chalta hai |
| Database | Apps Script me koi asli DB nahi (Sheets/Properties) | SQLite — per-business data isolation, 14 tests se verified |
| Business features | Plan me nahi ke barabar: sirf chat | Lead form, Appointment (owner Confirm/Reject), per-business FAQ Support, Admin dashboard, CSV export, Broadcast, plan limits |
| Website | Apps Script HTML (quota wali) | Isi Render service par nayi Website — `/` par chat UI, `/api/chat`, `/health` |
| Kharch | ₹0 (quota ke andar) | ₹0 (Render free + Groq/Gemini free tiers) |
| Live status | Kuch bana hi nahi | Telegram bot LIVE hai (health check OK, 6 Nov 2026); v2 website deploy ke baad usi URL par khulegi |

**Seedhi baat:** Apps Script wala raasta dobara banane se system kamzor hota.
Pro kaam ye tha — live bot ko website dekar diagram wala system
(Telegram + Website → ek backend → AI) sach me poora karna. Wahi kiya gaya hai.

## v2 me naya kya hai
1. **Website chat UI** (`web/index.html`) — mobile-first, Hindi/English, mode chips
   (Chat, Translator, YouTube, Business, Study, Research), Copy + Suno (TTS) button har jawab par.
2. **Voice input** — 🎤 button, browser ka free Speech Recognition (hi-IN).
   Koi paid API nahi. Chrome Android par behtar chalta hai.
3. **Ek hi backend** — website aur Telegram dono `ai_client.py` (Groq→Gemini) use karte hain.
   API key sirf Render env me; browser ko kabhi expose nahi hoti.
4. **Suraksha** — website par 30 jawab/din per visitor limit, 2000-char input cap,
   Support mode website par band (woh sirf business ke Telegram link se — data isolation bachane ke liye).

## Kya abhi bhi nahi hai (jhooth nahi bolenge)
- **Live internet research:** free tier me browsing nahi hai. Research mode sirf
  user ke diye text par kaam karta hai — website par ye saaf likha hai.
- **Image generation:** v2 me nahi hai. Gemini image ke liye alag setup/quota chahiye;
  bina test kiye feature list me nahi dala gaya.

## Deploy (owner ke taps — pehle wale bot jaisa hi)
v2 usi repo/service ka upgrade hai, naya account nahi chahiye:
1. Is `vishal-ai-pro` folder ka code `vishal-ai-assistant` GitHub repo me push (main branch).
2. Render service `vishal-ai-assistant` apne aap redeploy karegi (env vars wahi purane:
   TELEGRAM_BOT_TOKEN, ADMIN_IDS, GROQ_API_KEY — kuch naya dalna nahi hai).
3. Deploy ke baad check: `https://vishal-ai-assistant.onrender.com/health` → Pro v2 JSON,
   aur `/` kholne par website chat UI.
4. UptimeRobot monitor waisa hi rahega (ab `/` bhi 200 dega).

> Note: Push/deploy tabhi karna hai jab owner 'deploy karo' kahe.
> Tab tak ye pack sirf local tested hai — live bot v1 hi chal raha hai.

## Local test
```
python -m unittest discover tests -v
```
v1 ke 14 tests + v2 ke 4 web tests = 18 tests.
