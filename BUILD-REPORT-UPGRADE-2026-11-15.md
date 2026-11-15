# Vishal AI — Upgrade Pass Build Report (2026-11-15)

Source: Code With Mosh "Build AI-Powered Apps" course study (pack:
~/workspace/your_files/code-with-mosh-ai-apps-pack-2026-11-15.md).
User approved with "Karo" on 2026-11-15 ~22:37 SGT.
**Status: LOCAL BUILD + TESTS DONE. NOT DEPLOYED. Live bot untouched.**

## Kya bana
1. **Auto model-routing + token hisaab** (`ai_upgrade.py`, `bot.py`)
   - Pehle routing sirf mode par thi (fast/quality). Ab fast mode ka
     mushkil/lamba sawal apne aap quality model par upgrade hota hai:
     ~500+ token input, ya compare/research/script/code/"detail me
     samjhao" jaisi demand.
   - `estimate_tokens()` — heuristic (koi nayi dependency nahi);
     Devanagari text ke liye alag hisaab. Exact nahi, routing/cost
     awareness ke liye kaafi.
   - Dono raaste (Telegram `ask_ai` + website `web_chat`) par laga.
2. **Clarifying-question rule** (`ai_upgrade.py`, `bot.py`, `prompts.py`)
   - Support mode me adhoora sawal ("Fees batao", "Admission chahiye",
     "Timing kya hai") — bina course/class naam ke — to bot AI ko call
     hi nahi karta; pehle EK chhota sawal poochhta hai.
   - Specific sawal (Class 10, SSC, subject naam) kabhi roka nahi jata.
   - Agar business ke documents (RAG) me jawab mil jaye to bhi clarify
     nahi hota — pehle jawab.
   - Fayda: galat/lamba jawab bhi bacha, AI call + usage bhi bacha.
   - Support system prompt me bhi yahi niyam backup ke roop me joda.
3. **Structured JSON output** (`ai_client.py` → `chat_json()`)
   - Model se sirf JSON mangwana + `extract_json()` se parse karna
     (code-fence / aas-paas ki prose bardasht; na mile to None, crash
     kabhi nahi). Aane wale tools ke liye taiyaar buniyaad.

## Tests
- Baseline (shuru me): 85/85 OK
- Naye tests: `tests/test_upgrade.py` — 23 tests (token, routing,
  clarify, JSON, bot integration)
- **Final: 108/108 OK** (`python -m unittest discover -s tests`)

## Deploy — abhi NAHI hua
- Live bot (@VishalSahayakBot) purane code par hi chal raha hai.
- Deploy = GitHub push → Render auto-deploy. User ke final OK par hi
  hoga, uske baad uska apna /start test + purana pending two-phone
  isolation test usi ke haath rahega.
- Badle gaye files: `ai_upgrade.py` (naya), `ai_client.py`,
  `prompts.py`, `bot.py`, `tests/test_upgrade.py` (naya), ye report.
