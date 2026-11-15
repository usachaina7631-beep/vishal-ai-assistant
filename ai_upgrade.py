"""
Course-derived upgrades (Code With Mosh 'Build AI-Powered Apps' study,
2026-11-15 — technique adapted, nothing copied):

1. Token estimation + automatic model routing.
   The old routing was mode-only (fast vs quality). Now a nominally
   'fast' mode is upgraded to the quality model when the request is
   actually complex: long input, a comparison/research/script/code
   request, or an explicit 'explain in detail' ask. Simple short
   questions stay on the fast model, which protects the free quota.
   Token counts are a HEURISTIC estimate (no tokenizer dependency —
   this project is zero-dependency by design), good enough for
   routing and cost awareness, never billed as exact.

2. Clarifying questions (support mode).
   When a customer asks something too vague to answer correctly
   ('fees batao', 'admission', 'timing kya hai') with no course /
   class / service named, the bot asks ONE short clarifying question
   instead of guessing or dumping everything. Specific questions are
   never intercepted.

3. Structured JSON output.
   extract_json() parses a model reply into a dict, tolerating
   markdown code fences and surrounding prose, so future tools can
   ask the model for machine-readable answers safely.
"""
import json
import re

# ---------------------------------------------------------------- tokens
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for Latin text, ~2 for
    Devanagari-heavy text (each akshar costs more), at least 1."""
    if not text:
        return 0
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    per_char = 2.0 if devanagari > len(text) * 0.3 else 4.0
    by_chars = len(text) / per_char
    by_words = len(text.split()) * 1.3
    return max(1, int(max(by_chars, by_words)))


def estimate_messages_tokens(messages) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages) + 4 * len(messages)


# ---------------------------------------------------------------- routing
COMPLEX_HINTS = (
    "compare", "comparison", "difference between", "pros and cons",
    "research", "analysis", "analyse", "analyze", "strategy",
    "plan banao", "planning", "script", "code", "coding", "debug",
    "error aa", "kyun hota", "kyu hota", "kaise hota", "kaise kaam",
    "detail me", "detail mein", "vistar se", "in detail",
    "samjhao", "samjhaiye", "explain", "step by step", "step-by-step",
    "tulna", "antar batao", "फर्क", "तुलना", "विस्तार", "समझाओ",
    "क्यों होता", "कैसे होता",
)

LONG_INPUT_TOKENS = 500  # fast-mode input above this upgrades to quality


def choose_quality(mode: str, user_text: str, base_quality: bool) -> bool:
    """Decide fast vs quality model. base_quality (mode-based) always
    wins; a fast request is upgraded only on clear complexity signals."""
    if base_quality:
        return True
    if estimate_tokens(user_text or "") >= LONG_INPUT_TOKENS:
        return True
    low = (user_text or "").lower()
    return any(hint in low for hint in COMPLEX_HINTS)


# ---------------------------------------------------------------- clarify
_SPECIFIC_PATTERNS = (
    r"class\s*\d", r"\b\d{1,2}(st|nd|rd|th|vi|wi)\b",
    r"\bssc\b", r"\brailway\b", r"\bbpsc\b", r"\bneet\b", r"\bjee\b",
    r"\b10th\b", r"\b12th\b", r"\bba\b", r"\bbsc\b", r"\bma\b",
    r"maths?", r"english", r"hindi", r"science", r"physics", r"chemistry",
    r"biology", r"history", r"geography", r"polity", r"reasoning",
    r"spoken english", r"computer", r"typing", r"fitter", r"iti",
    r"कक्षा", r"गणित", r"अंग्रेज़ी", r"अंग्रेजी", r"विज्ञान",
)

_VAGUE_TOPICS = (
    (("fees", "fee", "price", "cost", "kitna paisa", "फीस", "शुल्क"),
     "🙏 Fees ki jaankari ke liye bas ek baat bata dein — kis course / class ki fees chahiye? "
     "(For fees, please tell me which course or class — e.g. Class 10, SSC, Spoken English.)"),
    (("admission", "admisssion", "enroll", "join karna", "प्रवेश", "दाखिला"),
     "🙏 Admission ke liye kis course / class me admission chahiye? Naam bata dein, phir main "
     "poori jaankari deta hoon. (Which course or class do you want admission in?)"),
    (("timing", "time kya", "batch", "kab hai", "समय", "बैच"),
     "🙏 Kis course / class ka batch timing chahiye? Naam bata dein. "
     "(Which course or class's batch timing do you need?)"),
    (("course", "courses", "कोर्स"),
     "🙏 Kaunsa course janna hai? Apna goal bata dein — jaise SSC, Railway, Board exam ya "
     "Spoken English — phir main sahi course ki jaankari deta hoon. "
     "(Which course are you interested in?)"),
)


def clarifying_question(mode: str, user_text: str):
    """Return ONE clarifying question for a too-vague support query,
    else None. Conservative by design: support mode only, short
    messages only, and never when a class/course/subject is named."""
    if mode != "support":
        return None
    text = (user_text or "").strip()
    if not text or len(text) > 60 or len(text.split()) > 8:
        return None
    low = text.lower()
    if any(re.search(p, low) for p in _SPECIFIC_PATTERNS):
        return None
    for words, question in _VAGUE_TOPICS:
        if any(w in low for w in words):
            return question
    return None


# ---------------------------------------------------------------- json out
def extract_json(text: str):
    """Parse a model reply into a dict/list. Tolerates ```json fences
    and prose around the JSON. Returns None when nothing parses —
    callers must fall back to the plain text, never crash."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except (ValueError, TypeError):
                continue
    return None


JSON_INSTRUCTION = (
    "Reply with ONLY a valid JSON object — no prose, no markdown — "
    "using exactly these keys: {keys}. Keep every value short."
)
