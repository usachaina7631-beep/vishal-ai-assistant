"""
AI provider layer with automatic provider fallback and model routing.

Order: Groq (free tier, no card) -> Gemini (free tier, no card).
Either key alone is enough. Model routing:
  - 'fast'    -> high-volume chat/support/translator calls
  - 'quality' -> youtube/research/business drafting
On HTTP 429 (rate limit) or any provider error, the next provider is
tried once. If everything fails, a clear bilingual error is returned -
the bot never crashes and never leaks the API key or raw error body.

DeepSeek is an OPTIONAL extra engine, deliberately kept OUT of the
default chain above: it is used only via chat_deepseek() when the user
explicitly picks it (/deepseek in the bot) and DEEPSEEK_API_KEY is set.
"""
import logging
import requests

from config import Config

log = logging.getLogger("ai_client")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ERROR_MSG = (
    "⚠️ AI abhi busy hai / AI is busy right now. "
    "Kripya 1 minute baad phir try karein. / Please try again in a minute."
)


class AIClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    # ---- Groq (OpenAI-compatible) ----
    def _groq(self, messages, quality=False, timeout=60):
        model = Config.GROQ_MODEL_QUALITY if quality else Config.GROQ_MODEL_FAST
        resp = self.session.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": 0.6, "max_tokens": 1200},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ---- Gemini ----
    def _gemini(self, messages, timeout=60):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        body = {"contents": contents}
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        resp = self.session.post(
            GEMINI_URL.format(model=Config.GEMINI_MODEL),
            headers={"x-goog-api-key": Config.GEMINI_API_KEY,
                     "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # ---- DeepSeek (optional, OpenAI-compatible) ----
    def _deepseek(self, messages, timeout=60):
        base = (Config.DEEPSEEK_API_BASE or "https://api.deepseek.com").rstrip("/")
        resp = self.session.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": Config.DEEPSEEK_MODEL, "messages": messages,
                  "temperature": 0.6, "max_tokens": 1200},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def deepseek_available():
        """True only when the admin has set DEEPSEEK_API_KEY."""
        return bool(Config.DEEPSEEK_API_KEY)

    def chat_deepseek(self, messages):
        """DeepSeek-only chat: (text, 'deepseek') or (ERROR_MSG, None).
        Never raises and never falls back to Groq/Gemini - the user
        explicitly chose this engine."""
        if not Config.DEEPSEEK_API_KEY:
            return ERROR_MSG, None
        try:
            return self._deepseek(messages), "deepseek"
        except Exception as exc:  # log type only - never the key/body
            log.warning("AI provider deepseek failed: %s", type(exc).__name__)
            return ERROR_MSG, None

    # ---- Public API ----
    def chat(self, messages, quality=False):
        """Return (text, provider_name) or (ERROR_MSG, None)."""
        attempts = []
        if Config.GROQ_API_KEY:
            attempts.append(("groq", lambda: self._groq(messages, quality)))
        if Config.GEMINI_API_KEY:
            attempts.append(("gemini", lambda: self._gemini(messages)))
        for name, fn in attempts:
            try:
                return fn(), name
            except Exception as exc:  # log type only - never the key/body
                log.warning("AI provider %s failed: %s", name, type(exc).__name__)
        return ERROR_MSG, None
