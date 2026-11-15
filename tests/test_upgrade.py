"""
Tests for the 2026-11-15 course-derived upgrade pass:
token estimation + auto model routing, clarifying questions,
structured JSON output. Run: python -m unittest discover tests -v
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-not-real")
os.environ.setdefault("ADMIN_IDS", "111")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from ai_upgrade import (choose_quality, clarifying_question,
                        estimate_messages_tokens, estimate_tokens,
                        extract_json)


class TestTokenEstimate(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_longer_text_more_tokens(self):
        self.assertGreater(estimate_tokens("word " * 200),
                           estimate_tokens("word " * 10))

    def test_messages_adds_overhead(self):
        msgs = [{"role": "user", "content": "hello world"}]
        self.assertGreater(estimate_messages_tokens(msgs),
                           estimate_tokens("hello world"))


class TestRouting(unittest.TestCase):
    def test_simple_fast_stays_fast(self):
        self.assertFalse(choose_quality("translator", "hello ka matlab?", False))

    def test_base_quality_never_downgrades(self):
        self.assertTrue(choose_quality("chat", "hi", True))

    def test_complex_hint_upgrades(self):
        self.assertTrue(choose_quality(
            "study", "photosynthesis aur respiration me antar batao, compare karo", False))

    def test_detail_ask_upgrades(self):
        self.assertTrue(choose_quality(
            "translator", "is paragraph ko detail me samjhao", False))

    def test_long_input_upgrades(self):
        self.assertTrue(choose_quality("support", "word " * 600, False))


class TestClarify(unittest.TestCase):
    def test_vague_fees_asks_which_course(self):
        q = clarifying_question("support", "Fees batao")
        self.assertIsNotNone(q)
        self.assertIn("course", q.lower())

    def test_vague_admission_asks(self):
        self.assertIsNotNone(clarifying_question("support", "Admission chahiye"))

    def test_specific_fees_not_intercepted(self):
        self.assertIsNone(clarifying_question(
            "support", "Class 10 ki monthly fees kitni hai?"))

    def test_specific_subject_not_intercepted(self):
        self.assertIsNone(clarifying_question("support", "SSC batch timing kya hai?"))

    def test_non_support_mode_never_clarifies(self):
        self.assertIsNone(clarifying_question("chat", "Fees batao"))

    def test_long_question_not_intercepted(self):
        long_q = "Mujhe fees ke baare me poori jaankari chahiye aur saath me timing bhi batao please"
        self.assertIsNone(clarifying_question("support", long_q))

    def test_unrelated_short_not_intercepted(self):
        self.assertIsNone(clarifying_question("support", "Cricket score kya hai?"))


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 2}\n```'), {"a": 2})

    def test_prose_around_json(self):
        self.assertEqual(extract_json('Sure! {"a": 3} done'), {"a": 3})

    def test_garbage_returns_none(self):
        self.assertIsNone(extract_json("koi json nahi hai"))
        self.assertIsNone(extract_json(""))


class TestChatJson(unittest.TestCase):
    def _resp(self, content):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"choices": [{"message": {"content": content}}]}
        return r

    def test_chat_json_parses(self):
        from ai_client import AIClient
        session = MagicMock()
        session.post.return_value = self._resp('{"title": "Namaste", "tags": ["a"]}')
        client = AIClient(session=session)
        data, provider = client.chat_json(
            [{"role": "user", "content": "topic: diwali"}], ["title", "tags"])
        self.assertEqual(provider, "groq")
        self.assertEqual(data["title"], "Namaste")

    def test_chat_json_non_json_returns_none_with_provider(self):
        from ai_client import AIClient
        session = MagicMock()
        session.post.return_value = self._resp("plain text jawab")
        client = AIClient(session=session)
        data, provider = client.chat_json(
            [{"role": "user", "content": "hi"}], ["title"])
        self.assertIsNone(data)
        self.assertEqual(provider, "groq")


class TestBotClarifyIntegration(unittest.TestCase):
    """ask_ai: vague support question -> clarifying reply, no AI call,
    no usage counted. Specific question -> normal AI path."""

    UID = 777001

    @classmethod
    def setUpClass(cls):
        import bot
        from db import DB
        cls.bot = bot
        cls.db = DB(":memory:")
        bot.db = cls.db
        cls.db.create_business("BIZ-TEST-0001", cls.UID, "Test Coaching",
                               "Coaching", "General coaching",
                               "+911234567890", "9-8")
        cls.db.upsert_user(cls.UID, "cust", "Cust")
        cls.db.bind_business(cls.UID, "BIZ-TEST-0001")

    def test_vague_fees_clarifies_without_ai(self):
        before = self.db.usage_today(self.UID)
        with patch.object(self.bot.ai, "chat") as mock_chat:
            reply = self.bot.ask_ai(self.UID, "support", "Fees batao")
        mock_chat.assert_not_called()
        self.assertIn("course", reply.lower())
        self.assertEqual(self.db.usage_today(self.UID), before)

    def test_specific_question_uses_ai(self):
        with patch.object(self.bot.ai, "chat",
                          return_value=("Fees 1500 hai.", "groq")) as mock_chat:
            reply = self.bot.ask_ai(self.UID, "support",
                                    "Class 10 ki fees kitni hai?")
        mock_chat.assert_called_once()
        self.assertIn("1500", reply)


if __name__ == "__main__":
    unittest.main()
