"""
Local tests - run with:  python -m unittest discover tests -v
No live token or API key is needed; network calls are mocked.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-not-real")
os.environ.setdefault("ADMIN_IDS", "111")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from utils import (business_code, detect_language, mask_phone,
                   normalize_phone_in, valid_date_ddmmyyyy, valid_time_hhmm)
from db import DB
from prompts import build_system_prompt


class TestUtils(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("मेरा नाम विशाल है"), "hi")
        self.assertEqual(detect_language("My name is Vishal"), "en")
        self.assertEqual(detect_language(""), "en")

    def test_phone_validation(self):
        self.assertEqual(normalize_phone_in("9876543210"), "+919876543210")
        self.assertEqual(normalize_phone_in("+91 98765 43210".replace(" ", "")), "+919876543210")
        self.assertEqual(normalize_phone_in("919876543210"), "+919876543210")
        self.assertIsNone(normalize_phone_in("1234567890"))   # wrong starting digit
        self.assertIsNone(normalize_phone_in("98765"))        # too short
        self.assertIsNone(normalize_phone_in("abcdefghij"))

    def test_mask_phone(self):
        self.assertEqual(mask_phone("+919876543210"), "+91*****3210")

    def test_date_time(self):
        self.assertTrue(valid_date_ddmmyyyy("15-11-2026"))
        self.assertFalse(valid_date_ddmmyyyy("31-02-2026"))
        self.assertFalse(valid_date_ddmmyyyy("2026-11-15"))
        self.assertTrue(valid_time_hhmm("17:30"))
        self.assertFalse(valid_time_hhmm("25:00"))
        self.assertFalse(valid_time_hhmm("5pm"))

    def test_business_code(self):
        self.assertEqual(business_code("Sharma Coaching", 7), "BIZ-SHARMA-0007")
        self.assertEqual(business_code("", 1), "BIZ-BIZ-0001")


class TestDBIsolation(unittest.TestCase):
    def setUp(self):
        self.db = DB(":memory:")

    def test_business_isolation(self):
        self.db.create_business("BIZ-A-0001", 1, "Shop A")
        self.db.create_business("BIZ-B-0002", 2, "Shop B")
        self.db.add_faq("BIZ-A-0001", "Fees?", "1000")
        self.db.add_faq("BIZ-B-0002", "Fees?", "2000")
        self.assertIn("1000", self.db.faq_context("BIZ-A-0001"))
        self.assertNotIn("2000", self.db.faq_context("BIZ-A-0001"))
        self.assertIn("2000", self.db.faq_context("BIZ-B-0002"))

    def test_leads_and_appointments_scoped(self):
        self.db.add_lead("BIZ-A-0001", 9, "Ram", "+919876543210", "Course", "5000")
        self.db.add_lead("BIZ-B-0002", 9, "Shyam", "+919876543211", "Shop", "1000")
        self.assertEqual(self.db.count_leads("BIZ-A-0001"), 1)
        self.assertEqual(self.db.count_leads(), 2)
        appt = self.db.add_appointment("BIZ-A-0001", 9, "Ram", "+919876543210",
                                       "15-11-2026", "17:30", "Demo")
        self.db.set_appointment_status(appt, "confirmed")
        self.assertEqual(self.db.get_appointment(appt)["status"], "confirmed")

    def test_usage_limit_counter(self):
        self.assertEqual(self.db.usage_today(42), 0)
        self.db.bump_usage(42)
        self.db.bump_usage(42)
        self.assertEqual(self.db.usage_today(42), 2)
        self.assertEqual(self.db.usage_today(43), 0)

    def test_memory_per_user_and_mode(self):
        self.db.add_memory(1, "chat", "user", "hello")
        self.db.add_memory(2, "chat", "user", "other user")
        mem = self.db.get_memory(1, "chat")
        self.assertEqual(len(mem), 1)
        self.assertEqual(mem[0]["content"], "hello")
        self.db.clear_memory(1)
        self.assertEqual(self.db.get_memory(1, "chat"), [])


class TestPrompts(unittest.TestCase):
    def test_support_prompt_includes_only_own_faq(self):
        biz = {"name": "Sharma Coaching", "category": "Coaching",
               "about": "SSC prep", "phone": "+911234567890", "timing": "9-8"}
        p = build_system_prompt("support", biz, "Q: Fees?\nA: 1000")
        self.assertIn("Sharma Coaching", p)
        self.assertIn("1000", p)
        self.assertIn("ONLY", p)

    def test_all_modes_exist(self):
        for mode in ("chat", "translator", "youtube", "business",
                     "support", "study", "research"):
            self.assertTrue(build_system_prompt(mode))


class TestAIClientFallback(unittest.TestCase):
    def _resp(self, status, payload):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload
        if status >= 400:
            r.raise_for_status.side_effect = Exception("http error")
        else:
            r.raise_for_status.return_value = None
        return r

    def test_groq_success(self):
        from ai_client import AIClient
        session = MagicMock()
        session.post.return_value = self._resp(
            200, {"choices": [{"message": {"content": "Namaste!"}}]})
        client = AIClient(session=session)
        text, provider = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "Namaste!")
        self.assertEqual(provider, "groq")

    def test_fallback_never_raises(self):
        from ai_client import AIClient, ERROR_MSG
        session = MagicMock()
        session.post.return_value = self._resp(429, {})
        with patch("ai_client.Config") as cfg:
            cfg.GROQ_API_KEY = "x"
            cfg.GEMINI_API_KEY = "y"
            cfg.GROQ_MODEL_FAST = "m"
            cfg.GEMINI_MODEL = "g"
            client = AIClient(session=session)
            text, provider = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(text, ERROR_MSG)
        self.assertIsNone(provider)


class TestFlowValidation(unittest.TestCase):
    def test_validate_field(self):
        import bot
        ok, val = bot.validate_field("lead", "phone", "9876543210")
        self.assertTrue(ok)
        self.assertEqual(val, "+919876543210")
        ok, _ = bot.validate_field("lead", "phone", "123")
        self.assertFalse(ok)
        ok, _ = bot.validate_field("book", "date", "15-11-2026")
        self.assertTrue(ok)
        ok, _ = bot.validate_field("book", "time", "9 baje")
        self.assertFalse(ok)


class TestPhotoEnhance(unittest.TestCase):
    def _jpeg_bytes(self, size=(40, 30)):
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.new("RGB", size, (120, 60, 30)).save(buf, "JPEG")
        return buf.getvalue()

    def test_enhance_doubles_size_and_returns_jpeg(self):
        import io as _io
        import bot
        from PIL import Image
        if not bot.PIL_AVAILABLE:
            self.skipTest("Pillow not installed")
        out = bot.enhance_image_bytes(self._jpeg_bytes())
        self.assertIsNotNone(out)
        img = Image.open(_io.BytesIO(out))
        self.assertEqual(img.size, (80, 60))  # 2x upscale
        self.assertEqual(img.format, "JPEG")

    def test_enhance_rejects_non_image(self):
        import bot
        if not bot.PIL_AVAILABLE:
            self.skipTest("Pillow not installed")
        self.assertIsNone(bot.enhance_image_bytes(b"this is not an image"))
        self.assertIsNone(bot.enhance_image_bytes(b""))


class TestBookSearch(unittest.TestCase):
    def _resp(self, payload):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = payload
        return r

    def test_search_formats_legal_links(self):
        import bot
        payload = {"items": [{"volumeInfo": {
            "title": "Ramayan", "authors": ["Valmiki"],
            "description": "The great epic.",
            "infoLink": "https://books.google.example/ramayan"}}]}
        with patch("bot.requests.get", return_value=self._resp(payload)):
            out = bot.search_books("Ramayan")
        self.assertIn("Ramayan", out)
        self.assertIn("Valmiki", out)
        self.assertIn("https://books.google.example/ramayan", out)
        self.assertIn("gutenberg.org", out)  # public-domain link always shown

    def test_search_no_results(self):
        import bot
        with patch("bot.requests.get", return_value=self._resp({"totalItems": 0})):
            out = bot.search_books("xyz-no-such-book")
        self.assertIn("nahi mili", out)
        self.assertIn("gutenberg.org", out)

    def test_search_network_failure_is_graceful(self):
        import bot
        with patch("bot.requests.get", side_effect=Exception("down")):
            out = bot.search_books("Ramayan")
        self.assertIn("kaam nahi kar raha", out)


class TestNewFeatureCommands(unittest.TestCase):
    UID = 987001

    def tearDown(self):
        import bot
        bot.states.pop(self.UID, None)
        bot.modes.pop(self.UID, None)
        bot.awaiting_photo.discard(self.UID)

    def _msg(self, text=None, **extra):
        m = {"chat": {"id": 555}, "from": {"id": self.UID,
                                           "username": "tester",
                                           "first_name": "Test"}}
        if text is not None:
            m["text"] = text
        m.update(extra)
        return m

    def test_chat_command_uses_quality_mode(self):
        import bot
        self.assertIn("chat", bot.QUALITY_MODES)
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/chat"))
        self.assertEqual(bot.modes.get(self.UID), "chat")
        self.assertIn("AI Chat", mock_send.call_args[0][1])
        with patch("bot.send"):
            bot.handle_message(self._msg("/stopchat"))
        self.assertNotIn(self.UID, bot.modes)

    def test_book_with_name_searches_bare_book_still_appointment(self):
        import bot
        with patch("bot.send") as mock_send, \
                patch("bot.typing"), \
                patch("bot.search_books", return_value="SEARCH-RESULT") as mock_search:
            bot.handle_message(self._msg("/book Ramayan"))
        mock_search.assert_called_once_with("Ramayan")
        self.assertEqual(mock_send.call_args[0][1], "SEARCH-RESULT")
        self.assertNotIn(self.UID, bot.states)
        with patch("bot.send"):
            bot.handle_message(self._msg("/book"))
        self.assertEqual(bot.states[self.UID]["flow"], "book")  # appointment

    def test_enhance_command_then_photo(self):
        import io as _io
        import bot
        from PIL import Image
        if not bot.PIL_AVAILABLE:
            self.skipTest("Pillow not installed")
        with patch("bot.send"):
            bot.handle_message(self._msg("/enhance"))
        self.assertIn(self.UID, bot.awaiting_photo)
        buf = _io.BytesIO()
        Image.new("RGB", (20, 20), (10, 20, 30)).save(buf, "JPEG")
        photo_msg = self._msg(photo=[{"file_id": "f1", "file_size": 100},
                                     {"file_id": "f2", "file_size": 200}])
        with patch("bot.send"), patch("bot.typing"), \
                patch("bot.download_file", return_value=buf.getvalue()) as dl, \
                patch("bot.send_photo", return_value={"ok": True}) as sp:
            bot.handle_message(photo_msg)
        dl.assert_called_once_with("f2")  # largest size is downloaded
        sp.assert_called_once()
        self.assertNotIn(self.UID, bot.awaiting_photo)


class TestDeepSeekClient(unittest.TestCase):
    def _resp(self, status, payload):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload
        if status >= 400:
            r.raise_for_status.side_effect = Exception("http error")
        else:
            r.raise_for_status.return_value = None
        return r

    def test_deepseek_configured_calls_official_api(self):
        from ai_client import AIClient
        from config import Config
        session = MagicMock()
        session.post.return_value = self._resp(
            200, {"choices": [{"message": {"content": "DeepSeek jawab"}}]})
        with patch.object(Config, "DEEPSEEK_API_KEY", "test-ds-key"), \
                patch.object(Config, "DEEPSEEK_API_BASE", "https://api.deepseek.com"), \
                patch.object(Config, "DEEPSEEK_MODEL", "deepseek-chat"):
            client = AIClient(session=session)
            self.assertTrue(client.deepseek_available())
            text, provider = client.chat_deepseek(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(text, "DeepSeek jawab")
        self.assertEqual(provider, "deepseek")
        url = session.post.call_args[0][0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        body = session.post.call_args[1]["json"]
        self.assertEqual(body["model"], "deepseek-chat")

    def test_deepseek_not_configured_no_network_call(self):
        from ai_client import AIClient, ERROR_MSG
        from config import Config
        session = MagicMock()
        with patch.object(Config, "DEEPSEEK_API_KEY", ""):
            client = AIClient(session=session)
            self.assertFalse(client.deepseek_available())
            text, provider = client.chat_deepseek(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(text, ERROR_MSG)
        self.assertIsNone(provider)
        session.post.assert_not_called()

    def test_deepseek_failure_is_graceful(self):
        from ai_client import AIClient, ERROR_MSG
        from config import Config
        session = MagicMock()
        session.post.return_value = self._resp(429, {})
        with patch.object(Config, "DEEPSEEK_API_KEY", "test-ds-key"):
            client = AIClient(session=session)
            text, provider = client.chat_deepseek(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(text, ERROR_MSG)
        self.assertIsNone(provider)


class TestDeepSeekCommands(unittest.TestCase):
    UID = 987777

    def tearDown(self):
        import bot
        bot.states.pop(self.UID, None)
        bot.modes.pop(self.UID, None)
        bot.engines.pop(self.UID, None)
        bot.awaiting_photo.discard(self.UID)

    def _msg(self, text):
        return {"chat": {"id": 556}, "from": {"id": self.UID,
                                              "username": "dstester",
                                              "first_name": "DS"},
                "text": text}

    def test_deepseek_not_configured_graceful_message(self):
        import bot
        with patch.object(bot.Config, "DEEPSEEK_API_KEY", ""), \
                patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/deepseek"))  # must not raise
        sent_text = mock_send.call_args[0][1]
        self.assertIn("DeepSeek", sent_text)
        self.assertIn("enable nahi hai", sent_text)
        self.assertNotIn(self.UID, bot.engines)
        self.assertNotIn(self.UID, bot.modes)

    def test_deepseek_configured_routes_chat_to_deepseek(self):
        import bot
        fake_ai = MagicMock()
        fake_ai.deepseek_available.return_value = True
        fake_ai.chat_deepseek.return_value = ("DeepSeek jawab", "deepseek")
        fake_ai.chat.return_value = ("Groq jawab", "groq")
        with patch.object(bot, "ai", fake_ai), patch("bot.send") as mock_send, \
                patch("bot.typing"):
            bot.handle_message(self._msg("/deepseek"))
            self.assertEqual(bot.modes.get(self.UID), "chat")
            self.assertEqual(bot.engines.get(self.UID), "deepseek")
            self.assertIn("DeepSeek", mock_send.call_args[0][1])
            bot.handle_message(self._msg("ek sawal"))
            fake_ai.chat_deepseek.assert_called_once()
            fake_ai.chat.assert_not_called()
            self.assertIn("DeepSeek jawab",
                          [c[0][1] for c in mock_send.call_args_list])
            # /chat switches back to the default Groq engine
            bot.handle_message(self._msg("/chat"))
            self.assertNotIn(self.UID, bot.engines)
            bot.handle_message(self._msg("dusra sawal"))
            fake_ai.chat.assert_called_once()
            self.assertEqual(fake_ai.chat_deepseek.call_count, 1)
            # /stopchat clears the engine too
            bot.handle_message(self._msg("/deepseek"))
            self.assertEqual(bot.engines.get(self.UID), "deepseek")
            bot.handle_message(self._msg("/stopchat"))
            self.assertNotIn(self.UID, bot.engines)
            self.assertNotIn(self.UID, bot.modes)


class TestClassSchedule(unittest.TestCase):
    UID = 987555

    def tearDown(self):
        import bot
        bot.states.pop(self.UID, None)
        bot.modes.pop(self.UID, None)

    def _ist(self, hh, mm=0):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime(2026, 11, 9, hh, mm, tzinfo=ZoneInfo("Asia/Kolkata"))

    def _msg(self, text):
        return {"chat": {"id": 557}, "from": {"id": self.UID,
                                              "username": "sched",
                                              "first_name": "S"},
                "text": text}

    def test_schedule_message_contains_all_slots(self):
        import bot
        out = bot.schedule_text(self._ist(10, 0))
        self.assertEqual(len(bot.CLASS_SCHEDULE), 16)
        for _start, subject in bot.CLASS_SCHEDULE:
            self.assertIn(subject, out)
        for label in ("7:00 AM", "11:00 AM", "12:00 PM", "1:00 PM",
                      "7:30 PM", "9:30 PM", "10:30 PM"):
            self.assertIn(label, out)

    def test_current_next_morning(self):
        import bot
        current, nxt = bot.current_and_next(self._ist(8, 30))
        self.assertEqual(current[1], "Science")   # 8:00 AM class chal rahi
        self.assertEqual(nxt[1], "Mix")           # 9:00 AM agli class

    def test_current_next_exact_start(self):
        import bot
        current, nxt = bot.current_and_next(self._ist(19, 30))
        self.assertEqual(current[1], "100 Question")
        self.assertEqual(nxt[1], "Polity")

    def test_current_next_late_night_wraps(self):
        import bot
        current, nxt = bot.current_and_next(self._ist(23, 0))
        self.assertEqual(current[1], "Mix")            # 10:30 PM wali class
        self.assertEqual(current[0].hour, 22)
        self.assertEqual(nxt[1], "Current Affairs")    # kal subah 7:00 AM
        self.assertEqual(nxt[0].hour, 7)

    def test_current_next_before_first_class_wraps(self):
        import bot
        current, nxt = bot.current_and_next(self._ist(6, 0))
        self.assertEqual(current[1], "Mix")            # pichhle din ki aakhri
        self.assertEqual(nxt[1], "Current Affairs")

    def test_schedule_message_marks_current_and_next(self):
        import bot
        out = bot.schedule_text(self._ist(8, 30))
        science = [ln for ln in out.splitlines() if "Science" in ln][0]
        mix9 = [ln for ln in out.splitlines() if ln.startswith("9:00 AM")][0]
        self.assertIn("abhi chal rahi hai", science)
        self.assertIn("agli class", mix9)

    def test_schedule_command_and_button(self):
        import bot
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/schedule"))
        sent = mock_send.call_args[0][1]
        self.assertIn("Class Time-Table", sent)
        self.assertIn("Current Affairs", sent)
        self.assertIn("Polity", sent)
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("🗓 Class Schedule"))
        self.assertIn("Current Affairs", mock_send.call_args[0][1])


class TestGKQuiz(unittest.TestCase):
    UID = 987333

    def tearDown(self):
        import bot
        bot.states.pop(self.UID, None)
        bot.modes.pop(self.UID, None)
        bot.engines.pop(self.UID, None)
        bot.quiz_sessions.pop(self.UID, None)
        bot.awaiting_photo.discard(self.UID)

    def _msg(self, text):
        return {"chat": {"id": 558}, "from": {"id": self.UID,
                                              "username": "quizzer",
                                              "first_name": "Q"},
                "text": text}

    def _current_q(self):
        import bot
        sess = bot.quiz_sessions[self.UID]
        return bot.QUIZ_QUESTIONS[sess["order"][sess["pos"]]]

    def _answer(self, correct=True):
        import bot
        q = self._current_q()
        idx = q["answer"] if correct else (q["answer"] + 1) % 4
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("ABCD"[idx]))
        return q, [c[0][1] for c in mock_send.call_args_list]

    def test_bank_integrity(self):
        import bot
        bank = bot.QUIZ_QUESTIONS
        self.assertGreaterEqual(len(bank), 30)
        seen_q = set()
        for entry in bank:
            self.assertEqual(len(entry["options"]), 4)
            self.assertIsInstance(entry["answer"], int)
            self.assertGreaterEqual(entry["answer"], 0)
            self.assertLess(entry["answer"], 4)
            self.assertTrue(entry["q"].strip())
            self.assertTrue(entry["why"].strip())
            self.assertTrue(all(o.strip() for o in entry["options"]))
            self.assertEqual(len(set(entry["options"])), 4)  # no dup options
            seen_q.add(entry["q"])
        self.assertEqual(len(seen_q), len(bank))  # no duplicate questions

    def test_quiz_starts_and_serves_question(self):
        import bot
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/quiz"))
        self.assertIn(self.UID, bot.quiz_sessions)
        sess = bot.quiz_sessions[self.UID]
        self.assertEqual(len(sess["order"]), bot.QUIZ_LENGTH)
        self.assertEqual(sess["pos"], 0)
        self.assertEqual(sess["score"], 0)
        sent = mock_send.call_args[0][1]
        self.assertIn("Sawal 1/10", sent)
        self.assertIn(self._current_q()["q"], sent)

    def test_quiz_button_starts_session(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("📝 Quiz"))
        self.assertIn(self.UID, bot.quiz_sessions)

    def test_correct_answer_increments_score(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
        _q, texts = self._answer(correct=True)
        self.assertEqual(bot.quiz_sessions[self.UID]["score"], 1)
        self.assertEqual(bot.quiz_sessions[self.UID]["pos"], 1)
        self.assertTrue(any("Sahi jawab" in t for t in texts))
        self.assertTrue(any("Sawal 2/10" in t for t in texts))

    def test_wrong_answer_reveals_correct_without_score(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
        q, texts = self._answer(correct=False)
        self.assertEqual(bot.quiz_sessions[self.UID]["score"], 0)
        self.assertEqual(bot.quiz_sessions[self.UID]["pos"], 1)
        joined = "\n".join(texts)
        self.assertIn("Galat jawab", joined)
        self.assertIn(q["options"][q["answer"]], joined)  # sahi jawab dikha
        self.assertIn(q["why"], joined)                   # vyakhya bhi

    def test_number_answer_also_works(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
        q = self._current_q()
        with patch("bot.send"):
            bot.handle_message(self._msg(str(q["answer"] + 1)))  # 1-4 format
        self.assertEqual(bot.quiz_sessions[self.UID]["score"], 1)

    def test_non_answer_reprompts_without_breaking_session(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("mujhe nahi pata"))
        self.assertIn(self.UID, bot.quiz_sessions)          # session zinda
        self.assertEqual(bot.quiz_sessions[self.UID]["pos"], 0)
        self.assertEqual(bot.quiz_sessions[self.UID]["score"], 0)
        texts = [c[0][1] for c in mock_send.call_args_list]
        self.assertTrue(any("samajh nahi aaya" in t for t in texts))
        self.assertTrue(any("Sawal 1/10" in t for t in texts))  # sawal dobara

    def test_quiz_ends_after_10_with_result(self):
        import bot
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/quiz"))
            for _ in range(10):
                q = self._current_q()
                bot.handle_message(self._msg("ABCD"[q["answer"]]))
        self.assertNotIn(self.UID, bot.quiz_sessions)  # session saf
        texts = [c[0][1] for c in mock_send.call_args_list]
        self.assertTrue(any("Quiz Khatm" in t for t in texts))
        self.assertTrue(any("Score: 10/10" in t for t in texts))

    def test_stopquiz_exits_mid_session(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
            q = self._current_q()
            bot.handle_message(self._msg("ABCD"[q["answer"]]))
        self.assertIn(self.UID, bot.quiz_sessions)
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/stopquiz"))
        self.assertNotIn(self.UID, bot.quiz_sessions)
        self.assertIn("band ho gaya", mock_send.call_args[0][1])

    def test_mode_switch_ends_quiz(self):
        import bot
        with patch("bot.send"):
            bot.handle_message(self._msg("/quiz"))
        self.assertIn(self.UID, bot.quiz_sessions)
        with patch("bot.send"):
            bot.handle_message(self._msg("💬 AI Chat"))  # mode button
        self.assertNotIn(self.UID, bot.quiz_sessions)
        self.assertEqual(bot.modes.get(self.UID), "chat")


class TestGroupQuiz(unittest.TestCase):
    CHAT = -100777001
    STARTER = 987111
    OTHER = 987222
    ADMIN = 111  # tests ke env ADMIN_IDS=111

    def tearDown(self):
        import bot
        bot.group_sessions.clear()
        bot.group_poll_chats.clear()

    def _msg(self, text, uid=None, chat=None):
        return {"chat": {"id": self.CHAT if chat is None else chat},
                "from": {"id": self.STARTER if uid is None else uid,
                         "username": "gq", "first_name": "G"},
                "text": text}

    def _fake_tg(self, calls):
        def fake_tg(method, **kwargs):
            calls.append((method, kwargs))
            if method == "sendPoll":
                n = sum(1 for m, _ in calls if m == "sendPoll")
                return {"ok": True, "result": {"poll": {"id": f"poll-{n}"}}}
            return {"ok": True, "result": {}}
        return fake_tg

    def _fake_thread_cls(self, bag):
        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None):
                self.target, self.args = target, args
                bag.append(self)

            def start(self):
                pass
        return FakeThread

    def _start(self, text="/groupquiz 5", uid=None, chat=None):
        import bot
        calls, threads = [], []
        with patch("bot.send") as mock_send, \
                patch("bot.tg", side_effect=self._fake_tg(calls)), \
                patch.object(bot.threading, "Thread",
                             self._fake_thread_cls(threads)):
            bot.handle_message(self._msg(text, uid=uid, chat=chat))
        polls = [kw for m, kw in calls if m == "sendPoll"]
        sent = [c[0][1] for c in mock_send.call_args_list]
        return polls, threads, sent

    # --- pure: poll payload ---
    def test_poll_payload_shape(self):
        import bot
        q = bot.QUIZ_QUESTIONS[0]
        p = bot.group_poll_payload(123, q, 0, 10)
        self.assertEqual(p["chat_id"], 123)
        self.assertEqual(p["type"], "quiz")
        self.assertEqual(p["correct_option_id"], q["answer"])
        self.assertEqual(p["open_period"], 30)
        self.assertIs(p["is_anonymous"], False)
        self.assertEqual(p["question"], f"Sawal 1/10: {q['q']}")
        self.assertEqual(list(p["options"]), q["options"])
        self.assertLessEqual(len(p["explanation"]), 200)

    def test_poll_payload_long_explanation_truncated(self):
        import bot
        long_q = {"q": "Lamba sawal?", "options": ["a", "b", "c", "d"],
                  "answer": 2, "why": "x" * 500}
        p = bot.group_poll_payload(1, long_q, 2, 7)
        self.assertEqual(p["question"], "Sawal 3/7: Lamba sawal?")
        self.assertEqual(p["correct_option_id"], 2)
        self.assertLessEqual(len(p["explanation"]), 200)

    # --- pure: scoring ---
    def _session(self):
        return {"poll_id": "p1", "correct": 2, "posted_at": 1000.0,
                "answered": set(), "scores": {}}

    def test_record_answer_correct_wrong_duplicate(self):
        import bot
        s = self._session()
        # sahi jawab: +1 score aur laga samay judta hai
        self.assertTrue(bot.group_record_answer(s, 1, "Ram", [2], now=1005.0))
        self.assertEqual(s["scores"][1]["correct"], 1)
        self.assertEqual(s["scores"][1]["time"], 5.0)
        # usi poll par dusra jawab ignore
        self.assertFalse(bot.group_record_answer(s, 1, "Ram", [0], now=1006.0))
        self.assertEqual(s["scores"][1]["correct"], 1)
        self.assertEqual(s["scores"][1]["time"], 5.0)
        # galat jawab: na score, na samay
        self.assertTrue(bot.group_record_answer(s, 2, "Shyam", [0], now=1007.0))
        self.assertEqual(s["scores"][2]["correct"], 0)
        self.assertEqual(s["scores"][2]["time"], 0.0)
        # khaali option_ids bhi crash nahi karta, galat hi ginti hai
        self.assertTrue(bot.group_record_answer(s, 3, "Gita", [], now=1008.0))
        self.assertEqual(s["scores"][3]["correct"], 0)
        # agla poll: wahi user phir jawab de sakta hai
        s["poll_id"] = "p2"
        self.assertTrue(bot.group_record_answer(s, 1, "Ram", [2], now=1010.0))
        self.assertEqual(s["scores"][1]["correct"], 2)

    def test_poll_answer_routing_and_unknown_poll_ignored(self):
        import bot
        q = bot.QUIZ_QUESTIONS[0]
        bot.group_sessions[self.CHAT] = {
            "questions": [q], "index": 0, "poll_id": "poll-X",
            "correct": q["answer"], "posted_at": 500.0,
            "scores": {}, "answered": set(), "starter_id": self.STARTER,
            "generation": 1, "active": True}
        bot.group_poll_chats["poll-X"] = self.CHAT
        bot.handle_poll_answer({"poll_id": "poll-X",
                                "user": {"id": 42, "first_name": "Ram"},
                                "option_ids": [q["answer"]]})
        entry = bot.group_sessions[self.CHAT]["scores"][42]
        self.assertEqual(entry["correct"], 1)
        self.assertEqual(entry["name"], "Ram")
        self.assertGreater(entry["time"], 0)
        # anjaan poll ka jawab: chup-chaap ignore, koi entry nahi
        bot.handle_poll_answer({"poll_id": "poll-UNKNOWN",
                                "user": {"id": 43, "first_name": "X"},
                                "option_ids": [0]})
        self.assertNotIn(43, bot.group_sessions[self.CHAT]["scores"])
        # wahi user dobara tap kare to score nahi badhta
        bot.handle_poll_answer({"poll_id": "poll-X",
                                "user": {"id": 42, "first_name": "Ram"},
                                "option_ids": [q["answer"]]})
        self.assertEqual(bot.group_sessions[self.CHAT]["scores"][42]["correct"], 1)

    # --- pure: leaderboard ---
    def test_leaderboard_sorting_medals_and_empty(self):
        import bot
        scores = {
            1: {"name": "Asha", "correct": 8, "time": 50.0},
            2: {"name": "Bina", "correct": 8, "time": 40.0},
            3: {"name": "Chand", "correct": 9, "time": 90.0},
            4: {"name": "Deep", "correct": 1, "time": 5.0},
        }
        lines = bot.group_leaderboard_text(scores, 10).splitlines()
        self.assertIn("Quiz khatm", lines[0])
        self.assertIn("🥇 Chand – 9/10 sahi (90 sec)", lines[1])
        self.assertIn("🥈 Bina – 8/10 sahi (40 sec)", lines[2])  # tez = upar
        self.assertIn("🥉 Asha – 8/10 sahi (50 sec)", lines[3])
        self.assertIn("4. Deep – 1/10 sahi (5 sec)", lines[4])
        empty = bot.group_leaderboard_text({}, 10)
        self.assertIn("kisi ne koi jawab nahi diya", empty)

    # --- session flow (API mocked) ---
    def test_groupquiz_start_posts_intro_and_first_poll(self):
        import bot
        polls, threads, sent = self._start("/groupquiz 5")
        self.assertTrue(any("GK Group Quiz shuru" in s and "5 sawal" in s
                            for s in sent))
        self.assertEqual(len(polls), 1)
        self.assertEqual(polls[0]["chat_id"], self.CHAT)
        self.assertEqual(polls[0]["type"], "quiz")
        self.assertTrue(polls[0]["question"].startswith("Sawal 1/5:"))
        session = bot.group_sessions[self.CHAT]
        self.assertEqual(len(session["questions"]), 5)
        self.assertEqual(session["poll_id"], "poll-1")
        self.assertEqual(session["starter_id"], self.STARTER)
        self.assertTrue(session["active"])
        self.assertEqual(bot.group_poll_chats.get("poll-1"), self.CHAT)
        self.assertEqual(len(threads), 1)  # advance timer laga

    def test_groupquiz_count_clamped(self):
        import bot
        self._start("/groupquiz 2", chat=-100777002)
        self.assertEqual(len(bot.group_sessions[-100777002]["questions"]),
                         bot.GROUP_QUIZ_MIN)
        self._start("/groupquiz 99", chat=-100777003)
        self.assertEqual(len(bot.group_sessions[-100777003]["questions"]), 35)

    def test_second_groupquiz_refused_while_active(self):
        import bot
        self._start("/groupquiz 5")
        calls = []
        with patch("bot.send") as mock_send, \
                patch("bot.tg", side_effect=self._fake_tg(calls)):
            bot.handle_message(self._msg("/groupquiz"))
        sent = [c[0][1] for c in mock_send.call_args_list]
        self.assertTrue(any("pehle se ek quiz chal raha hai" in s
                            for s in sent))
        self.assertEqual([m for m, _ in calls if m == "sendPoll"], [])
        self.assertEqual(len(bot.group_sessions[self.CHAT]["questions"]), 5)

    def test_timer_advances_and_finishes_quiz(self):
        import bot
        calls, threads = [], []
        with patch("bot.send") as mock_send, \
                patch("bot.tg", side_effect=self._fake_tg(calls)), \
                patch.object(bot.threading, "Thread",
                             self._fake_thread_cls(threads)), \
                patch("time.sleep", lambda s: None):
            bot.handle_message(self._msg("/groupquiz 3"))
            for _ in range(3):
                t = threads[-1]
                t.target(*t.args)  # timer turant chalao
        polls = [kw for m, kw in calls if m == "sendPoll"]
        self.assertEqual(len(polls), 3)
        self.assertTrue(polls[0]["question"].startswith("Sawal 1/3:"))
        self.assertTrue(polls[2]["question"].startswith("Sawal 3/3:"))
        sent = [c[0][1] for c in mock_send.call_args_list]
        self.assertTrue(any("Quiz khatm" in s for s in sent))
        self.assertNotIn(self.CHAT, bot.group_sessions)

    def test_stopgroupquiz_by_starter_clears_and_kills_timer(self):
        import bot
        _polls, threads, _sent = self._start("/groupquiz 5")
        self.assertIn(self.CHAT, bot.group_sessions)
        gen_before = bot._group_quiz_gen
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/stopgroupquiz"))
        self.assertIn("rok diya gaya", mock_send.call_args[0][1])
        self.assertNotIn(self.CHAT, bot.group_sessions)
        self.assertGreater(bot._group_quiz_gen, gen_before)  # gen badha
        # purana timer ab chal bhi jaye to kuch post nahi hona chahiye
        calls = []
        with patch("bot.tg", side_effect=self._fake_tg(calls)), \
                patch("bot.send"), patch("time.sleep", lambda s: None):
            threads[0].target(*threads[0].args)
        self.assertEqual([m for m, _ in calls if m == "sendPoll"], [])
        self.assertNotIn(self.CHAT, bot.group_sessions)

    def test_stopgroupquiz_denied_for_non_starter_admin_allowed(self):
        import bot
        self._start("/groupquiz 5")
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/stopgroupquiz", uid=self.OTHER))
        self.assertIn(self.CHAT, bot.group_sessions)  # quiz abhi bhi ON
        self.assertNotIn("rok diya gaya", mock_send.call_args[0][1])
        with patch("bot.send") as mock_send:
            bot.handle_message(self._msg("/stopgroupquiz", uid=self.ADMIN))
        self.assertNotIn(self.CHAT, bot.group_sessions)
        self.assertIn("rok diya gaya", mock_send.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
