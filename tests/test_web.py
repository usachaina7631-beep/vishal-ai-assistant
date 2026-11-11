"""
Pro v3 web-layer tests — run with: python -m unittest discover tests -v
No live token/API key needed; AI call is mocked at web_chat's ai object.
"""
import json
import os
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-not-real")
os.environ.setdefault("ADMIN_IDS", "111")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

import bot


class TestWebLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), bot.HealthHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return r.status, r.read().decode("utf-8")

    def _post(self, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_health_json(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "ok")
        # The live lineage's health shape - Render/UptimeRobot depend on it
        self.assertEqual(data["service"], "Vishal AI Assistant")

    def test_index_is_website_not_json(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("VISHAL AI", body)
        self.assertIn("/api/chat", body)
        self.assertIn("VishalSahayakBot", body)

    def test_chat_uses_ai_and_sanitizes_mode(self):
        with patch.object(bot, "ai") as fake:
            fake.chat.return_value = ("Namaste! Jawab taiyaar hai.", "groq")
            status, body = self._post({"mode": "support", "message": "hello"})
        self.assertEqual(status, 200)
        self.assertIn("Namaste", body["reply"])
        # support mode must NOT reach AI as support on the open website
        sent = fake.chat.call_args[0][0]
        self.assertNotIn("customer-support", str(sent).lower())

    def test_chat_rejects_empty_and_long(self):
        status, _ = self._post({"mode": "chat", "message": "  "})
        self.assertEqual(status, 400)
        status, _ = self._post({"mode": "chat", "message": "x" * 5000})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
