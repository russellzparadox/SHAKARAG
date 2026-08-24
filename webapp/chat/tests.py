from __future__ import annotations

from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from types import SimpleNamespace

from .crypto import decrypt, encrypt
from .models import ChatMessage, ChatSession, DatabaseProfile, LLMProfile


class CryptoTests(TestCase):
    def test_roundtrip(self):
        enc = encrypt("hunter2")
        self.assertNotEqual(enc, "hunter2")
        self.assertTrue(enc.startswith("enc1:"))
        self.assertEqual(decrypt(enc), "hunter2")

    def test_empty_and_plaintext_passthrough(self):
        self.assertEqual(encrypt(""), "")
        self.assertEqual(decrypt(""), "")
        self.assertEqual(decrypt("plain-value"), "plain-value")


class AuthFlowTests(TestCase):
    def test_register_creates_user_and_logs_in(self):
        resp = self.client.post(
            "/register/",
            {
                "username": "alice",
                "email": "a@example.com",
                "password1": "s3cretpass!x",
                "password2": "s3cretpass!x",
            },
            follow=True,
        )
        self.assertRedirects(resp, "/chat/")
        user = User.objects.get(username="alice")
        self.assertTrue(user.is_authenticated)

    def test_anonymous_redirects_to_login(self):
        for url in ["/chat/", "/db/", "/llm/"]:
            resp = self.client.get(url)
            self.assertEqual(resp.status_code, 302)
            self.assertIn("/accounts/login/", resp["Location"])

    def test_login_page_reachable(self):
        resp = self.client.get("/accounts/login/")
        self.assertEqual(resp.status_code, 200)


class ProfileAccessTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bob", password="pw12345!!")
        self.client.force_login(self.user)

    def test_create_db_profile_encrypts_password(self):
        resp = self.client.post(
            "/db/new/",
            {
                "name": "My MySQL",
                "dialect": "mysql",
                "host": "localhost",
                "port": 3306,
                "db_user": "ro",
                "password": "secret-pw",
                "db_name": "shop",
                "collection_name": "shop_schema",
            },
        )
        self.assertEqual(resp.status_code, 302)
        dbp = DatabaseProfile.objects.get(name="My MySQL")
        self.assertNotIn("secret-pw", dbp.password_enc)
        from .crypto import decrypt

        self.assertEqual(decrypt(dbp.password_enc), "secret-pw")

    def test_edit_keeps_password_when_blank(self):
        dbp = DatabaseProfile.objects.create(
            name="X", dialect="postgres", host="h", port=5432,
            db_user="u", password_enc=encrypt("orig"), db_name="d",
            collection_name="c1", owner=self.user,
        )
        self.client.post(
            f"/db/{dbp.pk}/edit/",
            {
                "name": "X", "dialect": "postgres", "host": "h", "port": 5432,
                "db_user": "u", "password": "", "db_name": "d", "collection_name": "c1",
            },
        )
        dbp.refresh_from_db()
        self.assertEqual(decrypt(dbp.password_enc), "orig")

    def test_shared_profiles_visible_to_others(self):
        DatabaseProfile.objects.create(name="shared", collection_name="shared_c", owner=None)
        resp = self.client.get("/db/")
        self.assertContains(resp, "shared")


def _fake_result(**kw):
    defaults = dict(
        sql="SELECT 1", explanation="why", columns=["one"], rows=[[1]], row_count=1,
        truncated=False, tables_used=["t"], answer="The answer is 1.", error=None, context="ctx",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class ChatSendTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("carol", password="pw12345!!")
        self.client.force_login(self.user)
        self.dbp = DatabaseProfile.objects.create(name="D", collection_name="cd", owner=self.user)
        self.llm = LLMProfile.objects.create(
            name="L", base_url="http://x/v1", model="m", api_key_enc=encrypt("k"), owner=self.user
        )

    def _mk_session(self):
        return ChatSession.objects.create(user=self.user, database=self.dbp, llm=self.llm)

    @patch("chat.rag_service.run_ask")
    def test_send_stores_messages_and_returns_json(self, mock_ask):
        mock_ask.return_value = {
            "sql": "SELECT 1", "explanation": "why", "columns": ["one"], "rows": [[1]],
            "row_count": 1, "truncated": False, "tables_used": ["t"],
            "answer": "The answer is 1.", "error": None,
        }
        session = self._mk_session()
        resp = self.client.post(f"/chat/{session.pk}/send/", data='{"question":"test me"}', content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["user"]["content"], "test me")
        self.assertEqual(data["assistant"]["content"], "The answer is 1.")
        self.assertEqual(data["assistant"]["meta"]["sql"], "SELECT 1")
        msgs = list(session.messages.all())
        self.assertEqual(len(msgs), 2)
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])

    @patch("chat.rag_service.run_ask")
    def test_send_handles_failure(self, mock_ask):
        mock_ask.side_effect = RuntimeError("boom")
        session = self._mk_session()
        resp = self.client.post(f"/chat/{session.pk}/send/", data='{"question":"q"}', content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("boom", data["assistant"]["content"])

    def test_empty_question_rejected(self):
        session = self._mk_session()
        resp = self.client.post(f"/chat/{session.pk}/send/", data='{"question":"  "}', content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    @patch("chat.rag_service.run_ask")
    def test_send_passes_session_language(self, mock_ask):
        mock_ask.return_value = {
            "sql": "SELECT 1", "explanation": None, "columns": [], "rows": [],
            "row_count": 0, "truncated": False, "tables_used": [],
            "answer": "پاسخ", "error": None,
        }
        session = self._mk_session()
        session.language = "fa"
        session.save(update_fields=["language"])
        self.client.post(
            f"/chat/{session.pk}/send/", data='{"question":"q"}', content_type="application/json"
        )
        mock_ask.assert_called_once()
        self.assertEqual(mock_ask.call_args.kwargs.get("answer_language"), "fa")

    def test_change_language(self):
        session = self._mk_session()
        resp = self.client.post(f"/chat/{session.pk}/language/", {"language": "fa"})
        self.assertEqual(resp.status_code, 302)
        session.refresh_from_db()
        self.assertEqual(session.language, "fa")

    def test_change_language_invalid_kept(self):
        session = self._mk_session()
        self.client.post(f"/chat/{session.pk}/language/", {"language": "klingon"})
        session.refresh_from_db()
        self.assertEqual(session.language, "auto")

    @patch("chat.rag_service.run_ask")
    def test_clarify_flow_saves_pending_and_meta(self, mock_ask):
        mock_ask.return_value = {
            "clarify": True, "clarify_question": "Which total do you mean?",
            "options": ["Amount", "Quantity"], "sql": None, "explanation": None,
            "columns": [], "rows": [], "row_count": 0, "truncated": False,
            "tables_used": [], "answer": None, "error": None,
        }
        session = self._mk_session()
        resp = self.client.post(
            f"/chat/{session.pk}/send/", data='{"question":"show totals"}', content_type="application/json"
        )
        data = resp.json()
        session.refresh_from_db()
        self.assertTrue(data["assistant"]["meta"]["type"] == "clarify")
        self.assertEqual(session.pending_question, "show totals")
        self.assertEqual(mock_ask.call_args.kwargs.get("allow_clarify"), True)

    @patch("chat.rag_service.run_ask")
    def test_answer_after_clarify_combines_questions(self, mock_ask):
        session = self._mk_session()
        session.pending_question = "show totals"
        session.save(update_fields=["pending_question"])
        mock_ask.return_value = {
            "clarify": False, "clarify_question": "", "options": [],
            "sql": "SELECT SUM(x) FROM t", "explanation": None,
            "columns": ["s"], "rows": [[5]], "row_count": 1, "truncated": False,
            "tables_used": ["t"], "answer": "Total is 5.", "error": None,
        }
        self.client.post(
            f"/chat/{session.pk}/send/", data='{"question":"amount"}', content_type="application/json"
        )
        kwargs = mock_ask.call_args
        self.assertIn("show totals", kwargs.args[2])
        self.assertIn("(User's clarification: amount)", kwargs.args[2])
        self.assertEqual(kwargs.kwargs.get("allow_clarify"), False)
        session.refresh_from_db()
        self.assertEqual(session.pending_question, "")

    def test_other_users_session_forbidden(self):
        other = User.objects.create_user("mallory", password="pw12345!!")
        session = ChatSession.objects.create(user=other, database=self.dbp, llm=self.llm)
        resp = self.client.post(
            f"/chat/{session.pk}/send/", data='{"question":"q"}', content_type="application/json"
        )
        self.assertEqual(resp.status_code, 404)
