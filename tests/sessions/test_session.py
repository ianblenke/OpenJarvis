"""Tests for session management dataclasses and SQLite-backed SessionStore."""

from __future__ import annotations

import time

import pytest

from openjarvis.sessions.session import (
    Session,
    SessionIdentity,
    SessionMessage,
    SessionStore,
)

# ---------------------------------------------------------------------------
# SessionIdentity dataclass
# ---------------------------------------------------------------------------


class TestSessionIdentity:
    @pytest.mark.spec("REQ-sessions.identity")
    @pytest.mark.spec("REQ-sessions.identity.create")
    def test_create_identity_with_all_fields(self) -> None:
        identity = SessionIdentity(
            user_id="u1",
            display_name="Alice",
            channel_ids={"telegram": "t123", "discord": "d456"},
        )
        assert identity.user_id == "u1"
        assert identity.display_name == "Alice"
        assert identity.channel_ids["telegram"] == "t123"
        assert identity.channel_ids["discord"] == "d456"

    @pytest.mark.spec("REQ-sessions.identity.defaults")
    def test_identity_defaults(self) -> None:
        identity = SessionIdentity(user_id="u2")
        assert identity.user_id == "u2"
        assert identity.display_name == ""
        assert identity.channel_ids == {}


# ---------------------------------------------------------------------------
# SessionMessage dataclass
# ---------------------------------------------------------------------------


class TestSessionMessage:
    @pytest.mark.spec("REQ-sessions.message.create")
    def test_create_message(self) -> None:
        msg = SessionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.channel == ""
        assert msg.timestamp == 0.0
        assert msg.metadata == {}

    @pytest.mark.spec("REQ-sessions.message.metadata")
    def test_message_with_metadata(self) -> None:
        msg = SessionMessage(
            role="assistant",
            content="Hi there",
            channel="telegram",
            timestamp=1234567890.0,
            metadata={"model": "gpt-4"},
        )
        assert msg.channel == "telegram"
        assert msg.timestamp == 1234567890.0
        assert msg.metadata["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


class TestSession:
    @pytest.mark.spec("REQ-sessions.session")
    @pytest.mark.spec("REQ-sessions.session.create")
    def test_create_session_defaults(self) -> None:
        session = Session()
        assert session.session_id == ""
        assert session.identity is None
        assert session.messages == []
        assert session.created_at == 0.0
        assert session.last_activity == 0.0
        assert session.metadata == {}

    @pytest.mark.spec("REQ-sessions.session.create")
    def test_create_session_with_id(self) -> None:
        session = Session(session_id="s1")
        assert session.session_id == "s1"
        assert len(session.messages) == 0

    @pytest.mark.spec("REQ-sessions.session.add-message")
    def test_add_message_appends_and_updates_activity(self) -> None:
        session = Session(session_id="s1")
        before = time.time()
        session.add_message("user", "Hello")
        after = time.time()

        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"
        assert session.messages[0].channel == ""
        assert before <= session.messages[0].timestamp <= after
        assert before <= session.last_activity <= after

    @pytest.mark.spec("REQ-sessions.session.add-message")
    def test_add_message_with_channel(self) -> None:
        session = Session(session_id="s1")
        session.add_message("user", "From Telegram", channel="telegram")
        assert session.messages[0].channel == "telegram"

    @pytest.mark.spec("REQ-sessions.session.add-message")
    def test_add_multiple_messages(self) -> None:
        session = Session(session_id="s1")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.add_message("user", "How are you?")
        assert len(session.messages) == 3
        assert session.messages[1].role == "assistant"
        assert session.messages[2].content == "How are you?"


# ---------------------------------------------------------------------------
# SessionStore (SQLite-backed persistence)
# ---------------------------------------------------------------------------


class TestSessionStore:
    def _make_store(self, tmp_path, **kwargs):
        return SessionStore(db_path=tmp_path / "sessions.db", **kwargs)

    @pytest.mark.spec("REQ-sessions.store.get-or-create")
    @pytest.mark.spec("REQ-sessions.store.create")
    def test_create_new_session(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create("user1", display_name="Alice")
        assert session.session_id != ""
        assert session.identity is not None
        assert session.identity.user_id == "user1"
        assert session.identity.display_name == "Alice"
        assert session.created_at > 0
        assert session.last_activity > 0
        store.close()

    @pytest.mark.spec("REQ-sessions.store.get-existing")
    def test_get_existing_session_returns_same(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        s1 = store.get_or_create("user1")
        s2 = store.get_or_create("user1")
        assert s1.session_id == s2.session_id
        store.close()

    @pytest.mark.spec("REQ-sessions.store.create")
    def test_create_session_with_channel(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create(
            "user1", channel="telegram", channel_user_id="t123"
        )
        assert session.identity.channel_ids.get("telegram") == "t123"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.messages")
    @pytest.mark.spec("REQ-sessions.store.save-message")
    def test_save_and_reload_messages(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create("user1")
        store.save_message(session.session_id, "user", "Hello")
        store.save_message(session.session_id, "assistant", "Hi there!")

        reloaded = store.get_or_create("user1")
        assert len(reloaded.messages) == 2
        assert reloaded.messages[0].role == "user"
        assert reloaded.messages[0].content == "Hello"
        assert reloaded.messages[1].role == "assistant"
        assert reloaded.messages[1].content == "Hi there!"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.save-message")
    def test_save_message_with_channel_and_metadata(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create("user1")
        store.save_message(
            session.session_id,
            "user",
            "Hello",
            channel="telegram",
            metadata={"source": "bot"},
        )
        reloaded = store.get_or_create("user1")
        assert reloaded.messages[0].channel == "telegram"
        assert reloaded.messages[0].metadata == {"source": "bot"}
        store.close()

    @pytest.mark.spec("REQ-sessions.store.channel-linking")
    @pytest.mark.spec("REQ-sessions.store.link-channel")
    def test_link_channel(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create("user1")
        store.link_channel(session.session_id, "telegram", "t123")
        store.link_channel(session.session_id, "discord", "d456")

        reloaded = store.get_or_create("user1")
        assert reloaded.identity.channel_ids.get("telegram") == "t123"
        assert reloaded.identity.channel_ids.get("discord") == "d456"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.expiry")
    def test_session_expiry_creates_new(self, tmp_path) -> None:
        store = self._make_store(tmp_path, max_age_hours=0.0001)  # ~0.36 seconds
        s1 = store.get_or_create("user1")
        time.sleep(0.5)
        s2 = store.get_or_create("user1")
        assert s1.session_id != s2.session_id
        store.close()

    @pytest.mark.spec("REQ-sessions.store.decay")
    def test_decay_removes_old_sessions(self, tmp_path) -> None:
        store = self._make_store(tmp_path, max_age_hours=0.0001)
        store.get_or_create("user1")
        time.sleep(0.5)
        removed = store.decay()
        assert removed >= 1
        store.close()

    @pytest.mark.spec("REQ-sessions.store.decay")
    def test_decay_with_custom_age(self, tmp_path) -> None:
        store = self._make_store(tmp_path, max_age_hours=24)
        store.get_or_create("user1")
        # With 24-hour window nothing should be removed
        removed = store.decay()
        assert removed == 0
        # With near-zero window everything should be removed
        time.sleep(0.5)
        removed = store.decay(max_age_hours=0.0001)
        assert removed >= 1
        store.close()

    @pytest.mark.spec("REQ-sessions.store.list")
    def test_list_sessions(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        store.get_or_create("user1")
        store.get_or_create("user2")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        # All sessions should have identity populated
        for s in sessions:
            assert s.identity is not None
        store.close()

    @pytest.mark.spec("REQ-sessions.store.list")
    def test_list_sessions_active_only(self, tmp_path) -> None:
        store = self._make_store(tmp_path, max_age_hours=0.0001)
        store.get_or_create("user1")
        time.sleep(0.5)
        sessions = store.list_sessions(active_only=True)
        assert len(sessions) == 0
        sessions_all = store.list_sessions(active_only=False)
        assert len(sessions_all) >= 1
        store.close()

    @pytest.mark.spec("REQ-sessions.store.consolidation")
    def test_consolidation_reduces_messages(self, tmp_path) -> None:
        store = self._make_store(tmp_path, consolidation_threshold=5)
        session = store.get_or_create("user1")
        for i in range(10):
            store.save_message(session.session_id, "user", f"msg {i}")
        reloaded = store.get_or_create("user1")
        assert len(reloaded.messages) < 10
        store.close()

    @pytest.mark.spec("REQ-sessions.store.cross-channel")
    def test_cross_channel_session(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        s1 = store.get_or_create("user1", channel="telegram", channel_user_id="t1")
        store.save_message(s1.session_id, "user", "From Telegram", channel="telegram")
        store.link_channel(s1.session_id, "discord", "d1")
        store.save_message(s1.session_id, "user", "From Discord", channel="discord")

        reloaded = store.get_or_create("user1")
        assert len(reloaded.messages) == 2
        assert reloaded.messages[0].channel == "telegram"
        assert reloaded.messages[1].channel == "discord"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.save-message")
    def test_save_message_updates_last_activity(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        session = store.get_or_create("user1")
        original_activity = session.last_activity
        time.sleep(0.05)
        store.save_message(session.session_id, "user", "ping")
        reloaded = store.get_or_create("user1")
        assert reloaded.last_activity > original_activity
        store.close()

    @pytest.mark.spec("REQ-sessions.store.get-existing-with-channel")
    def test_get_existing_session_updates_channel_ids(self, tmp_path) -> None:
        """Exercise lines 132-140: existing session gets channel_ids updated."""
        store = self._make_store(tmp_path)
        s1 = store.get_or_create("user1")
        # Second call with channel info should update channel_ids
        s2 = store.get_or_create(
            "user1",
            channel="telegram",
            channel_user_id="t999",
        )
        assert s1.session_id == s2.session_id
        assert s2.identity.channel_ids.get("telegram") == "t999"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.expiry-with-channel")
    def test_expired_session_with_channel_creates_new(self, tmp_path) -> None:
        """Exercise lines 124-129: expired session triggers _create_session."""
        store = self._make_store(tmp_path, max_age_hours=0.0001)
        s1 = store.get_or_create(
            "user1",
            channel="discord",
            channel_user_id="d123",
        )
        time.sleep(0.5)
        s2 = store.get_or_create(
            "user1",
            channel="telegram",
            channel_user_id="t456",
        )
        assert s1.session_id != s2.session_id
        assert s2.identity.channel_ids.get("telegram") == "t456"
        store.close()

    @pytest.mark.spec("REQ-sessions.store.consolidation")
    def test_consolidation_skips_when_below_half_threshold(self, tmp_path) -> None:
        """Exercise line 214: consolidate returns early when few messages."""
        store = self._make_store(tmp_path, consolidation_threshold=100)
        session = store.get_or_create("user1")
        # Add only 3 messages, well below threshold/2
        for i in range(3):
            store.save_message(session.session_id, "user", f"msg {i}")
        # Manually call consolidate -- should be a no-op
        store.consolidate(session.session_id)
        reloaded = store.get_or_create("user1")
        # All 3 messages should still be present
        assert len(reloaded.messages) == 3
        store.close()
