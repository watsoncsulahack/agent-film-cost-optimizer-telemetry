"""Unit tests for FR-3.2 Idle Timeout Watchdog and Negative Terminal Events."""

import asyncio
from uuid import uuid4
import pytest
from telemetry_agent.schemas import SessionInitPayload, TerminalPayload
from telemetry_agent.session_manager import TelemetrySessionManager


def test_session_idle_timeout_watchdog():
    """Tests that active session automatically transitions to user_accepted = 0 on timeout."""
    async def _test():
        # Use short timeout (0.1 seconds) for testing
        session_mgr = TelemetrySessionManager(idle_timeout_seconds=0.1)
        session_id = uuid4()

        init_payload = SessionInitPayload(
            session_id=session_id,
            prompt_text="Macro shot of hummingbird wings",
            suggested_model="Luma Ray 2",
            base_api_cost=0.0400,
        )
        await session_mgr.initialize_session(init_payload)
        assert session_mgr.get_active_session(session_id) is not None

        # Wait for watchdog timeout to fire
        await asyncio.sleep(0.25)

        # Active session should be removed and recorded as completed with user_accepted = 0
        assert session_mgr.get_active_session(session_id) is None
        record = session_mgr.get_completed_record(session_id)
        assert record is not None
        assert record.user_accepted == 0
        assert record.total_rerun_count == 0
        assert record.total_session_cost == 0.0400

    asyncio.run(_test())


def test_explicit_abandonment_cancels_watchdog():
    """Tests explicit user cancellation before timeout."""
    async def _test():
        session_mgr = TelemetrySessionManager(idle_timeout_seconds=5)
        session_id = uuid4()

        init_payload = SessionInitPayload(
            session_id=session_id,
            prompt_text="Futuristic space station interior",
            suggested_model="Runway Gen-3 Alpha",
            base_api_cost=0.0500,
        )
        await session_mgr.initialize_session(init_payload)

        # Explicit cancel
        record = await session_mgr.complete_session(
            TerminalPayload(session_id=session_id, user_accepted=False, reason="user_canceled")
        )

        assert record is not None
        assert record.user_accepted == 0
        assert session_mgr.get_active_session(session_id) is None

    asyncio.run(_test())
