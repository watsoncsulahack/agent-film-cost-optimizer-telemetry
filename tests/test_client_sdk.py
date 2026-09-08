"""Unit tests for high-level TelemetryClient SDK."""

import asyncio
from uuid import uuid4
import pytest
from telemetry_agent.client import TelemetryClient
from telemetry_agent.session_manager import TelemetrySessionManager


def test_telemetry_client_async_flow():
    async def _test():
        mgr = TelemetrySessionManager(idle_timeout_seconds=60)
        client = TelemetryClient(custom_session_manager=mgr)

        session_id = uuid4()
        # 1. Start session
        sid = await client.async_start_session(
            prompt_text="Cinematic car chase through tunnel",
            suggested_model="Runway Gen-3 Alpha",
            base_api_cost=0.0500,
            session_id=session_id,
        )
        assert sid == session_id

        # 2. Record rerun
        await client.async_record_rerun(session_id=sid, incremental_cost=0.0500, reason="Car motion jitter")
        state = mgr.get_active_session(sid)
        assert state.total_rerun_count == 1
        assert state.total_session_cost == 0.1000

        # 3. Complete session
        record = await client.async_complete_session(session_id=sid, accepted=True)
        assert record.user_accepted == 1
        assert record.total_session_cost == 0.1000

    asyncio.run(_test())


def test_telemetry_client_sync_flow():
    mgr = TelemetrySessionManager(idle_timeout_seconds=60)
    client = TelemetryClient(custom_session_manager=mgr)

    session_id = uuid4()
    # 1. Sync start
    sid = client.start_session(
        prompt_text="Forest morning sunbeams",
        suggested_model="Luma Ray 2",
        base_api_cost=0.0350,
        session_id=session_id,
    )
    assert sid == session_id

    # 2. Sync rerun
    client.record_rerun(session_id=sid, incremental_cost=0.0350, reason="Add soft fog")
    state = mgr.get_active_session(sid)
    assert state.total_rerun_count == 1
    assert state.total_session_cost == 0.0700

    # 3. Sync complete
    client.complete_session(session_id=sid, accepted=True)
    assert mgr.get_active_session(sid) is None
