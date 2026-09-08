"""Unit tests for Session Lifecycle Management (FR-1, FR-2, FR-3)."""

import asyncio
from uuid import uuid4
import pytest
from telemetry_agent.schemas import (
    RerunPayload,
    SessionInitPayload,
    TerminalPayload,
)
from telemetry_agent.session_manager import TelemetrySessionManager


def test_session_lifecycle_positive_acceptance():
    """Tests FR-1, FR-2, FR-3.1: Init -> Reruns -> Positive Acceptance -> Record Lock."""
    async def _test():
        session_mgr = TelemetrySessionManager(idle_timeout_seconds=300)
        session_id = uuid4()

        # 1. FR-1: Session Initialization
        init_payload = SessionInitPayload(
            session_id=session_id,
            prompt_text="Sunset over mountain peak with drone rotation",
            suggested_model="Kling 1.5 Pro",
            base_api_cost=0.0300,
        )
        state = await session_mgr.initialize_session(init_payload)

        assert state.session_id == session_id
        assert state.total_rerun_count == 0
        assert state.total_session_cost == 0.0300
        assert state.locked is False

        # 2. FR-2: Iteration & Rejection Tracking
        # Rerun #1
        await session_mgr.record_rerun(
            RerunPayload(
                session_id=session_id,
                incremental_cost=0.0300,
                reason="Color grading too washed out",
            )
        )
        assert state.total_rerun_count == 1
        assert state.total_session_cost == 0.0600

        # Rerun #2 with adjusted prompt
        await session_mgr.record_rerun(
            RerunPayload(
                session_id=session_id,
                incremental_cost=0.0300,
                reason="Adjust rotation speed",
                adjusted_prompt="Sunset over mountain peak with fast dynamic drone rotation",
            )
        )
        assert state.total_rerun_count == 2
        assert state.total_session_cost == 0.0900
        assert state.prompt_text == "Sunset over mountain peak with fast dynamic drone rotation"

        # 3. FR-3.1: Positive Terminal Event
        record = await session_mgr.complete_session(
            TerminalPayload(
                session_id=session_id,
                user_accepted=True,
                reason="Downloaded final render",
            )
        )

        assert record is not None
        assert record.session_id == session_id
        assert record.user_accepted == 1
        assert record.total_rerun_count == 2
        assert record.total_session_cost == 0.0900
        assert record.base_api_cost == 0.0300
        assert session_mgr.get_active_session(session_id) is None
        assert session_mgr.get_completed_record(session_id) is not None

    asyncio.run(_test())
