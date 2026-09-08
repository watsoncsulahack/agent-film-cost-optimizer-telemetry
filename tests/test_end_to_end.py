"""End-to-end simulation test verifying complete telemetry lifecycle."""

import asyncio
from uuid import uuid4
import pytest
from telemetry_agent.client import TelemetryClient
from telemetry_agent.session_manager import TelemetrySessionManager


def test_end_to_end_multimodel_telemetry_flow():
    async def _test():
        mgr = TelemetrySessionManager(idle_timeout_seconds=60)
        client = TelemetryClient(custom_session_manager=mgr)

        # Session A: Runway Gen-3 (1 rerun, accepted)
        sid_a = await client.async_start_session(
            prompt_text="Cinematic aerial drone shot sweeping over metropolis",
            suggested_model="Runway Gen-3 Alpha",
            base_api_cost=0.0500,
        )
        await client.async_record_rerun(sid_a, incremental_cost=0.0500, reason="Adjust sun angle")
        rec_a = await client.async_complete_session(sid_a, accepted=True)

        assert rec_a.total_rerun_count == 1
        assert rec_a.total_session_cost == 0.1000
        assert rec_a.user_accepted == 1

        # Session B: Luma Ray 2 (0 reruns, accepted)
        sid_b = await client.async_start_session(
            prompt_text="Cyberpunk character dialogue in rain",
            suggested_model="Luma Ray 2",
            base_api_cost=0.0350,
        )
        rec_b = await client.async_complete_session(sid_b, accepted=True)

        assert rec_b.total_rerun_count == 0
        assert rec_b.total_session_cost == 0.0350
        assert rec_b.user_accepted == 1

        # Session C: Kling 1.5 Pro (2 reruns, rejected/abandoned)
        sid_c = await client.async_start_session(
            prompt_text="Fluid dynamics water splash against rocks",
            suggested_model="Kling 1.5 Pro",
            base_api_cost=0.0280,
        )
        await client.async_record_rerun(sid_c, incremental_cost=0.0280, reason="Water unnatural")
        await client.async_record_rerun(sid_c, incremental_cost=0.0280, reason="Artifacts on rocks")
        rec_c = await client.async_abandon_session(sid_c, reason="Unusable physics")

        assert rec_c.total_rerun_count == 2
        assert rec_c.total_session_cost == 0.0840
        assert rec_c.user_accepted == 0

    asyncio.run(_test())
