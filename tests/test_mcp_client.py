"""Unit tests for ClickHouse MCP client and fault tolerance (FR-4, NFR-2)."""

import asyncio
import os
from uuid import uuid4
import pytest
from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import TelemetryRecord


def test_mcp_client_fallback_on_unreachable_server(tmp_path):
    """Verifies that unreachable ClickHouse MCP connection fails gracefully and writes fallback log (NFR-2)."""
    async def _test():
        fallback_file = str(tmp_path / "test_fallback.jsonl")
        client = ClickHouseMCPClient(
            mcp_command="false",  # Guaranteed to fail
            host="invalid-host-999.internal",
            port=9999,
        )
        client.fallback_file = fallback_file

        session_id = uuid4()
        record = TelemetryRecord(
            session_id=session_id,
            prompt_text="Test prompt for fallback",
            suggested_model="Sora",
            base_api_cost=0.0800,
            total_rerun_count=1,
            total_session_cost=0.1600,
            user_accepted=1,
        )

        # Should return False without raising an unhandled exception
        success = await client.insert_telemetry_record(record)
        assert success is False

        # Fallback log file should exist and contain the record
        assert os.path.exists(fallback_file)
        with open(fallback_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert str(session_id) in content
            assert "Sora" in content

    asyncio.run(_test())
