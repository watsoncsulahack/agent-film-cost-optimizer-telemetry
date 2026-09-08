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


def test_mcp_client_host_sanitization():
    """Verifies that full URLs with https/http and ports are sanitized properly."""
    # Test HTTPS full URL with port
    c1 = ClickHouseMCPClient(host="https://q48kam3v08.us-east1.gcp.clickhouse.cloud:8443")
    assert c1.host == "q48kam3v08.us-east1.gcp.clickhouse.cloud"
    assert c1.port == 8443
    assert c1.secure is True

    # Test HTTP full URL with trailing slash
    c2 = ClickHouseMCPClient(host="http://my-host.clickhouse.cloud:8123/", secure=False)
    assert c2.host == "my-host.clickhouse.cloud"
    assert c2.port == 8123

    # Test plain host without protocol
    c3 = ClickHouseMCPClient(host="localhost", port=8123, secure=False)
    assert c3.host == "localhost"
    assert c3.port == 8123
    assert c3.secure is False

