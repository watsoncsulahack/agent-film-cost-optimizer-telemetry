"""Cost Optimization Telemetry Agent Package.

Autonomous monitoring daemon capturing empirical ground-truth video generation metrics
and persisting to ClickHouse via Model Context Protocol (MCP).
"""

from telemetry_agent.agent import create_telemetry_agent, root_agent, telemetry_agent
from telemetry_agent.client import TelemetryClient, default_client
from telemetry_agent.config import TelemetryConfig, config
from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import (
    ModelEmpiricalStats,
    RerunPayload,
    SessionInitPayload,
    TelemetryRecord,
    TelemetrySummary,
    TerminalPayload,
)
from telemetry_agent.session_manager import ActiveSessionState, TelemetrySessionManager, session_manager
from telemetry_agent.tools import (
    get_telemetry_pipeline_summary,
    query_model_empirical_insights,
    record_generation_outcome,
    record_generation_rerun,
    record_generation_session_init,
)
from telemetry_agent.web_bridge import telemetry_router

__all__ = [
    "root_agent",
    "telemetry_agent",
    "create_telemetry_agent",
    "TelemetryClient",
    "default_client",
    "TelemetrySessionManager",
    "session_manager",
    "ActiveSessionState",
    "ClickHouseMCPClient",
    "TelemetryConfig",
    "config",
    "SessionInitPayload",
    "RerunPayload",
    "TerminalPayload",
    "TelemetryRecord",
    "ModelEmpiricalStats",
    "TelemetrySummary",
    "telemetry_router",
    "record_generation_session_init",
    "record_generation_rerun",
    "record_generation_outcome",
    "query_model_empirical_insights",
    "get_telemetry_pipeline_summary",
]
