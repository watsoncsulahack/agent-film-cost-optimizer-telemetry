"""Configuration settings for Cost Optimization Telemetry Agent."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TelemetryConfig:
    """Runtime configuration for ClickHouse, MCP, and Telemetry Agent."""

    # Google Gemini Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # ClickHouse Credentials & Network
    clickhouse_host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    clickhouse_port: int = int(os.getenv("CLICKHOUSE_PORT", "8443"))
    clickhouse_user: str = os.getenv("CLICKHOUSE_USER", "default")
    clickhouse_password: str = (
        os.getenv("CLICKHOUSE_PASSWORD")
        or os.getenv("CLICKHOUSE_API_KEY")
        or os.getenv("CLICKHOUSE_KEY")
        or os.getenv("CLICKHOUSE_TOKEN")
        or ""
    )
    clickhouse_api_key: str = (
        os.getenv("CLICKHOUSE_API_KEY")
        or os.getenv("CLICKHOUSE_KEY")
        or os.getenv("CLICKHOUSE_TOKEN")
        or ""
    )
    clickhouse_database: str = os.getenv("CLICKHOUSE_DATABASE", "default")
    clickhouse_secure: bool = os.getenv("CLICKHOUSE_SECURE", "false").lower() in ("true", "1", "yes")

    # ClickHouse MCP Server Command (stdio / JSON-RPC)
    mcp_command: str = os.getenv("MCP_CLICKHOUSE_COMMAND", "uvx mcp-clickhouse")
    mcp_sse_url: Optional[str] = os.getenv("MCP_CLICKHOUSE_SSE_URL") or None


    # Telemetry Lifecycle Settings
    idle_timeout_seconds: int = int(os.getenv("IDLE_TIMEOUT_SECONDS", "900"))  # Default: 15 minutes
    telemetry_table_name: str = os.getenv("TELEMETRY_TABLE_NAME", "generation_telemetry")
    fallback_log_file: str = os.getenv("TELEMETRY_FALLBACK_FILE", "telemetry_fallback.jsonl")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Global singleton instance
config = TelemetryConfig()
