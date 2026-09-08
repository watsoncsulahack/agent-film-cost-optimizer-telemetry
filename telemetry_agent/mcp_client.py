"""ClickHouse MCP Client for Telemetry Agent.

Integrates with the official ClickHouse Model Context Protocol server (mcp-clickhouse)
via stdio/JSON-RPC, executing DDL and telemetry ingestion queries (FR-4).
"""

import json
import logging
import os
import shlex
from datetime import datetime
from typing import Any, Dict, List, Optional

from telemetry_agent.config import config
from telemetry_agent.schemas import ModelEmpiricalStats, TelemetryRecord, TelemetrySummary

logger = logging.getLogger("telemetry_agent.mcp_client")
logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

# Table creation DDL from agent-telemetry-spec.md
CLICKHOUSE_DDL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    session_id UUID,
    prompt_text String,
    suggested_model LowCardinality(String),
    base_api_cost Float32,
    total_rerun_count Int32,
    total_session_cost Float32,
    user_accepted UInt8,
    feedback_category LowCardinality(String) DEFAULT 'unspecified',
    director_feedback String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (suggested_model, created_at);
"""


class ClickHouseMCPClient:
    """Manages MCP-based communication with ClickHouse."""

    def __init__(
        self,
        mcp_command: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        secure: Optional[bool] = None,
        table_name: Optional[str] = None,
    ):
        self.mcp_command = mcp_command or config.mcp_command
        self.host = host or config.clickhouse_host
        self.port = port or config.clickhouse_port
        self.user = user or config.clickhouse_user
        self.password = password if password is not None else config.clickhouse_password
        self.database = database or config.clickhouse_database
        self.secure = secure if secure is not None else config.clickhouse_secure
        self.table_name = table_name or config.telemetry_table_name
        self.fallback_file = config.fallback_log_file

        self._initialized_table = False
        self._ch_client = None  # Lazy clickhouse-connect instance if available

    def _get_env_vars(self) -> Dict[str, str]:
        """Prepares environment variables for the mcp-clickhouse child process."""
        env = os.environ.copy()
        env["CLICKHOUSE_HOST"] = self.host
        env["CLICKHOUSE_PORT"] = str(self.port)
        env["CLICKHOUSE_USER"] = self.user
        env["CLICKHOUSE_PASSWORD"] = self.password
        env["CLICKHOUSE_DATABASE"] = self.database
        env["CLICKHOUSE_SECURE"] = "true" if self.secure else "false"
        return env

    async def execute_query_via_mcp(self, query: str) -> Dict[str, Any]:
        """Executes a SQL query by communicating with the mcp-clickhouse server."""
        # Clean up whitespace
        clean_query = query.strip()
        logger.debug("Executing query via MCP: %s", clean_query)

        try:
            # Import MCP SDK components dynamically
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            # Parse command line string into binary and args
            cmd_parts = shlex.split(self.mcp_command)
            command = cmd_parts[0]
            args = cmd_parts[1:] if len(cmd_parts) > 1 else []

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=self._get_env_vars(),
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List available tools to detect query execution tool name
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    logger.debug("MCP Tools available: %s", tool_names)

                    # Official mcp-clickhouse uses 'run_query' or 'execute_query'
                    query_tool = "run_query" if "run_query" in tool_names else (
                        "execute_query" if "execute_query" in tool_names else "query"
                    )

                    if query_tool not in tool_names and tool_names:
                        # Fallback to first tool matching 'query' or first available
                        for name in tool_names:
                            if "query" in name:
                                query_tool = name
                                break
                        else:
                            query_tool = tool_names[0]

                    logger.debug("Calling MCP tool '%s' with query", query_tool)
                    result = await session.call_tool(
                        name=query_tool,
                        arguments={"query": clean_query, "sql": clean_query},
                    )

                    # Extract text content from MCP CallToolResult
                    text_content = ""
                    if hasattr(result, "content") and result.content:
                        for part in result.content:
                            if hasattr(part, "text"):
                                text_content += part.text

                    return {"success": True, "raw_result": text_content, "error": None}

        except ImportError:
            logger.warning("mcp Python package not available; using native ClickHouse fallback.")
            return await self._execute_native_clickhouse(clean_query)
        except Exception as e:
            logger.warning("MCP client connection/execution error: %s. Attempting fallback.", str(e))
            return await self._execute_native_clickhouse(clean_query, mcp_error=str(e))

    async def _execute_native_clickhouse(self, query: str, mcp_error: Optional[str] = None) -> Dict[str, Any]:
        """Native clickhouse-connect fallback client for resiliency (NFR-2)."""
        try:
            import clickhouse_connect

            if self._ch_client is None:
                self._ch_client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    database=self.database,
                    secure=self.secure,
                )

            res = self._ch_client.command(query)
            return {"success": True, "raw_result": str(res), "error": None}
        except Exception as e:
            err_msg = f"Native ClickHouse fallback also failed: {str(e)} (Initial MCP error: {mcp_error})"
            logger.error(err_msg)
            return {"success": False, "raw_result": None, "error": err_msg}

    def _write_fallback_log(self, record: TelemetryRecord, reason: str):
        """Persists failed telemetry records to a local JSON Lines log file (NFR-2)."""
        try:
            payload = record.model_dump()
            payload["_fallback_reason"] = reason
            payload["_logged_at"] = datetime.utcnow().isoformat()
            with open(self.fallback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
            logger.info("Telemetry record saved to local fallback log: %s", self.fallback_file)
        except Exception as log_err:
            logger.error("Failed to write to fallback log file: %s", str(log_err))

    async def ensure_table_exists(self) -> bool:
        """Verifies and creates the generation_telemetry ClickHouse table if needed."""
        if self._initialized_table:
            return True

        ddl = CLICKHOUSE_DDL.format(table_name=self.table_name).strip()
        logger.info("Verifying ClickHouse telemetry table '%s' exists...", self.table_name)
        result = await self.execute_query_via_mcp(ddl)

        if result["success"]:
            self._initialized_table = True
            logger.info("ClickHouse telemetry table '%s' ready.", self.table_name)
            return True
        else:
            logger.warning("Failed to initialize ClickHouse table via MCP: %s", result["error"])
            return False

    async def insert_telemetry_record(self, record: TelemetryRecord) -> bool:
        """Inserts a structured TelemetryRecord into ClickHouse via MCP (FR-4)."""
        # Ensure schema exists first
        await self.ensure_table_exists()

        insert_sql = record.to_sql_insert(table_name=self.table_name)
        logger.info(
            "Pushing telemetry record for session %s (model: %s, reruns: %d, cost: $%.4f, accepted: %d)",
            record.session_id,
            record.suggested_model,
            record.total_rerun_count,
            record.total_session_cost,
            record.user_accepted,
        )

        result = await self.execute_query_via_mcp(insert_sql)

        if result["success"]:
            logger.info("Successfully persisted telemetry record for session %s via MCP.", record.session_id)
            return True
        else:
            logger.error("Failed to insert telemetry to ClickHouse: %s", result["error"])
            # Save to local fallback without raising exception (NFR-2)
            self._write_fallback_log(record, reason=str(result["error"]))
            return False

    async def fetch_model_empirical_stats(self, model_name: Optional[str] = None) -> List[ModelEmpiricalStats]:
        """Queries ClickHouse for historical ground-truth performance statistics.

        Feeds empirical metrics back into the Cost Optimization Agent (Feedback Loop).
        """
        if model_name:
            escaped_name = model_name.replace("'", "''")
            where_clause = f"WHERE suggested_model = '{escaped_name}'"
        else:
            where_clause = ""

        query = f"""
        SELECT
            suggested_model,
            count() AS total_sessions,
            sum(user_accepted) AS accepted_sessions,
            avg(total_rerun_count) AS avg_rerun_count,
            avg(base_api_cost) AS avg_base_cost,
            avg(total_session_cost) AS avg_total_cost
        FROM {self.table_name}
        {where_clause}
        GROUP BY suggested_model
        ORDER BY total_sessions DESC
        """

        result = await self.execute_query_via_mcp(query)
        stats_list: List[ModelEmpiricalStats] = []

        if not result["success"] or not result["raw_result"]:
            # If database is offline or unpopulated, aggregate from local fallback log if present
            logger.debug("No empirical stats from ClickHouse; attempting to compute from fallback log: %s", self.fallback_file)
            return self._compute_stats_from_fallback_log(model_name)

        try:
            # Parse ClickHouse result rows from raw text or JSON
            raw = result["raw_result"]
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            for line in lines:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                if len(parts) < 6:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 6:
                    try:
                        m_name = parts[0]
                        total = int(parts[1])
                        accepted = int(parts[2])
                        avg_reruns = float(parts[3])
                        avg_base = float(parts[4])
                        avg_total = float(parts[5])
                        acc_rate = round(accepted / total if total > 0 else 0.0, 4)
                        multiplier = round(avg_total / avg_base if avg_base > 0 else 1.0, 4)

                        stats_list.append(
                            ModelEmpiricalStats(
                                suggested_model=m_name,
                                total_sessions=total,
                                accepted_sessions=accepted,
                                acceptance_rate=acc_rate,
                                avg_rerun_count=round(avg_reruns, 2),
                                avg_base_cost=round(avg_base, 4),
                                avg_total_cost=round(avg_total, 4),
                                effective_cost_multiplier=multiplier,
                            )
                        )
                    except (ValueError, IndexError):
                        continue
        except Exception as parse_err:
            logger.warning("Could not parse ClickHouse query output: %s", str(parse_err))

        if not stats_list:
            return self._compute_stats_from_fallback_log(model_name)

        return stats_list

    def _compute_stats_from_fallback_log(self, model_name: Optional[str] = None) -> List[ModelEmpiricalStats]:
        """Calculates empirical performance stats directly from local fallback JSON Lines log."""
        if not os.path.exists(self.fallback_file):
            return []

        model_groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        m = data.get("suggested_model")
                        if not m:
                            continue
                        if model_name and m != model_name:
                            continue
                        if m not in model_groups:
                            model_groups[m] = []
                        model_groups[m].append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Error reading fallback log for stats calculation: %s", str(e))
            return []

        stats_list: List[ModelEmpiricalStats] = []
        for m_name, entries in model_groups.items():
            total = len(entries)
            accepted = sum(1 for e in entries if e.get("user_accepted") == 1)
            total_reruns = sum(e.get("total_rerun_count", 0) for e in entries)
            total_base_cost = sum(e.get("base_api_cost", 0.0) for e in entries)
            total_realized_cost = sum(e.get("total_session_cost", 0.0) for e in entries)

            avg_reruns = round(total_reruns / total, 2) if total > 0 else 0.0
            avg_base = round(total_base_cost / total, 4) if total > 0 else 0.0
            avg_total = round(total_realized_cost / total, 4) if total > 0 else 0.0
            acc_rate = round(accepted / total, 4) if total > 0 else 0.0
            multiplier = round(avg_total / avg_base, 4) if avg_base > 0 else 1.0

            stats_list.append(
                ModelEmpiricalStats(
                    suggested_model=m_name,
                    total_sessions=total,
                    accepted_sessions=accepted,
                    acceptance_rate=acc_rate,
                    avg_rerun_count=avg_reruns,
                    avg_base_cost=avg_base,
                    avg_total_cost=avg_total,
                    effective_cost_multiplier=multiplier,
                )
            )

        # Sort by total sessions descending
        stats_list.sort(key=lambda s: s.total_sessions, reverse=True)
        return stats_list

    async def get_summary_metrics(self) -> TelemetrySummary:
        """Retrieves global telemetry summary across all models."""
        models_stats = await self.fetch_model_empirical_stats()
        total_sessions = sum(m.total_sessions for m in models_stats)
        total_accepted = sum(m.accepted_sessions for m in models_stats)
        total_spend = sum(m.avg_total_cost * m.total_sessions for m in models_stats)
        overall_acc_rate = round(total_accepted / total_sessions, 4) if total_sessions > 0 else 0.0

        return TelemetrySummary(
            total_sessions=total_sessions,
            total_accepted=total_accepted,
            overall_acceptance_rate=overall_acc_rate,
            total_pipeline_spend=round(total_spend, 4),
            models=models_stats,
        )
