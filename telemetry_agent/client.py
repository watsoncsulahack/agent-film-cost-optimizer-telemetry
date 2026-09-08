"""High-level Telemetry Client for application and upstream agent integration.

Enables the unified web app or Cost Optimization Agent to fire non-blocking telemetry events.
"""

import logging
from typing import List, Optional, Union
from uuid import UUID, uuid4

from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import (
    ModelEmpiricalStats,
    RerunPayload,
    SessionInitPayload,
    TelemetryRecord,
    TelemetrySummary,
    TerminalPayload,
)
from telemetry_agent.session_manager import TelemetrySessionManager, session_manager

logger = logging.getLogger("telemetry_agent.client")


class TelemetryClient:
    """Non-blocking telemetry client for session lifecycle monitoring and ClickHouse querying."""

    def __init__(
        self,
        custom_session_manager: Optional[TelemetrySessionManager] = None,
        custom_mcp_client: Optional[ClickHouseMCPClient] = None,
    ):
        self.session_manager = custom_session_manager or session_manager
        self.mcp_client = custom_mcp_client or self.session_manager.mcp_client

    def _normalize_uuid(self, session_id: Union[str, UUID]) -> UUID:
        """Converts string or UUID into a valid UUID object."""
        if isinstance(session_id, UUID):
            return session_id
        return UUID(str(session_id))

    # --- Async Lifecycle Methods ---

    async def async_start_session(
        self,
        prompt_text: str,
        suggested_model: str,
        base_api_cost: float,
        session_id: Optional[Union[str, UUID]] = None,
    ) -> UUID:
        """Initializes a telemetry session asynchronously (FR-1)."""
        uid = self._normalize_uuid(session_id) if session_id else uuid4()
        payload = SessionInitPayload(
            session_id=uid,
            prompt_text=prompt_text,
            suggested_model=suggested_model,
            base_api_cost=base_api_cost,
        )
        await self.session_manager.initialize_session(payload)
        return uid

    async def async_record_rerun(
        self,
        session_id: Union[str, UUID],
        incremental_cost: Optional[float] = None,
        reason: Optional[str] = None,
        adjusted_prompt: Optional[str] = None,
    ) -> None:
        """Records a regeneration rerun asynchronously (FR-2)."""
        uid = self._normalize_uuid(session_id)
        payload = RerunPayload(
            session_id=uid,
            incremental_cost=incremental_cost,
            reason=reason,
            adjusted_prompt=adjusted_prompt,
        )
        await self.session_manager.record_rerun(payload)

    async def async_complete_session(
        self,
        session_id: Union[str, UUID],
        accepted: bool = True,
        reason: Optional[str] = None,
    ) -> Optional[TelemetryRecord]:
        """Finalizes session and triggers ClickHouse MCP persistence (FR-3)."""
        uid = self._normalize_uuid(session_id)
        payload = TerminalPayload(
            session_id=uid,
            user_accepted=accepted,
            reason=reason or ("downloaded" if accepted else "rejected"),
        )
        return await self.session_manager.complete_session(payload)

    async def async_abandon_session(
        self,
        session_id: Union[str, UUID],
        reason: str = "user_abandoned",
    ) -> Optional[TelemetryRecord]:
        """Marks session as abandoned / closed without acceptance (FR-3.2)."""
        return await self.async_complete_session(session_id=session_id, accepted=False, reason=reason)

    async def async_get_model_insights(
        self, suggested_model: Optional[str] = None
    ) -> List[ModelEmpiricalStats]:
        """Fetches historical empirical stats from ClickHouse (Feedback Loop)."""
        return await self.mcp_client.fetch_model_empirical_stats(suggested_model)

    async def async_get_summary(self) -> TelemetrySummary:
        """Fetches overall telemetry summary."""
        return await self.mcp_client.get_summary_metrics()

    # --- Sync / Non-Blocking Fire-and-Forget Wrappers ---

    def start_session(
        self,
        prompt_text: str,
        suggested_model: str,
        base_api_cost: float,
        session_id: Optional[Union[str, UUID]] = None,
    ) -> UUID:
        """Synchronous wrapper to initialize session."""
        uid = self._normalize_uuid(session_id) if session_id else uuid4()
        payload = SessionInitPayload(
            session_id=uid,
            prompt_text=prompt_text,
            suggested_model=suggested_model,
            base_api_cost=base_api_cost,
        )
        self.session_manager.initialize_session_sync(payload)
        return uid

    def record_rerun(
        self,
        session_id: Union[str, UUID],
        incremental_cost: Optional[float] = None,
        reason: Optional[str] = None,
        adjusted_prompt: Optional[str] = None,
    ) -> None:
        """Synchronous wrapper to record rerun."""
        uid = self._normalize_uuid(session_id)
        payload = RerunPayload(
            session_id=uid,
            incremental_cost=incremental_cost,
            reason=reason,
            adjusted_prompt=adjusted_prompt,
        )
        self.session_manager.record_rerun_sync(payload)

    def complete_session(
        self,
        session_id: Union[str, UUID],
        accepted: bool = True,
        reason: Optional[str] = None,
    ) -> Optional[TelemetryRecord]:
        """Synchronous wrapper to complete session."""
        uid = self._normalize_uuid(session_id)
        payload = TerminalPayload(
            session_id=uid,
            user_accepted=accepted,
            reason=reason or ("downloaded" if accepted else "rejected"),
        )
        return self.session_manager.complete_session_sync(payload)

    def abandon_session(
        self,
        session_id: Union[str, UUID],
        reason: str = "user_abandoned",
    ) -> Optional[TelemetryRecord]:
        """Synchronous wrapper to mark session abandoned."""
        return self.complete_session(session_id=session_id, accepted=False, reason=reason)


# Singleton client instance
default_client = TelemetryClient()
