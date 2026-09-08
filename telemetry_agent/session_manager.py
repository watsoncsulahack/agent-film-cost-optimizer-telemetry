"""Session Lifecycle and State Management for Telemetry Agent.

Manages active in-memory session buffers, rerun counters, cost aggregation,
and the 15-minute idle timeout watchdog (FR-1, FR-2, FR-3).
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Callable, Dict, Optional
from uuid import UUID

from telemetry_agent.config import config
from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import (
    RerunPayload,
    SessionInitPayload,
    TelemetryRecord,
    TerminalPayload,
)

logger = logging.getLogger("telemetry_agent.session_manager")


class ActiveSessionState:
    """In-memory state buffer for a single active generation session."""

    def __init__(self, init_payload: SessionInitPayload):
        self.session_id: UUID = init_payload.session_id
        self.prompt_text: str = init_payload.prompt_text
        self.suggested_model: str = init_payload.suggested_model
        self.base_api_cost: float = init_payload.base_api_cost
        self.total_rerun_count: int = 0
        self.total_session_cost: float = init_payload.base_api_cost
        self.user_accepted: Optional[int] = None
        self.feedback_category: str = "unspecified"
        self.director_feedback: str = ""
        self.created_at: datetime = datetime.utcnow()
        self.last_activity_at: datetime = datetime.utcnow()
        self.locked: bool = False
        self.watchdog_task: Optional[asyncio.Task] = None

    def record_rerun(self, rerun_payload: RerunPayload) -> None:
        """Increments rerun count and accumulates incremental API cost with feedback (FR-2)."""
        if self.locked:
            logger.warning("Attempted to rerun already locked session %s", self.session_id)
            return

        cost_to_add = rerun_payload.incremental_cost if rerun_payload.incremental_cost is not None else self.base_api_cost
        self.total_rerun_count += 1
        self.total_session_cost = round(self.total_session_cost + cost_to_add, 4)
        self.last_activity_at = datetime.utcnow()

        if rerun_payload.adjusted_prompt:
            self.prompt_text = rerun_payload.adjusted_prompt
        if rerun_payload.feedback_category:
            self.feedback_category = rerun_payload.feedback_category
        if rerun_payload.director_feedback:
            self.director_feedback = rerun_payload.director_feedback

        logger.info(
            "Session %s rerun #%d recorded (+ $%.4f, total realized cost: $%.4f, category: %s)",
            self.session_id,
            self.total_rerun_count,
            cost_to_add,
            self.total_session_cost,
            self.feedback_category,
        )

    def lock_terminal_state(
        self,
        accepted: bool,
        feedback_category: Optional[str] = None,
        director_feedback: Optional[str] = None,
    ) -> TelemetryRecord:
        """Locks counters and creates the finalized TelemetryRecord (FR-3)."""
        self.locked = True
        self.user_accepted = 1 if accepted else 0
        if feedback_category is not None:
            self.feedback_category = feedback_category
        if director_feedback is not None:
            self.director_feedback = director_feedback

        # Cancel idle watchdog timer if active
        if self.watchdog_task and not self.watchdog_task.done():
            try:
                self.watchdog_task.cancel()
            except Exception:
                pass

        return TelemetryRecord(
            session_id=self.session_id,
            prompt_text=self.prompt_text,
            suggested_model=self.suggested_model,
            base_api_cost=round(self.base_api_cost, 4),
            total_rerun_count=self.total_rerun_count,
            total_session_cost=round(self.total_session_cost, 4),
            user_accepted=self.user_accepted,
            feedback_category=self.feedback_category or "unspecified",
            director_feedback=self.director_feedback or "",
            created_at=self.created_at,
        )


class TelemetrySessionManager:
    """Manages active generation sessions and coordinates ClickHouse MCP egress."""

    def __init__(
        self,
        mcp_client: Optional[ClickHouseMCPClient] = None,
        idle_timeout_seconds: Optional[int] = None,
        on_terminal_callback: Optional[Callable[[TelemetryRecord], None]] = None,
    ):
        self.mcp_client = mcp_client or ClickHouseMCPClient()
        self.idle_timeout_seconds = (
            idle_timeout_seconds if idle_timeout_seconds is not None else config.idle_timeout_seconds
        )
        self.on_terminal_callback = on_terminal_callback

        self._active_sessions: Dict[UUID, ActiveSessionState] = {}
        self._completed_records: Dict[UUID, TelemetryRecord] = {}
        self._lock = threading.Lock()

    def get_active_session(self, session_id: UUID) -> Optional[ActiveSessionState]:
        """Retrieves an active session state if present."""
        with self._lock:
            return self._active_sessions.get(session_id)

    def get_completed_record(self, session_id: UUID) -> Optional[TelemetryRecord]:
        """Retrieves a locked and completed telemetry record."""
        with self._lock:
            return self._completed_records.get(session_id)

    def initialize_session_sync(self, payload: SessionInitPayload) -> ActiveSessionState:
        """Synchronous session initialization."""
        with self._lock:
            state = ActiveSessionState(payload)
            self._active_sessions[payload.session_id] = state

        # Arm watchdog if running inside an event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                state.watchdog_task = loop.create_task(
                    self._idle_watchdog(payload.session_id, self.idle_timeout_seconds)
                )
        except RuntimeError:
            pass

        logger.info(
            "Initialized telemetry session %s (model: %s, base cost: $%.4f, idle timeout: %ds)",
            payload.session_id,
            payload.suggested_model,
            payload.base_api_cost,
            self.idle_timeout_seconds,
        )
        return state

    async def initialize_session(self, payload: SessionInitPayload) -> ActiveSessionState:
        """Initializes a new tracking session and arms the idle watchdog (FR-1)."""
        return self.initialize_session_sync(payload)

    def record_rerun_sync(self, payload: RerunPayload) -> Optional[ActiveSessionState]:
        """Synchronously handles user rejection / retry event (FR-2)."""
        with self._lock:
            state = self._active_sessions.get(payload.session_id)
            if not state:
                logger.warning("Cannot record rerun: session %s not found in active buffers", payload.session_id)
                return None

            state.record_rerun(payload)

        # Rearm the idle watchdog timer if running in an event loop
        try:
            loop = asyncio.get_running_loop()
            if state.watchdog_task and not state.watchdog_task.done():
                state.watchdog_task.cancel()
            if loop.is_running():
                state.watchdog_task = loop.create_task(
                    self._idle_watchdog(payload.session_id, self.idle_timeout_seconds)
                )
        except RuntimeError:
            pass

        return state

    async def record_rerun(self, payload: RerunPayload) -> Optional[ActiveSessionState]:
        """Handles user rejection / retry event (FR-2)."""
        return self.record_rerun_sync(payload)

    def complete_session_sync(self, payload: TerminalPayload) -> Optional[TelemetryRecord]:
        """Synchronously handles terminal event (FR-3)."""
        with self._lock:
            state = self._active_sessions.pop(payload.session_id, None)
            if not state:
                logger.warning("Cannot complete session: %s not found in active buffers", payload.session_id)
                return self._completed_records.get(payload.session_id)

            record = state.lock_terminal_state(
                accepted=payload.user_accepted,
                feedback_category=payload.feedback_category,
                director_feedback=payload.director_feedback,
            )
            self._completed_records[payload.session_id] = record

        # Dispatch egress in background
        self._dispatch_persistence(record)
        return record

    async def complete_session(self, payload: TerminalPayload) -> Optional[TelemetryRecord]:
        """Handles explicit acceptance or cancellation terminal event (FR-3)."""
        return self.complete_session_sync(payload)

    def _dispatch_persistence(self, record: TelemetryRecord) -> None:
        """Dispatches ClickHouse MCP write without blocking."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._persist_and_notify(record))
                return
        except RuntimeError:
            pass

        # Spawn daemon thread to persist without blocking caller
        threading.Thread(
            target=lambda: asyncio.run(self._persist_and_notify(record)),
            daemon=True,
        ).start()

    async def _idle_watchdog(self, session_id: UUID, timeout_seconds: int) -> None:
        """Asynchronous background watchdog monitoring idle sessions (FR-3.2)."""
        try:
            await asyncio.sleep(timeout_seconds)
            logger.info("Session %s reached idle timeout of %ds; locking as abandoned.", session_id, timeout_seconds)
            await self.complete_session(
                TerminalPayload(
                    session_id=session_id,
                    user_accepted=False,
                    reason=f"idle_timeout_{timeout_seconds}s",
                )
            )
        except asyncio.CancelledError:
            # Watchdog was cleanly reset by a new user event or manual completion
            pass

    async def _persist_and_notify(self, record: TelemetryRecord) -> None:
        """Asynchronously inserts record via MCP and executes optional callbacks."""
        try:
            await self.mcp_client.insert_telemetry_record(record)
        except Exception as e:
            logger.error("Error during background MCP persistence: %s", str(e))

        if self.on_terminal_callback:
            try:
                self.on_terminal_callback(record)
            except Exception as cb_err:
                logger.error("Error in terminal callback: %s", str(cb_err))


# Global default session manager
session_manager = TelemetrySessionManager()
