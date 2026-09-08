"""FastAPI Bridge and REST Router for Unified Web Application integration.

Enables mounting the Telemetry Agent directly into the Google Cloud hosted web app
or running as a decoupled microservice.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from telemetry_agent.client import default_client
from telemetry_agent.schemas import (
    ModelEmpiricalStats,
    RerunPayload,
    SessionInitPayload,
    TelemetrySummary,
    TerminalPayload,
)

telemetry_router = APIRouter(prefix="/api/telemetry", tags=["Telemetry"])


class StatusResponse(BaseModel):
    status: str
    message: str


@telemetry_router.post("/session/init", status_code=status.HTTP_201_CREATED)
async def init_session(payload: SessionInitPayload):
    """Initializes a new telemetry session buffer (FR-1)."""
    try:
        session_id = await default_client.async_start_session(
            prompt_text=payload.prompt_text,
            suggested_model=payload.suggested_model,
            base_api_cost=payload.base_api_cost,
            session_id=payload.session_id,
        )
        return {
            "status": "initialized",
            "session_id": str(session_id),
            "suggested_model": payload.suggested_model,
            "base_api_cost": payload.base_api_cost,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@telemetry_router.post("/session/rerun", status_code=status.HTTP_200_OK)
async def record_rerun(payload: RerunPayload):
    """Records a generation rerun / retry event (FR-2)."""
    try:
        await default_client.async_record_rerun(
            session_id=payload.session_id,
            incremental_cost=payload.incremental_cost,
            reason=payload.reason,
            adjusted_prompt=payload.adjusted_prompt,
            feedback_category=payload.feedback_category,
            director_feedback=payload.director_feedback,
        )
        state = default_client.session_manager.get_active_session(payload.session_id)
        return {
            "status": "rerun_recorded",
            "session_id": str(payload.session_id),
            "total_rerun_count": state.total_rerun_count if state else 0,
            "total_session_cost": state.total_session_cost if state else 0.0,
            "feedback_category": state.feedback_category if state else "unspecified",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@telemetry_router.post("/session/complete", status_code=status.HTTP_200_OK)
async def complete_session(payload: TerminalPayload):
    """Finalizes session and dispatches record to ClickHouse via MCP (FR-3, FR-4)."""
    try:
        record = await default_client.async_complete_session(
            session_id=payload.session_id,
            accepted=payload.user_accepted,
            reason=payload.reason,
            feedback_category=payload.feedback_category,
            director_feedback=payload.director_feedback,
        )
        if not record:
            return {
                "status": "completed",
                "session_id": str(payload.session_id),
                "message": "Session finalized.",
            }
        return {
            "status": "persisted",
            "session_id": str(record.session_id),
            "user_accepted": record.user_accepted,
            "total_rerun_count": record.total_rerun_count,
            "total_session_cost": record.total_session_cost,
            "feedback_category": record.feedback_category,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@telemetry_router.get("/metrics/models", response_model=List[ModelEmpiricalStats])
async def get_model_metrics(model: Optional[str] = Query(None, description="Optional model filter")):
    """Fetches empirical model performance from ClickHouse (Feedback Loop)."""
    try:
        return await default_client.async_get_model_insights(suggested_model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@telemetry_router.get("/metrics/summary", response_model=TelemetrySummary)
async def get_telemetry_summary():
    """Fetches overall telemetry summary across all recorded sessions."""
    try:
        return await default_client.async_get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@telemetry_router.get("/records")
async def get_all_records(limit: int = Query(100, ge=1, le=1000)):
    """Fetches individual session telemetry records from ClickHouse MCP."""
    try:
        return await default_client.mcp_client.fetch_all_records(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@telemetry_router.get("/health")
async def health_check():
    """Healthcheck verifying database readiness."""
    try:
        is_ready = await default_client.mcp_client.ensure_table_exists()
        return {
            "status": "healthy" if is_ready else "degraded",
            "clickhouse_table_ready": is_ready,
            "table_name": default_client.mcp_client.table_name,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "clickhouse_table_ready": False,
        }
