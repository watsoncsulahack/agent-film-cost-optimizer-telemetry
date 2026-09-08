"""Google ADK Tools for Cost Optimization Telemetry Agent."""

import json
from typing import Optional
from uuid import UUID

from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import (
    RerunPayload,
    SessionInitPayload,
    TerminalPayload,
)
from telemetry_agent.session_manager import session_manager

mcp_client = ClickHouseMCPClient()


def record_generation_session_init(
    session_id: str,
    prompt_text: str,
    suggested_model: str,
    base_api_cost: float,
) -> str:
    """Initializes a new video generation telemetry session buffer.

    Args:
        session_id: The unique UUID string of the generation session.
        prompt_text: The filmmaking prompt or shot description.
        suggested_model: The AI video generation model selected (e.g., Runway Gen-3, Luma Ray 2, Sora).
        base_api_cost: Baseline quoted cost for a single generation run in USD.

    Returns:
        JSON string confirming session initialization.
    """
    try:
        uid = UUID(session_id)
        payload = SessionInitPayload(
            session_id=uid,
            prompt_text=prompt_text,
            suggested_model=suggested_model,
            base_api_cost=base_api_cost,
        )
        # Note: initialize_session is an async method; we invoke it within active event loop or run synchronously
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(session_manager.initialize_session(payload))
        except RuntimeError:
            asyncio.run(session_manager.initialize_session(payload))

        return json.dumps({
            "status": "initialized",
            "session_id": str(uid),
            "suggested_model": suggested_model,
            "base_api_cost": base_api_cost,
            "total_rerun_count": 0,
            "total_session_cost": base_api_cost,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def record_generation_rerun(
    session_id: str,
    incremental_cost: Optional[float] = None,
    reason: Optional[str] = None,
    adjusted_prompt: Optional[str] = None,
) -> str:
    """Records a user rejection / retry / regeneration event for an active session.

    Args:
        session_id: Active session UUID string.
        incremental_cost: Optional additional API cost for this rerun; defaults to base cost.
        reason: Optional user feedback reason for regeneration.
        adjusted_prompt: Optional modified shot prompt.

    Returns:
        JSON string with updated rerun count and cumulative realized cost.
    """
    try:
        uid = UUID(session_id)
        payload = RerunPayload(
            session_id=uid,
            incremental_cost=incremental_cost,
            reason=reason,
            adjusted_prompt=adjusted_prompt,
        )
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            state = session_manager.get_active_session(uid)
            if state:
                state.record_rerun(payload)
                return json.dumps({
                    "status": "rerun_recorded",
                    "session_id": str(uid),
                    "total_rerun_count": state.total_rerun_count,
                    "total_session_cost": state.total_session_cost,
                })
            else:
                return json.dumps({"status": "not_found", "session_id": session_id})
        except RuntimeError:
            state = asyncio.run(session_manager.record_rerun(payload))
            if state:
                return json.dumps({
                    "status": "rerun_recorded",
                    "session_id": str(uid),
                    "total_rerun_count": state.total_rerun_count,
                    "total_session_cost": state.total_session_cost,
                })
            return json.dumps({"status": "not_found", "session_id": session_id})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def record_generation_outcome(
    session_id: str,
    accepted: bool,
    reason: Optional[str] = None,
) -> str:
    """Records the final terminal outcome (acceptance or abandonment) and triggers ClickHouse MCP persistence.

    Args:
        session_id: Active session UUID string.
        accepted: True if user downloaded/accepted the shot, False if abandoned/canceled.
        reason: Optional description of outcome.

    Returns:
        JSON string summarizing finalized session telemetry.
    """
    try:
        uid = UUID(session_id)
        payload = TerminalPayload(
            session_id=uid,
            user_accepted=accepted,
            reason=reason,
        )
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(session_manager.complete_session(payload))
            return json.dumps({
                "status": "terminal_recorded",
                "session_id": str(uid),
                "user_accepted": 1 if accepted else 0,
                "message": "Persisting telemetry to ClickHouse via MCP.",
            })
        except RuntimeError:
            record = asyncio.run(session_manager.complete_session(payload))
            if record:
                return json.dumps({
                    "status": "persisted",
                    "session_id": str(uid),
                    "user_accepted": record.user_accepted,
                    "total_rerun_count": record.total_rerun_count,
                    "total_session_cost": record.total_session_cost,
                })
            return json.dumps({"status": "not_found", "session_id": session_id})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def query_model_empirical_insights(suggested_model: Optional[str] = None) -> str:
    """Queries ClickHouse for historical ground-truth performance statistics.

    Provides empirical data to the Cost Optimization Agent for smarter model recommendations.

    Args:
        suggested_model: Optional specific model name (e.g. 'Runway Gen-3 Alpha') or None for all.

    Returns:
        JSON string containing models, average rerun rates, actual realized costs, and acceptance rates.
    """
    try:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If in running loop, run in thread or async
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stats = pool.submit(
                    asyncio.run, mcp_client.fetch_model_empirical_stats(suggested_model)
                ).result()
        except RuntimeError:
            stats = asyncio.run(mcp_client.fetch_model_empirical_stats(suggested_model))

        stats_dicts = [s.model_dump() for s in stats]
        return json.dumps({
            "status": "success",
            "count": len(stats_dicts),
            "empirical_model_stats": stats_dicts,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def get_telemetry_pipeline_summary() -> str:
    """Retrieves an aggregate summary of all video generation sessions stored in ClickHouse.

    Returns:
        JSON string with overall metrics: total sessions, total spend, acceptance rate, and breakdown by model.
    """
    try:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                summary = pool.submit(asyncio.run, mcp_client.get_summary_metrics()).result()
        except RuntimeError:
            summary = asyncio.run(mcp_client.get_summary_metrics())

        return json.dumps({
            "status": "success",
            "summary": summary.model_dump(),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
