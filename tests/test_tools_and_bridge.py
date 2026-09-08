"""Unit tests for ADK tools and FastAPI web bridge."""

import json
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from telemetry_agent.tools import (
    record_generation_session_init,
    record_generation_rerun,
    record_generation_outcome,
    query_model_empirical_insights,
    get_telemetry_pipeline_summary,
)
from telemetry_agent.web_bridge import telemetry_router


def test_tools_lifecycle():
    session_id = str(uuid4())

    # 1. Init tool
    res_init = json.loads(record_generation_session_init(
        session_id=session_id,
        prompt_text="Macro eye shot",
        suggested_model="Runway Gen-3 Alpha",
        base_api_cost=0.05,
    ))
    assert res_init["status"] == "initialized"

    # 2. Rerun tool
    res_rerun = json.loads(record_generation_rerun(
        session_id=session_id,
        incremental_cost=0.05,
        reason="Pupil too dilated",
    ))
    assert res_rerun["status"] == "rerun_recorded"
    assert res_rerun["total_rerun_count"] == 1
    assert res_rerun["total_session_cost"] == 0.10

    # 3. Outcome tool
    res_outcome = json.loads(record_generation_outcome(
        session_id=session_id,
        accepted=True,
    ))
    assert res_outcome["status"] in ("persisted", "terminal_recorded")


def test_fastapi_web_bridge():
    app = FastAPI()
    app.include_router(telemetry_router)
    client = TestClient(app)

    session_id = str(uuid4())

    # POST /api/telemetry/session/init
    resp = client.post("/api/telemetry/session/init", json={
        "session_id": session_id,
        "prompt_text": "Aerial mountain sweep",
        "suggested_model": "Luma Ray 2",
        "base_api_cost": 0.04,
    })
    assert resp.status_code == 201
    assert resp.json()["status"] == "initialized"

    # POST /api/telemetry/session/rerun
    resp = client.post("/api/telemetry/session/rerun", json={
        "session_id": session_id,
        "incremental_cost": 0.04,
        "reason": "Sunflare too harsh",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "rerun_recorded"

    # POST /api/telemetry/session/complete
    resp = client.post("/api/telemetry/session/complete", json={
        "session_id": session_id,
        "user_accepted": True,
        "reason": "approved",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] in ("persisted", "completed")

    # GET /api/telemetry/health
    resp = client.get("/api/telemetry/health")
    assert resp.status_code == 200
