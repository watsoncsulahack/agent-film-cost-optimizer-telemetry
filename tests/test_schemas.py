"""Unit tests for Telemetry Agent schemas and precision requirements."""

from datetime import datetime
from uuid import uuid4
import pytest
from telemetry_agent.schemas import (
    RerunPayload,
    SessionInitPayload,
    TelemetryRecord,
    TerminalPayload,
    ModelEmpiricalStats,
)


def test_session_init_payload_precision():
    uid = uuid4()
    payload = SessionInitPayload(
        session_id=uid,
        prompt_text="Drone shot over futuristic city",
        suggested_model="Runway Gen-3 Alpha",
        base_api_cost=0.050012345,
    )
    assert payload.session_id == uid
    assert payload.base_api_cost == 0.0500


def test_rerun_payload_precision():
    uid = uuid4()
    payload = RerunPayload(
        session_id=uid,
        incremental_cost=0.0350999,
        reason="Camera too fast",
    )
    assert payload.incremental_cost == 0.0351


def test_telemetry_record_sql_insert():
    uid = uuid4()
    dt = datetime(2026, 9, 5, 20, 0, 0)
    record = TelemetryRecord(
        session_id=uid,
        prompt_text="Cinematic close-up of a hero's face in rain",
        suggested_model="Luma Ray 2",
        base_api_cost=0.0400,
        total_rerun_count=2,
        total_session_cost=0.1200,
        user_accepted=1,
        feedback_category="lighting_inconsistency",
        director_feedback="Lighting shifted midway through shot",
        created_at=dt,
    )

    sql = record.to_sql_insert(table_name="generation_telemetry")
    assert "INSERT INTO generation_telemetry" in sql
    assert str(uid) in sql
    # Verify quotes are escaped in prompt
    assert "hero''s face" in sql
    assert "0.0400" in sql
    assert "0.1200" in sql
    assert "lighting_inconsistency" in sql
    assert "Lighting shifted midway through shot" in sql
    assert "2026-09-05 20:00:00" in sql


def test_feedback_survey_payloads():
    uid = uuid4()
    rerun = RerunPayload(
        session_id=uid,
        incremental_cost=0.04,
        feedback_category="motion_artifact",
        director_feedback="Severe character limb distortion on fast camera pan",
    )
    assert rerun.feedback_category == "motion_artifact"
    assert "limb distortion" in rerun.director_feedback

    term = TerminalPayload(
        session_id=uid,
        user_accepted=False,
        feedback_category="budget_exceeded",
        director_feedback="Compute cost limit reached for shot",
    )
    assert term.user_accepted is False
    assert term.feedback_category == "budget_exceeded"


def test_model_empirical_stats():
    stats = ModelEmpiricalStats(
        suggested_model="Runway Gen-3 Alpha",
        total_sessions=10,
        accepted_sessions=8,
        acceptance_rate=0.8,
        avg_rerun_count=1.5,
        avg_base_cost=0.05,
        avg_total_cost=0.125,
        effective_cost_multiplier=2.5,
    )
    assert stats.acceptance_rate == 0.8
    assert stats.effective_cost_multiplier == 2.5
