"""Pydantic data schemas and payload models for Telemetry Agent."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class SessionInitPayload(BaseModel):
    """Payload to initialize a telemetry tracking session (FR-1)."""
    session_id: UUID = Field(default_factory=uuid4, description="Unique UUID for generation session")
    prompt_text: str = Field(..., min_length=1, description="Raw text prompt for video shot")
    suggested_model: str = Field(..., min_length=1, description="Video model selected (e.g., Runway Gen-3, Sora, Luma Ray 2)")
    base_api_cost: float = Field(..., ge=0.0, description="Baseline API quote for single generation")

    @field_validator("base_api_cost")
    @classmethod
    def round_precision(cls, v: float) -> float:
        return round(float(v), 4)


class RerunPayload(BaseModel):
    """Payload for generation retry / rejection event with director feedback (FR-2)."""
    session_id: UUID = Field(..., description="Active session UUID")
    incremental_cost: Optional[float] = Field(None, ge=0.0, description="Cost of rerun; defaults to base_api_cost if omitted")
    reason: Optional[str] = Field(None, description="Optional filmmaker rejection reason")
    adjusted_prompt: Optional[str] = Field(None, description="Optional prompt adjustment")
    feedback_category: Optional[str] = Field(default="unspecified", description="Issue category (e.g. motion_artifact, physics_defect, lighting, camera_motion, prompt_hallucination)")
    director_feedback: Optional[str] = Field(default="", description="Detailed qualitative feedback notes from filmmaker")

    @field_validator("incremental_cost")
    @classmethod
    def round_incremental_cost(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(float(v), 4)
        return v


class TerminalPayload(BaseModel):
    """Payload for terminal event - acceptance or abandonment with feedback (FR-3)."""
    session_id: UUID = Field(..., description="Active session UUID")
    user_accepted: bool = Field(..., description="True if accepted/downloaded, False if abandoned/timed out")
    reason: Optional[str] = Field(None, description="Optional terminal outcome reason (e.g., 'downloaded', 'idle_timeout', 'canceled')")
    feedback_category: Optional[str] = Field(default="unspecified", description="Reason category for discard or approval")
    director_feedback: Optional[str] = Field(default="", description="Detailed qualitative feedback notes")


class TelemetryRecord(BaseModel):
    """Structured telemetry record matching the ClickHouse generation_telemetry schema."""
    session_id: UUID = Field(..., description="UUID for generation workflow")
    prompt_text: str = Field(..., description="Raw text prompt used for generation run")
    suggested_model: str = Field(..., description="Video model executed")
    base_api_cost: float = Field(..., description="Estimated baseline cost for single generation")
    total_rerun_count: int = Field(default=0, ge=0, description="Total number of regeneration retries attempted")
    total_session_cost: float = Field(..., ge=0.0, description="Cumulative API cost incurred across all retries")
    user_accepted: int = Field(..., ge=0, le=1, description="1 if accepted, 0 if rejected/abandoned")
    feedback_category: str = Field(default="unspecified", description="Issue or outcome category")
    director_feedback: str = Field(default="", description="Director qualitative notes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="UTC commit timestamp")

    @field_validator("base_api_cost", "total_session_cost")
    @classmethod
    def enforce_micro_precision(cls, v: float) -> float:
        """Enforces 4-decimal precision float values (NFR-3)."""
        return round(float(v), 4)

    def to_sql_insert(self, table_name: str = "generation_telemetry") -> str:
        """Generates a sanitized ClickHouse SQL INSERT query string."""
        sanitized_prompt = self.prompt_text.replace("'", "''").replace("\n", " ")
        sanitized_model = self.suggested_model.replace("'", "''")
        sanitized_cat = (self.feedback_category or "unspecified").replace("'", "''")
        sanitized_feedback = (self.director_feedback or "").replace("'", "''").replace("\n", " ")
        dt_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"INSERT INTO {table_name} "
            f"(session_id, prompt_text, suggested_model, base_api_cost, total_rerun_count, total_session_cost, user_accepted, feedback_category, director_feedback, created_at) "
            f"VALUES ('{str(self.session_id)}', '{sanitized_prompt}', '{sanitized_model}', "
            f"{self.base_api_cost:.4f}, {self.total_rerun_count}, {self.total_session_cost:.4f}, "
            f"{self.user_accepted}, '{sanitized_cat}', '{sanitized_feedback}', '{dt_str}')"
        )


class ModelEmpiricalStats(BaseModel):
    """Empirical ground-truth performance statistics aggregated from ClickHouse."""
    suggested_model: str
    total_sessions: int
    accepted_sessions: int
    acceptance_rate: float
    avg_rerun_count: float
    avg_base_cost: float
    avg_total_cost: float
    effective_cost_multiplier: float


class TelemetrySummary(BaseModel):
    """System-wide telemetry summary for dashboard and Cost Optimizer feedback."""
    total_sessions: int
    total_accepted: int
    overall_acceptance_rate: float
    total_pipeline_spend: float
    models: List[ModelEmpiricalStats] = []
