"""Pydantic models for API request/response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class JobType(str, Enum):
    DATA_PREPARATION = "data_preparation"
    FULL_TRAINING = "full_training_pipeline"
    PLANNING = "planning"
    RETRAIN_AND_PLAN = "retrain_and_plan"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELED = "CANCELED"
    NOT_STARTED = "NOT_STARTED"


class DataPrepConfig(BaseModel):
    """Configuration for data preparation jobs."""

    standardization_config: str = Field(
        description="Path to standardization YAML config"
    )
    filtering_config: str = Field(description="Path to filtering YAML config")
    input_reactions: str = Field(description="Path to raw reaction data file")
    num_cpus: int = Field(default=4, ge=1)
    batch_size: int = Field(default=100, ge=1)


class TrainingConfig(BaseModel):
    """Configuration for model training jobs."""

    policy_config: str = Field(description="Path to policy training YAML config")
    rule_extraction_config: str = Field(
        description="Path to rule extraction YAML config"
    )
    input_reactions: str = Field(description="Path to mapped reaction data")
    standardization_config: str | None = Field(
        default=None, description="If provided, runs data prep first"
    )
    filtering_config: str | None = None
    num_cpus: int = Field(default=4, ge=1)


class PlanningConfig(BaseModel):
    """Configuration for planning jobs."""

    planning_config: str = Field(description="Path to planning YAML config")
    targets: str = Field(description="Path to target molecules file (.smi)")
    reaction_rules: str = Field(description="Path to reaction rules file")
    building_blocks: str = Field(description="Path to building blocks file")
    policy_network: str = Field(description="Path to trained policy weights")
    value_network: str | None = Field(
        default=None, description="Path to value network weights (optional)"
    )
    cluster_results: bool = Field(
        default=True, description="Whether to cluster results after planning"
    )


class SubmitJobRequest(BaseModel):
    """Request to submit a new job."""

    job_type: JobType
    data_prep_config: DataPrepConfig | None = None
    training_config: TrainingConfig | None = None
    planning_config: PlanningConfig | None = None


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    run_id: str
    job_type: str
    status: JobStatus
    started_at: str | None = None
    ended_at: str | None = None
    error_message: str | None = None
    result_metadata: dict | None = None


class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""

    run_id: str
    job_type: str
    status: str = "QUEUED"
    message: str = "Job submitted successfully"
