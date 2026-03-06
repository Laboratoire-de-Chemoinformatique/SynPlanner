"""FastAPI application for submitting and monitoring SynPlanner Dagster jobs.

This API acts as a bridge between the frontend and Dagster's GraphQL API,
providing a simpler REST interface for job management.

Run with:
    uvicorn dagster_synplanner.api.app:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dagster_synplanner.api.models import (
    JobStatusResponse,
    JobSubmitResponse,
    JobType,
    PlanningConfig,
    SubmitJobRequest,
)

DAGSTER_GRAPHQL_URL = os.environ.get(
    "DAGSTER_GRAPHQL_URL", "http://localhost:3000/graphql"
)
UPLOAD_DIR = os.environ.get("SYNPLANNER_UPLOAD_DIR", "/data/synplanner/uploads")
RESULTS_DIR = os.environ.get("SYNPLANNER_OUTPUT_DIR", "/data/synplanner")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="SynPlanner Orchestration API",
    description="REST API for submitting and monitoring SynPlanner computational jobs via Dagster.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Dagster GraphQL helpers --------------------------------------------------

LAUNCH_RUN_MUTATION = """
mutation LaunchRun($executionParams: ExecutionParams!) {
  launchRun(executionParams: $executionParams) {
    __typename
    ... on LaunchRunSuccess {
      run {
        runId
        status
      }
    }
    ... on PythonError {
      message
      stack
    }
    ... on InvalidStepError {
      invalidStepKey
    }
    ... on InvalidOutputError {
      invalidOutputName
    }
  }
}
"""

RUN_STATUS_QUERY = """
query RunStatus($runId: ID!) {
  runOrError(runId: $runId) {
    __typename
    ... on Run {
      runId
      status
      startTime
      endTime
      tags {
        key
        value
      }
    }
    ... on RunNotFoundError {
      message
    }
    ... on PythonError {
      message
    }
  }
}
"""

RUNS_LIST_QUERY = """
query RunsList($limit: Int!, $cursor: String) {
  runsOrError(limit: $limit, cursor: $cursor) {
    __typename
    ... on Runs {
      results {
        runId
        jobName
        status
        startTime
        endTime
        tags {
          key
          value
        }
      }
    }
  }
}
"""


async def _graphql_request(query: str, variables: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            DAGSTER_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
        )
        resp.raise_for_status()
        return resp.json()


def _build_run_config(request: SubmitJobRequest) -> dict:
    """Build Dagster run_config from the API request."""
    if request.job_type == JobType.PLANNING and request.planning_config:
        pc = request.planning_config
        config = {
            "ops": {
                "planning_op": {
                    "inputs": {
                        "config_path": pc.planning_config,
                        "targets": pc.targets,
                        "reaction_rules": pc.reaction_rules,
                        "building_blocks": pc.building_blocks,
                        "policy_network": pc.policy_network,
                    }
                }
            }
        }
        if pc.value_network:
            config["ops"]["planning_op"]["inputs"]["value_network"] = pc.value_network
        return config

    if request.job_type == JobType.DATA_PREPARATION and request.data_prep_config:
        dc = request.data_prep_config
        return {
            "ops": {
                "reaction_standardizing_op": {
                    "inputs": {
                        "config_path": dc.standardization_config,
                        "input_file": dc.input_reactions,
                    }
                },
                "reaction_filtering_op": {
                    "inputs": {
                        "config_path": dc.filtering_config,
                    }
                },
            }
        }

    return {}


# -- API Endpoints -------------------------------------------------------------


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/jobs/submit", response_model=JobSubmitResponse)
async def submit_job(request: SubmitJobRequest):
    """Submit a new SynPlanner job to Dagster."""
    run_config = _build_run_config(request)

    variables = {
        "executionParams": {
            "selector": {
                "repositoryLocationName": "dagster_synplanner.repository",
                "repositoryName": "__repository__",
                "jobName": request.job_type.value,
            },
            "runConfigData": run_config,
            "tags": [{"key": "source", "value": "api"}],
        }
    }

    try:
        result = await _graphql_request(LAUNCH_RUN_MUTATION, variables)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach Dagster: {e}",
        )

    launch = result.get("data", {}).get("launchRun", {})
    typename = launch.get("__typename", "")

    if typename == "LaunchRunSuccess":
        run = launch["run"]
        return JobSubmitResponse(
            run_id=run["runId"],
            job_type=request.job_type.value,
            status=run["status"],
        )

    error_msg = launch.get("message", "Unknown error launching run")
    raise HTTPException(status_code=400, detail=error_msg)


@app.get("/jobs/{run_id}", response_model=JobStatusResponse)
async def get_job_status(run_id: str):
    """Get the status of a specific job run."""
    try:
        result = await _graphql_request(RUN_STATUS_QUERY, {"runId": run_id})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dagster unreachable: {e}")

    run_or_error = result.get("data", {}).get("runOrError", {})
    typename = run_or_error.get("__typename", "")

    if typename == "RunNotFoundError":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if typename == "PythonError":
        raise HTTPException(status_code=500, detail=run_or_error.get("message"))

    tags = {t["key"]: t["value"] for t in run_or_error.get("tags", [])}
    return JobStatusResponse(
        run_id=run_or_error["runId"],
        job_type=tags.get("dagster/job", "unknown"),
        status=run_or_error["status"],
        started_at=str(run_or_error.get("startTime")) if run_or_error.get("startTime") else None,
        ended_at=str(run_or_error.get("endTime")) if run_or_error.get("endTime") else None,
    )


@app.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(limit: int = 20):
    """List recent job runs."""
    try:
        result = await _graphql_request(RUNS_LIST_QUERY, {"limit": limit})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dagster unreachable: {e}")

    runs_data = result.get("data", {}).get("runsOrError", {})
    if runs_data.get("__typename") != "Runs":
        return []

    jobs = []
    for run in runs_data.get("results", []):
        tags = {t["key"]: t["value"] for t in run.get("tags", [])}
        jobs.append(
            JobStatusResponse(
                run_id=run["runId"],
                job_type=run.get("jobName", tags.get("dagster/job", "unknown")),
                status=run["status"],
                started_at=str(run.get("startTime")) if run.get("startTime") else None,
                ended_at=str(run.get("endTime")) if run.get("endTime") else None,
            )
        )
    return jobs


@app.post("/upload/{category}")
async def upload_file(category: str, file: UploadFile):
    """Upload a data file (targets, configs, building_blocks, etc.)."""
    allowed = {"targets", "configs", "building_blocks", "reactions"}
    if category not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Category must be one of: {allowed}",
        )

    dest_dir = Path(UPLOAD_DIR) / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"path": str(dest_path), "size": len(content)}


@app.get("/results/{run_id}")
async def list_results(run_id: str):
    """List result files for a completed run."""
    results_path = Path(RESULTS_DIR) / run_id
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    files = []
    for f in results_path.rglob("*"):
        if f.is_file():
            files.append({
                "name": f.name,
                "path": str(f.relative_to(results_path)),
                "size": f.stat().st_size,
            })
    return {"run_id": run_id, "files": files}


@app.get("/results/{run_id}/download/{file_path:path}")
async def download_result(run_id: str, file_path: str):
    """Download a specific result file."""
    full_path = Path(RESULTS_DIR) / run_id / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path, filename=full_path.name)


@app.get("/presets")
async def list_presets():
    """List available HuggingFace presets."""
    return {
        "presets": [
            {
                "name": "synplanner-article",
                "description": "Pre-trained models and data from the SynPlanner article",
            }
        ]
    }
