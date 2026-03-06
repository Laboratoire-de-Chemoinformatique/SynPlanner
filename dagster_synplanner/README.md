# SynPlanner Dagster Platform

A Dagster-based orchestration platform for [SynPlanner](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner) retrosynthetic planning workflows. Provides a web UI for submitting and monitoring long-running computational chemistry tasks, with deployment support for AWS/Kubernetes.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  React UI   │────▶│  FastAPI      │────▶│  Dagster        │
│  (frontend) │     │  (REST API)   │     │  (orchestrator) │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  SynPlanner Ops  │
                                          │  (user code)     │
                                          └────────┬────────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              │                    │                    │
                     ┌────────▼──────┐  ┌─────────▼────────┐  ┌───────▼───────┐
                     │ Data Prep     │  │ Model Training   │  │ Planning      │
                     │ • standardize │  │ • rule extraction│  │ • MCTS search │
                     │ • filter      │  │ • policy training│  │ • clustering  │
                     │ • atom map    │  │ • value tuning   │  │               │
                     └───────────────┘  └──────────────────┘  └───────────────┘
```

## Components

### Dagster Ops (computational tasks)

| Op | Description | Duration |
|----|-------------|----------|
| `reaction_standardizing_op` | Standardize & deduplicate reactions | Minutes-hours |
| `reaction_filtering_op` | Filter erroneous reactions | Minutes-hours |
| `reaction_mapping_op` | Neural atom mapping (GPU) | Hours |
| `building_blocks_standardizing_op` | Standardize building blocks | Minutes |
| `rule_extracting_op` | Extract reaction templates | Minutes-hours |
| `ranking_policy_training_op` | Train ranking policy GNN (GPU) | Hours |
| `filtering_policy_training_op` | Train filtering policy GNN (GPU) | Hours |
| `value_network_tuning_op` | RL fine-tuning of value network (GPU) | Hours-days |
| `planning_op` | MCTS retrosynthetic planning | Minutes-hours |
| `clustering_op` | Cluster synthesis routes | Minutes |

### Pre-built Jobs (workflows)

| Job | Pipeline |
|-----|----------|
| `data_preparation` | standardize → filter → map |
| `full_training_pipeline` | data_prep → rules → policy training |
| `planning` | MCTS planning → clustering |
| `retrain_and_plan` | data_prep → training → planning (end-to-end) |

### Sensors

- **`new_targets_sensor`** - Watches a directory for new `.smi` files and auto-triggers planning jobs

## Quick Start (Docker Compose)

```bash
# From the repo root:
docker compose -f dagster_synplanner/deploy/docker-compose.yml up --build

# Services available at:
#   Frontend:  http://localhost:5173
#   API:       http://localhost:8000
#   Dagster:   http://localhost:3000
```

## Kubernetes Deployment (Helm)

```bash
# Add Dagster Helm repo
helm repo add dagster https://dagster-io.github.io/helm
helm repo update

# Build and push the Docker image
docker build -t your-registry/synplanner:latest -f dagster_synplanner/Dockerfile .
docker push your-registry/synplanner:latest

# Install
cd dagster_synplanner/deploy/helm
helm dependency update synplanner
helm install synplanner synplanner/ \
  --set image.repository=your-registry/synplanner \
  --set image.tag=latest \
  --set synplanner.device=cuda \
  --set gpuWorkers.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=synplanner.your-domain.com
```

### AWS-specific setup

For AWS EKS deployment:

1. **Storage**: Use EFS (Elastic File System) for shared `ReadWriteMany` persistent volumes:
   ```yaml
   synplanner:
     persistence:
       storageClass: efs-sc
       accessMode: ReadWriteMany
       size: 100Gi
   ```

2. **GPU nodes**: Use `p3.2xlarge` or `g4dn.xlarge` instances in a managed node group:
   ```yaml
   gpuWorkers:
     enabled: true
     nodeSelector:
       node.kubernetes.io/instance-type: g4dn.xlarge
   ```

3. **Ingress**: Use AWS ALB Ingress Controller:
   ```yaml
   ingress:
     enabled: true
     className: alb
     annotations:
       alb.ingress.kubernetes.io/scheme: internet-facing
       alb.ingress.kubernetes.io/target-type: ip
   ```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs/submit` | Submit a new job |
| `GET` | `/jobs` | List recent jobs |
| `GET` | `/jobs/{run_id}` | Get job status |
| `POST` | `/upload/{category}` | Upload data files |
| `GET` | `/results/{run_id}` | List result files |
| `GET` | `/results/{run_id}/download/{path}` | Download result file |
| `GET` | `/presets` | List available presets |
| `GET` | `/health` | Health check |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNPLANNER_OUTPUT_DIR` | `/data/synplanner` | Base output directory |
| `SYNPLANNER_NUM_CPUS` | `4` | CPUs for parallel processing |
| `SYNPLANNER_BATCH_SIZE` | `100` | Batch size for data processing |
| `SYNPLANNER_DEVICE` | `cpu` | PyTorch device (`cpu`/`cuda`) |
| `SYNPLANNER_TARGETS_DIR` | `/data/synplanner/targets` | Directory watched by file sensor |
| `DAGSTER_PG_HOST` | - | PostgreSQL host |
| `DAGSTER_PG_PORT` | `5432` | PostgreSQL port |
| `DAGSTER_PG_USER` | - | PostgreSQL user |
| `DAGSTER_PG_PASSWORD` | - | PostgreSQL password |
| `DAGSTER_PG_DB` | - | PostgreSQL database |

## Development

```bash
# Install dependencies
pip install -e ".[dev]"
pip install dagster dagster-webserver dagster-postgres fastapi uvicorn httpx

# Run Dagster dev server (local, no Docker)
dagster dev -m dagster_synplanner.repository

# Run API server
uvicorn dagster_synplanner.api.app:app --reload

# Run frontend
cd dagster_synplanner/frontend
npm install && npm run dev
```

## Project Structure

```
dagster_synplanner/
├── ops/                    # Dagster ops wrapping SynPlanner tasks
│   ├── data_processing.py  # Standardization, filtering, mapping
│   ├── model_training.py   # Rule extraction, policy/value training
│   ├── planning.py         # MCTS planning and clustering
│   └── data_download.py    # HuggingFace preset downloads
├── jobs/
│   └── pipelines.py        # Pre-built job definitions
├── resources/
│   └── config.py           # SynPlannerResource (shared config)
├── sensors/
│   └── file_sensors.py     # File-watching automation
├── schedules/              # (extensible) Periodic jobs
├── api/
│   ├── app.py              # FastAPI application
│   └── models.py           # Request/response schemas
├── frontend/               # React + Vite + TypeScript UI
│   └── src/
│       ├── App.tsx
│       ├── api.ts           # API client
│       └── pages/           # JobList, SubmitJob, JobDetail
├── deploy/
│   ├── docker-compose.yml   # Local development stack
│   ├── dagster.yaml         # Dagster instance config
│   ├── workspace.yaml       # Dagster workspace config
│   └── helm/synplanner/     # Kubernetes Helm chart
├── repository.py            # Dagster Definitions entry point
└── Dockerfile               # Multi-stage build
```
