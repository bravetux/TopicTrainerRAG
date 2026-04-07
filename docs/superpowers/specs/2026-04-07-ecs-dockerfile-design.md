# ECS Dockerfile Design — TechTrainer AI

**Date:** 2026-04-07  
**Author:** B.Vignesh Kumar  
**Status:** Approved

---

## Overview

Add a production-ready `Dockerfile` and `.dockerignore` to run TechTrainer AI on AWS ECS (EC2 launch type). No application code changes are required.

---

## Architecture

### Container Image

| Aspect | Decision |
|---|---|
| Base image | `python:3.11-slim` |
| Package manager | `uv` copied from `ghcr.io/astral-sh/uv:latest` |
| Dependency install | `uv sync --frozen --no-dev` (uses `uv.lock` for exact reproducibility) |
| Working directory | `/app` |
| Runtime user | Non-root `appuser` (UID 1000) — ECS security best practice |
| Exposed port | `8501` (Streamlit default) |
| Health check | `GET http://localhost:8501/_stcore/health` (Streamlit built-in endpoint) |
| Entrypoint | `uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501` |

### Artifacts

- `Dockerfile` — in project root
- `.dockerignore` — in project root

---

## Data Persistence

All persistent data lives under `/app/data` inside the container:

| Path | Content |
|---|---|
| `/app/data/chroma/` | ChromaDB vector store |
| `/app/data/sessions/` | Chat session files |
| `/app/data/generated/` | AI-generated training modules |
| `/app/data/progress.db` | SQLite progress database |
| `/app/data/documents/` | Uploaded training documents |

An **Amazon EFS volume** is mounted at `/app/data` via the ECS task definition (`efsVolumeConfiguration`). `src/config.py` uses `./data/...` paths which resolve correctly to `/app/data/...` since `WORKDIR` is `/app` — no code changes needed.

---

## AWS Credential Handling

The Dockerfile contains no credentials. Two patterns are supported at the ECS task definition level:

### Pattern 1 — IAM Task Role (recommended)
- Assign an IAM role to the ECS task with permissions:
  - `bedrock:InvokeModel`
  - `bedrock:InvokeModelWithResponseStream`
  - `bedrock:ListFoundationModels`
- boto3 automatically picks up credentials via the ECS container metadata endpoint.
- Only `AWS_REGION` is passed as a plain environment variable in the task definition.

### Pattern 2 — AWS Secrets Manager / Parameter Store
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` stored as secrets in Secrets Manager or SSM Parameter Store.
- Injected via the `secrets` block in the ECS task definition (never in plaintext env vars).
- `AWS_REGION` passed as a plain environment variable.

Both patterns work with zero application code changes — `src/config.py` reads all values via `os.getenv()`.

---

## .dockerignore

The following are excluded from the image build context to keep the image lean and prevent credential leakage:

- `.venv/` — not needed; `uv sync` installs deps at build time
- `data/` — runtime data; provided by EFS at runtime
- `.env` — local credentials must never enter the image
- `tests/` — not needed in production
- `.git/`, `__pycache__/`, `*.pyc` — build artifacts
- `.claude/`, `.superpowers/` — tooling artifacts

---

## ECS Task Definition Notes (out of scope for Dockerfile)

These are configured in the task definition, not the Dockerfile:

- **EFS volume:** `efsVolumeConfiguration` pointing to your EFS file system ID, mounted at `/app/data`
- **CPU / Memory:** Minimum recommended 1 vCPU / 2 GB RAM for Streamlit + ChromaDB + Bedrock calls
- **Port mapping:** Host port 8501 → Container port 8501
- **Log driver:** `awslogs` to CloudWatch Logs
- **Secrets injection:** (Pattern 2 only) reference Secrets Manager ARNs in the `secrets` block

---

## Out of Scope

- ECS task definition JSON / Terraform — infrastructure provisioning
- EFS filesystem creation and access point configuration
- ECR repository setup and image push pipeline (CI/CD)
- ALB / target group configuration for the service
