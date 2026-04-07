# ECS Dockerfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-ready `Dockerfile` and `.dockerignore` so TechTrainer AI can be built and run as a container on AWS ECS (EC2 launch type).

**Architecture:** Single-stage image based on `python:3.11-slim` with `uv` copied in from its official image. Dependencies are installed at build time via `uv sync --frozen --no-dev`. The `/app/data` directory is a mount point for an EFS volume configured in the ECS task definition — the image itself creates the empty directory. AWS credentials are never baked into the image; they are provided at runtime via IAM Task Role or Secrets Manager injection.

**Tech Stack:** Docker, Python 3.11-slim, uv, Streamlit 1.35+, AWS ECS (EC2), Amazon EFS

---

### Task 1: Create `.dockerignore`

**Files:**
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore` in the project root**

```
# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Runtime data — provided by EFS at runtime, must not be baked into image
data/

# Local credentials — must never enter the image
.env

# Test and dev artifacts
tests/
.pytest_cache/
htmlcov/
.coverage

# Version control and tooling
.git/
.gitignore
.claude/
.superpowers/

# Docs and scripts — not needed at runtime
docs/
scripts/
README.md
"use case.txt"
```

- [ ] **Step 2: Verify the ignore rules are correct**

Run from the project root (requires Docker installed):
```bash
docker build --no-cache --dry-run . 2>&1 | head -40
```

If Docker dry-run is unavailable, verify manually:
```bash
# These files must NOT appear in the build context:
# .env, .venv/, data/, tests/, .git/
# These files MUST be present in the context:
# app.py, pyproject.toml, uv.lock, src/
```

- [ ] **Step 3: Commit**

```bash
git add .dockerignore
git commit -m "chore: add .dockerignore for ECS image build"
```

---

### Task 2: Create `Dockerfile`

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create `Dockerfile` in the project root**

```dockerfile
# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 07 April 2026
# =============================================================================

# ── Stage: pull uv binary ─────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:latest AS uv

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Copy uv binary from official image
COPY --from=uv /uv /bin/uv

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy lockfile and project metadata first to leverage Docker layer caching.
# If pyproject.toml and uv.lock are unchanged, this layer is reused on rebuild.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app.py ./

# ── Data directory ────────────────────────────────────────────────────────────
# Create the mount point for the EFS volume.
# In ECS, this is overridden by the efsVolumeConfiguration in the task definition.
# At runtime the EFS volume at /app/data provides:
#   /app/data/chroma/      — ChromaDB vector store
#   /app/data/sessions/    — chat session files
#   /app/data/generated/   — AI-generated training modules
#   /app/data/progress.db  — SQLite progress database
#   /app/data/documents/   — uploaded training documents
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# ── Runtime user ──────────────────────────────────────────────────────────────
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# Streamlit exposes a built-in health endpoint at /_stcore/health.
# ECS uses this to determine whether the task is healthy.
#   --interval=30s   check every 30 seconds
#   --timeout=10s    fail the check if no response within 10 seconds
#   --start-period=30s  give Streamlit time to start before counting failures
#   --retries=3      mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --server.address 0.0.0.0   bind to all interfaces (required for ECS port mapping)
# --server.port 8501         explicit port
# --server.headless true     suppress browser-open prompt (no browser in container)
# --server.fileWatcherType none  disable inotify file watching (not needed in production)
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", \
            "--server.address", "0.0.0.0", \
            "--server.port", "8501", \
            "--server.headless", "true", \
            "--server.fileWatcherType", "none"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for AWS ECS EC2 deployment"
```

---

### Task 3: Verify the image builds

**Files:** (none created — verification only)

- [ ] **Step 1: Build the image**

```bash
docker build -t techtrainer-ai:local .
```

Expected output ends with:
```
Successfully built <image-id>
Successfully tagged techtrainer-ai:local
```

If build fails on `uv sync`, confirm `uv.lock` is present and not in `.dockerignore`:
```bash
grep "uv.lock" .dockerignore
# Should return nothing — uv.lock must NOT be ignored
```

- [ ] **Step 2: Verify image size is reasonable**

```bash
docker images techtrainer-ai:local
```

Expected: image size under 2 GB (typical range 800 MB–1.5 GB given chromadb and its native deps).

- [ ] **Step 3: Verify no credentials or data baked into the image**

```bash
# Check .env is not present
docker run --rm techtrainer-ai:local ls /app/.env 2>&1
# Expected: ls: cannot access '/app/.env': No such file or directory

# Check data/ is an empty mount point
docker run --rm techtrainer-ai:local ls /app/data/
# Expected: empty output (no files — EFS provides them at runtime)

# Check non-root user
docker run --rm techtrainer-ai:local whoami
# Expected: appuser
```

- [ ] **Step 4: Verify the health check endpoint responds**

Run the container locally with dummy AWS env vars (Bedrock calls will fail, but Streamlit will start):
```bash
docker run --rm -d \
  -p 8501:8501 \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=dummy \
  -e AWS_SECRET_ACCESS_KEY=dummy \
  --name techtrainer-test \
  techtrainer-ai:local

# Wait 30 seconds for Streamlit to start, then probe the health endpoint
sleep 30
curl -f http://localhost:8501/_stcore/health
# Expected: {"status": "ok"}

# Stop the test container
docker stop techtrainer-test
```

- [ ] **Step 5: Commit verification note**

No code to commit — verification passed. Proceed to Task 4.

---

### Task 4: Push image to ECR and register in ECS

**Files:** (none — operational steps)

> **Note:** These steps require AWS CLI configured with sufficient IAM permissions (`ecr:*`, `ecs:RegisterTaskDefinition`).

- [ ] **Step 1: Create an ECR repository (one-time)**

```bash
aws ecr create-repository \
  --repository-name techtrainer-ai \
  --region us-east-1
```

Note the `repositoryUri` from the output, e.g.:
```
123456789012.dkr.ecr.us-east-1.amazonaws.com/techtrainer-ai
```

- [ ] **Step 2: Authenticate Docker to ECR**

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.us-east-1.amazonaws.com
```

Expected: `Login Succeeded`

- [ ] **Step 3: Tag and push the image**

```bash
docker tag techtrainer-ai:local \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/techtrainer-ai:latest

docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/techtrainer-ai:latest
```

- [ ] **Step 4: ECS task definition — key fields to set**

In your ECS task definition (console or JSON), configure these fields:

**Container definition:**
```json
{
  "name": "techtrainer-ai",
  "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/techtrainer-ai:latest",
  "portMappings": [{ "containerPort": 8501, "hostPort": 8501, "protocol": "tcp" }],
  "environment": [
    { "name": "AWS_REGION", "value": "us-east-1" }
  ],
  "mountPoints": [
    { "sourceVolume": "techtrainer-data", "containerPath": "/app/data", "readOnly": false }
  ],
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/techtrainer-ai",
      "awslogs-region": "us-east-1",
      "awslogs-stream-prefix": "ecs"
    }
  },
  "healthCheck": {
    "command": ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')\" || exit 1"],
    "interval": 30,
    "timeout": 10,
    "startPeriod": 30,
    "retries": 3
  }
}
```

**Volume (EFS):**
```json
{
  "name": "techtrainer-data",
  "efsVolumeConfiguration": {
    "fileSystemId": "fs-xxxxxxxxx",
    "rootDirectory": "/",
    "transitEncryption": "ENABLED"
  }
}
```

**Credential injection — choose one:**

*Option A — IAM Task Role (recommended):*
Set `taskRoleArn` to an IAM role with:
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream",
    "bedrock:ListFoundationModels"
  ],
  "Resource": "*"
}
```
No secrets block needed — boto3 picks up credentials automatically.

*Option B — Secrets Manager:*
Add to the container definition `secrets` block:
```json
"secrets": [
  {
    "name": "AWS_ACCESS_KEY_ID",
    "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:techtrainer/aws-credentials:AWS_ACCESS_KEY_ID::"
  },
  {
    "name": "AWS_SECRET_ACCESS_KEY",
    "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:techtrainer/aws-credentials:AWS_SECRET_ACCESS_KEY::"
  }
]
```
The ECS task execution role must have `secretsmanager:GetSecretValue` permission on those ARNs.

- [ ] **Step 5: Verify the running task**

After launching an ECS service or standalone task with this definition:
```bash
# Check task health in ECS console or via CLI
aws ecs describe-tasks \
  --cluster <your-cluster-name> \
  --tasks <task-arn> \
  --query "tasks[0].containers[0].healthStatus"
# Expected: "HEALTHY"
```

Then open `http://<ec2-instance-public-ip>:8501` in a browser. TechTrainer AI should load.

---

## Notes for First ECS Run

- The EFS volume at `/app/data` will be empty on first run. The app creates required subdirectories on startup via `src/config.py` (`mkdir -p` calls).
- `data/settings.json` and `data/topics_registry.json` are generated by the app on first use — no manual seeding required.
- Training documents must be uploaded via the app UI or copied directly to the EFS filesystem at `/app/data/documents/<topic>/` before indexing.
