<!--
  Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
  Date   : 26 March 2026
-->
# TechTrainer AI — Setup and Run Guide

**Project:** Applied AI
**Date:** 2026-03-28

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Python 3.10 or higher** — `python --version`
- [ ] **uv package manager** — [installation instructions below](#1-install-uv)
- [ ] **AWS account** with Amazon Bedrock access
- [ ] **Bedrock model access enabled** for:
  - `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - `amazon.titan-embed-text-v2:0` (Titan Embeddings v2)
- [ ] **AWS IAM user** with the following permissions:
  - `bedrock:InvokeModel`
  - `bedrock:InvokeModelWithResponseStream`
  - `bedrock:ListFoundationModels`

---

## Step 1 — Install uv

uv is a fast Python package manager. Install it once on your machine.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

---

## Step 2 — Initialize the Project

Navigate to the project directory and install all dependencies:

```bash
cd D:/Downloads/Projects/ai_arena/techtrainer-ai

# Install all dependencies from pyproject.toml
uv sync
```

This creates a `.venv` virtual environment and installs:
- `strands-agents` — core agent framework
- `strands-agents-tools` — community tools (retrieve, file operations)
- `strands-agents-evals` — evaluation SDK
- `streamlit` — web UI
- `chromadb` — local vector store
- `boto3` — AWS SDK
- `pypdf`, `python-docx`, `python-pptx`, `openpyxl` — document parsers
- `langchain-text-splitters` — text chunking
- `pydantic` — data validation
- `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-cov` — testing

---

## Step 3 — Enable Bedrock Models

Before using the app, you must request access to the required models in the AWS Console.

1. Go to **AWS Console → Amazon Bedrock → Model access**
2. Click **"Manage model access"**
3. Enable the following models:
   - **Anthropic Claude Sonnet 4** (cross-region inference)
   - **Amazon Titan Text Embeddings V2**
4. Wait for access to be granted (usually instant for Titan, may take minutes for Claude)

**Verify access via AWS CLI:**
```bash
aws bedrock list-foundation-models --region us-east-1 \
  --query "modelSummaries[?modelId=='amazon.titan-embed-text-v2:0'].modelId"
```

---

## Step 4 — Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in a text editor and set:

```env
# ─── REQUIRED ────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=AKIA...your_key_here...
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# ─── OPTIONAL (defaults shown) ───────────────────────────────────────────────
# Change the region inference prefix if your AWS region differs
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# Embedding model for document indexing
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Bedrock Guardrails — leave empty to disable content filtering
BEDROCK_GUARDRAIL_ID=
BEDROCK_GUARDRAIL_VERSION=1

# Local storage paths
CHROMA_PERSIST_DIR=./data/chroma
SESSIONS_DIR=./data/sessions
GENERATED_DIR=./data/generated
PROGRESS_DB=./data/progress.db

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

**Finding your AWS credentials:**
1. Go to **AWS Console → IAM → Users → Your User → Security credentials**
2. Click **"Create access key"**
3. Copy the Access Key ID and Secret Access Key

**Important:** Never commit `.env` to version control. It is listed in `.gitignore`.

---

## Step 5 — Add Training Documents

Place your training materials in the appropriate folder:

```
data/
└── documents/
    ├── qa/          ← QA testing materials go here
    │   ├── selenium_webdriver_guide.pdf
    │   ├── tosca_user_manual.docx
    │   ├── playwright_documentation.pdf
    │   └── ...
    └── etl/         ← Data engineering materials go here
        ├── aws_glue_developer_guide.pdf
        ├── apache_spark_tutorials.docx
        ├── dbt_documentation.pdf
        ├── informatica_guide.pdf
        └── ...
```

**Supported formats:**
| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | All text extracted per page |
| Word Document | `.docx` | Paragraphs and tables |
| PowerPoint | `.pptx` | Slide text and notes |
| Excel | `.xlsx` | All sheets as text |
| Plain text | `.txt` | Direct read |
| Markdown | `.md` | Direct read |

**Tip:** For best results, use text-based PDFs (not scanned images). OCR support can be added via the `pytesseract` library if needed.

---

## Step 6 — Index Training Documents

Run the ingestion pipeline to parse, embed, and index all documents:

```bash
uv run python -m src.tools.document_ingestion --reindex
```

**What this does:**
1. Walks `./data/documents/qa/` and `./data/documents/etl/`
2. Parses each document by file format
3. Splits text into 1000-token chunks with 200-token overlap
4. Calls Bedrock Titan Embeddings to create vector embeddings
5. Stores embeddings in ChromaDB at `./data/chroma/`

**Expected output:**
```
[INFO] Starting document ingestion...
[INFO] Found 8 files in ./data/documents/qa/
[INFO] Found 6 files in ./data/documents/etl/
[INFO] Parsing selenium_webdriver_guide.pdf... 1,247 chunks
[INFO] Parsing tosca_user_manual.docx... 834 chunks
...
[INFO] Indexing complete. Total chunks: 8,432
[INFO] ChromaDB collection 'qa_training': 4,891 chunks
[INFO] ChromaDB collection 'etl_training': 3,541 chunks
```

**Re-indexing after adding new documents:**
```bash
uv run python -m src.tools.document_ingestion --reindex
```

**Check index without re-indexing:**
```bash
uv run python -m src.tools.document_ingestion --status
```

---

## Step 7 — Run the Application

```bash
uv run streamlit run app.py
```

The app will start and open in your browser at:
```
http://localhost:8501
```

**What you'll see on first launch:**
1. Sidebar with empty progress dashboard
2. Chat tab active and ready for questions
3. All four tabs accessible: Chat, Quiz, Learning Path, Content Author

---

## Step 8 — Verify the Setup

Run through this quick smoke test:

1. **Chat tab:** Type `"What is Selenium WebDriver?"` — you should get a detailed answer with document sources cited
2. **Quiz tab:** Select "Selenium" + "Beginner" + 5 questions → click Generate → MCQs should appear
3. **Learning Path tab:** Click "Refresh My Path" → beginner-level recommendations appear
4. **Content Author tab:** Enter "Introduction to Playwright" + Playwright + Beginner → click Generate → Markdown module appears

If all four work, the setup is complete.

---

## Running Tests

### Unit and Integration Tests (no AWS required)
```bash
uv run pytest tests/ -v \
  --ignore=tests/test_evals.py \
  --ignore=tests/test_e2e.py
```

### With Coverage Report
```bash
uv run pytest tests/ -v \
  --ignore=tests/test_evals.py \
  --ignore=tests/test_e2e.py \
  --cov=src \
  --cov-report=html
# Open htmlcov/index.html in browser
```

### Evaluation Tests (requires AWS + training documents)
```bash
uv run pytest tests/test_evals.py -v -s
```

### Full End-to-End Tests (requires AWS + training documents)
```bash
uv run pytest tests/test_e2e.py -v --slow
```

---

## Common Issues and Fixes

### Issue: `NoCredentialsError` or `InvalidClientTokenId`
**Cause:** AWS credentials not configured or incorrect.
**Fix:** Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` are correct.

```bash
# Verify credentials work
uv run python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

---

### Issue: `AccessDeniedException` on Bedrock
**Cause:** Bedrock model access not enabled, or IAM user lacks permissions.
**Fix:**
1. Go to AWS Console → Bedrock → Model access → Enable required models
2. Attach `AmazonBedrockFullAccess` policy to your IAM user (or a custom policy with `bedrock:InvokeModel`)

---

### Issue: `ValidationException: The provided model identifier is invalid`
**Cause:** Model ID prefix doesn't match your AWS region.
**Fix:** Update `BEDROCK_MODEL_ID` in `.env`:

| AWS Region | Model ID Prefix |
|---|---|
| us-east-1, us-west-2 | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| eu-west-1, eu-central-1 | `eu.anthropic.claude-sonnet-4-20250514-v1:0` |
| ap-northeast-1, ap-southeast-2 | `ap.anthropic.claude-sonnet-4-20250514-v1:0` |

---

### Issue: ChromaDB `Collection not found`
**Cause:** Documents haven't been indexed yet, or `CHROMA_PERSIST_DIR` path changed.
**Fix:**
```bash
uv run python -m src.tools.document_ingestion --reindex
```

---

### Issue: `streamlit: command not found`
**Cause:** Running `streamlit` directly instead of through uv.
**Fix:** Always use `uv run streamlit run app.py` (not just `streamlit run app.py`)

---

### Issue: Slow response times (>15 seconds)
**Cause:** Bedrock throttling or large context window.
**Fix:**
- Check your Bedrock quota in AWS Console → Service Quotas → Bedrock
- Reduce `window_size` in `SlidingWindowConversationManager` in `src/config.py`
- Use `us.anthropic.claude-haiku-4-5-20251001` for faster (but less capable) responses

---

### Issue: Session not persisting after browser refresh
**Cause:** Streamlit session ID changes on refresh.
**Fix:** The app uses a stable session ID stored in `st.session_state`. If you need truly persistent cross-browser sessions, set a custom `SESSION_ID` environment variable.

---

## Production Deployment Notes

For deploying beyond a local machine (e.g., internal server):

```bash
# Run on a specific port with external access
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Run as background process
nohup uv run streamlit run app.py > logs/app.log 2>&1 &

# Or use a process manager like PM2 or systemd
```

For Docker deployment, a `Dockerfile` can be added with:
```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY . .
RUN uv sync --no-dev
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

## Updating Training Materials

1. Add/replace files in `data/documents/qa/` or `data/documents/etl/`
2. Run re-indexing:
   ```bash
   uv run python -m src.tools.document_ingestion --reindex
   ```
3. No app restart needed — ChromaDB is queried at request time

---

## Directory Permissions

Ensure the following directories are writable by the process running the app:
- `./data/chroma/` — ChromaDB vector store
- `./data/sessions/` — conversation session files
- `./data/generated/` — AI-generated training modules
- `./data/` — SQLite database file
