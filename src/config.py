"""Central configuration loaded from environment variables."""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── AWS / Bedrock ──────────────────────────────────────────────────────────────
AWS_REGION: str = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID: str = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
BEDROCK_EMBEDDING_MODEL: str = os.getenv(
    "BEDROCK_EMBEDDING_MODEL",
    "amazon.titan-embed-text-v2:0"
)
BEDROCK_GUARDRAIL_ID: str = os.getenv("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION: str = os.getenv("BEDROCK_GUARDRAIL_VERSION", "1")

# ── Local Storage ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
SESSIONS_DIR: str = os.getenv("SESSIONS_DIR", "./data/sessions")
GENERATED_DIR: str = os.getenv("GENERATED_DIR", "./data/generated")
PROGRESS_DB: str = os.getenv("PROGRESS_DB", "./data/progress.db")
TOPICS_REGISTRY_PATH: str = os.getenv("TOPICS_REGISTRY_PATH", "./data/topics_registry.json")

# ── Built-in Topics ────────────────────────────────────────────────────────────
BUILTIN_TOPICS: list[dict] = [
    {"id": "selenium",    "display_name": "Selenium",           "collection": "tech_selenium",    "doc_dir": "./data/documents/selenium"},
    {"id": "tosca",       "display_name": "Tosca",              "collection": "tech_tosca",       "doc_dir": "./data/documents/tosca"},
    {"id": "playwright",  "display_name": "Playwright",         "collection": "tech_playwright",  "doc_dir": "./data/documents/playwright"},
    {"id": "aws_glue",    "display_name": "AWS Glue",           "collection": "tech_aws_glue",    "doc_dir": "./data/documents/aws_glue"},
    {"id": "spark",       "display_name": "Spark",              "collection": "tech_spark",       "doc_dir": "./data/documents/spark"},
    {"id": "dbt",         "display_name": "dbt",                "collection": "tech_dbt",         "doc_dir": "./data/documents/dbt"},
    {"id": "informatica", "display_name": "Informatica",        "collection": "tech_informatica", "doc_dir": "./data/documents/informatica"},
    {"id": "ssis",        "display_name": "SSIS",               "collection": "tech_ssis",        "doc_dir": "./data/documents/ssis"},
    {"id": "talend",      "display_name": "Talend",             "collection": "tech_talend",      "doc_dir": "./data/documents/talend"},
    {"id": "adf",         "display_name": "Azure Data Factory", "collection": "tech_adf",         "doc_dir": "./data/documents/adf"},
]

QA_TOPIC_IDS: list[str] = ["selenium", "tosca", "playwright"]
ETL_TOPIC_IDS: list[str] = ["aws_glue", "spark", "dbt", "informatica", "ssis", "talend", "adf"]

# ── Backward Compatibility (deprecated, will be removed) ──────────────────────
# These are maintained for tasks 4 and 5 refactoring. Tasks should use BUILTIN_TOPICS.
CHROMA_QA_COLLECTION: str = "qa_training"
CHROMA_ETL_COLLECTION: str = "etl_training"
DOCUMENTS_QA_DIR: str = "./data/documents/qa"
DOCUMENTS_ETL_DIR: str = "./data/documents/etl"

# ── Agent Settings ─────────────────────────────────────────────────────────────
ORCHESTRATOR_WINDOW_SIZE: int = 10
QA_AGENT_WINDOW_SIZE: int = 15
ETL_AGENT_WINDOW_SIZE: int = 15
MAX_TOOLS_PER_TURN: int = 10
SUBAGENT_TIMEOUT_SECONDS: int = 60
AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
AGENT_TOP_P: float = float(os.getenv("AGENT_TOP_P", "0.9"))
AGENT_MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))

# ── Ingestion Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
RETRIEVAL_TOP_K: int = 5

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    handlers=[logging.StreamHandler()],
)

# ── Ensure runtime directories exist ──────────────────────────────────────────
for _dir in [SESSIONS_DIR, GENERATED_DIR, CHROMA_PERSIST_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)

for _topic in BUILTIN_TOPICS:
    Path(_topic["doc_dir"]).mkdir(parents=True, exist_ok=True)
