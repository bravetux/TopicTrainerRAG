"""Provider configuration and model factory for TechTrainer AI.

Supports: AWS Bedrock, Local Ollama, LM Studio, OpenRouter,
          Google Gemini, OpenAI / ChatGPT, Custom OpenAI-compatible endpoints.

Priority (high → low):  .env  >  data/settings.json  >  built-in defaults
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Overridable by tests via monkeypatch
SETTINGS_PATH: str = "./data/settings.json"

DEFAULTS: dict = {
    "active_provider": "bedrock",
    "model_name": "",
    "base_url": "",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 32768,
    "topic_classifications": {},
}

PROVIDER_LABELS: dict = {
    "bedrock": "AWS Bedrock",
    "ollama": "Local Ollama",
    "lmstudio": "LM Studio",
    "openrouter": "OpenRouter",
    "gemini": "Google Gemini",
    "openai": "OpenAI / ChatGPT",
    "custom": "Custom",
}

PROVIDER_IDS: list = list(PROVIDER_LABELS.keys())


def load_settings() -> dict:
    """Read data/settings.json. Returns defaults if the file is missing or corrupt."""
    try:
        path = Path(SETTINGS_PATH)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in DEFAULTS.items():
                data.setdefault(k, v)
            return data
    except Exception as exc:
        logger.warning("Could not load settings.json: %s — using defaults.", exc)
    return dict(DEFAULTS)


_SECRET_KEYS = frozenset({
    "aws_access_key_id", "aws_secret_access_key", "llm_api_key",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "LLM_API_KEY",
    "embedding_api_key", "EMBEDDING_API_KEY",
})

def save_settings(settings: dict) -> None:
    """Write settings dict to data/settings.json (non-sensitive values only).

    Secret keys (credentials) are stripped before writing.
    """
    safe = {k: v for k, v in settings.items() if k not in _SECRET_KEYS}
    path = Path(SETTINGS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    logger.info("Settings saved to %s", SETTINGS_PATH)


def write_env_values(updates: dict) -> None:
    """Write key=value pairs to the .env file using dotenv.set_key.

    Preserves existing entries and comments. Creates .env if absent.
    """
    from dotenv import set_key, find_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        dotenv_path = ".env"
        Path(dotenv_path).touch()
    for key, value in updates.items():
        set_key(dotenv_path, key, str(value) if value is not None else "")
    logger.info("Updated .env with keys: %s", list(updates.keys()))


def get_effective_config() -> dict:
    """Merge defaults → settings.json → .env and return the active configuration.

    .env values always win over settings.json which always win over defaults.
    """
    import os

    settings = load_settings()
    cfg: dict = dict(DEFAULTS)

    # Layer 2: settings.json (non-sensitive runtime config)
    cfg["active_provider"] = settings.get("active_provider", cfg["active_provider"])
    cfg["llm_model"] = settings.get("model_name", "")
    cfg["llm_base_url"] = settings.get("base_url", "")
    cfg["temperature"] = float(settings.get("temperature", cfg["temperature"]))
    cfg["top_p"] = float(settings.get("top_p", cfg["top_p"]))
    cfg["max_tokens"] = int(settings.get("max_tokens", cfg["max_tokens"]))

    # Layer 1: .env (reload to pick up runtime writes, but don't override
    # already-set environment variables so process env / monkeypatch wins)
    from dotenv import load_dotenv
    load_dotenv(override=False)

    _apply_env(cfg, os)

    # Set meaningful defaults before falling back to empty strings
    cfg.setdefault("aws_region", "us-east-1")
    cfg.setdefault("bedrock_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    cfg.setdefault("bedrock_guardrail_version", "1")

    # Ensure all other credential keys exist (empty string if not set)
    for key in ("bedrock_guardrail_id", "aws_access_key_id",
                "aws_secret_access_key", "llm_api_key", "llm_base_url", "llm_model"):
        cfg.setdefault(key, "")

    # Validate provider; fall back to bedrock on unrecognised value
    if cfg["active_provider"] not in PROVIDER_IDS:
        logger.warning("Unknown provider '%s' — falling back to bedrock.", cfg["active_provider"])
        cfg["active_provider"] = "bedrock"

    cfg["provider_label"] = PROVIDER_LABELS[cfg["active_provider"]]
    return cfg


def _apply_env(cfg: dict, os_mod) -> None:
    """Apply non-empty environment variables onto cfg in-place."""
    _map = {
        "ACTIVE_PROVIDER": ("active_provider", str),
        "AWS_REGION": ("aws_region", str),
        "AWS_DEFAULT_REGION": ("aws_region", str),
        "AWS_ACCESS_KEY_ID": ("aws_access_key_id", str),
        "AWS_SECRET_ACCESS_KEY": ("aws_secret_access_key", str),
        "BEDROCK_MODEL_ID": ("bedrock_model_id", str),
        "BEDROCK_GUARDRAIL_ID": ("bedrock_guardrail_id", str),
        "BEDROCK_GUARDRAIL_VERSION": ("bedrock_guardrail_version", str),
        "LLM_API_KEY": ("llm_api_key", str),
        "LLM_BASE_URL": ("llm_base_url", str),
        "LLM_MODEL": ("llm_model", str),
        "AGENT_TEMPERATURE": ("temperature", float),
        "AGENT_TOP_P": ("top_p", float),
        "AGENT_MAX_TOKENS": ("max_tokens", int),
        "EMBEDDING_PROVIDER": ("embedding_provider", str),
        "EMBEDDING_MODEL": ("embedding_model", str),
        "EMBEDDING_BASE_URL": ("embedding_base_url", str),
    }
    for env_key, (cfg_key, cast) in _map.items():
        val = os_mod.getenv(env_key)
        if val:
            try:
                cfg[cfg_key] = cast(val)
            except (ValueError, TypeError) as exc:
                logger.warning("Could not apply env var %s: %s", env_key, exc)


def _resolve_litellm_model(provider: str, cfg: dict) -> str:
    """Build the LiteLLM model_id string for a given provider."""
    model = cfg.get("llm_model", "").strip()
    if provider == "ollama":
        return f"ollama/{model or 'llama3.2'}"
    if provider == "lmstudio":
        return f"openai/{model or 'local-model'}"
    if provider == "openrouter":
        return f"openrouter/{model or 'anthropic/claude-3-sonnet'}"
    if provider == "gemini":
        return f"gemini/{model or 'gemini-1.5-pro'}"
    if provider == "openai":
        return f"openai/{model or 'gpt-4o'}"
    if provider == "custom":
        return f"openai/{model or 'local-model'}"
    return f"openai/{model or 'gpt-4o'}"


# Lazy shim so patch("src.tools.provider_manager.BedrockModel") works in tests
def BedrockModel(**kwargs):  # type: ignore[override]
    from strands.models.bedrock import BedrockModel as _BM
    return _BM(**kwargs)


def get_model(temperature: Optional[float] = None):
    """Return the configured AI model instance.

    Args:
        temperature: Optional per-agent override. When given, replaces the
                     global temperature for this model instance only.

    Returns:
        BedrockModel for AWS Bedrock, LiteLLMModel for all other providers.
    """
    cfg = get_effective_config()
    provider = cfg["active_provider"]
    effective_temp = temperature if temperature is not None else cfg["temperature"]

    if provider == "bedrock":
        kwargs: dict = {
            "model_id": cfg["bedrock_model_id"],
            "region_name": cfg["aws_region"],
            "temperature": effective_temp,
            "top_p": cfg["top_p"],
            "max_tokens": cfg["max_tokens"],
            "streaming": True,
        }
        if cfg.get("bedrock_guardrail_id"):
            kwargs["guardrail_id"] = cfg["bedrock_guardrail_id"]
            kwargs["guardrail_version"] = cfg["bedrock_guardrail_version"]
        return BedrockModel(**kwargs)

    # All non-Bedrock providers use LiteLLM
    try:
        from strands.models.litellm import LiteLLMModel
    except ImportError as exc:
        raise ImportError(
            "litellm package is required for non-Bedrock providers. "
            "Run: uv add litellm"
        ) from exc

    model_id = _resolve_litellm_model(provider, cfg)
    litellm_kwargs: dict = {
        "model_id": model_id,
        "temperature": effective_temp,
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "streaming": True,
    }
    if cfg.get("llm_api_key"):
        litellm_kwargs["api_key"] = cfg["llm_api_key"]
    if cfg.get("llm_base_url"):
        litellm_kwargs["base_url"] = cfg["llm_base_url"]
    return LiteLLMModel(**litellm_kwargs)


def test_connection() -> dict:
    """Test connectivity to the configured provider.

    Returns:
        {"ok": bool, "message": str, "latency_ms": int | None}
    """
    cfg = get_effective_config()
    provider = cfg["active_provider"]
    t0 = time.time()
    try:
        if provider == "bedrock":
            return _test_bedrock(cfg, t0)
        if provider == "ollama":
            return _test_http_get(cfg.get("llm_base_url") or "http://localhost:11434", "/api/tags", {}, t0)
        if provider == "lmstudio":
            base = cfg.get("llm_base_url") or "http://localhost:1234/v1"
            return _test_http_get(base, "/models", {}, t0)
        if provider == "openrouter":
            headers = {}
            if cfg.get("llm_api_key"):
                headers["Authorization"] = f"Bearer {cfg['llm_api_key']}"
            return _test_http_get("https://openrouter.ai/api/v1", "/models", headers, t0)
        if provider == "openai":
            base = (cfg.get("llm_base_url") or "https://api.openai.com").rstrip("/")
            headers = {}
            if cfg.get("llm_api_key"):
                headers["Authorization"] = f"Bearer {cfg['llm_api_key']}"
            return _test_http_get(base, "/v1/models", headers, t0)
        if provider == "gemini":
            return _test_gemini(cfg, t0)
        if provider == "custom":
            base = cfg.get("llm_base_url", "").rstrip("/")
            if not base:
                return {"ok": False, "message": "Base URL is required for Custom provider", "latency_ms": None}
            headers = {}
            if cfg.get("llm_api_key"):
                headers["Authorization"] = f"Bearer {cfg['llm_api_key']}"
            return _test_http_get(base, "/models", headers, t0)
        return {"ok": False, "message": f"Unknown provider: {provider}", "latency_ms": None}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "latency_ms": None}


def _test_bedrock(cfg: dict, t0: float) -> dict:
    import json as _json
    import boto3
    session_kwargs: dict = {"region_name": cfg["aws_region"]}
    if cfg.get("aws_access_key_id") and cfg.get("aws_secret_access_key"):
        session_kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        session_kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
    client = boto3.client("bedrock-runtime", **session_kwargs)
    client.invoke_model(
        modelId=cfg["bedrock_model_id"],
        body=_json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hi"}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    ms = int((time.time() - t0) * 1000)
    return {"ok": True, "message": f"Connected · {ms}ms", "latency_ms": ms}


def _test_http_get(base_url: str, path: str, headers: dict, t0: float) -> dict:
    import urllib.request
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    ms = int((time.time() - t0) * 1000)
    return {"ok": True, "message": f"Connected · {ms}ms", "latency_ms": ms}


def _test_gemini(cfg: dict, t0: float) -> dict:
    import urllib.request
    api_key = cfg.get("llm_api_key", "")
    model = (cfg.get("llm_model") or "gemini-1.5-pro").replace("gemini/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
    req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    ms = int((time.time() - t0) * 1000)
    return {"ok": True, "message": f"Connected · {ms}ms", "latency_ms": ms}
