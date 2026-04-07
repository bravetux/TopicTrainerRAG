# Settings Tab & Multi-Provider AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `⚙️ Settings` tab to TechTrainer AI that lets users configure and switch AI providers (AWS Bedrock, Ollama, LM Studio, OpenRouter, Gemini, OpenAI/ChatGPT, Custom), save credentials to `.env`, test connections, and see live status in the sidebar. Change the app subtitle.

**Architecture:** New `src/tools/provider_manager.py` owns the provider config, `.env` write logic, `get_model()` factory, and `test_connection()`. All agents replace their hardcoded `BedrockModel(...)` calls with `get_model(temperature=X)`. `app.py` gains a 6th Settings tab and a sidebar connection indicator. `litellm` is added as a dependency for non-Bedrock providers.

**Tech Stack:** `strands-agents` (`BedrockModel` + `LiteLLMModel`), `litellm`, `python-dotenv` (already installed), Streamlit, `urllib` (stdlib, no new dep for HTTP tests).

---

## File Map

| File | Change |
|------|--------|
| `pyproject.toml` | Add `litellm>=1.0.0` dependency |
| `src/tools/provider_manager.py` | **New** — settings load/save, `.env` writer, `get_model()`, `test_connection()` |
| `tests/test_provider_manager.py` | **New** — unit tests for provider_manager |
| `src/agents/orchestrator.py` | Replace `_build_bedrock_model()` with `get_model()` |
| `src/agents/qa_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.3)` |
| `src/agents/etl_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.3)` |
| `src/agents/quiz_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.5)` |
| `src/agents/learning_path_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.2)` |
| `src/agents/content_author_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.6)` |
| `src/agents/progress_agent.py` | Replace `BedrockModel(...)` with `get_model(temperature=0.1)` |
| `app.py` | Change subtitle; add connection check at session start; sidebar status; 6th Settings tab |

---

### Task 1: Add `litellm` dependency and create `provider_manager.py`

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tools/provider_manager.py`
- Create: `tests/test_provider_manager.py`

- [ ] **Step 1: Add litellm to pyproject.toml**

In `pyproject.toml`, in the `dependencies` list, add `"litellm>=1.0.0",` after the `python-dotenv` line:

```toml
    "python-dotenv>=1.0.0",
    "litellm>=1.0.0",
```

- [ ] **Step 2: Install the new dependency**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv sync`
Expected: Resolves and installs `litellm` and its transitive deps. No errors.

- [ ] **Step 3: Verify LiteLLMModel is importable from Strands**

Run: `uv run python -c "from strands.models.litellm import LiteLLMModel; print('LiteLLMModel OK')"`
Expected: `LiteLLMModel OK`

If this fails (older strands version), run: `uv run python -c "import strands.models; print(dir(strands.models))"` to find the correct import path and use that path throughout this plan instead of `strands.models.litellm`.

- [ ] **Step 4: Write failing tests**

Create `tests/test_provider_manager.py`:

```python
"""Tests for provider_manager — settings, config merging, model resolution."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadSettings:
    def test_returns_defaults_when_file_missing(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        result = pm.load_settings()
        assert result["active_provider"] == "bedrock"
        assert result["temperature"] == 0.3
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 4096

    def test_loads_saved_provider(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "ollama", "temperature": 0.7,
            "top_p": 0.9, "max_tokens": 2048, "model_name": "llama3", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        result = pm.load_settings()
        assert result["active_provider"] == "ollama"
        assert result["temperature"] == 0.7

    def test_returns_defaults_on_corrupt_json(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("not valid {{{ json")
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        result = pm.load_settings()
        assert result["active_provider"] == "bedrock"


class TestSaveSettings:
    def test_writes_json_file(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        pm.save_settings({
            "active_provider": "gemini", "temperature": 0.5,
            "top_p": 0.9, "max_tokens": 4096, "model_name": "gemini-1.5-pro", "base_url": "",
        })
        data = json.loads((tmp_path / "settings.json").read_text())
        assert data["active_provider"] == "gemini"
        assert data["model_name"] == "gemini-1.5-pro"


class TestResolveLitellmModel:
    def test_ollama_with_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("ollama", {"llm_model": "llama3.2"}) == "ollama/llama3.2"

    def test_ollama_default_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("ollama", {"llm_model": ""}) == "ollama/llama3.2"

    def test_lmstudio_no_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("lmstudio", {"llm_model": ""}) == "openai/local-model"

    def test_lmstudio_with_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("lmstudio", {"llm_model": "mistral-7b"}) == "openai/mistral-7b"

    def test_openrouter(self):
        from src.tools.provider_manager import _resolve_litellm_model
        result = _resolve_litellm_model("openrouter", {"llm_model": "anthropic/claude-3-sonnet"})
        assert result == "openrouter/anthropic/claude-3-sonnet"

    def test_openrouter_default(self):
        from src.tools.provider_manager import _resolve_litellm_model
        result = _resolve_litellm_model("openrouter", {"llm_model": ""})
        assert result == "openrouter/anthropic/claude-3-sonnet"

    def test_gemini(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("gemini", {"llm_model": "gemini-1.5-pro"}) == "gemini/gemini-1.5-pro"

    def test_openai(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("openai", {"llm_model": "gpt-4o"}) == "openai/gpt-4o"

    def test_custom_with_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("custom", {"llm_model": "my-model"}) == "openai/my-model"

    def test_custom_no_model(self):
        from src.tools.provider_manager import _resolve_litellm_model
        assert _resolve_litellm_model("custom", {"llm_model": ""}) == "openai/local-model"


class TestGetEffectiveConfig:
    def test_env_overrides_settings(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "ollama", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.setenv("ACTIVE_PROVIDER", "gemini")
        result = pm.get_effective_config()
        assert result["active_provider"] == "gemini"

    def test_provider_label_derived(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "bedrock", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.delenv("ACTIVE_PROVIDER", raising=False)
        result = pm.get_effective_config()
        assert result["provider_label"] == "AWS Bedrock"

    def test_unknown_provider_falls_back_to_bedrock(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "nonexistent", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.delenv("ACTIVE_PROVIDER", raising=False)
        result = pm.get_effective_config()
        assert result["active_provider"] == "bedrock"

    def test_temperature_env_override(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "bedrock", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.setenv("AGENT_TEMPERATURE", "0.8")
        monkeypatch.delenv("ACTIVE_PROVIDER", raising=False)
        result = pm.get_effective_config()
        assert result["temperature"] == 0.8


class TestGetModel:
    def test_bedrock_provider_returns_bedrock_model(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "bedrock", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.delenv("ACTIVE_PROVIDER", raising=False)

        mock_model = MagicMock()
        with patch("src.tools.provider_manager.BedrockModel", return_value=mock_model) as mock_cls:
            result = pm.get_model()
        mock_cls.assert_called_once()
        assert result is mock_model

    def test_temperature_override_passed_to_model(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "active_provider": "bedrock", "temperature": 0.3, "top_p": 0.9,
            "max_tokens": 4096, "model_name": "", "base_url": "",
        }))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        monkeypatch.delenv("ACTIVE_PROVIDER", raising=False)
        monkeypatch.delenv("AGENT_TEMPERATURE", raising=False)

        with patch("src.tools.provider_manager.BedrockModel") as mock_cls:
            pm.get_model(temperature=0.7)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.7
```

- [ ] **Step 5: Run tests to confirm they fail**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/test_provider_manager.py -v`
Expected: All FAIL with `ModuleNotFoundError` or `ImportError` — `provider_manager` doesn't exist yet.

- [ ] **Step 6: Create `src/tools/provider_manager.py`**

Create the file with this exact content:

```python
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
    "max_tokens": 4096,
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


def save_settings(settings: dict) -> None:
    """Write settings dict to data/settings.json (non-sensitive values only)."""
    path = Path(SETTINGS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
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

    # Layer 1: .env (highest priority — reload to pick up runtime writes)
    from dotenv import load_dotenv
    load_dotenv(override=True)

    _apply_env(cfg, os)

    # Ensure all credential keys exist (empty string if not set)
    for key in ("aws_region", "bedrock_model_id", "bedrock_guardrail_id",
                "bedrock_guardrail_version", "aws_access_key_id",
                "aws_secret_access_key", "llm_api_key", "llm_base_url", "llm_model"):
        cfg.setdefault(key, "")

    cfg.setdefault("aws_region", "us-east-1")
    cfg.setdefault("bedrock_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    cfg.setdefault("bedrock_guardrail_version", "1")

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
    }
    for env_key, (cfg_key, cast) in _map.items():
        val = os_mod.getenv(env_key)
        if val:
            try:
                cfg[cfg_key] = cast(val)
            except (ValueError, TypeError):
                pass


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


# Lazy import so the module loads even if strands.models.bedrock is unavailable in tests
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
        from strands.models.bedrock import BedrockModel as _BM
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
        return _BM(**kwargs)

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
            return _test_http_get(
                "https://openrouter.ai/api/v1", "/models",
                {"Authorization": f"Bearer {cfg.get('llm_api_key', '')}"}, t0,
            )
        if provider == "openai":
            base = (cfg.get("llm_base_url") or "https://api.openai.com").rstrip("/")
            return _test_http_get(
                base, "/v1/models",
                {"Authorization": f"Bearer {cfg.get('llm_api_key', '')}"}, t0,
            )
        if provider == "gemini":
            return _test_gemini(cfg, t0)
        if provider == "custom":
            base = cfg.get("llm_base_url", "").rstrip("/")
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}?key={api_key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    ms = int((time.time() - t0) * 1000)
    return {"ok": True, "message": f"Connected · {ms}ms", "latency_ms": ms}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/test_provider_manager.py -v`
Expected: All tests PASS.

If `TestGetModel::test_bedrock_provider_returns_bedrock_model` fails because `BedrockModel` is imported directly at module level rather than being patchable, update the test patch target. The function `get_model()` imports `BedrockModel` as `from strands.models.bedrock import BedrockModel as _BM` inside the function body, so the patch target is `"strands.models.bedrock.BedrockModel"` — update the two `TestGetModel` test patches to use `"strands.models.bedrock.BedrockModel"` if `"src.tools.provider_manager.BedrockModel"` doesn't work.

- [ ] **Step 8: Run the full test suite**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/ -q`
Expected: All tests pass (or same failures as before this task).

- [ ] **Step 9: Commit**

```bash
cd "D:\Downloads\Projects\ai_arena\887" && git add pyproject.toml src/tools/provider_manager.py tests/test_provider_manager.py && git commit -m "feat(provider): add provider_manager with get_model() factory and multi-provider support"
```

---

### Task 2: Update all agent files to use `get_model()`

**Files:**
- Modify: `src/agents/orchestrator.py`
- Modify: `src/agents/qa_agent.py`
- Modify: `src/agents/etl_agent.py`
- Modify: `src/agents/quiz_agent.py`
- Modify: `src/agents/learning_path_agent.py`
- Modify: `src/agents/content_author_agent.py`
- Modify: `src/agents/progress_agent.py`

- [ ] **Step 1: Update `src/agents/orchestrator.py`**

In `src/agents/orchestrator.py`:

1. Remove this import line:
```python
from strands.models.bedrock import BedrockModel
```

2. Remove these config imports (they're no longer needed in orchestrator — `get_model()` reads them internally):
```python
    BEDROCK_MODEL_ID, AWS_REGION,
    BEDROCK_GUARDRAIL_ID, BEDROCK_GUARDRAIL_VERSION,
```
Keep `ORCHESTRATOR_WINDOW_SIZE, SESSIONS_DIR, AGENT_TEMPERATURE, AGENT_TOP_P, AGENT_MAX_TOKENS` — wait, actually all Bedrock-specific params move to provider_manager so remove all of them and add the provider_manager import.

Replace the entire import block at the top of orchestrator.py with:

```python
"""Orchestrator agent — single entry point routing to specialist sub-agents."""
import logging
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session.file_session_manager import FileSessionManager

from src.config import ORCHESTRATOR_WINDOW_SIZE, SESSIONS_DIR
from src.tools.provider_manager import get_model
from src.hooks.logging_throttle import LoggingThrottleHook
from src.agents.qa_agent import qa_training_agent
from src.agents.etl_agent import etl_training_agent
from src.agents.quiz_agent import quiz_agent
from src.agents.learning_path_agent import learning_path_agent
from src.agents.content_author_agent import content_author_agent
from src.agents.progress_agent import progress_agent
from src.tools.retrieval import retrieve_topic
```

3. Replace the entire `_build_bedrock_model()` function (lines ~106-119) with nothing — delete it.

4. In `build_orchestrator()`, replace:
```python
    model = _build_bedrock_model()
```
with:
```python
    model = get_model()
```

The final `build_orchestrator` function should look like:

```python
def build_orchestrator(session_id: str) -> Agent:
    """Build and return the orchestrator agent for a given session."""
    model = get_model()
    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=SESSIONS_DIR,
    )
    return Agent(
        model=model,
        system_prompt=_build_system_prompt(),
        tools=_ALL_TOOLS,
        conversation_manager=SlidingWindowConversationManager(window_size=ORCHESTRATOR_WINDOW_SIZE),
        session_manager=session_manager,
        hooks=[LoggingThrottleHook()],
        trace_attributes={
            "session.id": session_id,
            "app.name": "techtrainer-ai",
            "app.version": "1.0.0",
        },
    )
```

- [ ] **Step 2: Update `src/agents/qa_agent.py`**

1. Replace:
```python
from strands.models.bedrock import BedrockModel
```
with:
```python
from src.tools.provider_manager import get_model
```

2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from the config imports line. The import becomes:
```python
from src.config import AGENT_MAX_TOKENS, QA_AGENT_WINDOW_SIZE
```

3. In `build_qa_agent()`, replace:
```python
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.3, max_tokens=AGENT_MAX_TOKENS)
```
with:
```python
    model = get_model(temperature=0.3)
```

- [ ] **Step 3: Update `src/agents/etl_agent.py`**

Read the file first. Apply the same pattern as qa_agent.py:

1. Replace `from strands.models.bedrock import BedrockModel` with `from src.tools.provider_manager import get_model`
2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from config imports
3. Replace `BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.3, max_tokens=AGENT_MAX_TOKENS)` with `get_model(temperature=0.3)`

- [ ] **Step 4: Update `src/agents/quiz_agent.py`**

Read the file first. Apply the same pattern:

1. Replace `from strands.models.bedrock import BedrockModel` with `from src.tools.provider_manager import get_model`
2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from config imports
3. Replace `BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.5, max_tokens=AGENT_MAX_TOKENS)` with `get_model(temperature=0.5)`

- [ ] **Step 5: Update `src/agents/learning_path_agent.py`**

Read the file first. Apply the same pattern:

1. Replace `from strands.models.bedrock import BedrockModel` with `from src.tools.provider_manager import get_model`
2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from config imports
3. Replace `BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.2, max_tokens=AGENT_MAX_TOKENS)` with `get_model(temperature=0.2)`

- [ ] **Step 6: Update `src/agents/content_author_agent.py`**

Read the file first. Apply the same pattern:

1. Replace `from strands.models.bedrock import BedrockModel` with `from src.tools.provider_manager import get_model`
2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from config imports
3. Replace `BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.6, max_tokens=AGENT_MAX_TOKENS)` with `get_model(temperature=0.6)`

- [ ] **Step 7: Update `src/agents/progress_agent.py`**

Read the file first. Apply the same pattern:

1. Replace `from strands.models.bedrock import BedrockModel` with `from src.tools.provider_manager import get_model`
2. Remove `BEDROCK_MODEL_ID, AWS_REGION,` from config imports
3. Replace `BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.1, max_tokens=AGENT_MAX_TOKENS)` with `get_model(temperature=0.1)`

- [ ] **Step 8: Verify all imports work**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run python -c "from src.agents.orchestrator import build_orchestrator; from src.agents.qa_agent import qa_training_agent; from src.agents.etl_agent import etl_training_agent; from src.agents.quiz_agent import quiz_agent; from src.agents.learning_path_agent import learning_path_agent; from src.agents.content_author_agent import content_author_agent; from src.agents.progress_agent import progress_agent; print('All agents OK')"`
Expected: `All agents OK`

- [ ] **Step 9: Run the full test suite**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
cd "D:\Downloads\Projects\ai_arena\887" && git add src/agents/ && git commit -m "refactor(agents): replace hardcoded BedrockModel with get_model() factory for multi-provider support"
```

---

### Task 3: Update `app.py` — subtitle, session-start connection check, sidebar status

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add provider_manager import to app.py**

Near the top of `app.py`, after the existing imports, add:

```python
from src.tools.provider_manager import get_effective_config, test_connection
```

- [ ] **Step 2: Change the subtitle text**

Find:
```python
st.caption("Your intelligent training assistant for QA Testing & Data Engineering")
```

Replace with:
```python
st.caption("Your intelligent training assistant for Skill Engineering")
```

- [ ] **Step 3: Add connection check at session start**

Find the session initialisation block (around line 32):
```python
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
```

Extend it so it also runs `test_connection()` once per session:
```python
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.connection_status = {"ok": None, "message": "Checking...", "latency_ms": None}

if st.session_state.get("connection_status", {}).get("ok") is None:
    st.session_state.connection_status = test_connection()
```

- [ ] **Step 4: Add sidebar connection status indicator**

In `app.py`, find the sidebar block. Find this line near the bottom of the sidebar:
```python
    st.caption("TechTrainer AI v1.0")
```

Insert the connection status indicator ABOVE that line:
```python
    st.divider()
    _status = st.session_state.get("connection_status", {"ok": None})
    _cfg = get_effective_config()
    _label = _cfg.get("provider_label", "AI Provider")
    if _status.get("ok") is True:
        st.markdown(
            f'<span style="color:#4ade80">● {_label} · Connected</span>',
            unsafe_allow_html=True,
        )
    elif _status.get("ok") is False:
        st.markdown(
            f'<span style="color:#ef4444">● {_label} · Disconnected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:#94a3b8">● {_label} · Checking...</span>',
            unsafe_allow_html=True,
        )
```

- [ ] **Step 5: Verify syntax**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 6: Run the full test suite**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
cd "D:\Downloads\Projects\ai_arena\887" && git add app.py && git commit -m "feat(ui): update subtitle to Skill Engineering; add sidebar connection status indicator"
```

---

### Task 4: Add Settings tab to `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `save_settings` and `write_env_values` to the provider_manager import**

Find:
```python
from src.tools.provider_manager import get_effective_config, test_connection
```

Replace with:
```python
from src.tools.provider_manager import (
    get_effective_config,
    test_connection,
    save_settings,
    write_env_values,
    PROVIDER_IDS,
    PROVIDER_LABELS,
)
```

- [ ] **Step 2: Add `tab_settings` to the tab bar**

Find:
```python
tab_chat, tab_quiz, tab_path, tab_author, tab_kb = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author", "📚 Knowledge Base"]
)
```

Replace with:
```python
tab_chat, tab_quiz, tab_path, tab_author, tab_kb, tab_settings = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author", "📚 Knowledge Base", "⚙️ Settings"]
)
```

- [ ] **Step 3: Append the Settings tab block at the end of `app.py`**

After the last `with tab_kb:` block, append this entire block:

```python
# ── TAB: Settings ─────────────────────────────────────────────────────────────
with tab_settings:
    st.subheader("⚙️ AI Provider Settings")
    st.caption(
        "Choose your AI provider and enter credentials. "
        "Credentials are saved to .env. Non-sensitive settings are saved to data/settings.json."
    )

    # ── Connection status card ───────────────────────────────────────────────
    _conn = st.session_state.get("connection_status", {"ok": None, "message": "Unknown", "latency_ms": None})
    _cfg_now = get_effective_config()
    col_conn, col_test_btn = st.columns([4, 1])
    with col_conn:
        if _conn.get("ok") is True:
            st.success(f"✓ Connected to **{_cfg_now['provider_label']}** · {_conn.get('message', '')}")
        elif _conn.get("ok") is False:
            st.error(f"✗ Cannot reach **{_cfg_now['provider_label']}**: {_conn.get('message', '')}")
        else:
            st.info(f"● **{_cfg_now['provider_label']}** — connection not tested yet")
    with col_test_btn:
        st.write("")
        if st.button("🔌 Test Connection", key="settings_test_conn"):
            with st.spinner("Testing..."):
                result = test_connection()
                st.session_state["connection_status"] = result
            st.rerun()

    st.divider()

    # ── Provider selector ────────────────────────────────────────────────────
    st.markdown("### Provider")
    provider_names = [PROVIDER_LABELS[p] for p in PROVIDER_IDS]
    current_provider = _cfg_now.get("active_provider", "bedrock")
    current_idx = PROVIDER_IDS.index(current_provider) if current_provider in PROVIDER_IDS else 0
    selected_idx = st.radio(
        "Select Provider",
        range(len(PROVIDER_IDS)),
        format_func=lambda i: provider_names[i],
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="settings_provider_radio",
    )
    selected_provider = PROVIDER_IDS[selected_idx]

    st.divider()
    st.markdown(f"### {provider_names[selected_idx]} Credentials")

    # Initialize all form variables from current config before the form
    _api_key = _cfg_now.get("llm_api_key", "")
    _base_url = _cfg_now.get("llm_base_url", "")
    _model_name = _cfg_now.get("llm_model", "")
    _aws_region = _cfg_now.get("aws_region", "us-east-1")
    _bedrock_model = _cfg_now.get("bedrock_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    _guardrail_id = _cfg_now.get("bedrock_guardrail_id", "")
    # Never pre-fill secrets — user must re-enter
    _aws_access_key = ""
    _aws_secret_key = ""

    with st.form("settings_form"):
        if selected_provider == "bedrock":
            c1, c2 = st.columns(2)
            with c1:
                _aws_region = st.text_input("AWS Region", value=_aws_region, key="s_region")
                _aws_access_key = st.text_input("Access Key ID", value="", type="password", key="s_ak",
                                                 placeholder="Leave blank to keep existing")
                _guardrail_id = st.text_input("Guardrail ID (optional)", value=_guardrail_id, key="s_gid")
            with c2:
                _bedrock_model = st.text_input("Model ID", value=_bedrock_model, key="s_model_id")
                _aws_secret_key = st.text_input("Secret Access Key", value="", type="password", key="s_sk",
                                                 placeholder="Leave blank to keep existing")

        elif selected_provider == "ollama":
            _base_url = st.text_input("Base URL", value=_base_url or "http://localhost:11434", key="s_base")
            st.caption("Ollama does not require an API key.")

        elif selected_provider == "lmstudio":
            c1, c2 = st.columns(2)
            with c1:
                _base_url = st.text_input("Base URL", value=_base_url or "http://localhost:1234/v1", key="s_base")
            with c2:
                _model_name = st.text_input("Model Name (optional)", value=_model_name,
                                             placeholder="Leave blank to use loaded model", key="s_model")

        elif selected_provider == "openrouter":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="sk-or-...")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "anthropic/claude-3-sonnet", key="s_model")

        elif selected_provider == "gemini":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="AIza...")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "gemini-1.5-pro", key="s_model")

        elif selected_provider == "openai":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="sk-...")
                _base_url = st.text_input("Base URL (optional — for Enterprise/Azure)",
                                           value=_base_url, key="s_base")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "gpt-4o", key="s_model")

        elif selected_provider == "custom":
            c1, c2 = st.columns(2)
            with c1:
                _base_url = st.text_input("Base URL *", value=_base_url,
                                           placeholder="https://my-api.example.com/v1", key="s_base")
                _api_key = st.text_input("API Key (optional)", value="", type="password", key="s_apikey")
            with c2:
                _model_name = st.text_input("Model Name (optional)", value=_model_name, key="s_model")

        st.divider()
        st.markdown("### Model Parameters")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            _temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=1.0, step=0.05,
                value=float(_cfg_now.get("temperature", 0.3)), key="s_temp",
            )
        with pc2:
            _top_p = st.number_input(
                "Top P", min_value=0.0, max_value=1.0, step=0.05,
                value=float(_cfg_now.get("top_p", 0.9)), key="s_top_p",
            )
        with pc3:
            _max_tokens = st.number_input(
                "Max Tokens", min_value=256, max_value=32768, step=256,
                value=int(_cfg_now.get("max_tokens", 4096)), key="s_max_tokens",
            )

        save_btn = st.form_submit_button("💾 Save to .env & settings.json", type="primary")

    # ── Save handler ─────────────────────────────────────────────────────────
    if save_btn:
        env_updates: dict = {"ACTIVE_PROVIDER": selected_provider}
        if selected_provider == "bedrock":
            env_updates["AWS_REGION"] = _aws_region
            env_updates["BEDROCK_MODEL_ID"] = _bedrock_model
            env_updates["BEDROCK_GUARDRAIL_ID"] = _guardrail_id
            if _aws_access_key:
                env_updates["AWS_ACCESS_KEY_ID"] = _aws_access_key
            if _aws_secret_key:
                env_updates["AWS_SECRET_ACCESS_KEY"] = _aws_secret_key
        else:
            env_updates["LLM_BASE_URL"] = _base_url
            env_updates["LLM_MODEL"] = _model_name
            if _api_key:
                env_updates["LLM_API_KEY"] = _api_key

        env_updates["AGENT_TEMPERATURE"] = str(_temperature)
        env_updates["AGENT_TOP_P"] = str(_top_p)
        env_updates["AGENT_MAX_TOKENS"] = str(int(_max_tokens))

        try:
            write_env_values(env_updates)
        except Exception as e:
            st.error(f"Cannot write to .env — check file permissions: {e}")

        save_settings({
            "active_provider": selected_provider,
            "model_name": _model_name,
            "base_url": _base_url,
            "temperature": _temperature,
            "top_p": _top_p,
            "max_tokens": int(_max_tokens),
        })
        st.success("Settings saved.")
        st.info("Click **Apply & Restart Session** below for provider changes to take effect.")

    # ── Apply & Restart ───────────────────────────────────────────────────────
    if st.button("↺ Apply & Restart Session", key="settings_restart"):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()

    st.caption(
        "Credentials are written to .env (gitignored). "
        "Non-sensitive config is written to data/settings.json. "
        "Values already in .env are pre-loaded in fields above. "
        "Password fields are always blank — re-enter only if you want to change them."
    )
```

- [ ] **Step 4: Verify syntax**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 5: Run the full test suite**

Run: `cd "D:\Downloads\Projects\ai_arena\887" && uv run --with pytest pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
cd "D:\Downloads\Projects\ai_arena\887" && git add app.py && git commit -m "feat(ui): add Settings tab with multi-provider selector, credential input, connection test, and restart"
```
