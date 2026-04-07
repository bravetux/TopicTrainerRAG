<!--
  Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
  Date   : 26 March 2026
-->
# Settings Tab & Multi-Provider AI Design

**Last updated:** 2026-03-30 — Added embedding provider section, test connection restriction, developer credit.

**Goal:** Add a `⚙️ Settings` tab to TechTrainer AI that lets users select an AI provider, enter credentials (saved to `.env`), test the connection, and see live connection status in both the tab and the sidebar. Change the app subtitle. All agents switch from hardcoded `BedrockModel` to a `get_model()` factory. Settings tab also includes a configurable embedding provider section.

**Architecture:** A new `provider_manager.py` module owns provider config, the `get_model()` factory, and `.env` write logic. AWS Bedrock uses the existing `BedrockModel`; all other providers (Ollama, LM Studio, OpenRouter, Gemini, OpenAI/ChatGPT Enterprise, Custom) use Strands' `LiteLLMModel`. Non-sensitive settings persist in `data/settings.json`; credentials persist in `.env`.

**Tech Stack:** Strands `LiteLLMModel` + `BedrockModel`, `litellm` package, `python-dotenv`, Streamlit `st.text_input(type="password")`, existing `src/config.py`.

---

## 1. Subtitle Change

In `app.py`, change:
```python
st.caption("Your intelligent training assistant for QA Testing & Data Engineering")
```
to:
```python
st.caption("Your intelligent training assistant for Skill Engineering")
```

---

## 2. Provider Model

### 2.1 Supported Providers

| id | display_name | protocol | notes |
|----|-------------|----------|-------|
| `bedrock` | AWS Bedrock | Bedrock API | default; uses `BedrockModel` |
| `ollama` | Local Ollama | OpenAI-compatible | default base URL: `http://localhost:11434` |
| `lmstudio` | LM Studio | OpenAI-compatible | default base URL: `http://localhost:1234/v1` |
| `openrouter` | OpenRouter | OpenAI-compatible | base URL: `https://openrouter.ai/api/v1` |
| `gemini` | Google Gemini | LiteLLM Gemini | model prefix: `gemini/` |
| `openai` | OpenAI / ChatGPT | OpenAI API | optional custom base URL for Enterprise/Azure |
| `custom` | Custom | OpenAI-compatible | user-supplied base URL required |

### 2.2 Settings Storage

**`data/settings.json`** — non-sensitive config, persists across restarts:
```json
{
  "active_provider": "bedrock",
  "model_name": "",
  "base_url": "",
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 4096,
  "embedding_provider": "bedrock",
  "embedding_model": "",
  "embedding_base_url": ""
}
```

**`.env`** — all credentials, written by the Settings UI "Save" button:
```
ACTIVE_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_GUARDRAIL_ID=
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
EMBEDDING_PROVIDER=bedrock
EMBEDDING_MODEL=
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
```

### 2.3 Override Priority (high → low)

1. `.env` file (set by Settings UI or manual edit)
2. `data/settings.json` (provider choice, model, base URL, parameters)
3. Built-in defaults in `provider_manager.py`

---

## 3. New File: `src/tools/provider_manager.py`

Responsibilities:
- `load_settings() -> dict` — reads `data/settings.json`; returns defaults if missing
- `save_settings(settings: dict)` — writes `data/settings.json`
- `write_env_values(updates: dict)` — uses `dotenv.set_key(dotenv_path, key, value)` for each entry so existing comments and unrelated keys are preserved.
- `get_effective_config() -> dict` — merges .env + settings.json + defaults; returns the winning value for each key
- `get_model()` — returns a `BedrockModel` when `active_provider == "bedrock"`, else returns `LiteLLMModel` configured for the active provider
- `test_connection() -> dict` — returns `{"ok": bool, "message": str, "latency_ms": int}`

### 3.1 `get_model()` logic

```python
def get_model():
    cfg = get_effective_config()
    provider = cfg["active_provider"]

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel
        kwargs = {
            "model_id": cfg["bedrock_model_id"],
            "region_name": cfg["aws_region"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "max_tokens": cfg["max_tokens"],
            "streaming": True,
        }
        if cfg.get("bedrock_guardrail_id"):
            kwargs["guardrail_id"] = cfg["bedrock_guardrail_id"]
            kwargs["guardrail_version"] = cfg["bedrock_guardrail_version"]
        return BedrockModel(**kwargs)

    # All other providers via LiteLLM
    from strands.models.litellm import LiteLLMModel
    model_id = _resolve_litellm_model(provider, cfg)
    litellm_kwargs = {
        "model_id": model_id,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "streaming": True,
    }
    if cfg.get("llm_api_key"):
        litellm_kwargs["api_key"] = cfg["llm_api_key"]
    if cfg.get("llm_base_url"):
        litellm_kwargs["base_url"] = cfg["llm_base_url"]
    return LiteLLMModel(**litellm_kwargs)
```

### 3.2 LiteLLM model ID mapping

| Provider | LiteLLM model_id format | Example |
|---|---|---|
| `ollama` | `ollama/{model}` | `ollama/llama3.2` |
| `lmstudio` | `openai/{model}` with custom base_url | `openai/local-model` |
| `openrouter` | `openrouter/{model}` | `openrouter/anthropic/claude-3-sonnet` |
| `gemini` | `gemini/{model}` | `gemini/gemini-1.5-pro` |
| `openai` | `openai/{model}` | `openai/gpt-4o` |
| `custom` | `openai/{model}` with custom base_url | `openai/my-model` |

If `model_name` is empty for local providers (Ollama, LM Studio), use `"local-model"` as placeholder — the server ignores it and uses the currently loaded model.

### 3.3 `test_connection()` logic

The **Test Connection** button is only shown in the UI for local and custom providers (`ollama`, `lmstudio`, `custom`). Managed cloud providers (Bedrock, OpenRouter, Gemini, OpenAI) show a static "provider configured" status instead.

| Provider | Test method | Button shown |
|---|---|---|
| `bedrock` | `bedrock-runtime.invoke_model` with prompt `"Hi"`, `max_tokens=1` | No |
| `ollama` | `GET {base_url}/api/tags` (no auth) | Yes |
| `lmstudio` | `GET {base_url}/v1/models` (no auth) | Yes |
| `openrouter` | `GET https://openrouter.ai/api/v1/models` with Bearer token | No |
| `gemini` | LiteLLM completion, `max_tokens=1` | No |
| `openai` | `GET {base_url}/v1/models` with Bearer token | No |
| `custom` | `GET {base_url}/models` with optional Bearer token | Yes |

Returns:
```python
{"ok": True, "message": "Connected · 312ms", "latency_ms": 312}
# or
{"ok": False, "message": "Connection refused — is Ollama running?", "latency_ms": None}
```

---

## 4. Modified Agent Files

Each agent that constructs a `BedrockModel` directly is updated to call `provider_manager.get_model()` instead.

Files to update:
- `src/agents/orchestrator.py` — replace `_build_bedrock_model()` with `get_model()` call
- `src/agents/qa_agent.py`
- `src/agents/etl_agent.py`
- `src/agents/quiz_agent.py`
- `src/agents/content_author_agent.py`
- `src/agents/learning_path_agent.py`
- `src/agents/progress_agent.py`

Each agent's per-agent temperature (0.3 for QA, 0.5 for quiz, 0.6 for content author) is preserved by passing `temperature=` as an override kwarg to `get_model(temperature=0.5)`. The `get_model()` function accepts an optional `temperature` override parameter for this purpose.

---

## 5. `app.py` Changes

### 5.1 Subtitle

```python
st.caption("Your intelligent training assistant for Skill Engineering")
```

### 5.2 Sidebar Connection Status

Added at the bottom of the sidebar, above the version line:

```python
# st.session_state["connection_status"] is set once at session start
# and refreshed whenever the user clicks "Test Connection" in the Settings tab.
status = st.session_state.get("connection_status", {"ok": None})
cfg = get_effective_config()
label = cfg["provider_label"]
if status["ok"] is True:
    st.markdown(f'<span style="color:#4ade80">● {label} · Connected</span>', unsafe_allow_html=True)
elif status["ok"] is False:
    st.markdown(f'<span style="color:#ef4444">● {label} · Disconnected</span>', unsafe_allow_html=True)
else:
    st.markdown(f'<span style="color:#94a3b8">● {label} · Checking...</span>', unsafe_allow_html=True)
```

Connection status is checked once at session start (by calling `test_connection()` in the app initialisation block and storing the result in `st.session_state["connection_status"]`). Clicking "Test Connection" in the Settings tab re-runs `test_connection()` and updates `st.session_state["connection_status"]`.

### 5.3 Settings Tab — 6th Tab

Tab bar becomes:
```python
tab_chat, tab_quiz, tab_path, tab_author, tab_kb, tab_settings = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author", "📚 Knowledge Base", "⚙️ Settings"]
)
```

### 5.4 Settings Tab Layout

```
⚙️ Settings

[Connection Status Card]          [Active Provider: AWS Bedrock · ● Provider configured]
                                  (Test Connection button only for Ollama / LM Studio / Custom)

── Provider ──────────────────────────────────────────────────────────────
  ○ AWS Bedrock  ○ Local Ollama  ○ LM Studio  ○ OpenRouter
  ○ Google Gemini  ○ OpenAI / ChatGPT  ○ Custom

── Embedding Provider ────────────────────────────────────────────────────
  ○ AWS Bedrock Titan  ○ Ollama (local)  ○ OpenAI  ○ Custom (OpenAI-compatible)
  ⚠ Changing provider requires re-indexing all Knowledge Base collections
  Embedding Model: [text-embedding-3-small]   Base URL: [only for Ollama/Custom]
  API Key: [only for OpenAI/Custom]

── Credentials (AWS Bedrock) ─────────────────────────────────────────────
  Region: [us-east-1]   Model ID: [us.anthropic.claude-...]
  Access Key ID: [••••••••ABCD]  Secret Key: [••••••••••••]
  Guardrail ID: [optional]

── Model Parameters ──────────────────────────────────────────────────────
  Temperature: [0.3]   Top P: [0.9]   Max Tokens: [4096]

  [💾 Save to .env & settings.json]   [↺ Apply & Restart Session]

ℹ️  Values already set in .env are pre-filled. Save overwrites .env.
    Changes take effect after "Apply & Restart Session".
```

Credential fields shown per provider:

| Provider | Fields shown |
|---|---|
| `bedrock` | Region, Model ID, Access Key ID (password), Secret Key (password), Guardrail ID |
| `ollama` | Base URL |
| `lmstudio` | Base URL, Model Name (optional) |
| `openrouter` | API Key (password), Model |
| `gemini` | API Key (password), Model |
| `openai` | API Key (password), Model, Base URL (optional, for Enterprise), Org ID (optional) |
| `custom` | Base URL (required), API Key (optional, password), Model Name (optional) |

---

## 6. Error Handling

| Scenario | Behaviour |
|---|---|
| `.env` not writable | `st.error("Cannot write to .env — check file permissions.")` |
| `settings.json` corrupt | Silently reset to defaults, log warning |
| LiteLLM not installed | `st.error("litellm package not installed. Run: uv add litellm")` |
| Connection test timeout (>10s) | Returns `{"ok": False, "message": "Timeout after 10s"}` |
| Unknown provider in settings | Fall back to `bedrock`, log warning |

---

## 7. New File: `src/tools/embedding_manager.py`

Provides a provider-agnostic embedding abstraction used by both `document_ingestion.py` and `retrieval.py`.

**Supported providers:**

| id | Display Name | Backend | Default Model |
|---|---|---|---|
| `bedrock` | AWS Bedrock Titan | boto3 `invoke_model` | `amazon.titan-embed-text-v1` |
| `ollama` | Ollama (local) | LiteLLM `litellm.embedding` | `nomic-embed-text` |
| `openai` | OpenAI | LiteLLM `litellm.embedding` | `text-embedding-3-small` |
| `custom` | Custom (OpenAI-compatible) | LiteLLM `litellm.embedding` | user-supplied |

**Key functions:**
- `get_embedding_config() -> dict` — reads `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` from `.env` / `settings.json`
- `embed_texts(texts, cfg=None) -> list` — dispatches to `_embed_bedrock` or `_embed_litellm`

**Override priority:** `.env` > `settings.json` > built-in defaults

**Note:** `EMBEDDING_API_KEY` is treated as a secret and stripped from `settings.json` by `save_settings`.

---

## 8. Files Updated for Embedding Abstraction

- `src/tools/document_ingestion.py` — removed local `embed_texts(texts, bedrock_client)` and `boto3` import; now calls `embedding_manager.embed_texts(chunks, emb_cfg)`; `bedrock_client` param removed from `index_directory` and `index_technology`
- `src/tools/retrieval.py` — removed `_embed`, `_get_bedrock`, `_bedrock_client`; `_embed()` now delegates to `embedding_manager.embed_texts()`; `boto3` and `json` imports removed
- `src/tools/provider_manager.py` — `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` added to `_apply_env`; `embedding_api_key` added to `_SECRET_KEYS`

---

## 9. Sidebar Developer Credit

Added at the bottom of the sidebar, below the version line:
```
Developed by B.Vignesh Kumar
ic19939@gmail.com
```

---

## 10. Files Not Changed

- `src/tools/kb_manager.py`
- `src/models/schemas.py`
- `src/hooks/logging_throttle.py`
- All test files for existing modules (new tests added for `provider_manager`)
