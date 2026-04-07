# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Embedding provider abstraction for TechTrainer AI.

Supports: AWS Bedrock Titan, Ollama (local), OpenAI, Custom (OpenAI-compatible).
Priority (high → low): .env > data/settings.json > built-in defaults

Switching embedding providers requires re-indexing all KB collections because
stored vector dimensions must match the query embedding dimensions.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER_LABELS: dict = {
    "bedrock": "AWS Bedrock Titan",
    "ollama": "Ollama (local)",
    "openai": "OpenAI",
    "custom": "Custom (OpenAI-compatible)",
}

EMBEDDING_PROVIDER_IDS: list = list(EMBEDDING_PROVIDER_LABELS.keys())

EMBEDDING_MODEL_DEFAULTS: dict = {
    "bedrock": "amazon.titan-embed-text-v1",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
    "custom": "",
}


def get_embedding_config() -> dict:
    """Return the active embedding configuration.

    Merges settings.json → .env (env wins).
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(override=False)

    try:
        from src.tools.provider_manager import load_settings
        settings = load_settings()
    except Exception:
        settings = {}

    provider = (
        os.getenv("EMBEDDING_PROVIDER")
        or settings.get("embedding_provider", "bedrock")
    )
    if provider not in EMBEDDING_PROVIDER_IDS:
        logger.warning("Unknown embedding provider '%s' — falling back to bedrock.", provider)
        provider = "bedrock"

    model = (
        os.getenv("EMBEDDING_MODEL")
        or settings.get("embedding_model")
        or EMBEDDING_MODEL_DEFAULTS[provider]
    )
    base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or settings.get("embedding_base_url", "")
    )
    api_key = os.getenv("EMBEDDING_API_KEY", "")
    aws_region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or settings.get("aws_region", "us-east-1")
    )

    return {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_base_url": base_url,
        "embedding_api_key": api_key,
        "aws_region": aws_region,
        "provider_label": EMBEDDING_PROVIDER_LABELS[provider],
    }


def embed_texts(texts: list, cfg: Optional[dict] = None) -> list:
    """Embed a list of texts using the configured provider.

    Args:
        texts: List of strings to embed.
        cfg:   Optional pre-loaded config dict (avoids re-reading settings on every call).

    Returns:
        List of float vectors, one per input text.
    """
    if cfg is None:
        cfg = get_embedding_config()
    provider = cfg["embedding_provider"]

    if provider == "bedrock":
        return _embed_bedrock(texts, cfg)
    return _embed_litellm(texts, cfg)


def _embed_bedrock(texts: list, cfg: dict) -> list:
    import json
    import boto3
    client = boto3.client("bedrock-runtime", region_name=cfg["aws_region"])
    results = []
    for text in texts:
        response = client.invoke_model(
            modelId=cfg["embedding_model"],
            body=json.dumps({"inputText": text[:8192]}),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        results.append(body["embedding"])
    return results


def _embed_litellm(texts: list, cfg: dict) -> list:
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "litellm is required for non-Bedrock embeddings. Run: uv add litellm"
        ) from exc

    provider = cfg["embedding_provider"]
    model = cfg["embedding_model"]
    kwargs: dict = {}

    if provider == "ollama":
        litellm_model = f"ollama/{model}"
        kwargs["api_base"] = cfg.get("embedding_base_url") or "http://localhost:11434"
    elif provider == "openai":
        litellm_model = model
        if cfg.get("embedding_api_key"):
            kwargs["api_key"] = cfg["embedding_api_key"]
    elif provider == "custom":
        litellm_model = f"openai/{model}" if model else "openai/local-model"
        if cfg.get("embedding_base_url"):
            kwargs["api_base"] = cfg["embedding_base_url"]
        if cfg.get("embedding_api_key"):
            kwargs["api_key"] = cfg["embedding_api_key"]
    else:
        litellm_model = model

    response = litellm.embedding(model=litellm_model, input=texts, **kwargs)
    return [item["embedding"] for item in response.data]
