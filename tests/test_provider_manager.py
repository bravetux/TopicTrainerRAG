# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
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
        assert result["max_tokens"] == 32768

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

    def test_defaults_contain_topic_classifications(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        result = pm.load_settings()
        assert "topic_classifications" in result
        assert result["topic_classifications"] == {}


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
