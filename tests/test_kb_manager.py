"""Tests for Knowledge Base manager."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock


class TestSanitiseId:
    def test_spaces_become_underscores(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("My Topic") == "my_topic"

    def test_special_chars_removed(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("C# & .NET") == "c_net"

    def test_already_clean(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("kubernetes") == "kubernetes"


class TestRegistryIO:
    def test_load_empty_registry(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        result = kb_manager._load_registry()
        assert result == {"custom": []}

    def test_save_and_reload(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        reg_path = str(tmp_path / "reg.json")
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", reg_path)
        data = {"custom": [{"id": "test", "display_name": "Test"}]}
        kb_manager._save_registry(data)
        assert kb_manager._load_registry() == data


class TestCreateCustomTopic:
    def test_creates_topic_in_registry(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        topic = kb_manager.create_custom_topic("Kubernetes", "K8s training")
        assert topic["id"] == "kubernetes"
        assert topic["collection"] == "tech_kubernetes"
        assert topic["display_name"] == "Kubernetes"
        registry = kb_manager._load_registry()
        assert any(t["id"] == "kubernetes" for t in registry["custom"])

    def test_duplicate_raises(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        kb_manager.create_custom_topic("Kubernetes")
        with pytest.raises(ValueError, match="already exists"):
            kb_manager.create_custom_topic("Kubernetes")

    def test_builtin_name_collision_raises(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            kb_manager.create_custom_topic("Selenium")


class TestDeleteCustomTopic:
    def test_delete_removes_from_registry(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.chdir(tmp_path)
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value.count.return_value = 0
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_custom_topic("jenkins")
        registry = kb_manager._load_registry()
        assert not any(t["id"] == "jenkins" for t in registry["custom"])

    def test_delete_builtin_raises(self, tmp_path, monkeypatch):
        import src.tools.provider_manager as pm
        from src.tools import kb_manager
        fake_builtin = [{"id": "selenium", "display_name": "Selenium",
                         "collection": "tech_selenium", "doc_dir": str(tmp_path / "selenium")}]
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", fake_builtin)
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client",
                            MagicMock(get_collection=MagicMock(side_effect=Exception())))
        with pytest.raises(ValueError, match="Built-in"):
            kb_manager.delete_custom_topic("selenium")


class TestSaveUploadedFile:
    def test_saves_file_to_topic_dir(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        fake_topic = {
            "id": "selenium", "display_name": "Selenium",
            "collection": "tech_selenium",
            "doc_dir": str(tmp_path / "selenium"),
        }
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [fake_topic])
        saved = kb_manager.save_uploaded_file("selenium", "guide.pdf", b"pdf content")
        assert Path(saved).exists()
        assert Path(saved).read_bytes() == b"pdf content"

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        fake_topic = {
            "id": "selenium", "display_name": "Selenium",
            "collection": "tech_selenium",
            "doc_dir": str(tmp_path / "selenium"),
        }
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [fake_topic])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        saved = kb_manager.save_uploaded_file("selenium", "../../evil.pdf", b"bad")
        assert ".." not in saved
        assert str(tmp_path / "selenium") in str(Path(saved).parent)


class TestDeleteTopic:
    def test_delete_user_created_custom_removes_from_registry(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.chdir(tmp_path)
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value.count.return_value = 0
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_topic("jenkins")
        registry = kb_manager._load_registry()
        assert not any(t["id"] == "jenkins" for t in registry["custom"])

    def test_delete_user_created_custom_drops_collection(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.chdir(tmp_path)
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value.count.return_value = 0
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_topic("jenkins")
        mock_chroma.delete_collection.assert_called_once_with("tech_jenkins")

    def test_delete_demoted_builtin_drops_collection_only(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        fake_builtin = [{"id": "selenium", "display_name": "Selenium",
                         "collection": "tech_selenium",
                         "doc_dir": str(tmp_path / "selenium")}]
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", fake_builtin)
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"selenium": "custom"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value.count.return_value = 0
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.delete_topic("selenium")
        # collection dropped
        mock_chroma.delete_collection.assert_called_once_with("tech_selenium")
        # NOT removed from registry
        registry = kb_manager._load_registry()
        assert registry == {"custom": []}

    def test_delete_builtin_classified_topic_raises(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        fake_builtin = [{"id": "selenium", "display_name": "Selenium",
                         "collection": "tech_selenium",
                         "doc_dir": str(tmp_path / "selenium")}]
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", fake_builtin)
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        with pytest.raises(ValueError, match="Built-in"):
            kb_manager.delete_topic("selenium")


class TestTopicClassificationOverrides:
    def _make_fake_builtin(self, tmp_path):
        return [{"id": "selenium", "display_name": "Selenium",
                 "collection": "tech_selenium",
                 "doc_dir": str(tmp_path / "selenium")}]

    def test_builtin_default_is_builtin(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", self._make_fake_builtin(tmp_path))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        topics = kb_manager.load_all_topics()
        selenium = next(t for t in topics if t["id"] == "selenium")
        assert selenium["is_builtin"] is True

    def test_demoted_builtin_is_not_builtin(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", self._make_fake_builtin(tmp_path))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"selenium": "custom"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        topics = kb_manager.load_all_topics()
        selenium = next(t for t in topics if t["id"] == "selenium")
        assert selenium["is_builtin"] is False

    def test_promoted_custom_is_builtin(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        reg_path = tmp_path / "reg.json"
        reg_path.write_text(json.dumps({"custom": [
            {"id": "vxworks", "display_name": "VxWorks",
             "collection": "tech_vxworks", "doc_dir": str(tmp_path / "vxworks")}
        ]}))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(reg_path))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"vxworks": "builtin"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        topics = kb_manager.load_all_topics()
        vxworks = next(t for t in topics if t["id"] == "vxworks")
        assert vxworks["is_builtin"] is True
