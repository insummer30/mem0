"""Tests for REST API startup configuration loading."""

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


DEFAULT_CONFIG_ENV = {
    "VECTOR_STORE_PROVIDER": "qdrant",
    "VECTOR_STORE_HOST": "localhost",
    "VECTOR_STORE_PORT": "6333",
    "LLM_PROVIDER": "gemini",
    "LLM_MODEL": "gemini-2.0-flash",
    "EMBEDDER_PROVIDER": "gemini",
    "EMBEDDER_MODEL": "models/gemini-embedding-001",
}


def _reload_server(env_overrides: dict, mock_instance: MagicMock):
    """Reload server.main with the provided environment."""
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, env_overrides, clear=False):
        with patch("mem0.Memory.from_config", return_value=mock_instance) as from_config:
            server_main = importlib.import_module("server.main")

    return server_main, from_config


def test_startup_uses_default_config_file_when_mem0_config_path_is_unset():
    mock_instance = MagicMock()

    server_main, from_config = _reload_server(
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "CONFIG_PATH": "", **DEFAULT_CONFIG_ENV},
        mock_instance,
    )

    assert from_config.call_count == 1
    loaded_config = from_config.call_args.args[0]
    assert loaded_config["vector_store"]["provider"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_PROVIDER"]
    assert loaded_config["vector_store"]["config"]["host"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_HOST"]
    assert loaded_config["vector_store"]["config"]["port"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_PORT"]
    assert loaded_config["llm"]["provider"] == DEFAULT_CONFIG_ENV["LLM_PROVIDER"]
    assert loaded_config["llm"]["config"]["model"] == DEFAULT_CONFIG_ENV["LLM_MODEL"]
    assert loaded_config["embedder"]["provider"] == DEFAULT_CONFIG_ENV["EMBEDDER_PROVIDER"]
    assert loaded_config["embedder"]["config"]["model"] == DEFAULT_CONFIG_ENV["EMBEDDER_MODEL"]


def test_startup_uses_json_config_when_mem0_config_path_is_set(tmp_path: Path):
    mock_instance = MagicMock()
    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "env:VECTOR_STORE_PROVIDER",
            "config": {
                "host": "env:VECTOR_STORE_HOST",
                "port": "env:VECTOR_STORE_PORT",
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
        "llm": {
            "provider": "env:LLM_PROVIDER",
            "config": {
                "model": "env:LLM_MODEL",
            },
        },
        "embedder": {
            "provider": "env:EMBEDDER_PROVIDER",
            "config": {
                "model": "env:EMBEDDER_MODEL",
                "output_dimensionality": 768,
            },
        },
    }
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    _, from_config = _reload_server(
        {
            "ADMIN_API_KEY": "",
            "CONFIG_PATH": str(config_path),
            **DEFAULT_CONFIG_ENV,
        },
        mock_instance,
    )

    assert from_config.call_count == 1
    assert from_config.call_args.args[0] == {
        "version": "v1.1",
        "vector_store": {
            "provider": DEFAULT_CONFIG_ENV["VECTOR_STORE_PROVIDER"],
            "config": {
                "host": DEFAULT_CONFIG_ENV["VECTOR_STORE_HOST"],
                "port": DEFAULT_CONFIG_ENV["VECTOR_STORE_PORT"],
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
        "llm": {
            "provider": DEFAULT_CONFIG_ENV["LLM_PROVIDER"],
            "config": {
                "model": DEFAULT_CONFIG_ENV["LLM_MODEL"],
            },
        },
        "embedder": {
            "provider": DEFAULT_CONFIG_ENV["EMBEDDER_PROVIDER"],
            "config": {
                "model": DEFAULT_CONFIG_ENV["EMBEDDER_MODEL"],
                "output_dimensionality": 768,
            },
        },
    }


def test_startup_fails_for_missing_env_reference(tmp_path: Path):
    config = {
        "version": "v1.1",
        "llm": {"provider": "gemini", "config": {"model": "env:MISSING_MODEL_ENV"}},
        "embedder": {
            "provider": "gemini",
            "config": {"model": "models/gemini-embedding-001", "output_dimensionality": 768},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"host": "localhost", "port": 6333, "collection_name": "memories", "embedding_model_dims": 768},
        },
    }
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with patch.dict(
        os.environ,
        {"ADMIN_API_KEY": "", "CONFIG_PATH": str(config_path)},
        clear=False,
    ):
        with patch("mem0.Memory.from_config", return_value=MagicMock()):
            sys.modules.pop("server.main", None)

            with pytest.raises(ValueError, match="MISSING_MODEL_ENV"):
                importlib.import_module("server.main")


def test_startup_fails_for_missing_mem0_config_path(tmp_path: Path):
    missing_path = tmp_path / "missing-config.json"

    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "CONFIG_PATH": str(missing_path)},
        clear=False,
    ):
        with patch("mem0.Memory.from_config", return_value=MagicMock()):
            sys.modules.pop("server.main", None)

            with pytest.raises(FileNotFoundError, match="CONFIG_PATH"):
                importlib.import_module("server.main")


def test_startup_loads_only_server_dotenv_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps({"version": "v1.1"}), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    server_env_path = repo_root / "server" / ".env"
    root_env_path = repo_root / ".env"
    original_exists = Path.exists

    def fake_exists(path_self: Path) -> bool:
        if path_self == server_env_path:
            return True
        if path_self == root_env_path:
            return True
        return original_exists(path_self)

    def fake_load_dotenv(dotenv_path=None, override=False):
        assert dotenv_path == server_env_path
        assert override is False
        os.environ.setdefault("GOOGLE_API_KEY", "dotenv-google-key")
        return True

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "CONFIG_PATH": str(config_path)}, clear=False):
        with patch("dotenv.load_dotenv", side_effect=fake_load_dotenv) as load_dotenv_mock:
            with patch.object(Path, "exists", autospec=True, side_effect=fake_exists):
                with patch("mem0.Memory.from_config", return_value=MagicMock()):
                    importlib.import_module("server.main")

    load_dotenv_mock.assert_called_once_with(server_env_path, override=False)


def test_startup_does_not_fall_back_to_root_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps({"version": "v1.1"}), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    server_env_path = repo_root / "server" / ".env"
    root_env_path = repo_root / ".env"
    original_exists = Path.exists

    def fake_exists(path_self: Path) -> bool:
        if path_self == server_env_path:
            return False
        if path_self == root_env_path:
            return True
        return original_exists(path_self)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "CONFIG_PATH": str(config_path)}, clear=False):
        with patch("dotenv.load_dotenv") as load_dotenv_mock:
            with patch.object(Path, "exists", autospec=True, side_effect=fake_exists):
                with patch("mem0.Memory.from_config", return_value=MagicMock()):
                    importlib.import_module("server.main")

    load_dotenv_mock.assert_not_called()
    assert "GOOGLE_API_KEY" not in os.environ


def test_startup_server_dotenv_does_not_override_exported_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps({"version": "v1.1"}), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    server_env_path = repo_root / "server" / ".env"
    original_exists = Path.exists

    def fake_exists(path_self: Path) -> bool:
        if path_self == server_env_path:
            return True
        return original_exists(path_self)

    def fake_load_dotenv(dotenv_path=None, override=False):
        assert dotenv_path == server_env_path
        assert override is False
        os.environ.setdefault("GOOGLE_API_KEY", "dotenv-google-key")
        return True

    monkeypatch.setenv("GOOGLE_API_KEY", "exported-key")
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "CONFIG_PATH": str(config_path)}, clear=False):
        with patch("dotenv.load_dotenv", side_effect=fake_load_dotenv):
            with patch.object(Path, "exists", autospec=True, side_effect=fake_exists):
                with patch("mem0.Memory.from_config", return_value=MagicMock()):
                    importlib.import_module("server.main")

    assert os.environ["GOOGLE_API_KEY"] == "exported-key"


def test_startup_loads_server_dotenv_before_memory_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "env:VECTOR_STORE_PROVIDER",
            "config": {
                "host": "env:VECTOR_STORE_HOST",
                "port": "env:VECTOR_STORE_PORT",
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
        "llm": {"provider": "env:LLM_PROVIDER", "config": {"model": "env:LLM_MODEL"}},
        "embedder": {
            "provider": "env:EMBEDDER_PROVIDER",
            "config": {"model": "env:EMBEDDER_MODEL", "output_dimensionality": 768},
        },
    }
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    server_env_path = repo_root / "server" / ".env"
    original_exists = Path.exists

    def fake_exists(path_self: Path) -> bool:
        if path_self == server_env_path:
            return True
        return original_exists(path_self)

    def fake_load_dotenv(dotenv_path=None, override=False):
        assert dotenv_path == server_env_path
        assert override is False
        os.environ.setdefault("GOOGLE_API_KEY", "dotenv-google-key")
        os.environ.setdefault("VECTOR_STORE_PROVIDER", DEFAULT_CONFIG_ENV["VECTOR_STORE_PROVIDER"])
        os.environ.setdefault("VECTOR_STORE_HOST", DEFAULT_CONFIG_ENV["VECTOR_STORE_HOST"])
        os.environ.setdefault("VECTOR_STORE_PORT", DEFAULT_CONFIG_ENV["VECTOR_STORE_PORT"])
        os.environ.setdefault("LLM_PROVIDER", DEFAULT_CONFIG_ENV["LLM_PROVIDER"])
        os.environ.setdefault("LLM_MODEL", DEFAULT_CONFIG_ENV["LLM_MODEL"])
        os.environ.setdefault("EMBEDDER_PROVIDER", DEFAULT_CONFIG_ENV["EMBEDDER_PROVIDER"])
        os.environ.setdefault("EMBEDDER_MODEL", DEFAULT_CONFIG_ENV["EMBEDDER_MODEL"])
        return True

    def from_config_side_effect(config_dict):
        assert os.environ["GOOGLE_API_KEY"] == "dotenv-google-key"
        assert config_dict["vector_store"]["provider"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_PROVIDER"]
        assert config_dict["vector_store"]["config"]["host"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_HOST"]
        assert config_dict["vector_store"]["config"]["port"] == DEFAULT_CONFIG_ENV["VECTOR_STORE_PORT"]
        assert config_dict["llm"]["provider"] == DEFAULT_CONFIG_ENV["LLM_PROVIDER"]
        assert config_dict["llm"]["config"]["model"] == DEFAULT_CONFIG_ENV["LLM_MODEL"]
        assert config_dict["embedder"]["provider"] == DEFAULT_CONFIG_ENV["EMBEDDER_PROVIDER"]
        assert config_dict["embedder"]["config"]["model"] == DEFAULT_CONFIG_ENV["EMBEDDER_MODEL"]
        return MagicMock()

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "CONFIG_PATH": str(config_path)}, clear=False):
        with patch("dotenv.load_dotenv", side_effect=fake_load_dotenv):
            with patch.object(Path, "exists", autospec=True, side_effect=fake_exists):
                with patch("mem0.Memory.from_config", side_effect=from_config_side_effect):
                    importlib.import_module("server.main")
