"""Tests for REST API startup configuration loading."""

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": ""},
        mock_instance,
    )

    assert from_config.call_count == 1
    assert from_config.call_args.args[0] == json.loads(server_main.DEFAULT_STARTUP_CONFIG_PATH.read_text(encoding="utf-8"))


def test_startup_uses_json_config_when_mem0_config_path_is_set(tmp_path: Path):
    mock_instance = MagicMock()
    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.0-flash",
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "output_dimensionality": 768,
            },
        },
    }
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    _, from_config = _reload_server(
        {"ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": str(config_path)},
        mock_instance,
    )

    assert from_config.call_count == 1
    assert from_config.call_args.args[0] == config


def test_startup_fails_for_missing_mem0_config_path(tmp_path: Path):
    missing_path = tmp_path / "missing-config.json"

    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": str(missing_path)},
        clear=False,
    ):
        with patch("mem0.Memory.from_config", return_value=MagicMock()):
            sys.modules.pop("server.main", None)

            with pytest.raises(FileNotFoundError, match="MEM0_CONFIG_PATH"):
                importlib.import_module("server.main")


def test_load_server_environment_prefers_server_env_and_ignores_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    server_main, _ = _reload_server(
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": ""},
        MagicMock(),
    )

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=root-key\n", encoding="utf-8")
    (server_dir / ".env").write_text("GOOGLE_API_KEY=server-key\n", encoding="utf-8")
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()

    monkeypatch.chdir(other_cwd)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    loaded_path = server_main.load_server_environment(server_dir=server_dir)

    assert loaded_path == server_dir / ".env"
    assert os.environ["GOOGLE_API_KEY"] == "server-key"


def test_load_server_environment_falls_back_to_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    server_main, _ = _reload_server(
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": ""},
        MagicMock(),
    )

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=root-key\n", encoding="utf-8")

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    loaded_path = server_main.load_server_environment(server_dir=server_dir)

    assert loaded_path == tmp_path / ".env"
    assert os.environ["GOOGLE_API_KEY"] == "root-key"


def test_load_server_environment_does_not_override_exported_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    server_main, _ = _reload_server(
        {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": ""},
        MagicMock(),
    )

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / ".env").write_text("GOOGLE_API_KEY=server-key\n", encoding="utf-8")

    monkeypatch.setenv("GOOGLE_API_KEY", "exported-key")

    loaded_path = server_main.load_server_environment(server_dir=server_dir)

    assert loaded_path == server_dir / ".env"
    assert os.environ["GOOGLE_API_KEY"] == "exported-key"


def test_startup_loads_server_dotenv_before_memory_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
        "llm": {"provider": "gemini", "config": {"model": "gemini-2.0-flash"}},
        "embedder": {
            "provider": "gemini",
            "config": {"model": "models/gemini-embedding-001", "output_dimensionality": 768},
        },
    }
    config_path = tmp_path / "mem0-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    server_env_path = repo_root / "server" / ".env"
    root_env_path = repo_root / ".env"
    original_exists = Path.exists

    def fake_exists(path_self: Path) -> bool:
        if path_self == server_env_path:
            return True
        if path_self == root_env_path:
            return False
        return original_exists(path_self)

    def fake_load_dotenv(dotenv_path=None, override=False):
        assert dotenv_path == server_env_path
        assert override is False
        os.environ.setdefault("GOOGLE_API_KEY", "dotenv-google-key")
        return True

    def from_config_side_effect(config_dict):
        assert os.environ["GOOGLE_API_KEY"] == "dotenv-google-key"
        assert config_dict["llm"]["provider"] == "gemini"
        assert "api_key" not in config_dict["llm"]["config"]
        return MagicMock()

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sys.modules.pop("server.main", None)

    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "MEM0_CONFIG_PATH": str(config_path)}, clear=False):
        with patch("dotenv.load_dotenv", side_effect=fake_load_dotenv):
            with patch.object(Path, "exists", autospec=True, side_effect=fake_exists):
                with patch("mem0.Memory.from_config", side_effect=from_config_side_effect):
                    importlib.import_module("server.main")
