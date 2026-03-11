"""Tests for src.core.project_manager – DLC project creation & info."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.core.project_manager import ProjectManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_PROJECT_NAME = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{2,19}", fullmatch=True)
_EXPERIMENTER = st.from_regex(r"[A-Za-z][A-Za-z0-9]{1,14}", fullmatch=True)


# ===================================================================
# ProjectManager._create_project_structure
# ===================================================================


class TestProjectManagerStructure:
    """Test project directory scaffold."""

    def test_creates_expected_subfolders(self, tmp_path: Path) -> None:
        mgr = ProjectManager()
        mgr._create_project_structure(tmp_path)

        for folder in ["models", "frames", "output", "dataset"]:
            assert (tmp_path / folder).is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        mgr = ProjectManager()
        mgr._create_project_structure(tmp_path)
        mgr._create_project_structure(tmp_path)  # second call should not fail

        for folder in ["models", "frames", "output", "dataset"]:
            assert (tmp_path / folder).is_dir()


# ===================================================================
# ProjectManager.get_project_info
# ===================================================================


class TestProjectManagerGetInfo:
    """Test info retrieval from config."""

    def test_returns_expected_keys(self, tmp_config: Path) -> None:
        mgr = ProjectManager()
        info = mgr.get_project_info(str(tmp_config))

        assert info["project_name"] == "test_project"
        assert info["experimenter"] == "tester"
        assert info["project_path"] == str(tmp_config.parent)
        assert isinstance(info["bodyparts"], list)
        assert info["multianimal"] is False

    def test_multianimal_flag(self, tmp_multianimal_config: Path) -> None:
        mgr = ProjectManager()
        info = mgr.get_project_info(str(tmp_multianimal_config))
        assert info["multianimal"] is True

    def test_missing_fields_have_defaults(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        mgr = ProjectManager()
        info = mgr.get_project_info(str(cfg_path))
        assert info["project_name"] == "Unknown"
        assert info["experimenter"] == "Unknown"


# ===================================================================
# ProjectManager.create_project (mocked DLC)
# ===================================================================


class TestProjectManagerCreate:
    """Test project creation with mocked deeplabcut."""

    @patch("src.core.project_manager.deeplabcut")
    def test_create_project_calls_dlc(
        self, mock_dlc: MagicMock, tmp_path: Path
    ) -> None:
        fake_config = str(tmp_path / "project" / "config.yaml")
        (tmp_path / "project").mkdir()
        Path(fake_config).write_text("Task: x\n")

        mock_dlc.create_new_project.return_value = fake_config

        mgr = ProjectManager()
        result = mgr.create_project(
            "TestProj", "Tester", [], str(tmp_path)
        )

        assert result == fake_config
        mock_dlc.create_new_project.assert_called_once()

    @given(name=_PROJECT_NAME, experimenter=_EXPERIMENTER)
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_names_forwarded_to_dlc(
        self,
        tmp_path: Path,
        name: str,
        experimenter: str,
    ) -> None:
        with patch("src.core.project_manager.deeplabcut") as mock_dlc:
            fake_config = str(tmp_path / "p" / "config.yaml")
            (tmp_path / "p").mkdir(exist_ok=True)
            Path(fake_config).write_text("Task: x\n")
            mock_dlc.create_new_project.return_value = fake_config

            mgr = ProjectManager()
            mgr.create_project(name, experimenter, [], str(tmp_path))

            call_args = mock_dlc.create_new_project.call_args
            assert call_args[0][0] == name
            assert call_args[0][1] == experimenter

