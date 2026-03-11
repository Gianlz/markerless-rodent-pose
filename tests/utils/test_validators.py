"""Tests for src.utils.validators – path validation helpers."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.utils.validators import validate_config_path, validate_video_path


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_YAML_EXTENSIONS = st.sampled_from([".yaml", ".yml"])
_VIDEO_EXTENSIONS = st.sampled_from([".mp4", ".avi", ".mov", ".mkv"])
_INVALID_EXTENSIONS = st.sampled_from([".txt", ".png", ".json", ".py", ".csv", ".xml"])
_SAFE_STEM = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_]{0,29}", fullmatch=True)


# ===================================================================
# validate_config_path
# ===================================================================


class TestValidateConfigPath:
    """Test suite for validate_config_path."""

    def test_empty_string_returns_invalid(self) -> None:
        valid, msg = validate_config_path("")
        assert valid is False
        assert msg is not None
        assert "required" in msg.lower()

    def test_nonexistent_path_returns_invalid(self, tmp_path: Path) -> None:
        fake = str(tmp_path / "nonexistent" / "config.yaml")
        valid, msg = validate_config_path(fake)
        assert valid is False
        assert "not found" in msg.lower()

    def test_non_yaml_extension_returns_invalid(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "config.json"
        bad_file.touch()
        valid, msg = validate_config_path(str(bad_file))
        assert valid is False
        assert "YAML" in msg

    def test_valid_yaml_path(self, tmp_config: Path) -> None:
        valid, msg = validate_config_path(str(tmp_config))
        assert valid is True
        assert msg is None

    def test_valid_yml_extension(self, tmp_path: Path) -> None:
        yml_file = tmp_path / "settings.yml"
        yml_file.write_text("key: value\n")
        valid, msg = validate_config_path(str(yml_file))
        assert valid is True
        assert msg is None

    @given(ext=_INVALID_EXTENSIONS, stem=_SAFE_STEM)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_extensions_rejected(
        self, tmp_path: Path, ext: str, stem: str
    ) -> None:
        bad_file = tmp_path / f"{stem}{ext}"
        bad_file.touch()
        valid, _ = validate_config_path(str(bad_file))
        assert valid is False

    @given(ext=_YAML_EXTENSIONS, stem=_SAFE_STEM)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_existing_yaml_accepted(
        self, tmp_path: Path, ext: str, stem: str
    ) -> None:
        good_file = tmp_path / f"{stem}{ext}"
        good_file.write_text("x: 1\n")
        valid, msg = validate_config_path(str(good_file))
        assert valid is True
        assert msg is None


# ===================================================================
# validate_video_path
# ===================================================================


class TestValidateVideoPath:
    """Test suite for validate_video_path."""

    def test_empty_string_returns_invalid(self) -> None:
        valid, msg = validate_video_path("")
        assert valid is False
        assert msg is not None
        assert "required" in msg.lower()

    def test_nonexistent_path_returns_invalid(self, tmp_path: Path) -> None:
        fake = str(tmp_path / "nope.mp4")
        valid, msg = validate_video_path(fake)
        assert valid is False
        assert "not found" in msg.lower()

    def test_invalid_extension(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "video.gif"
        bad_file.touch()
        valid, msg = validate_video_path(str(bad_file))
        assert valid is False
        assert "Invalid video format" in msg

    def test_valid_mp4(self, tmp_video_file: Path) -> None:
        valid, msg = validate_video_path(str(tmp_video_file))
        assert valid is True
        assert msg is None

    @given(ext=_VIDEO_EXTENSIONS, stem=_SAFE_STEM)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_all_valid_extensions_accepted(
        self, tmp_path: Path, ext: str, stem: str
    ) -> None:
        vid = tmp_path / f"{stem}{ext}"
        vid.write_bytes(b"\x00")
        valid, msg = validate_video_path(str(vid))
        assert valid is True
        assert msg is None

    @given(ext=_INVALID_EXTENSIONS, stem=_SAFE_STEM)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_extensions_rejected(
        self, tmp_path: Path, ext: str, stem: str
    ) -> None:
        vid = tmp_path / f"{stem}{ext}"
        vid.touch()
        valid, _ = validate_video_path(str(vid))
        assert valid is False

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Uppercase extension should also be accepted."""
        vid = tmp_path / "clip.MP4"
        vid.write_bytes(b"\x00")
        valid, msg = validate_video_path(str(vid))
        assert valid is True
        assert msg is None
