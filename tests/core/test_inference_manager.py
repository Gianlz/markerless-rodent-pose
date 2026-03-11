"""Tests for src.core.inference_manager – video analysis & snapshot resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.core.inference_manager import InferenceManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_SHUFFLE_INTS = st.integers(min_value=1, max_value=10)


# ===================================================================
# InferenceManager.get_bodyparts
# ===================================================================


class TestInferenceManagerGetBodyparts:
    """Tests for config-driven bodypart retrieval."""

    def test_returns_bodyparts(self, tmp_config: Path) -> None:
        mgr = InferenceManager()
        parts = mgr.get_bodyparts(str(tmp_config))
        assert parts == ["nose", "left_ear", "right_ear", "tail_base"]

    def test_empty_bodyparts(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"Task": "x"}))
        mgr = InferenceManager()
        assert mgr.get_bodyparts(str(cfg_path)) == []


# ===================================================================
# InferenceManager.check_analysis_exists
# ===================================================================


class TestInferenceManagerCheckAnalysis:
    """Tests for checking whether a video has analysis results."""

    def test_no_h5_returns_false(self, tmp_video_file: Path) -> None:
        mgr = InferenceManager()
        assert mgr.check_analysis_exists(str(tmp_video_file), "cfg.yaml") is False

    def test_h5_present_returns_true(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.touch()
        h5 = tmp_path / "clipDLC_resnet50.h5"
        h5.touch()

        mgr = InferenceManager()
        assert mgr.check_analysis_exists(str(video), "cfg.yaml") is True


# ===================================================================
# InferenceManager.get_best_snapshot
# ===================================================================


class TestInferenceManagerGetBestSnapshot:
    """Tests for snapshot resolution logic."""

    def test_no_models_dir_returns_none(self, tmp_config: Path) -> None:
        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config)) is None

    def test_empty_iterations_returns_none(self, tmp_config: Path) -> None:
        models_dir = tmp_config.parent / "dlc-models-pytorch"
        models_dir.mkdir()
        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config)) is None

    def test_no_shuffle_folder_returns_none(self, tmp_config: Path) -> None:
        models_dir = (
            tmp_config.parent / "dlc-models-pytorch" / "iteration-0"
        )
        models_dir.mkdir(parents=True)
        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config)) is None

    def test_no_train_folder_returns_none(self, tmp_config: Path) -> None:
        shuffle_dir = (
            tmp_config.parent
            / "dlc-models-pytorch"
            / "iteration-0"
            / "testDec1-trainset95shuffle1"
        )
        shuffle_dir.mkdir(parents=True)
        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config)) is None

    def test_best_snapshot_found(self, tmp_config: Path) -> None:
        train_dir = (
            tmp_config.parent
            / "dlc-models-pytorch"
            / "iteration-0"
            / "testDec1-trainset95shuffle1"
            / "train"
        )
        train_dir.mkdir(parents=True)
        (train_dir / "snapshot-best-100.pt").touch()
        (train_dir / "snapshot-best-200.pt").touch()

        mgr = InferenceManager()
        result = mgr.get_best_snapshot(str(tmp_config))
        assert result is not None
        assert "snapshot-best-200.pt" in result

    def test_fallback_to_regular_snapshot(self, tmp_config: Path) -> None:
        train_dir = (
            tmp_config.parent
            / "dlc-models-pytorch"
            / "iteration-0"
            / "testDec1-trainset95shuffle1"
            / "train"
        )
        train_dir.mkdir(parents=True)
        (train_dir / "snapshot-50000.pt").touch()

        mgr = InferenceManager()
        result = mgr.get_best_snapshot(str(tmp_config))
        assert result is not None
        assert "snapshot-50000.pt" in result

    def test_no_snapshots_returns_none(self, tmp_config: Path) -> None:
        train_dir = (
            tmp_config.parent
            / "dlc-models-pytorch"
            / "iteration-0"
            / "testDec1-trainset95shuffle1"
            / "train"
        )
        train_dir.mkdir(parents=True)
        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config)) is None

    @given(shuffle=_SHUFFLE_INTS)
    @settings(
        max_examples=5,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_shuffle_parameter_respected(
        self, tmp_config: Path, shuffle: int
    ) -> None:
        """Ensure only matching shuffle dirs are considered."""
        iter_dir = tmp_config.parent / "dlc-models-pytorch" / "iteration-0"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Create a dir that does NOT match the requested shuffle
        wrong = iter_dir / f"test-shuffle{shuffle + 100}"
        wrong.mkdir(exist_ok=True)

        mgr = InferenceManager()
        assert mgr.get_best_snapshot(str(tmp_config), shuffle=shuffle) is None


# ===================================================================
# InferenceManager.analyze_videos (mocked DLC calls)
# ===================================================================


class TestInferenceManagerAnalyze:
    """Tests for analyze_videos with mocked deeplabcut."""

    @patch("src.core.inference_manager.deeplabcut")
    def test_analyze_calls_dlc(self, mock_dlc: MagicMock) -> None:
        mgr = InferenceManager()
        mgr.analyze_videos("cfg.yaml", ["/v.mp4"])

        mock_dlc.analyze_videos.assert_called_once()
        mock_dlc.filterpredictions.assert_called_once()

    @patch("src.core.inference_manager.deeplabcut")
    def test_filter_failure_does_not_propagate(
        self, mock_dlc: MagicMock
    ) -> None:
        mock_dlc.filterpredictions.side_effect = RuntimeError("oops")
        mgr = InferenceManager()
        # Should not raise
        mgr.analyze_videos("cfg.yaml", ["/v.mp4"])

    @patch("src.core.inference_manager.deeplabcut")
    def test_create_labeled_video_calls_dlc(
        self, mock_dlc: MagicMock
    ) -> None:
        mgr = InferenceManager()
        mgr.create_labeled_video("cfg.yaml", ["/v.mp4"])
        mock_dlc.create_labeled_video.assert_called_once()
