"""Tests for src.core.frame_extractor – frame extraction logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import yaml
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.core.frame_extractor import FrameExtractor


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_NUM_FRAMES = st.integers(min_value=1, max_value=100)
_STEP_SIZES = st.integers(min_value=1, max_value=10)
_RESIZE_WIDTHS = st.integers(min_value=10, max_value=100)


# ===================================================================
# FrameExtractor.__init__
# ===================================================================


class TestFrameExtractorInit:
    """Test construction."""

    def test_default_config_none(self) -> None:
        fe = FrameExtractor()
        assert fe.config_path is None

    def test_with_config_path(self) -> None:
        fe = FrameExtractor(config_path="/some/config.yaml")
        assert fe.config_path == "/some/config.yaml"


# ===================================================================
# FrameExtractor.extract_frames – dispatch logic
# ===================================================================


class TestFrameExtractorDispatch:
    """Test mode/algo dispatch without heavy I/O."""

    @patch("src.core.frame_extractor.deeplabcut")
    def test_manual_mode_delegates_to_dlc(
        self, mock_dlc: MagicMock
    ) -> None:
        fe = FrameExtractor()
        fe.extract_frames("cfg.yaml", mode="manual")
        mock_dlc.extract_frames.assert_called_once_with(
            "cfg.yaml", mode="manual", userfeedback=False
        )

    @patch.object(FrameExtractor, "_extract_uniform")
    def test_uniform_algo_calls_private(
        self, mock_uniform: MagicMock
    ) -> None:
        fe = FrameExtractor()
        fe.extract_frames("cfg.yaml", mode="automatic", algo="uniform", num_frames=10)
        mock_uniform.assert_called_once_with("cfg.yaml", 10, 1)

    @patch("src.core.frame_extractor.FAISS_AVAILABLE", True)
    @patch.object(FrameExtractor, "_extract_kmeans_faiss")
    def test_kmeans_algo_calls_private(
        self, mock_kmeans: MagicMock
    ) -> None:
        fe = FrameExtractor()
        fe.extract_frames(
            "cfg.yaml",
            mode="automatic",
            algo="kmeans",
            num_frames=15,
            cluster_step=2,
            cluster_resize_width=40,
            cluster_color=True,
        )
        mock_kmeans.assert_called_once_with("cfg.yaml", 15, 2, 40, True)

    @patch("src.core.frame_extractor.FAISS_AVAILABLE", False)
    def test_kmeans_without_faiss_raises(self) -> None:
        fe = FrameExtractor()
        with pytest.raises(ImportError, match="FAISS"):
            fe.extract_frames("cfg.yaml", algo="kmeans")


# ===================================================================
# FrameExtractor._extract_uniform
# ===================================================================


class TestFrameExtractorUniform:
    """Test uniform extraction with mocked cv2."""

    @patch("src.core.frame_extractor.cv2")
    def test_extracts_correct_number_of_frames(
        self, mock_cv2: MagicMock, tmp_config: Path
    ) -> None:
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # 100 total frames
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7  # cv2 constant

        fe = FrameExtractor()
        fe._extract_uniform(str(tmp_config), num_frames=5, step=1)

        # Should write 5 frames
        assert mock_cv2.imwrite.call_count == 5

    @patch("src.core.frame_extractor.cv2")
    def test_skips_unopened_video(
        self, mock_cv2: MagicMock, tmp_config: Path
    ) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        fe = FrameExtractor()
        fe._extract_uniform(str(tmp_config), num_frames=5, step=1)

        mock_cv2.imwrite.assert_not_called()

    @patch("src.core.frame_extractor.cv2")
    def test_handles_read_failure(
        self, mock_cv2: MagicMock, tmp_config: Path
    ) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7

        fe = FrameExtractor()
        fe._extract_uniform(str(tmp_config), num_frames=5, step=1)

        mock_cv2.imwrite.assert_not_called()

    @given(n=_NUM_FRAMES)
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_num_frames_parameter(
        self, tmp_config: Path, n: int
    ) -> None:
        with patch("src.core.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = max(n * 2, 200)
            mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_COUNT = 7

            fe = FrameExtractor()
            fe._extract_uniform(str(tmp_config), num_frames=n, step=1)

            assert mock_cv2.imwrite.call_count == n


# ===================================================================
# FrameExtractor.extract_outlier_frames (mocked DLC)
# ===================================================================


class TestFrameExtractorOutliers:
    """Test outlier frame extraction delegation."""

    @patch("src.core.frame_extractor.deeplabcut")
    def test_delegates_to_dlc(self, mock_dlc: MagicMock) -> None:
        fe = FrameExtractor()
        fe.extract_outlier_frames("cfg.yaml", ["/v.mp4"])
        mock_dlc.extract_outlier_frames.assert_called_once()

    @patch("src.core.frame_extractor.deeplabcut")
    def test_forwards_parameters(self, mock_dlc: MagicMock) -> None:
        fe = FrameExtractor()
        fe.extract_outlier_frames(
            "cfg.yaml",
            ["/v.mp4"],
            outlier_algorithm="fitting",
            epsilon=0.5,
            p_bound=0.05,
            automatic=True,
        )

        call_kwargs = mock_dlc.extract_outlier_frames.call_args[1]
        assert call_kwargs["outlieralgorithm"] == "fitting"
        assert call_kwargs["epsilon"] == 0.5
        assert call_kwargs["p_bound"] == 0.05
        assert call_kwargs["automatic"] is True
