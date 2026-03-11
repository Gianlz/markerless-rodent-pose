"""Tests for src.utils.video_utils – video re-encoding & integrity checks."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.utils.video_utils import reencode_video, check_video_integrity


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_CRF_VALUES = st.integers(min_value=0, max_value=51)
_PRESETS = st.sampled_from(["ultrafast", "fast", "medium", "slow", "veryslow"])
_CODECS = st.sampled_from(["libx264", "libx265", "libvpx"])


# ===================================================================
# reencode_video
# ===================================================================


class TestReencodeVideo:
    """Test suite for reencode_video."""

    @patch("src.utils.video_utils.subprocess.run")
    def test_default_output_path_uses_reencoded_suffix(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        input_file = tmp_path / "clip.mp4"
        input_file.touch()

        result = reencode_video(str(input_file))

        expected = str(tmp_path / "clip_reencoded.mp4")
        assert result == expected
        mock_run.assert_called_once()

    @patch("src.utils.video_utils.subprocess.run")
    def test_custom_output_path(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        input_file = tmp_path / "clip.avi"
        input_file.touch()
        output_file = tmp_path / "output" / "result.mp4"

        result = reencode_video(str(input_file), output_path=str(output_file))

        assert result == str(output_file)

    @patch("src.utils.video_utils.subprocess.run")
    def test_ffmpeg_command_structure(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        input_file = tmp_path / "vid.mp4"
        input_file.touch()

        reencode_video(str(input_file), codec="libx265", crf=18, preset="slow")

        args = mock_run.call_args
        cmd = args[0][0]  # positional arg 0
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "libx265" in cmd
        assert "18" in cmd
        assert "slow" in cmd
        assert "-y" in cmd

    @patch(
        "src.utils.video_utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"encode error"
        ),
    )
    def test_ffmpeg_failure_raises_runtime_error(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        input_file = tmp_path / "bad.mp4"
        input_file.touch()

        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            reencode_video(str(input_file))

    @patch(
        "src.utils.video_utils.subprocess.run",
        side_effect=FileNotFoundError(),
    )
    def test_ffmpeg_missing_raises_runtime_error(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        input_file = tmp_path / "no_ffmpeg.mp4"
        input_file.touch()

        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            reencode_video(str(input_file))

    @given(crf=_CRF_VALUES, preset=_PRESETS, codec=_CODECS)
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_various_params_produce_valid_command(
        self,
        tmp_path: Path,
        crf: int,
        preset: str,
        codec: str,
    ) -> None:
        with patch("src.utils.video_utils.subprocess.run") as mock_run:
            input_file = tmp_path / "v.mp4"
            input_file.touch()

            result = reencode_video(
                str(input_file), codec=codec, crf=crf, preset=preset
            )

            assert result.endswith("_reencoded.mp4")
            cmd = mock_run.call_args[0][0]
            assert str(crf) in cmd
            assert preset in cmd
            assert codec in cmd


# ===================================================================
# check_video_integrity
# ===================================================================


class TestCheckVideoIntegrity:
    """Test suite for check_video_integrity."""

    @patch("src.utils.video_utils.subprocess.run")
    def test_parses_ffprobe_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            stdout="30/1,1920,1080,120.5,3600", returncode=0
        )

        info = check_video_integrity("/fake/video.mp4")

        assert info["fps"] == pytest.approx(30.0)
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["duration"] == pytest.approx(120.5)
        assert info["packets"] == 3600

    @patch("src.utils.video_utils.subprocess.run")
    def test_single_fps_value(self, mock_run: MagicMock) -> None:
        """When fps is a single number (no denominator '/')."""
        mock_run.return_value = MagicMock(
            stdout="25,640,480,60.0,1500", returncode=0
        )

        info = check_video_integrity("/fake/video.avi")

        assert info["fps"] == pytest.approx(25.0)

    @patch("src.utils.video_utils.subprocess.run")
    def test_short_output_returns_empty(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="30/1,640", returncode=0)

        info = check_video_integrity("/fake/short.mp4")

        assert info == {}

    @patch(
        "src.utils.video_utils.subprocess.run",
        side_effect=FileNotFoundError(),
    )
    def test_ffprobe_missing_returns_empty(self, mock_run: MagicMock) -> None:
        info = check_video_integrity("/fake/noffprobe.mp4")
        assert info == {}

    @patch(
        "src.utils.video_utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffprobe"),
    )
    def test_ffprobe_error_returns_empty(self, mock_run: MagicMock) -> None:
        info = check_video_integrity("/fake/error.mp4")
        assert info == {}

    @patch("src.utils.video_utils.subprocess.run")
    def test_empty_duration_handled(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            stdout="30/1,1920,1080,,3600", returncode=0
        )

        info = check_video_integrity("/fake/nodur.mp4")

        assert info["duration"] == 0
        assert info["packets"] == 3600

    @patch("src.utils.video_utils.subprocess.run")
    def test_missing_packets_field(self, mock_run: MagicMock) -> None:
        """When output has exactly 4 fields (no packets count)."""
        mock_run.return_value = MagicMock(
            stdout="30/1,1920,1080,60.0", returncode=0
        )

        info = check_video_integrity("/fake/nopkt.mp4")

        assert info["fps"] == pytest.approx(30.0)
        assert info["packets"] == 0
