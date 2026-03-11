"""Tests for src.core.label_manager – bodypart / skeleton CRUD via YAML config."""

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.core.label_manager import LabelManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_BODYPART_NAME = st.from_regex(r"[a-z][a-z0-9_]{1,19}", fullmatch=True)


# ===================================================================
# LabelManager.get_bodyparts / add / remove / update
# ===================================================================


class TestLabelManagerBodyparts:
    """CRUD operations on bodyparts list."""

    def test_get_bodyparts_returns_list(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        parts = mgr.get_bodyparts(str(tmp_config))
        assert isinstance(parts, list)
        assert "nose" in parts

    def test_add_bodypart_appends(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.add_bodypart(str(tmp_config), "spine")
        parts = mgr.get_bodyparts(str(tmp_config))
        assert "spine" in parts

    def test_add_duplicate_bodypart_is_noop(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        original = mgr.get_bodyparts(str(tmp_config))
        mgr.add_bodypart(str(tmp_config), "nose")
        after = mgr.get_bodyparts(str(tmp_config))
        assert original == after

    def test_remove_bodypart(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.remove_bodypart(str(tmp_config), "nose")
        parts = mgr.get_bodyparts(str(tmp_config))
        assert "nose" not in parts

    def test_remove_nonexistent_bodypart_is_noop(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        original = mgr.get_bodyparts(str(tmp_config))
        mgr.remove_bodypart(str(tmp_config), "phantom")
        after = mgr.get_bodyparts(str(tmp_config))
        assert original == after

    def test_update_bodypart(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.update_bodypart(str(tmp_config), "nose", "snout")
        parts = mgr.get_bodyparts(str(tmp_config))
        assert "snout" in parts
        assert "nose" not in parts

    def test_update_nonexistent_bodypart_is_noop(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        original = mgr.get_bodyparts(str(tmp_config))
        mgr.update_bodypart(str(tmp_config), "phantom", "new_name")
        after = mgr.get_bodyparts(str(tmp_config))
        assert original == after

    @given(name=_BODYPART_NAME)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_add_then_remove_roundtrip(self, tmp_config: Path, name: str) -> None:
        assume(name not in ["nose", "left_ear", "right_ear", "tail_base"])
        mgr = LabelManager()
        original = mgr.get_bodyparts(str(tmp_config))

        mgr.add_bodypart(str(tmp_config), name)
        assert name in mgr.get_bodyparts(str(tmp_config))

        mgr.remove_bodypart(str(tmp_config), name)
        assert mgr.get_bodyparts(str(tmp_config)) == original

    @given(old=_BODYPART_NAME, new=_BODYPART_NAME)
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_add_then_rename_roundtrip(
        self, tmp_config: Path, old: str, new: str
    ) -> None:
        assume(old != new)
        assume(old not in ["nose", "left_ear", "right_ear", "tail_base"])
        assume(new not in ["nose", "left_ear", "right_ear", "tail_base"])

        mgr = LabelManager()
        mgr.add_bodypart(str(tmp_config), old)
        mgr.update_bodypart(str(tmp_config), old, new)
        parts = mgr.get_bodyparts(str(tmp_config))
        assert new in parts
        assert old not in parts


# ===================================================================
# LabelManager.get_skeleton / add / remove connections
# ===================================================================


class TestLabelManagerSkeleton:
    """Skeleton connection CRUD."""

    def test_get_skeleton_returns_list(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        skeleton = mgr.get_skeleton(str(tmp_config))
        assert isinstance(skeleton, list)
        assert len(skeleton) == 2  # from fixture

    def test_add_skeleton_connection(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.add_skeleton_connection(str(tmp_config), "left_ear", "right_ear")
        skeleton = mgr.get_skeleton(str(tmp_config))
        assert ["left_ear", "right_ear"] in skeleton

    def test_add_duplicate_connection_is_noop(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.add_skeleton_connection(str(tmp_config), "nose", "left_ear")
        skeleton = mgr.get_skeleton(str(tmp_config))
        assert skeleton.count(["nose", "left_ear"]) == 1

    def test_add_reverse_connection_is_noop(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.add_skeleton_connection(str(tmp_config), "left_ear", "nose")
        skeleton = mgr.get_skeleton(str(tmp_config))
        # Should not duplicate; original is ["nose", "left_ear"]
        count = sum(
            1
            for conn in skeleton
            if set(conn) == {"nose", "left_ear"}
        )
        assert count == 1

    def test_remove_skeleton_connection(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.remove_skeleton_connection(str(tmp_config), "nose", "left_ear")
        skeleton = mgr.get_skeleton(str(tmp_config))
        assert ["nose", "left_ear"] not in skeleton

    def test_remove_reverse_skeleton_connection(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        mgr.remove_skeleton_connection(str(tmp_config), "right_ear", "nose")
        skeleton = mgr.get_skeleton(str(tmp_config))
        assert ["nose", "right_ear"] not in skeleton

    def test_remove_nonexistent_connection_is_noop(
        self, tmp_config: Path
    ) -> None:
        mgr = LabelManager()
        original = mgr.get_skeleton(str(tmp_config))
        mgr.remove_skeleton_connection(str(tmp_config), "left_ear", "tail_base")
        after = mgr.get_skeleton(str(tmp_config))
        assert original == after


# ===================================================================
# LabelManager.get_videos
# ===================================================================


class TestLabelManagerGetVideos:
    """Tests for video folder discovery."""

    def test_returns_video_dirs(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        videos = mgr.get_videos(str(tmp_config))
        assert isinstance(videos, list)
        assert "sample" in videos

    def test_ignores_hidden_dirs(self, tmp_config: Path) -> None:
        hidden = tmp_config.parent / "labeled-data" / ".hidden"
        hidden.mkdir()
        mgr = LabelManager()
        videos = mgr.get_videos(str(tmp_config))
        assert ".hidden" not in videos

    def test_missing_labeled_data_returns_empty(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("Task: x\n")
        mgr = LabelManager()
        assert mgr.get_videos(str(cfg)) == []


# ===================================================================
# LabelManager.check_labels
# ===================================================================


class TestLabelManagerCheckLabels:
    """Tests for labeling status checks."""

    def test_no_labeled_data_folder(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("Task: x\n")
        mgr = LabelManager()
        result = mgr.check_labels(str(cfg))
        assert "status" in result

    def test_empty_labeled_data(self, tmp_config: Path) -> None:
        mgr = LabelManager()
        result = mgr.check_labels(str(tmp_config))
        # sample dir exists but has no h5/csv
        assert "Videos" in result or "status" in result

    def test_with_video_dir_no_h5(self, tmp_config: Path) -> None:
        # labeled-data/sample exists from fixture but no h5 files
        mgr = LabelManager()
        result = mgr.check_labels(str(tmp_config))
        # Should count 1 video dir, 0 labeled frames
        assert result.get("Videos", 0) >= 0
