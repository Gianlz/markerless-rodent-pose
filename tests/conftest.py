"""Shared fixtures for the test suite."""

import json
import textwrap
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal DLC-style config.yaml and return its path."""
    video_path = tmp_path / "videos" / "sample.avi"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    cfg = {
        "Task": "test_project",
        "scorer": "tester",
        "bodyparts": ["nose", "left_ear", "right_ear", "tail_base"],
        "skeleton": [["nose", "left_ear"], ["nose", "right_ear"]],
        "video_sets": {str(video_path): {"crop": "0, 640, 0, 480"}},
        "multianimalproject": False,
        "net_type": "resnet_50",
        "init_weights": "imagenet",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Create labeled-data skeleton
    labeled_dir = tmp_path / "labeled-data" / "sample"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    return config_path


@pytest.fixture()
def tmp_multianimal_config(tmp_path: Path) -> Path:
    """Create a multi-animal DLC-style config.yaml."""
    video_path = tmp_path / "videos" / "multi.avi"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    cfg = {
        "Task": "multi_test",
        "scorer": "tester",
        "bodyparts": ["nose", "tail"],
        "skeleton": [],
        "video_sets": {str(video_path): {"crop": "0, 640, 0, 480"}},
        "multianimalproject": True,
        "net_type": "dlcrnet_ms5",
        "init_weights": "imagenet",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


@pytest.fixture()
def models_json(tmp_path: Path) -> Path:
    """Create a models.json config and return its path."""
    data = {
        "networks": {
            "single_animal": ["resnet_50", "resnet_101"],
            "multi_animal": ["dlcrnet_ms5", "efficientnet-b0"],
        },
        "augmenters": ["default", "imgaug"],
        "weight_init": [
            "Transfer Learning - SuperAnimal TopViewMouse",
            "Transfer Learning - ImageNet",
            "Random Initialization",
        ],
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "models.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture()
def tmp_video_file(tmp_path: Path) -> Path:
    """Create a dummy video file at a realistic path."""
    video = tmp_path / "videos" / "test_video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"\x00" * 128)
    return video


@pytest.fixture()
def tmp_yaml_file(tmp_path: Path) -> Path:
    """Create a dummy YAML file (not a DLC config, just a valid YAML)."""
    path = tmp_path / "dummy.yaml"
    path.write_text("key: value\n")
    return path
