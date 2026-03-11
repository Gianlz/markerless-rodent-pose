"""Tests for src.core.train_manager – training orchestration & snapshot discovery."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.train_manager import TrainManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_SHUFFLE_INTS = st.integers(min_value=1, max_value=10)


# ===================================================================
# TrainManager.get_available_shuffles
# ===================================================================


class TestTrainManagerShuffles:
    """Test shuffle discovery from filesystem."""

    def test_no_training_datasets_returns_default(self, tmp_config: Path) -> None:
        mgr = TrainManager()
        assert mgr.get_available_shuffles(str(tmp_config)) == [1]

    def test_empty_iteration_returns_default(self, tmp_config: Path) -> None:
        td = tmp_config.parent / "training-datasets"
        td.mkdir()
        mgr = TrainManager()
        assert mgr.get_available_shuffles(str(tmp_config)) == [1]

    def test_discovers_shuffles(self, tmp_config: Path) -> None:
        iter_dir = (
            tmp_config.parent / "training-datasets" / "iteration-0"
        )
        iter_dir.mkdir(parents=True)
        (iter_dir / "UnaugmentedDataSet_testDec1-trainset95shuffle1").mkdir()
        (iter_dir / "UnaugmentedDataSet_testDec1-trainset95shuffle3").mkdir()

        mgr = TrainManager()
        shuffles = mgr.get_available_shuffles(str(tmp_config))
        assert 1 in shuffles
        assert 3 in shuffles
        assert shuffles == sorted(shuffles)


# ===================================================================
# TrainManager.get_available_snapshots
# ===================================================================


class TestTrainManagerSnapshots:
    """Test snapshot file discovery."""

    def test_no_training_datasets_returns_empty(self, tmp_config: Path) -> None:
        mgr = TrainManager()
        assert mgr.get_available_snapshots(str(tmp_config)) == []

    def test_finds_snapshot_index_files(self, tmp_config: Path) -> None:
        train_dir = (
            tmp_config.parent
            / "training-datasets"
            / "iteration-0"
            / "UnaugmentedDataSet_testDec1-trainset95shuffle1"
            / "train"
        )
        train_dir.mkdir(parents=True)
        (train_dir / "snapshot-50000.index").touch()
        (train_dir / "snapshot-100000.index").touch()

        mgr = TrainManager()
        snaps = mgr.get_available_snapshots(str(tmp_config))
        assert len(snaps) == 2
        assert "snapshot-50000" in snaps
        assert "snapshot-100000" in snaps


# ===================================================================
# TrainManager.get_training_info
# ===================================================================


class TestTrainManagerInfo:
    """Test training info retrieval."""

    def test_returns_expected_keys(self, tmp_config: Path) -> None:
        mgr = TrainManager()
        info = mgr.get_training_info(str(tmp_config))

        assert info["net_type"] == "resnet_50"
        assert info["init_weights"] == "imagenet"
        assert info["multianimal"] is False
        assert isinstance(info["training_dataset_exists"], bool)

    def test_multianimal_project(self, tmp_multianimal_config: Path) -> None:
        mgr = TrainManager()
        assert mgr.is_multianimal_project(str(tmp_multianimal_config)) is True

    def test_single_animal_project(self, tmp_config: Path) -> None:
        mgr = TrainManager()
        assert mgr.is_multianimal_project(str(tmp_config)) is False


# ===================================================================
# TrainManager.train_network (mocked DLC)
# ===================================================================


class TestTrainManagerTrain:
    """Mocked calls to deeplabcut.train_network."""

    @patch("src.core.train_manager.deeplabcut")
    def test_train_calls_dlc(
        self, mock_dlc: MagicMock, tmp_config: Path
    ) -> None:
        mgr = TrainManager()
        mgr.train_network(str(tmp_config), maxiters=100)
        mock_dlc.train_network.assert_called_once()

    @patch("src.core.train_manager.deeplabcut")
    def test_train_forwards_parameters(
        self, mock_dlc: MagicMock, tmp_config: Path
    ) -> None:
        mgr = TrainManager()
        mgr.train_network(
            str(tmp_config),
            shuffle=2,
            maxiters=500,
            displayiters=10,
            saveiters=100,
        )

        call_kwargs = mock_dlc.train_network.call_args[1]
        assert call_kwargs["shuffle"] == 2
        assert call_kwargs["epochs"] == 500
        assert call_kwargs["displayiters"] == 10
        assert call_kwargs["save_epochs"] == 100

    @patch("src.core.train_manager.deeplabcut")
    def test_superanimal_warning_logged(
        self, mock_dlc: MagicMock, tmp_config: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Set init_weights to superanimal
        with open(tmp_config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["init_weights"] = "superanimal"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        import logging

        with caplog.at_level(logging.WARNING):
            mgr = TrainManager()
            mgr.train_network(str(tmp_config))

        assert any("SuperAnimal" in r.message for r in caplog.records)
