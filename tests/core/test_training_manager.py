"""Tests for src.core.training_manager – dataset creation & model config."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.core.training_manager import TrainingManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_NET_TYPES = st.sampled_from(["resnet_50", "resnet_101", "efficientnet-b0"])
_AUGMENTERS = st.sampled_from(["default", "imgaug"])
_NUM_SHUFFLES = st.integers(min_value=1, max_value=5)


# ===================================================================
# TrainingManager.__init__ & config loading
# ===================================================================


class TestTrainingManagerConfig:
    """Tests for model configuration loading."""

    def test_default_config_when_file_missing(self, tmp_path: Path) -> None:
        with patch.object(
            TrainingManager,
            "__init__",
            lambda self: setattr(self, "config_path", tmp_path / "nope.json")
            or setattr(self, "models_config", self._load_models_config()),
        ):
            mgr = TrainingManager()

        assert "networks" in mgr.models_config
        assert "augmenters" in mgr.models_config

    def test_loads_from_json_file(self, models_json: Path) -> None:
        with patch.object(
            TrainingManager,
            "__init__",
            lambda self: setattr(self, "config_path", models_json)
            or setattr(self, "models_config", self._load_models_config()),
        ):
            mgr = TrainingManager()

        assert "resnet_50" in mgr.models_config["networks"]["single_animal"]

    def test_reload_config(self, models_json: Path) -> None:
        with patch.object(
            TrainingManager,
            "__init__",
            lambda self: setattr(self, "config_path", models_json)
            or setattr(self, "models_config", self._load_models_config()),
        ):
            mgr = TrainingManager()

        # Modify the file
        data = json.loads(models_json.read_text())
        data["networks"]["single_animal"].append("new_net")
        models_json.write_text(json.dumps(data))

        mgr.reload_config()
        assert "new_net" in mgr.models_config["networks"]["single_animal"]


# ===================================================================
# TrainingManager.get_available_*
# ===================================================================


class TestTrainingManagerGetters:
    """Tests for convenience getters."""

    def _make_mgr(self, models_json: Path) -> TrainingManager:
        """Helper to build a manager with custom config path."""

        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        return _TM(models_json)

    def test_get_available_networks_single(self, models_json: Path) -> None:
        mgr = self._make_mgr(models_json)
        nets = mgr.get_available_networks(multianimal=False)
        assert "resnet_50" in nets

    def test_get_available_networks_multi(self, models_json: Path) -> None:
        mgr = self._make_mgr(models_json)
        nets = mgr.get_available_networks(multianimal=True)
        assert "dlcrnet_ms5" in nets

    def test_get_available_augmenters(self, models_json: Path) -> None:
        mgr = self._make_mgr(models_json)
        augs = mgr.get_available_augmenters()
        assert "default" in augs
        assert "imgaug" in augs

    def test_get_available_weight_init(self, models_json: Path) -> None:
        mgr = self._make_mgr(models_json)
        inits = mgr.get_available_weight_init()
        assert any("ImageNet" in w for w in inits)


# ===================================================================
# TrainingManager.is_multianimal_project
# ===================================================================


class TestTrainingManagerMultianimal:
    """Tests for multi-animal boolean check."""

    def _make_mgr(self, models_json: Path) -> TrainingManager:
        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        return _TM(models_json)

    def test_single_animal(
        self, tmp_config: Path, models_json: Path
    ) -> None:
        mgr = self._make_mgr(models_json)
        assert mgr.is_multianimal_project(str(tmp_config)) is False

    def test_multi_animal(
        self, tmp_multianimal_config: Path, models_json: Path
    ) -> None:
        mgr = self._make_mgr(models_json)
        assert mgr.is_multianimal_project(str(tmp_multianimal_config)) is True


# ===================================================================
# TrainingManager.check_training_dataset_exists
# ===================================================================


class TestTrainingManagerCheckDataset:
    """Tests for training dataset existence checks."""

    def _make_mgr(self, models_json: Path) -> TrainingManager:
        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        return _TM(models_json)

    def test_no_training_datasets_dir(
        self, tmp_config: Path, models_json: Path
    ) -> None:
        mgr = self._make_mgr(models_json)
        result = mgr.check_training_dataset_exists(str(tmp_config))
        assert result["exists"] is False

    def test_empty_training_datasets_dir(
        self, tmp_config: Path, models_json: Path
    ) -> None:
        (tmp_config.parent / "training-datasets").mkdir()
        mgr = self._make_mgr(models_json)
        result = mgr.check_training_dataset_exists(str(tmp_config))
        assert result["exists"] is False

    def test_dataset_exists(
        self, tmp_config: Path, models_json: Path
    ) -> None:
        trainset_dir = (
            tmp_config.parent
            / "training-datasets"
            / "iteration-0"
            / "UnaugmentedDataSet_testDec1-trainset95shuffle1"
        )
        trainset_dir.mkdir(parents=True)
        mgr = self._make_mgr(models_json)
        result = mgr.check_training_dataset_exists(str(tmp_config))
        assert result["exists"] is True
        assert result["iterations"] == 1


# ===================================================================
# TrainingManager.create_training_dataset (mocked DLC)
# ===================================================================


class TestTrainingManagerCreate:
    """Mocked DLC calls for dataset creation."""

    @patch("src.core.training_manager.deeplabcut")
    def test_create_calls_dlc(
        self, mock_dlc: MagicMock, tmp_config: Path, models_json: Path
    ) -> None:
        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        mgr = _TM(models_json)
        mgr.create_training_dataset(str(tmp_config))

        mock_dlc.create_training_dataset.assert_called_once()

    @patch("src.core.training_manager.deeplabcut")
    def test_config_updated_after_creation(
        self, mock_dlc: MagicMock, tmp_config: Path, models_json: Path
    ) -> None:
        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        mgr = _TM(models_json)
        mgr.create_training_dataset(
            str(tmp_config), net_type="resnet_101", init_weights="random"
        )

        with open(tmp_config) as f:
            cfg = yaml.safe_load(f)
        assert cfg["net_type"] == "resnet_101"
        assert cfg["init_weights"] == "random"

    @given(net=_NET_TYPES, aug=_AUGMENTERS, n=_NUM_SHUFFLES)
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_params_forwarded(
        self,
        tmp_config: Path,
        models_json: Path,
        net: str,
        aug: str,
        n: int,
    ) -> None:
        with patch("src.core.training_manager.deeplabcut") as mock_dlc:
            class _TM(TrainingManager):
                def __init__(self, path: Path) -> None:
                    self.config_path = path
                    self.models_config = self._load_models_config()

            mgr = _TM(models_json)
            mgr.create_training_dataset(
                str(tmp_config), num_shuffles=n, net_type=net, augmenter_type=aug
            )

            call_kwargs = mock_dlc.create_training_dataset.call_args[1]
            assert call_kwargs["num_shuffles"] == n
            assert call_kwargs["net_type"] == net
            assert call_kwargs["augmenter_type"] == aug

    @patch("src.core.training_manager.deeplabcut")
    def test_create_multianimal_calls_dlc(
        self,
        mock_dlc: MagicMock,
        tmp_multianimal_config: Path,
        models_json: Path,
    ) -> None:
        class _TM(TrainingManager):
            def __init__(self, path: Path) -> None:
                self.config_path = path
                self.models_config = self._load_models_config()

        mgr = _TM(models_json)
        mgr.create_multianimal_training_dataset(str(tmp_multianimal_config))

        mock_dlc.create_multianimaltraining_dataset.assert_called_once()
