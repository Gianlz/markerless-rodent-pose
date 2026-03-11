"""DeepLabCut training dataset management"""

from pathlib import Path
import logging
import json
import deeplabcut
import yaml
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingManager:
    """Handles training dataset creation and management"""

    def __init__(self):
        """Initialize training manager and load model config"""
        self.config_path = (
            Path(__file__).parent.parent.parent / "config" / "models.json"
        )
        self.models_config = self._load_models_config()

    def _load_models_config(self) -> dict:
        """Load models configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(
                    f"Models config not found at {self.config_path}, using defaults"
                )
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading models config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration if JSON file is not available"""
        return {
            "networks": {
                "single_animal": ["resnet_50", "resnet_101", "resnet_152"],
                "multi_animal": ["dlcrnet_ms5", "efficientnet-b0"],
            },
            "augmenters": ["default", "imgaug"],
            "weight_init": [
                "Transfer Learning - SuperAnimal TopViewMouse",
                "Transfer Learning - ImageNet",
                "Random Initialization",
            ],
        }

    def reload_config(self):
        """Reload models configuration from JSON file"""
        self.models_config = self._load_models_config()

    def create_training_dataset(
        self,
        config: str,
        num_shuffles: int = 1,
        net_type: str = "resnet_50",
        augmenter_type: str = "default",
        init_weights: str = "imagenet",
    ) -> None:
        """
        Create training dataset from labeled frames

        Args:
            config: Path to config.yaml
            num_shuffles: Number of shuffles for training
            net_type: Network architecture
            augmenter_type: Data augmentation method
            init_weights: Weight initialization ('imagenet', 'superanimal', 'random')
        """
        weight_init = None
        if init_weights == "superanimal":
            try:
                from deeplabcut.modelzoo import build_weight_init
                from deeplabcut import auxiliaryfunctions

                logger.info(
                    "Building SuperAnimal TopViewMouse weight initialization..."
                )
                cfg = auxiliaryfunctions.read_config(config)

                model_name_map = {
                    "resnet_50": "resnet_50",
                    "resnet_101": "resnet_50",
                    "resnet_152": "resnet_50",
                    "hrnet_w32": "hrnet_w32",
                    "mobilenet_v2_1.0": "resnet_50",
                    "mobilenet_v2_0.75": "resnet_50",
                    "efficientnet-b0": "resnet_50",
                }

                model_name = model_name_map.get(net_type, "hrnet_w32")

                weight_init = build_weight_init(
                    cfg=cfg,
                    super_animal="superanimal_topviewmouse",
                    model_name=model_name,
                    detector_name="fasterrcnn_resnet50_fpn_v2",
                    with_decoder=False,
                )
                logger.info(f"SuperAnimal weights configured: model={model_name}")
            except Exception as e:
                logger.warning(f"Could not build SuperAnimal weights: {e}")
                logger.info("Falling back to ImageNet weights")
                weight_init = None

        # Create training dataset
        if weight_init is not None:
            deeplabcut.create_training_dataset(
                config,
                num_shuffles=num_shuffles,
                net_type=net_type,
                augmenter_type=augmenter_type,
                weight_init=weight_init,
            )
        else:
            deeplabcut.create_training_dataset(
                config,
                num_shuffles=num_shuffles,
                net_type=net_type,
                augmenter_type=augmenter_type,
            )

        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["init_weights"] = init_weights
        cfg["net_type"] = net_type
        if init_weights == "superanimal":
            cfg["superanimal_name"] = "superanimal_topviewmouse"

        with open(config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated config: net_type={net_type}, init_weights={init_weights}")

    def create_multianimal_training_dataset(
        self,
        config: str,
        num_shuffles: int = 1,
        net_type: str = "dlcrnet_ms5",
        augmenter_type: str = "imgaug",
        init_weights: str = "imagenet",
    ) -> None:
        """
        Create training dataset for multi-animal project

        Args:
            config: Path to config.yaml
            num_shuffles: Number of shuffles for training
            net_type: Network architecture for multi-animal
            augmenter_type: Data augmentation method
            init_weights: Weight initialization ('imagenet', 'superanimal', 'random')
        """
        deeplabcut.create_multianimaltraining_dataset(
            config,
            num_shuffles=num_shuffles,
            net_type=net_type,
            augmenter_type=augmenter_type,
        )

        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["init_weights"] = init_weights
        cfg["net_type"] = net_type

        with open(config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated config: net_type={net_type}, init_weights={init_weights}")

    def get_available_networks(self, multianimal: bool = False) -> list[str]:
        """Get list of available network architectures from config"""
        key = "multi_animal" if multianimal else "single_animal"
        return self.models_config.get("networks", {}).get(key, ["resnet_50"])

    def get_available_augmenters(self) -> list[str]:
        """Get list of available augmentation methods from config"""
        return self.models_config.get("augmenters", ["default"])

    def get_available_weight_init(self) -> list[str]:
        """Get list of available weight initialization options from config"""
        return self.models_config.get("weight_init", ["Transfer Learning - ImageNet"])

    def is_multianimal_project(self, config: str) -> bool:
        """Check if project is multi-animal"""
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("multianimalproject", False)

    def check_training_dataset_exists(self, config: str) -> dict:
        """Check if training dataset has been created"""
        project_path = Path(config).parent
        training_datasets_path = project_path / "training-datasets"

        if not training_datasets_path.exists():
            return {"exists": False, "message": "No training-datasets folder found"}

        # Look for iteration folders
        iterations = list(training_datasets_path.glob("iteration-*"))

        if not iterations:
            return {"exists": False, "message": "No iteration folders found"}

        # Check latest iteration
        latest = sorted(iterations)[-1]
        train_folders = list(latest.glob("*trainset*"))

        if not train_folders:
            return {"exists": False, "message": f"No training data in {latest.name}"}

        return {
            "exists": True,
            "iterations": len(iterations),
            "latest_iteration": latest.name,
            "train_folders": len(train_folders),
        }
