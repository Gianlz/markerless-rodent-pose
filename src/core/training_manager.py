"""DeepLabCut training dataset management"""
from pathlib import Path
from typing import Optional
import logging
import deeplabcut
import yaml

logger = logging.getLogger(__name__)


class TrainingManager:
    """Handles training dataset creation and management"""
    
    def create_training_dataset(
        self,
        config: str,
        num_shuffles: int = 1,
        net_type: str = 'resnet_50',
        augmenter_type: str = 'default',
        init_weights: str = 'imagenet'
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
        if init_weights == 'superanimal':
            try:
                from deeplabcut.modelzoo import build_weight_init
                from deeplabcut import auxiliaryfunctions
                
                logger.info("Building SuperAnimal TopViewMouse weight initialization...")
                cfg = auxiliaryfunctions.read_config(config)
                
                model_name_map = {
                    'resnet_50': 'resnet_50',
                    'resnet_101': 'resnet_50',
                    'resnet_152': 'resnet_50',
                    'hrnet_w32': 'hrnet_w32',
                    'mobilenet_v2_1.0': 'resnet_50',
                    'mobilenet_v2_0.75': 'resnet_50',
                    'efficientnet-b0': 'resnet_50',
                }
                
                model_name = model_name_map.get(net_type, 'hrnet_w32')
                
                weight_init = build_weight_init(
                    cfg=cfg,
                    super_animal='superanimal_topviewmouse',
                    model_name=model_name,
                    detector_name='fasterrcnn_resnet50_fpn_v2',
                    with_decoder=False
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
                weight_init=weight_init
            )
        else:
            deeplabcut.create_training_dataset(
                config,
                num_shuffles=num_shuffles,
                net_type=net_type,
                augmenter_type=augmenter_type
            )
        
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        cfg['init_weights'] = init_weights
        cfg['net_type'] = net_type
        if init_weights == 'superanimal':
            cfg['superanimal_name'] = 'superanimal_topviewmouse'
        
        with open(config, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated config: net_type={net_type}, init_weights={init_weights}")
    
    def create_multianimal_training_dataset(
        self,
        config: str,
        num_shuffles: int = 1,
        net_type: str = 'dlcrnet_ms5',
        augmenter_type: str = 'imgaug',
        init_weights: str = 'imagenet'
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
            augmenter_type=augmenter_type
        )
        
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        cfg['init_weights'] = init_weights
        cfg['net_type'] = net_type
        
        with open(config, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated config: net_type={net_type}, init_weights={init_weights}")
    
    def get_available_networks(self, multianimal: bool = False) -> list[str]:
        """Get list of available network architectures"""
        if multianimal:
            return [
                'dlcrnet_ms5',
                'efficientnet-b0',
                'efficientnet-b1',
                'efficientnet-b2',
                'efficientnet-b3',
                'efficientnet-b4',
                'efficientnet-b5',
                'efficientnet-b6'
            ]
        else:
            return [
                'resnet_50',
                'resnet_101',
                'resnet_152',
                'mobilenet_v2_1.0',
                'mobilenet_v2_0.75',
                'mobilenet_v2_0.5',
                'mobilenet_v2_0.35',
                'efficientnet-b0',
                'efficientnet-b1',
                'efficientnet-b2',
                'efficientnet-b3',
                'efficientnet-b4',
                'efficientnet-b5',
                'efficientnet-b6'
            ]
    
    def get_available_augmenters(self) -> list[str]:
        """Get list of available augmentation methods"""
        return ['default', 'imgaug', 'albumentations', 'tensorpack', 'deterministic']
    
    def is_multianimal_project(self, config: str) -> bool:
        """Check if project is multi-animal"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('multianimalproject', False)
    
    def check_training_dataset_exists(self, config: str) -> dict:
        """Check if training dataset has been created"""
        project_path = Path(config).parent
        training_datasets_path = project_path / 'training-datasets'
        
        if not training_datasets_path.exists():
            return {'exists': False, 'message': 'No training-datasets folder found'}
        
        # Look for iteration folders
        iterations = list(training_datasets_path.glob('iteration-*'))
        
        if not iterations:
            return {'exists': False, 'message': 'No iteration folders found'}
        
        # Check latest iteration
        latest = sorted(iterations)[-1]
        train_folders = list(latest.glob('*trainset*'))
        
        if not train_folders:
            return {'exists': False, 'message': f'No training data in {latest.name}'}
        
        return {
            'exists': True,
            'iterations': len(iterations),
            'latest_iteration': latest.name,
            'train_folders': len(train_folders)
        }
