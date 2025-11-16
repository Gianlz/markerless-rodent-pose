"""DeepLabCut model training management"""
from pathlib import Path
from typing import Optional
import deeplabcut
import yaml
import os


class TrainManager:
    """Handles model training"""
    
    def train_network(
        self,
        config: str,
        shuffle: int = 1,
        trainingsetindex: int = 0,
        max_snapshots_to_keep: int = 5,
        displayiters: int = 1000,
        saveiters: int = 50000,
        maxiters: int = 200000,
        allow_growth: bool = True,
        gputouse: Optional[int] = None
    ) -> None:
        """
        Train the network
        
        Args:
            config: Path to config.yaml
            shuffle: Shuffle index
            trainingsetindex: Training set index
            max_snapshots_to_keep: Number of snapshots to keep
            displayiters: Display iterations
            saveiters: Save iterations
            maxiters: Maximum iterations
            allow_growth: Allow GPU memory growth
            gputouse: GPU device to use (None for default)
        """
        # Log training configuration
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        init_weights = cfg.get('init_weights', 'imagenet')
        net_type = cfg.get('net_type', 'resnet_50')
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Network: {net_type}")
        print(f"Init Weights: {init_weights}")
        print(f"Shuffle: {shuffle}")
        print(f"Max Iterations: {maxiters}")
        print(f"Display Iterations: {displayiters}")
        print(f"Save Iterations: {saveiters}")
        print(f"Snapshots to Keep: {max_snapshots_to_keep}")
        
        # Handle SuperAnimal weights
        if init_weights == 'superanimal':
            print("\n" + "!"*60)
            print("WARNING: SuperAnimal weights must be set during dataset creation!")
            print("If you created the training dataset with ImageNet weights,")
            print("you need to recreate it with SuperAnimal weights selected.")
            print("Go back to 'Create Training Dataset' tab and recreate the dataset.")
            print("!"*60)
        
        print("="*60 + "\n")
        
        deeplabcut.train_network(
            config,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            max_snapshots_to_keep=max_snapshots_to_keep,
            displayiters=displayiters,
            saveiters=saveiters,
            maxiters=maxiters,
            allow_growth=allow_growth,
            gputouse=gputouse
        )
    
    def get_available_shuffles(self, config: str) -> list[int]:
        """Get list of available shuffles from training-datasets"""
        project_path = Path(config).parent
        training_datasets_path = project_path / 'training-datasets'
        
        if not training_datasets_path.exists():
            return [1]
        
        # Look for latest iteration
        iterations = list(training_datasets_path.glob('iteration-*'))
        if not iterations:
            return [1]
        
        latest = sorted(iterations)[-1]
        
        # Find shuffle folders
        shuffles = set()
        for folder in latest.iterdir():
            if folder.is_dir() and 'shuffle' in folder.name:
                # Extract shuffle number from folder name
                parts = folder.name.split('shuffle')
                if len(parts) > 1:
                    try:
                        shuffle_num = int(parts[1].split('_')[0])
                        shuffles.add(shuffle_num)
                    except (ValueError, IndexError):
                        pass
        
        return sorted(list(shuffles)) if shuffles else [1]
    
    def get_available_snapshots(self, config: str, shuffle: int = 1) -> list[str]:
        """Get list of available training snapshots"""
        project_path = Path(config).parent
        
        # Find the training folder
        training_datasets_path = project_path / 'training-datasets'
        if not training_datasets_path.exists():
            return []
        
        iterations = list(training_datasets_path.glob('iteration-*'))
        if not iterations:
            return []
        
        latest = sorted(iterations)[-1]
        
        # Find shuffle folder
        shuffle_folders = list(latest.glob(f'*shuffle{shuffle}*'))
        if not shuffle_folders:
            return []
        
        shuffle_folder = shuffle_folders[0]
        
        # Look for train folder
        train_folder = shuffle_folder / 'train'
        if not train_folder.exists():
            return []
        
        # Find snapshot files
        snapshots = []
        for snapshot in train_folder.glob('snapshot-*.index'):
            snapshot_name = snapshot.stem  # Remove .index extension
            snapshots.append(snapshot_name)
        
        return sorted(snapshots)
    
    def get_training_info(self, config: str) -> dict:
        """Get training configuration info"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        project_path = Path(config).parent
        training_datasets_path = project_path / 'training-datasets'
        
        info = {
            'project_path': str(project_path),
            'net_type': cfg.get('net_type', 'resnet_50'),
            'init_weights': cfg.get('init_weights', 'imagenet'),
            'multianimal': cfg.get('multianimalproject', False),
            'training_dataset_exists': training_datasets_path.exists()
        }
        
        return info
    
    def is_multianimal_project(self, config: str) -> bool:
        """Check if project is multi-animal"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('multianimalproject', False)
