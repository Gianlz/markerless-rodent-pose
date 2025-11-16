"""DeepLabCut video inference management"""
from pathlib import Path
from typing import Optional
import deeplabcut
import yaml


class InferenceManager:
    """Handles video analysis and labeled video creation"""
    
    def analyze_videos(
        self,
        config: str,
        videos: list[str],
        shuffle: int = 1,
        trainingsetindex: int = 0,
        gputouse: Optional[int] = None,
        save_as_csv: bool = True,
        destfolder: Optional[str] = None
    ) -> None:
        """
        Analyze videos using trained model
        
        Args:
            config: Path to config.yaml
            videos: List of video paths to analyze
            shuffle: Shuffle index
            trainingsetindex: Training set index
            gputouse: GPU device to use
            save_as_csv: Save results as CSV
            destfolder: Destination folder for results
        """
        deeplabcut.analyze_videos(
            config,
            videos,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=gputouse,
            save_as_csv=save_as_csv,
            destfolder=destfolder
        )
        
        # Filter predictions for smoother tracking
        print("[InferenceManager] Filtering predictions...")
        try:
            deeplabcut.filterpredictions(
                config,
                videos,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex
            )
            print("[InferenceManager] Filtering completed")
        except Exception as e:
            print(f"[InferenceManager] Warning: Could not filter predictions: {e}")
    
    def create_labeled_video(
        self,
        config: str,
        videos: list[str],
        shuffle: int = 1,
        trainingsetindex: int = 0,
        filtered: bool = True,
        draw_skeleton: bool = True,
        trailpoints: int = 0,
        displayedbodyparts: str = 'all',
        destfolder: Optional[str] = None
    ) -> None:
        """
        Create labeled video with pose overlay
        
        Args:
            config: Path to config.yaml
            videos: List of video paths
            shuffle: Shuffle index
            trainingsetindex: Training set index
            filtered: Use filtered predictions
            draw_skeleton: Draw skeleton connections
            trailpoints: Number of trail points (0 = no trail)
            displayedbodyparts: Which bodyparts to display ('all' or list)
            destfolder: Destination folder for videos
        """
        deeplabcut.create_labeled_video(
            config,
            videos,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            filtered=filtered,
            draw_skeleton=draw_skeleton,
            trailpoints=trailpoints,
            displayedbodyparts=displayedbodyparts,
            destfolder=destfolder
        )
    
    def get_best_snapshot(self, config: str, shuffle: int = 1) -> Optional[str]:
        """Get path to best snapshot"""
        project_path = Path(config).parent
        
        # Check dlc-models-pytorch folder (PyTorch models)
        dlc_models_path = project_path / 'dlc-models-pytorch'
        
        print(f"[InferenceManager] Looking for models in: {dlc_models_path}")
        
        if not dlc_models_path.exists():
            print(f"[InferenceManager] dlc-models-pytorch not found")
            return None
        
        iterations = list(dlc_models_path.glob('iteration-*'))
        print(f"[InferenceManager] Found iterations: {[i.name for i in iterations]}")
        
        if not iterations:
            return None
        
        latest = sorted(iterations)[-1]
        shuffle_folders = list(latest.glob(f'*shuffle{shuffle}*'))
        print(f"[InferenceManager] Found shuffle folders: {[f.name for f in shuffle_folders]}")
        
        if not shuffle_folders:
            return None
        
        train_folder = shuffle_folders[0] / 'train'
        print(f"[InferenceManager] Train folder: {train_folder}")
        
        if not train_folder.exists():
            print(f"[InferenceManager] Train folder does not exist")
            return None
        
        # Look for best snapshot
        best_snapshots = list(train_folder.glob('snapshot-best-*.pt'))
        print(f"[InferenceManager] Best snapshots: {[s.name for s in best_snapshots]}")
        
        if best_snapshots:
            snapshot = str(sorted(best_snapshots)[-1])
            print(f"[InferenceManager] Using best snapshot: {snapshot}")
            return snapshot
        
        # Fallback to any snapshot
        snapshots = list(train_folder.glob('snapshot-*.pt'))
        print(f"[InferenceManager] All snapshots: {[s.name for s in snapshots]}")
        
        if snapshots:
            snapshot = str(sorted(snapshots)[-1])
            print(f"[InferenceManager] Using snapshot: {snapshot}")
            return snapshot
        
        print(f"[InferenceManager] No snapshots found")
        return None
    
    def get_bodyparts(self, config: str) -> list[str]:
        """Get list of bodyparts from config"""
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('bodyparts', [])
    
    def check_analysis_exists(self, video_path: str, config: str, shuffle: int = 1) -> bool:
        """Check if video has already been analyzed"""
        video_path = Path(video_path)
        
        # Look for h5 file with DLC naming pattern
        # Pattern: videoname + DLC_scorer + .h5
        h5_pattern = f"{video_path.stem}DLC*.h5"
        h5_files = list(video_path.parent.glob(h5_pattern))
        
        print(f"[InferenceManager] Checking for analysis: {h5_pattern}")
        print(f"[InferenceManager] Found {len(h5_files)} h5 files")
        
        return len(h5_files) > 0
