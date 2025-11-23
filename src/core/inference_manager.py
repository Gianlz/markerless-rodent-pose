"""DeepLabCut video inference management"""

from pathlib import Path
from typing import Optional
import logging
import deeplabcut
import yaml

logger = logging.getLogger(__name__)


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
        destfolder: Optional[str] = None,
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
            destfolder=destfolder,
        )

        logger.info("Filtering predictions...")
        try:
            deeplabcut.filterpredictions(
                config, videos, shuffle=shuffle, trainingsetindex=trainingsetindex
            )
            logger.info("Filtering completed")
        except Exception as e:
            logger.warning(f"Could not filter predictions: {e}")

    def create_labeled_video(
        self,
        config: str,
        videos: list[str],
        shuffle: int = 1,
        trainingsetindex: int = 0,
        filtered: bool = True,
        draw_skeleton: bool = True,
        trailpoints: int = 0,
        displayedbodyparts: str = "all",
        destfolder: Optional[str] = None,
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
            destfolder=destfolder,
        )

    def get_best_snapshot(self, config: str, shuffle: int = 1) -> Optional[str]:
        """Get path to best snapshot"""
        project_path = Path(config).parent
        dlc_models_path = project_path / "dlc-models-pytorch"

        logger.debug(f"Looking for models in: {dlc_models_path}")

        if not dlc_models_path.exists():
            logger.debug("dlc-models-pytorch not found")
            return None

        iterations = list(dlc_models_path.glob("iteration-*"))
        logger.debug(f"Found iterations: {[i.name for i in iterations]}")

        if not iterations:
            return None

        latest = sorted(iterations)[-1]
        shuffle_folders = list(latest.glob(f"*shuffle{shuffle}*"))
        logger.debug(f"Found shuffle folders: {[f.name for f in shuffle_folders]}")

        if not shuffle_folders:
            return None

        train_folder = shuffle_folders[0] / "train"

        if not train_folder.exists():
            logger.debug("Train folder does not exist")
            return None

        best_snapshots = list(train_folder.glob("snapshot-best-*.pt"))

        if best_snapshots:
            snapshot = str(sorted(best_snapshots)[-1])
            logger.info(f"Using best snapshot: {snapshot}")
            return snapshot

        snapshots = list(train_folder.glob("snapshot-*.pt"))

        if snapshots:
            snapshot = str(sorted(snapshots)[-1])
            logger.info(f"Using snapshot: {snapshot}")
            return snapshot

        logger.debug("No snapshots found")
        return None

    def get_bodyparts(self, config: str) -> list[str]:
        """Get list of bodyparts from config"""
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("bodyparts", [])

    def check_analysis_exists(
        self, video_path: str, config: str, shuffle: int = 1
    ) -> bool:
        """Check if video has already been analyzed"""
        video_path = Path(video_path)
        h5_pattern = f"{video_path.stem}DLC*.h5"
        h5_files = list(video_path.parent.glob(h5_pattern))

        logger.debug(f"Checking for analysis: {h5_pattern}")
        logger.debug(f"Found {len(h5_files)} h5 files")

        return len(h5_files) > 0
