"""DeepLabCut frame extraction core functionality"""
from pathlib import Path
from typing import Optional, Literal
import deeplabcut


class FrameExtractor:
    """Handles frame extraction using DeepLabCut methods"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        
    def extract_frames(
        self,
        config: str,
        mode: Literal['automatic', 'manual'] = 'automatic',
        algo: Literal['uniform', 'kmeans'] = 'kmeans',
        crop: bool = False,
        use_shelve: bool = False,
        cluster_step: int = 1,
        cluster_resize_width: int = 30,
        cluster_color: bool = False,
        opencv: bool = True,
        slider_width: int = 25
    ) -> None:
        """
        Extract frames from videos using DeepLabCut
        
        Args:
            config: Path to config.yaml
            mode: 'automatic' or 'manual' extraction
            algo: 'uniform' or 'kmeans' for automatic mode
            crop: Whether to crop frames
            use_shelve: Use shelve for storing data
            cluster_step: Step size for clustering
            cluster_resize_width: Width for resizing during clustering
            cluster_color: Use color for clustering
            opencv: Use OpenCV for display
            slider_width: Width of slider in manual mode
        """
        deeplabcut.extract_frames(
            config,
            mode=mode,
            algo=algo,
            crop=crop,
            userfeedback=False,
            cluster_step=cluster_step,
            cluster_resizewidth=cluster_resize_width,
            cluster_color=cluster_color,
            opencv=opencv,
            slider_width=slider_width
        )
    
    def extract_outlier_frames(
        self,
        config: str,
        videos: list[str],
        outlier_algorithm: str = 'jump',
        epsilon: float = 0,
        p_bound: float = 0.01,
        automatic: bool = False,
        cluster_step: int = 1,
        cluster_resize_width: int = 30,
        cluster_color: bool = False,
        opencv: bool = True,
        save_frames: bool = True
    ) -> None:
        """
        Extract outlier frames for refinement
        
        Args:
            config: Path to config.yaml
            videos: List of video paths
            outlier_algorithm: Algorithm for outlier detection
            epsilon: Epsilon value for outlier detection
            p_bound: P-value bound
            automatic: Automatic extraction
            cluster_step: Step size for clustering
            cluster_resize_width: Width for resizing
            cluster_color: Use color for clustering
            opencv: Use OpenCV
            save_frames: Save extracted frames
        """
        deeplabcut.extract_outlier_frames(
            config,
            videos,
            outlieralgorithm=outlier_algorithm,
            epsilon=epsilon,
            p_bound=p_bound,
            automatic=automatic,
            cluster_step=cluster_step,
            cluster_resizewidth=cluster_resize_width,
            cluster_color=cluster_color,
            opencv=opencv,
            save_frames=save_frames
        )
