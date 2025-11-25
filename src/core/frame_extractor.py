"""Frame extraction using FAISS K-means clustering"""

import logging
from pathlib import Path
from typing import Literal, Optional

import cv2
import deeplabcut
import numpy as np
import yaml

from ..utils.device import get_faiss_clustering, get_faiss_index, is_cuda_available

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
    logger.info("FAISS available")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")


class FrameExtractor:
    """Handles frame extraction using FAISS K-means"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path

    def extract_frames(
        self,
        config: str,
        mode: Literal["automatic", "manual"] = "automatic",
        algo: Literal["uniform", "kmeans"] = "kmeans",
        num_frames: int = 20,
        cluster_step: int = 1,
        cluster_resize_width: int = 30,
        cluster_color: bool = False,
    ) -> None:
        """
        Extract frames from videos using FAISS K-means
        Args:
            config: Path to config.yaml
            mode: 'automatic' or 'manual' extraction
            algo: 'uniform' or 'kmeans' for automatic mode
            num_frames: Number of frames to extract per video
            cluster_step: Step size for frame sampling
            cluster_resize_width: Width for resizing during clustering
            cluster_color: Use color for clustering (otherwise grayscale)
        """
        logger.info(f"Starting frame extraction - Mode: {mode}, Algo: {algo}")
        logger.info(f"Config: {config}")
        if mode == "manual":
            logger.info("Using manual extraction mode")
            deeplabcut.extract_frames(config, mode="manual", userfeedback=False)
            return
        if algo == "uniform":
            logger.info(f"Using uniform extraction - {num_frames} frames")
            self._extract_uniform(config, num_frames, cluster_step)
        elif algo == "kmeans":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS not available. Install with: uv add faiss-cpu")
            logger.info(f"Using FAISS K-means - {num_frames} clusters")
            self._extract_kmeans_faiss(
                config, num_frames, cluster_step, cluster_resize_width, cluster_color
            )
        logger.info("Frame extraction completed")

    def _extract_uniform(self, config: str, num_frames: int, step: int) -> None:
        """Extract frames uniformly from videos"""
        logger.info("Loading config file")
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        videos = list(cfg.get("video_sets", {}).keys())
        project_path = Path(config).parent
        logger.info(f"Found {len(videos)} video(s) to process")
        for video_path in videos:
            logger.info(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Total frames: {total_frames}")
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            output_dir = project_path / "labeled-data" / Path(video_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    output_path = output_dir / f"img{idx:04d}.png"
                    cv2.imwrite(str(output_path), frame)
                    logger.info(f"Extracted frame {i + 1}/{num_frames} at index {idx}")
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            cap.release()
            logger.info(f"Completed video: {video_path}")

    def _extract_kmeans_faiss(
        self,
        config: str,
        num_frames: int,
        step: int,
        resize_width: int,
        use_color: bool,
    ) -> None:
        """Extract frames using FAISS K-means clustering (sequential read, CPU)"""
        logger.info("Loading config file")
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        videos = list(cfg.get("video_sets", {}).keys())
        project_path = Path(config).parent
        logger.info(f"Found {len(videos)} video(s) to process")
        for video_path in videos:
            logger.info(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Total frames: {total_frames}")
            # Sequential sampling
            frames_data = []
            valid_indices = []
            frame_count = 0
            logger.info(f"Sampling with step={step}")
            while True:
                if frame_count % step == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize
                    h, w = frame.shape[:2]
                    new_h = int(h * resize_width / w)
                    resized = cv2.resize(frame, (resize_width, new_h))
                    # Convert to feature vector
                    if use_color:
                        feature = resized.flatten()
                    else:
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        feature = gray.flatten()
                    frames_data.append(feature)
                    valid_indices.append(frame_count)
                    if len(frames_data) % 100 == 0:
                        logger.info(f"Sampled {len(frames_data)} frames")
                else:
                    if not cap.grab():
                        break
                frame_count += 1
            cap.release()
            logger.info(f"Sampled {len(frames_data)} valid frames")
            if len(frames_data) < num_frames:
                logger.warning(
                    f"Requested {num_frames} frames but only {len(frames_data)} available"
                )
                num_frames = len(frames_data)
            # FAISS K-means (GPU if available)
            use_gpu = is_cuda_available()
            device_str = "GPU" if use_gpu else "CPU"
            logger.info(f"Running FAISS {device_str} K-means with {num_frames} clusters")
            features = np.array(frames_data, dtype="float32")
            d = features.shape[1]
            logger.info(f"Feature dimension: {d}")

            kmeans, quantizer = get_faiss_clustering(d, num_frames, use_gpu=use_gpu)
            kmeans.train(features, quantizer)
            centroids = faiss.vector_float_to_array(kmeans.centroids)
            centroids = np.array(
                [centroids[i * d : (i + 1) * d] for i in range(num_frames)]
            )
            # Find nearest frame to each centroid
            logger.info("Finding nearest frames to centroids")
            index = get_faiss_index(d, use_gpu=use_gpu)
            index.add(features)
            D, nearest_indices = index.search(centroids, 1)
            selected_indices = [valid_indices[i] for i in nearest_indices.flatten()]
            logger.info(f"Selected {len(selected_indices)} representative frames")
            # Extract and save selected frames (sequential seek minimal)
            cap = cv2.VideoCapture(video_path)
            output_dir = project_path / "labeled-data" / Path(video_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            for i, idx in enumerate(selected_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    output_path = output_dir / f"img{idx:04d}.png"
                    cv2.imwrite(str(output_path), frame)
                    logger.info(
                        f"Saved frame {i + 1}/{len(selected_indices)} at index {idx}"
                    )
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            cap.release()
            logger.info(f"Completed video: {video_path}")

    def extract_outlier_frames(
        self,
        config: str,
        videos: list[str],
        outlier_algorithm: str = "jump",
        epsilon: float = 0,
        p_bound: float = 0.01,
        automatic: bool = False,
        cluster_step: int = 1,
        cluster_resize_width: int = 30,
        cluster_color: bool = False,
        opencv: bool = True,
        save_frames: bool = True,
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
            save_frames=save_frames,
        )
