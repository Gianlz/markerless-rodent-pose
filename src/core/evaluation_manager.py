"""Model evaluation manager for computing metrics"""

import logging
from pathlib import Path
from typing import Callable, Optional

import deeplabcut
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationManager:
    """Handles model evaluation and metrics computation"""

    def evaluate_network(
        self,
        config: str,
        shuffle: int = 1,
        trainingsetindex: int = 0,
        plotting: bool = False,
        show_errors: bool = True,
        comparisonbodyparts: str = "all",
        gputouse: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Evaluate network and compute metrics.

        Returns dict with metrics: precision, recall, f1, iou, rmse, etc.
        """
        if progress_callback:
            progress_callback("Running DeepLabCut evaluation...")

        deeplabcut.evaluate_network(
            config,
            Shuffles=[shuffle],
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=show_errors,
            comparisonbodyparts=comparisonbodyparts,
            gputouse=gputouse,
        )

        if progress_callback:
            progress_callback("Loading evaluation results...")

        # Find evaluation results
        results = self._load_evaluation_results(config, shuffle)

        if progress_callback:
            progress_callback("Computing additional metrics...")

        # Compute precision, recall, F1, IoU from the evaluation data
        metrics = self._compute_metrics(config, shuffle, results)

        return metrics

    def _load_evaluation_results(self, config: str, shuffle: int) -> dict:
        """Load evaluation results from DLC output"""
        project_path = Path(config).parent
        eval_path = project_path / "evaluation-results"

        if not eval_path.exists():
            return {}

        # Find latest iteration
        iterations = list(eval_path.glob("iteration-*"))
        if not iterations:
            return {}

        latest = sorted(iterations)[-1]

        # Find CSV results
        csv_files = list(latest.glob("*shuffle*/*.csv"))
        if not csv_files:
            csv_files = list(latest.glob("*.csv"))

        results = {}
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                results[csv_file.stem] = df
            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")

        return results

    def _compute_metrics(self, config: str, shuffle: int, eval_results: dict) -> dict:
        """Compute precision, recall, F1, IoU from evaluation data"""
        project_path = Path(config).parent

        metrics = {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "iou": None,
            "train_error": None,
            "test_error": None,
            "pcutoff": None,
        }

        # Load DLC evaluation CSV if available
        if eval_results:
            for name, df in eval_results.items():
                if "Train" in df.columns or "train" in df.columns.str.lower():
                    train_col = [c for c in df.columns if "train" in c.lower()]
                    test_col = [c for c in df.columns if "test" in c.lower()]
                    if train_col:
                        metrics["train_error"] = df[train_col[0]].iloc[0]
                    if test_col:
                        metrics["test_error"] = df[test_col[0]].iloc[0]

        # Compute metrics from labeled vs predicted data
        labeled_data_path = project_path / "labeled-data"
        if labeled_data_path.exists():
            precision, recall, f1, iou = self._compute_detection_metrics(
                config, shuffle, labeled_data_path
            )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1
            metrics["iou"] = iou

        return metrics

    def _compute_detection_metrics(
        self,
        config: str,
        shuffle: int,
        labeled_data_path: Path,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute detection metrics by comparing predictions to ground truth.

        Uses a threshold-based approach where a detection is considered:
        - True Positive: predicted point within threshold of ground truth
        - False Positive: predicted point with no nearby ground truth
        - False Negative: ground truth point with no nearby prediction
        """
        project_path = Path(config).parent

        # Find evaluation predictions
        eval_path = project_path / "evaluation-results"
        if not eval_path.exists():
            return None, None, None, None

        iterations = list(eval_path.glob("iteration-*"))
        if not iterations:
            return None, None, None, None

        latest = sorted(iterations)[-1]

        # Find prediction h5 files
        h5_files = list(latest.glob("**/*DLC*.h5"))
        if not h5_files:
            return None, None, None, None

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou_sum = 0.0
        total_iou_count = 0

        threshold = 10.0  # pixels - detection threshold
        pcutoff = 0.6  # likelihood cutoff

        for h5_file in h5_files:
            try:
                pred_df = pd.read_hdf(h5_file)

                # Find corresponding ground truth
                # The h5 file name contains the image folder name
                folder_name = None
                for folder in labeled_data_path.iterdir():
                    if folder.is_dir() and folder.name in h5_file.stem:
                        folder_name = folder
                        break

                if folder_name is None:
                    continue

                # Load ground truth
                gt_file = folder_name / "CollectedData.h5"
                if not gt_file.exists():
                    gt_file = folder_name / "CollectedData.csv"
                    if gt_file.exists():
                        gt_df = pd.read_csv(gt_file, header=[0, 1, 2], index_col=0)
                    else:
                        continue
                else:
                    gt_df = pd.read_hdf(gt_file)

                # Compare predictions to ground truth
                scorer = pred_df.columns.levels[0][0]
                bodyparts = pred_df.columns.levels[1]

                for idx in pred_df.index:
                    if idx not in gt_df.index:
                        continue

                    for bp in bodyparts:
                        try:
                            # Get prediction
                            pred_x = pred_df.loc[idx, (scorer, bp, "x")]
                            pred_y = pred_df.loc[idx, (scorer, bp, "y")]
                            pred_likelihood = pred_df.loc[
                                idx, (scorer, bp, "likelihood")
                            ]

                            # Get ground truth (try different column structures)
                            gt_x = None
                            gt_y = None
                            for col in gt_df.columns:
                                if bp in str(col) and "x" in str(col).lower():
                                    gt_x = gt_df.loc[idx, col]
                                elif bp in str(col) and "y" in str(col).lower():
                                    gt_y = gt_df.loc[idx, col]

                            if gt_x is None or gt_y is None:
                                continue

                            # Skip if ground truth is NaN (unlabeled)
                            if pd.isna(gt_x) or pd.isna(gt_y):
                                if pred_likelihood >= pcutoff:
                                    total_fp += 1
                                continue

                            # Check if prediction is valid
                            pred_valid = pred_likelihood >= pcutoff

                            if pred_valid:
                                # Calculate distance
                                dist = np.sqrt(
                                    (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2
                                )

                                if dist <= threshold:
                                    total_tp += 1
                                    # IoU approximation using Gaussian overlap
                                    iou = np.exp(-(dist**2) / (2 * threshold**2))
                                    total_iou_sum += iou
                                    total_iou_count += 1
                                else:
                                    total_fp += 1
                                    total_fn += 1
                            else:
                                total_fn += 1

                        except (KeyError, IndexError):
                            continue

            except Exception as e:
                logger.warning(f"Error processing {h5_file}: {e}")
                continue

        # Calculate metrics
        precision = None
        recall = None
        f1 = None
        iou = None

        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)

        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        if total_iou_count > 0:
            iou = total_iou_sum / total_iou_count

        return precision, recall, f1, iou

    def get_evaluation_summary(self, config: str, shuffle: int = 1) -> Optional[dict]:
        """Get summary of existing evaluation results"""
        project_path = Path(config).parent
        eval_path = project_path / "evaluation-results"

        if not eval_path.exists():
            return None

        iterations = list(eval_path.glob("iteration-*"))
        if not iterations:
            return None

        latest = sorted(iterations)[-1]

        summary = {
            "iteration": latest.name,
            "shuffle": shuffle,
            "files": [],
        }

        for f in latest.rglob("*"):
            if f.is_file() and f.suffix in [".csv", ".h5", ".png"]:
                summary["files"].append(str(f.relative_to(eval_path)))

        return summary
