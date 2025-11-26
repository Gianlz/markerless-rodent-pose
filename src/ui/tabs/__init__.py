"""Tab widgets"""

from .clean_video_tab import CleanVideoTab
from .evaluation_tab import EvaluationTab
from .extract_tab import ExtractTab
from .freezing_tab import FreezingTab
from .inference_tab import InferenceTab
from .label_tab import LabelTab
from .outlier_tab import OutlierTab
from .project_tab import ProjectTab
from .system_info_tab import SystemInfoTab
from .train_tab import TrainTab
from .training_tab import TrainingTab

__all__ = [
    "CleanVideoTab",
    "ProjectTab",
    "ExtractTab",
    "OutlierTab",
    "LabelTab",
    "TrainingTab",
    "TrainTab",
    "InferenceTab",
    "EvaluationTab",
    "OutlierTab",
    "FreezingTab",
    "SystemInfoTab",
]
