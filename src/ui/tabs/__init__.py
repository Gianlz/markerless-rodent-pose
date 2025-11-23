"""Tab widgets"""

from .clean_video_tab import CleanVideoTab
from .project_tab import ProjectTab
from .extract_tab import ExtractTab
from .outlier_tab import OutlierTab
from .label_tab import LabelTab
from .training_tab import TrainingTab
from .train_tab import TrainTab
from .inference_tab import InferenceTab
from .system_info_tab import SystemInfoTab

__all__ = [
    "CleanVideoTab",
    "ProjectTab",
    "ExtractTab",
    "OutlierTab",
    "LabelTab",
    "TrainingTab",
    "TrainTab",
    "InferenceTab",
    "SystemInfoTab",
]
