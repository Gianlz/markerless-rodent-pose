"""Application theme and styles"""
from pathlib import Path


def load_stylesheet() -> str:
    """Load QSS stylesheet from file"""
    qss_path = Path(__file__).parent.parent.parent.parent / "assets" / "styles" / "main.qss"
    
    if qss_path.exists():
        with open(qss_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# Widget object names for specific styling
SECONDARY_BUTTON = "secondaryButton"
DANGER_BUTTON = "dangerButton"
SUCCESS_LABEL = "successLabel"
ERROR_LABEL = "errorLabel"
INFO_LABEL = "infoLabel"
VIDEO_LIST_LABEL = "videoListLabel"
