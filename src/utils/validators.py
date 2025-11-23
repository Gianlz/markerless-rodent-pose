"""Input validation utilities"""

from pathlib import Path
from typing import Optional


def validate_config_path(path: str) -> tuple[bool, Optional[str]]:
    """Validate config.yaml path"""
    if not path:
        return False, "Config path is required"

    p = Path(path)
    if not p.exists():
        return False, f"Config file not found: {path}"

    if p.suffix not in [".yaml", ".yml"]:
        return False, "Config must be a YAML file"

    return True, None


def validate_video_path(path: str) -> tuple[bool, Optional[str]]:
    """Validate video file path"""
    if not path:
        return False, "Video path is required"

    p = Path(path)
    if not p.exists():
        return False, f"Video file not found: {path}"

    valid_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    if p.suffix.lower() not in valid_extensions:
        return False, f"Invalid video format. Supported: {', '.join(valid_extensions)}"

    return True, None
