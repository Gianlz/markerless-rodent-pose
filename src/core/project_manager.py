"""DeepLabCut project management"""

from pathlib import Path
import deeplabcut


class ProjectManager:
    """Handles DeepLabCut project creation and management"""

    def create_project(
        self,
        project_name: str,
        experimenter: str,
        videos: list[str],
        working_directory: str,
        copy_videos: bool = False,
        multianimal: bool = False,
    ) -> str:
        """
        Create a new DeepLabCut project

        Args:
            project_name: Name of the project
            experimenter: Name of the experimenter
            videos: List of video paths to include
            working_directory: Directory where project will be created
            copy_videos: Whether to copy videos to project folder
            multianimal: Create multi-animal project

        Returns:
            Path to config.yaml file
        """
        config_path = deeplabcut.create_new_project(
            project_name,
            experimenter,
            videos,
            working_directory=working_directory,
            copy_videos=copy_videos,
            multianimal=multianimal,
        )

        # Create additional subfolders
        project_path = Path(config_path).parent
        self._create_project_structure(project_path)

        return config_path

    def _create_project_structure(self, project_path: Path) -> None:
        """Create additional project subfolders"""
        subfolders = ["models", "frames", "output", "dataset"]

        for folder in subfolders:
            folder_path = project_path / folder
            folder_path.mkdir(exist_ok=True)

    def get_project_info(self, config_path: str) -> dict:
        """Get project information from config"""
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        project_path = Path(config_path).parent

        return {
            "project_name": config.get("Task", "Unknown"),
            "experimenter": config.get("scorer", "Unknown"),
            "project_path": str(project_path),
            "videos": config.get("video_sets", {}),
            "bodyparts": config.get("bodyparts", []),
            "multianimal": config.get("multianimalproject", False),
        }
