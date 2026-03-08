"""Project Manager Tab"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QThread, Signal

from ...core.project_manager import ProjectManager
from ..styles import SECONDARY_BUTTON, SUCCESS_LABEL, ERROR_LABEL, INFO_LABEL


class ProjectCreationWorker(QThread):
    """Worker thread for project creation"""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, project_manager, **kwargs):
        super().__init__()
        self.project_manager = project_manager
        self.kwargs = kwargs

    def run(self):
        try:
            config_path = self.project_manager.create_project(**self.kwargs)
            self.finished.emit(config_path)
        except Exception as e:
            self.error.emit(str(e))


class ProjectTab(QWidget):
    """Project manager tab widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.config_created = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)

        # --- Project Information ---
        info_group = QGroupBox("Project Information")
        info_layout = QFormLayout()
        info_layout.setSpacing(12)
        info_layout.setContentsMargins(16, 24, 16, 16)

        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("e.g., MouseTracking")
        info_layout.addRow("Project Name:", self.project_name_input)

        self.experimenter_input = QLineEdit()
        self.experimenter_input.setPlaceholderText("Your Name")
        info_layout.addRow("Experimenter:", self.experimenter_input)

        # Working Directory
        dir_layout = QHBoxLayout()
        self.working_dir_input = QLineEdit()
        self.working_dir_input.setPlaceholderText("Select parent directory...")

        dir_btn = QPushButton("Browse")
        dir_btn.setObjectName(SECONDARY_BUTTON)
        dir_btn.setMinimumWidth(100)
        dir_btn.clicked.connect(self.browse_working_dir)

        dir_layout.addWidget(self.working_dir_input)
        dir_layout.addWidget(dir_btn)
        info_layout.addRow("Working Directory:", dir_layout)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- Videos ---
        video_group = QGroupBox("Videos")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(16, 24, 16, 16)
        video_layout.setSpacing(12)

        # Video List
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.video_list.setAlternatingRowColors(True)
        self.video_list.setMinimumHeight(150)
        video_layout.addWidget(self.video_list)

        # Video Buttons
        video_btn_layout = QHBoxLayout()

        add_video_btn = QPushButton("Add Videos")
        add_video_btn.clicked.connect(self.add_videos)

        remove_video_btn = QPushButton("Remove Selected")
        remove_video_btn.setObjectName(SECONDARY_BUTTON)
        remove_video_btn.clicked.connect(self.remove_selected_videos)

        clear_video_btn = QPushButton("Clear All")
        clear_video_btn.setObjectName(SECONDARY_BUTTON)
        clear_video_btn.clicked.connect(self.clear_videos)

        video_btn_layout.addWidget(add_video_btn)
        video_btn_layout.addWidget(remove_video_btn)
        video_btn_layout.addWidget(clear_video_btn)
        video_btn_layout.addStretch()

        video_layout.addLayout(video_btn_layout)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # --- Options & Actions ---
        options_layout = QHBoxLayout()
        self.copy_videos_check = QCheckBox("Copy videos to project folder")
        self.multianimal_check = QCheckBox("Multi-animal project")

        options_layout.addWidget(self.copy_videos_check)
        options_layout.addWidget(self.multianimal_check)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Create Button
        self.create_btn = QPushButton("Create Project")
        self.create_btn.setMinimumHeight(40)
        self.create_btn.clicked.connect(self.create_project)
        layout.addWidget(self.create_btn)

        # Status
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.hide()  # Hidden by default
        layout.addWidget(self.status_label)

        layout.addStretch()

    def browse_working_dir(self):
        """Browse for working directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if dir_path:
            self.working_dir_input.setText(dir_path)

    def add_videos(self):
        """Add videos to list"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_paths:
            for path in file_paths:
                # Avoid duplicates
                existing_items = [
                    self.video_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.video_list.count())
                ]
                if path not in existing_items:
                    item = QListWidgetItem(Path(path).name)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    item.setToolTip(path)
                    self.video_list.addItem(item)

    def remove_selected_videos(self):
        """Remove selected videos from list"""
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def clear_videos(self):
        """Clear video list"""
        self.video_list.clear()

    def get_video_paths(self) -> list[str]:
        """Get list of video paths"""
        return [
            self.video_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.video_list.count())
        ]

    def create_project(self):
        """Create new DeepLabCut project"""
        project_name = self.project_name_input.text().strip()
        experimenter = self.experimenter_input.text().strip()
        working_dir = self.working_dir_input.text().strip()
        videos = self.get_video_paths()

        # Validation
        if not project_name:
            self.show_error("Project name is required")
            return
        if not experimenter:
            self.show_error("Experimenter name is required")
            return
        if not working_dir:
            self.show_error("Working directory is required")
            return
        if not Path(working_dir).exists():
            self.show_error("Working directory does not exist")
            return
        if not videos:
            self.show_error("Please add at least one video")
            return

        self.create_btn.setEnabled(False)
        self.show_info("Creating project... This may take a while if copying videos.")

        kwargs = {
            "project_name": project_name,
            "experimenter": experimenter,
            "videos": videos,
            "working_directory": working_dir,
            "copy_videos": self.copy_videos_check.isChecked(),
            "multianimal": self.multianimal_check.isChecked(),
        }

        self.worker = ProjectCreationWorker(self.project_manager, **kwargs)
        self.worker.finished.connect(self.on_project_created)
        self.worker.error.connect(self.on_project_creation_error)
        self.worker.start()

    def on_project_created(self, config_path: str):
        """Handle successful project creation"""
        self.create_btn.setEnabled(True)
        self.config_created = config_path
        self.show_success(f"Project created successfully!\nConfig: {config_path}")

        QMessageBox.information(
            self,
            "Success",
            f"Project created successfully!\n\nConfig: {config_path}\n\nSubfolders:\n→ models/\n→ frames/\n→ output/\n→ dataset/",
        )

    def on_project_creation_error(self, error_msg: str):
        """Handle project creation error"""
        self.create_btn.setEnabled(True)
        self.show_error(f"Failed to create project: {error_msg}")
        QMessageBox.critical(self, "Error", f"Failed to create project:\n{error_msg}")

    def get_config_path(self) -> str:
        """Get created config path"""
        return self.config_created or ""

    def show_error(self, message: str):
        self.status_label.setText(message)
        self.status_label.setObjectName(ERROR_LABEL)
        self.status_label.setStyleSheet("")  # Force style update
        self.status_label.show()

    def show_success(self, message: str):
        self.status_label.setText(message)
        self.status_label.setObjectName(SUCCESS_LABEL)
        self.status_label.setStyleSheet("")
        self.status_label.show()

    def show_info(self, message: str):
        self.status_label.setText(message)
        self.status_label.setObjectName(INFO_LABEL)
        self.status_label.setStyleSheet("")
        self.status_label.show()
