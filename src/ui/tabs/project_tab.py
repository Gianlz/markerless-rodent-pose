"""Project Manager Tab"""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QGroupBox, QFileDialog, QMessageBox, QCheckBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt

from ...core.project_manager import ProjectManager
from ..styles import SECONDARY_BUTTON, SUCCESS_LABEL, ERROR_LABEL, VIDEO_LIST_LABEL


class ProjectTab(QWidget):
    """Project manager tab widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.video_paths = []
        self.config_created = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Project info
        info_group = QGroupBox("Project Information")
        info_layout = QGridLayout()
        info_layout.setSpacing(6)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setColumnStretch(1, 1)
        
        # Project name
        name_label = QLabel("Project Name:")
        name_label.setFixedWidth(100)
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("e.g., MouseTracking")
        info_layout.addWidget(name_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_layout.addWidget(self.project_name_input, 0, 1)
        
        # Experimenter
        exp_label = QLabel("Experimenter:")
        exp_label.setFixedWidth(100)
        self.experimenter_input = QLineEdit()
        self.experimenter_input.setPlaceholderText("Your name")
        info_layout.addWidget(exp_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_layout.addWidget(self.experimenter_input, 1, 1)
        
        # Working directory
        dir_label = QLabel("Working Dir:")
        dir_label.setFixedWidth(100)
        self.working_dir_input = QLineEdit()
        self.working_dir_input.setPlaceholderText("Project parent directory")
        dir_btn = QPushButton("Browse")
        dir_btn.setObjectName(SECONDARY_BUTTON)
        dir_btn.setFixedWidth(80)
        dir_btn.clicked.connect(self.browse_working_dir)
        
        dir_container = QHBoxLayout()
        dir_container.setSpacing(4)
        dir_container.addWidget(self.working_dir_input)
        dir_container.addWidget(dir_btn)
        
        info_layout.addWidget(dir_label, 2, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_layout.addLayout(dir_container, 2, 1)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Videos
        video_group = QGroupBox("Videos")
        video_layout = QVBoxLayout()
        video_layout.setSpacing(6)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.video_list_label = QLabel("No videos added")
        self.video_list_label.setObjectName(VIDEO_LIST_LABEL)
        self.video_list_label.setWordWrap(True)
        self.video_list_label.setFixedHeight(50)
        video_layout.addWidget(self.video_list_label)
        
        video_btn_layout = QHBoxLayout()
        video_btn_layout.setSpacing(4)
        add_video_btn = QPushButton("Add Videos")
        add_video_btn.setFixedWidth(100)
        add_video_btn.clicked.connect(self.add_videos)
        clear_video_btn = QPushButton("Clear")
        clear_video_btn.setObjectName(SECONDARY_BUTTON)
        clear_video_btn.setFixedWidth(70)
        clear_video_btn.clicked.connect(self.clear_videos)
        video_btn_layout.addWidget(add_video_btn)
        video_btn_layout.addWidget(clear_video_btn)
        video_btn_layout.addStretch()
        video_layout.addLayout(video_btn_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        options_layout.setSpacing(16)
        options_layout.setContentsMargins(8, 8, 8, 8)
        self.copy_videos_check = QCheckBox("Copy videos to project")
        self.multianimal_check = QCheckBox("Multi-animal project")
        options_layout.addWidget(self.copy_videos_check)
        options_layout.addWidget(self.multianimal_check)
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Create button
        self.create_btn = QPushButton("Create Project")
        self.create_btn.setFixedHeight(32)
        self.create_btn.clicked.connect(self.create_project)
        layout.addWidget(self.create_btn)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setFixedHeight(30)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def browse_working_dir(self):
        """Browse for working directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if dir_path:
            self.working_dir_input.setText(dir_path)
    
    def add_videos(self):
        """Add videos to project"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_paths:
            self.video_paths.extend(file_paths)
            self.update_video_list()
    
    def clear_videos(self):
        """Clear video list"""
        self.video_paths.clear()
        self.update_video_list()
    
    def update_video_list(self):
        """Update video list display"""
        if not self.video_paths:
            self.video_list_label.setText("No videos added")
        else:
            video_names = [Path(v).name for v in self.video_paths]
            self.video_list_label.setText(
                f"{len(video_names)} video(s):\n" + "\n".join(video_names)
            )
    
    def create_project(self):
        """Create new DeepLabCut project"""
        project_name = self.project_name_input.text().strip()
        experimenter = self.experimenter_input.text().strip()
        working_dir = self.working_dir_input.text().strip()
        
        if not project_name:
            QMessageBox.warning(self, "Validation Error", "Project name is required")
            return
        
        if not experimenter:
            QMessageBox.warning(self, "Validation Error", "Experimenter name is required")
            return
        
        if not working_dir:
            QMessageBox.warning(self, "Validation Error", "Working directory is required")
            return
        
        if not Path(working_dir).exists():
            QMessageBox.warning(self, "Validation Error", "Working directory does not exist")
            return
        
        if not self.video_paths:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video")
            return
        
        self.create_btn.setEnabled(False)
        self.status_label.setText("Creating project...")
        
        try:
            config_path = self.project_manager.create_project(
                project_name=project_name,
                experimenter=experimenter,
                videos=self.video_paths,
                working_directory=working_dir,
                copy_videos=self.copy_videos_check.isChecked(),
                multianimal=self.multianimal_check.isChecked()
            )
            
            self.config_created = config_path
            self.status_label.setText(f"✓ Project created!\nConfig: {config_path}")
            self.status_label.setObjectName(SUCCESS_LABEL)
            
            QMessageBox.information(
                self,
                "Success",
                f"Project created successfully!\n\nConfig: {config_path}\n\nSubfolders:\n→ models/\n→ frames/\n→ output/\n→ dataset/"
            )
            
        except Exception as e:
            self.status_label.setText(f"✗ Error: {str(e)}")
            self.status_label.setObjectName(ERROR_LABEL)
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{str(e)}")
        
        finally:
            self.create_btn.setEnabled(True)
    
    def get_config_path(self) -> str:
        """Get created config path"""
        return self.config_created or ""
