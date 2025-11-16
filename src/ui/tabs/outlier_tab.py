"""Outlier Frame Extraction Tab"""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.frame_extractor import FrameExtractor
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON, VIDEO_LIST_LABEL


class OutlierWorker(QThread):
    """Worker thread for outlier extraction"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, extractor: FrameExtractor, **kwargs):
        super().__init__()
        self.extractor = extractor
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.extractor.extract_outlier_frames(**self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class OutlierTab(QWidget):
    """Outlier extraction tab widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.extractor = FrameExtractor()
        self.worker = None
        self.video_paths = []
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Config file
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setSpacing(4)
        config_layout.setContentsMargins(8, 8, 8, 8)
        
        config_label = QLabel("Config:")
        config_label.setFixedWidth(60)
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.setFixedWidth(80)
        config_btn.clicked.connect(self.browse_config)
        
        config_layout.addWidget(config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Video files
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
        
        # Outlier settings
        settings_group = QGroupBox("Outlier Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        
        # Algorithm
        algo_label = QLabel("Algorithm:")
        algo_label.setFixedWidth(80)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['jump', 'fitting', 'uncertain'])
        settings_layout.addWidget(algo_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.algo_combo, 0, 1)
        
        # P-Bound
        param_label = QLabel("P-Bound (%):")
        param_label.setFixedWidth(80)
        self.p_bound_spin = QSpinBox()
        self.p_bound_spin.setRange(0, 100)
        self.p_bound_spin.setValue(1)
        settings_layout.addWidget(param_label, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.p_bound_spin, 0, 3)
        
        # Checkboxes
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(16)
        self.automatic_check = QCheckBox("Automatic")
        self.save_frames_check = QCheckBox("Save Frames")
        self.save_frames_check.setChecked(True)
        checkbox_layout.addWidget(self.automatic_check)
        checkbox_layout.addWidget(self.save_frames_check)
        checkbox_layout.addStretch()
        
        settings_layout.addLayout(checkbox_layout, 1, 0, 1, 4)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Extract button
        self.extract_btn = QPushButton("Extract Outlier Frames")
        self.extract_btn.setFixedHeight(32)
        self.extract_btn.clicked.connect(self.extract_outliers)
        layout.addWidget(self.extract_btn)
        
        layout.addStretch()
    
    def browse_config(self):
        """Browse for config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config File",
            "",
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)
    
    def set_config_path(self, path: str):
        """Set config path from external source"""
        self.config_input.setText(path)
    
    def add_videos(self):
        """Add videos"""
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
    
    def extract_outliers(self):
        """Start outlier extraction"""
        config = self.config_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        if not self.video_paths:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video")
            return
        
        self.extract_btn.setEnabled(False)
        
        kwargs = {
            'config': config,
            'videos': self.video_paths,
            'outlier_algorithm': self.algo_combo.currentText(),
            'p_bound': self.p_bound_spin.value() / 100.0,
            'automatic': self.automatic_check.isChecked(),
            'save_frames': self.save_frames_check.isChecked()
        }
        
        self.worker = OutlierWorker(self.extractor, **kwargs)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self):
        """Handle extraction completion"""
        self.extract_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Outlier frame extraction completed")
    
    def on_error(self, error: str):
        """Handle extraction error"""
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Extraction failed:\n{error}")
