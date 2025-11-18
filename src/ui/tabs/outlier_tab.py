"""Outlier Frame Extraction Tab"""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox, QListWidget,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar, QAbstractItemView
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.frame_extractor import FrameExtractor
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON, INFO_LABEL


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
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)
        
        # --- Configuration ---
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(16, 24, 16, 16)
        
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.clicked.connect(self.browse_config)
        
        config_layout.addWidget(QLabel("Config:"))
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # --- Videos ---
        video_group = QGroupBox("Videos to Scan")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(16, 24, 16, 16)
        video_layout.setSpacing(12)
        
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.video_list.setAlternatingRowColors(True)
        self.video_list.setMinimumHeight(120)
        video_layout.addWidget(self.video_list)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Videos")
        add_btn.clicked.connect(self.add_videos)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setObjectName(SECONDARY_BUTTON)
        remove_btn.clicked.connect(self.remove_video)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName(SECONDARY_BUTTON)
        clear_btn.clicked.connect(self.clear_videos)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        video_layout.addLayout(btn_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # --- Settings ---
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['jump', 'fitting', 'uncertain'])
        settings_layout.addRow("Algorithm:", self.algo_combo)
        
        self.p_bound_spin = QSpinBox()
        self.p_bound_spin.setRange(0, 100)
        self.p_bound_spin.setValue(1)
        self.p_bound_spin.setSuffix(" %")
        settings_layout.addRow("P-Bound:", self.p_bound_spin)
        
        self.automatic_check = QCheckBox("Automatic Extraction")
        settings_layout.addRow("", self.automatic_check)
        
        self.save_frames_check = QCheckBox("Save Frames")
        self.save_frames_check.setChecked(True)
        settings_layout.addRow("", self.save_frames_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # --- Action ---
        self.extract_btn = QPushButton("Extract Outlier Frames")
        self.extract_btn.setMinimumHeight(40)
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
            existing = [self.video_list.item(i).text() for i in range(self.video_list.count())]
            for path in file_paths:
                if path not in existing:
                    self.video_list.addItem(path)
    
    def remove_video(self):
        """Remove selected videos"""
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
    
    def clear_videos(self):
        """Clear video list"""
        self.video_list.clear()
    
    def extract_outliers(self):
        """Start outlier extraction"""
        config = self.config_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "Validation Error", "Please add at least one video")
            return
        
        self.set_busy(True)
        self.status_label.setText("Extracting outliers...")
        
        videos = [self.video_list.item(i).text() for i in range(self.video_list.count())]
        
        kwargs = {
            'config': config,
            'videos': videos,
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
        self.set_busy(False)
        self.status_label.setText("Completed!")
        QMessageBox.information(self, "Success", "Outlier frame extraction completed")
    
    def on_error(self, error: str):
        """Handle extraction error"""
        self.set_busy(False)
        self.status_label.setText("Error occurred.")
        QMessageBox.critical(self, "Error", f"Extraction failed:\n{error}")
        
    def set_busy(self, busy: bool):
        """Toggle UI"""
        self.extract_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
