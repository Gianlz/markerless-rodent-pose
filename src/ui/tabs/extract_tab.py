"""Frame Extraction Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.frame_extractor import FrameExtractor
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON, INFO_LABEL


class ExtractionWorker(QThread):
    """Worker thread for frame extraction"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, extractor: FrameExtractor, **kwargs):
        super().__init__()
        self.extractor = extractor
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.extractor.extract_frames(**self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ExtractTab(QWidget):
    """Frame extraction tab widget"""
    
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
        
        config_layout.addWidget(QLabel("Config Path:"))
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # --- Extraction Settings ---
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)
        
        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['automatic', 'manual'])
        settings_layout.addRow("Extraction Mode:", self.mode_combo)
        
        # Algorithm
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['kmeans', 'uniform'])
        settings_layout.addRow("Algorithm:", self.algo_combo)
        
        # Parameters
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(1, 1000)
        self.num_frames_spin.setValue(20)
        self.num_frames_spin.setSuffix(" frames")
        settings_layout.addRow("Number of Frames:", self.num_frames_spin)
        
        self.cluster_step_spin = QSpinBox()
        self.cluster_step_spin.setRange(1, 100)
        self.cluster_step_spin.setValue(1)
        settings_layout.addRow("Cluster Step:", self.cluster_step_spin)
        
        self.cluster_width_spin = QSpinBox()
        self.cluster_width_spin.setRange(10, 500)
        self.cluster_width_spin.setValue(30)
        self.cluster_width_spin.setSuffix(" px")
        settings_layout.addRow("Cluster Resize Width:", self.cluster_width_spin)
        
        self.cluster_color_check = QCheckBox("Use Color Features")
        settings_layout.addRow("", self.cluster_color_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # --- Actions ---
        self.extract_btn = QPushButton("Start Frame Extraction")
        self.extract_btn.setMinimumHeight(40)
        self.extract_btn.clicked.connect(self.extract_frames)
        layout.addWidget(self.extract_btn)
        
        # Progress/Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
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
    
    def extract_frames(self):
        """Start frame extraction"""
        config = self.config_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        self.set_busy(True)
        self.status_label.setText("Extracting frames... This may take a while.")
        
        kwargs = {
            'config': config,
            'mode': self.mode_combo.currentText(),
            'algo': self.algo_combo.currentText(),
            'num_frames': self.num_frames_spin.value(),
            'cluster_step': self.cluster_step_spin.value(),
            'cluster_resize_width': self.cluster_width_spin.value(),
            'cluster_color': self.cluster_color_check.isChecked()
        }
        
        self.worker = ExtractionWorker(self.extractor, **kwargs)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self):
        """Handle extraction completion"""
        self.set_busy(False)
        self.status_label.setText("Extraction completed successfully.")
        QMessageBox.information(self, "Success", "Frame extraction completed!\nYou can now proceed to labeling.")
    
    def on_error(self, error: str):
        """Handle extraction error"""
        self.set_busy(False)
        self.status_label.setText("Extraction failed.")
        QMessageBox.critical(self, "Error", f"Extraction failed:\n{error}")
        
    def set_busy(self, busy: bool):
        """Update UI state during processing"""
        self.extract_btn.setEnabled(not busy)
        self.config_input.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
