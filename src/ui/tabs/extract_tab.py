"""Frame Extraction Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.frame_extractor import FrameExtractor
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


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
        
        # Extraction settings
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        
        # Mode
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(80)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['automatic', 'manual'])
        settings_layout.addWidget(mode_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.mode_combo, 0, 1)
        
        # Algorithm
        algo_label = QLabel("Algorithm:")
        algo_label.setFixedWidth(80)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['kmeans', 'uniform'])
        settings_layout.addWidget(algo_label, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.algo_combo, 0, 3)
        
        # Cluster settings
        cluster_step_label = QLabel("Cluster Step:")
        cluster_step_label.setFixedWidth(80)
        self.cluster_step_spin = QSpinBox()
        self.cluster_step_spin.setRange(1, 100)
        self.cluster_step_spin.setValue(1)
        settings_layout.addWidget(cluster_step_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.cluster_step_spin, 1, 1)
        
        cluster_width_label = QLabel("Resize Width:")
        cluster_width_label.setFixedWidth(80)
        self.cluster_width_spin = QSpinBox()
        self.cluster_width_spin.setRange(10, 200)
        self.cluster_width_spin.setValue(30)
        settings_layout.addWidget(cluster_width_label, 1, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.cluster_width_spin, 1, 3)
        
        # Checkboxes
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(16)
        self.crop_check = QCheckBox("Crop")
        self.opencv_check = QCheckBox("Use OpenCV")
        self.opencv_check.setChecked(True)
        self.cluster_color_check = QCheckBox("Cluster Color")
        checkbox_layout.addWidget(self.crop_check)
        checkbox_layout.addWidget(self.opencv_check)
        checkbox_layout.addWidget(self.cluster_color_check)
        checkbox_layout.addStretch()
        
        settings_layout.addLayout(checkbox_layout, 2, 0, 1, 4)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Extract button
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.setFixedHeight(32)
        self.extract_btn.clicked.connect(self.extract_frames)
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
    
    def extract_frames(self):
        """Start frame extraction"""
        config = self.config_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        self.extract_btn.setEnabled(False)
        
        kwargs = {
            'config': config,
            'mode': self.mode_combo.currentText(),
            'algo': self.algo_combo.currentText(),
            'crop': self.crop_check.isChecked(),
            'cluster_step': self.cluster_step_spin.value(),
            'cluster_resize_width': self.cluster_width_spin.value(),
            'cluster_color': self.cluster_color_check.isChecked(),
            'opencv': self.opencv_check.isChecked()
        }
        
        self.worker = ExtractionWorker(self.extractor, **kwargs)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self):
        """Handle extraction completion"""
        self.extract_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Frame extraction completed")
    
    def on_error(self, error: str):
        """Handle extraction error"""
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Extraction failed:\n{error}")
