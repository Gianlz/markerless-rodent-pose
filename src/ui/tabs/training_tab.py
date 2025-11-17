"""Training Dataset Creation Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QTextEdit
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.training_manager import TrainingManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class TrainingDatasetWorker(QThread):
    """Worker thread for training dataset creation"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, manager: TrainingManager, config: str, is_multianimal: bool, **kwargs):
        super().__init__()
        self.manager = manager
        self.config = config
        self.is_multianimal = is_multianimal
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.is_multianimal:
                self.manager.create_multianimal_training_dataset(self.config, **self.kwargs)
            else:
                self.manager.create_training_dataset(self.config, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class TrainingTab(QWidget):
    """Training dataset creation tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = TrainingManager()
        self.worker = None
        self.is_multianimal = False
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
        self.config_input.textChanged.connect(self.on_config_changed)
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.setFixedWidth(80)
        config_btn.clicked.connect(self.browse_config)
        
        config_layout.addWidget(config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Dataset settings
        settings_group = QGroupBox("Dataset Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        
        # Shuffle
        shuffle_label = QLabel("Shuffles:")
        shuffle_label.setFixedWidth(120)
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(1, 10)
        self.shuffle_spin.setValue(1)
        settings_layout.addWidget(shuffle_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.shuffle_spin, 0, 1)
        
        # Network architecture
        net_label = QLabel("Network:")
        net_label.setFixedWidth(120)
        self.net_combo = QComboBox()
        settings_layout.addWidget(net_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.net_combo, 1, 1, 1, 3)
        
        # Augmentation
        aug_label = QLabel("Augmentation:")
        aug_label.setFixedWidth(120)
        self.aug_combo = QComboBox()
        self.aug_combo.addItems(self.manager.get_available_augmenters())
        settings_layout.addWidget(aug_label, 2, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.aug_combo, 2, 1, 1, 3)
        
        # Weight initialization
        weight_label = QLabel("Weight Init:")
        weight_label.setFixedWidth(120)
        self.weight_combo = QComboBox()
        self.weight_combo.addItems(['Transfer Learning - SuperAnimal TopViewMouse', 'Transfer Learning - ImageNet', 'Random Initialization'])
        settings_layout.addWidget(weight_label, 3, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.weight_combo, 3, 1, 1, 3)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Status
        status_group = QGroupBox("Dataset Status")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(6)
        status_layout.setContentsMargins(8, 8, 8, 8)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(80)
        self.status_text.setPlaceholderText("Dataset status will appear here...")
        status_layout.addWidget(self.status_text)
        
        self.check_status_btn = QPushButton("Check Status")
        self.check_status_btn.setObjectName(SECONDARY_BUTTON)
        self.check_status_btn.setFixedHeight(28)
        self.check_status_btn.clicked.connect(self.check_status)
        status_layout.addWidget(self.check_status_btn)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Create button
        self.create_btn = QPushButton("Create Training Dataset")
        self.create_btn.setFixedHeight(32)
        self.create_btn.clicked.connect(self.create_dataset)
        layout.addWidget(self.create_btn)
        
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
    
    def on_config_changed(self):
        """Handle config path change"""
        config = self.config_input.text()
        if config:
            valid, _ = validate_config_path(config)
            if valid:
                self.is_multianimal = self.manager.is_multianimal_project(config)
                self.update_network_list()
                self.check_status()
    
    def update_network_list(self):
        """Update network architecture list based on project type"""
        self.net_combo.clear()
        networks = self.manager.get_available_networks(self.is_multianimal)
        self.net_combo.addItems(networks)
        
        # Set default
        if self.is_multianimal:
            self.net_combo.setCurrentText('dlcrnet_ms5')
        else:
            self.net_combo.setCurrentText('resnet_50')
    
    def check_status(self):
        """Check training dataset status"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        
        if not valid:
            self.status_text.setPlainText(f"Error: {error}")
            return
        
        try:
            status = self.manager.check_training_dataset_exists(config)
            
            if status['exists']:
                text = f"✓ Training dataset exists\n"
                text += f"Iterations: {status['iterations']}\n"
                text += f"Latest: {status['latest_iteration']}\n"
                text += f"Train folders: {status['train_folders']}"
            else:
                text = f"✗ {status['message']}"
            
            self.status_text.setPlainText(text)
        except Exception as e:
            self.status_text.setPlainText(f"Error checking status: {str(e)}")
    
    def create_dataset(self):
        """Create training dataset"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        self.create_btn.setEnabled(False)
        self.status_text.setPlainText("Creating training dataset...")
        
        weight_choice = self.weight_combo.currentText()
        
        if 'SuperAnimal' in weight_choice:
            init_weights = 'superanimal'
        elif 'Random' in weight_choice:
            init_weights = 'random'
        else:
            init_weights = 'imagenet'
        
        kwargs = {
            'num_shuffles': self.shuffle_spin.value(),
            'net_type': self.net_combo.currentText(),
            'augmenter_type': self.aug_combo.currentText(),
            'init_weights': init_weights
        }
        
        self.worker = TrainingDatasetWorker(
            self.manager, config, self.is_multianimal, **kwargs
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self):
        """Handle dataset creation completion"""
        self.create_btn.setEnabled(True)
        self.check_status()
        QMessageBox.information(self, "Success", "Training dataset created successfully")
    
    def on_error(self, error: str):
        """Handle dataset creation error"""
        self.create_btn.setEnabled(True)
        self.status_text.setPlainText(f"Error: {error}")
        QMessageBox.critical(self, "Error", f"Failed to create training dataset:\n{error}")
