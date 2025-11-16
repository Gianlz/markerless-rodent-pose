"""Model Training Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QGroupBox,
    QFileDialog, QMessageBox, QTextEdit
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.train_manager import TrainManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class TrainingWorker(QThread):
    """Worker thread for model training"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, manager: TrainManager, config: str, **kwargs):
        super().__init__()
        self.manager = manager
        self.config = config
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.manager.train_network(self.config, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class TrainTab(QWidget):
    """Model training tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = TrainManager()
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
        
        # Training info
        info_group = QGroupBox("Training Info")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 8)
        
        self.info_label = QLabel("Load a config file to see training info")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        params_layout.setSpacing(6)
        params_layout.setContentsMargins(8, 8, 8, 8)
        params_layout.setColumnStretch(1, 1)
        params_layout.setColumnStretch(3, 1)
        
        # Shuffle
        shuffle_label = QLabel("Shuffle:")
        shuffle_label.setFixedWidth(120)
        self.shuffle_combo = QComboBox()
        params_layout.addWidget(shuffle_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        params_layout.addWidget(self.shuffle_combo, 0, 1)
        
        # Display iterations
        display_label = QLabel("Display Iterations:")
        display_label.setFixedWidth(120)
        self.display_spin = QSpinBox()
        self.display_spin.setRange(10, 100000)
        self.display_spin.setSingleStep(10)
        self.display_spin.setValue(1000)
        params_layout.addWidget(display_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        params_layout.addWidget(self.display_spin, 1, 1)
        
        # Snapshots to keep
        snapshots_label = QLabel("Snapshots to Keep:")
        snapshots_label.setFixedWidth(120)
        self.snapshots_spin = QSpinBox()
        self.snapshots_spin.setRange(1, 20)
        self.snapshots_spin.setValue(5)
        params_layout.addWidget(snapshots_label, 1, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        params_layout.addWidget(self.snapshots_spin, 1, 3)
        
        # Maximum epochs
        max_epochs_label = QLabel("Maximum Epochs:")
        max_epochs_label.setFixedWidth(120)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(100, 10000000)
        self.max_epochs_spin.setSingleStep(1000)
        self.max_epochs_spin.setValue(200000)
        params_layout.addWidget(max_epochs_label, 2, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        params_layout.addWidget(self.max_epochs_spin, 2, 1)
        
        # Save epochs
        save_epochs_label = QLabel("Save Epochs:")
        save_epochs_label.setFixedWidth(120)
        self.save_epochs_spin = QSpinBox()
        self.save_epochs_spin.setRange(100, 1000000)
        self.save_epochs_spin.setSingleStep(1000)
        self.save_epochs_spin.setValue(50000)
        params_layout.addWidget(save_epochs_label, 2, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        params_layout.addWidget(self.save_epochs_spin, 2, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Resume training
        resume_group = QGroupBox("Resume Training (Optional)")
        resume_layout = QVBoxLayout()
        resume_layout.setSpacing(6)
        resume_layout.setContentsMargins(8, 8, 8, 8)
        
        snapshot_layout = QHBoxLayout()
        snapshot_layout.setSpacing(4)
        
        snapshot_label = QLabel("Snapshot:")
        snapshot_label.setFixedWidth(80)
        self.snapshot_combo = QComboBox()
        self.snapshot_combo.addItem("Start from scratch")
        
        snapshot_layout.addWidget(snapshot_label)
        snapshot_layout.addWidget(self.snapshot_combo)
        resume_layout.addLayout(snapshot_layout)
        
        resume_group.setLayout(resume_layout)
        layout.addWidget(resume_group)
        
        # Train button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setFixedHeight(32)
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)
        
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
                self.load_training_info()
                self.load_shuffles()
                self.load_snapshots()
    
    def load_training_info(self):
        """Load and display training info from config file"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        
        if not valid:
            return
        
        try:
            import yaml
            
            # Read config file directly
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
            
            # Get actual values from config
            net_type = cfg.get('net_type', 'resnet_50')
            init_weights = cfg.get('init_weights', 'imagenet')
            multianimal = cfg.get('multianimalproject', False)
            
            # Check if training dataset exists
            info = self.manager.get_training_info(config)
            
            # Display actual config values
            text = f"Network: {net_type} | Engine: pytorch\n"
            text += f"Init Weights: {init_weights}\n"
            text += f"Project Type: {'Multi-animal' if multianimal else 'Single-animal'}\n"
            text += f"Training Dataset: {'✓ Created' if info['training_dataset_exists'] else '✗ Not created'}"
            
            self.info_label.setText(text)
        except Exception as e:
            self.info_label.setText(f"Error loading info: {str(e)}")
    
    def load_shuffles(self):
        """Load available shuffles"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        
        if not valid:
            return
        
        try:
            shuffles = self.manager.get_available_shuffles(config)
            self.shuffle_combo.clear()
            self.shuffle_combo.addItems([str(s) for s in shuffles])
            
            # Update snapshots when shuffle changes
            self.shuffle_combo.currentTextChanged.connect(self.load_snapshots)
        except Exception as e:
            pass
    
    def load_snapshots(self):
        """Load available snapshots for selected shuffle"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        
        if not valid:
            return
        
        try:
            shuffle = int(self.shuffle_combo.currentText()) if self.shuffle_combo.currentText() else 1
            snapshots = self.manager.get_available_snapshots(config, shuffle)
            
            self.snapshot_combo.clear()
            self.snapshot_combo.addItem("Start from scratch")
            self.snapshot_combo.addItems(snapshots)
        except Exception as e:
            pass
    
    def start_training(self):
        """Start model training"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        # Check if training dataset exists
        info = self.manager.get_training_info(config)
        if not info['training_dataset_exists']:
            QMessageBox.warning(
                self, "Warning",
                "Training dataset not found. Please create training dataset first."
            )
            return
        
        reply = QMessageBox.question(
            self, "Confirm",
            "Start training? This may take several hours.\n\n"
            "The training will run in the background. You can monitor progress in the terminal.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training in progress...")
        
        kwargs = {
            'shuffle': int(self.shuffle_combo.currentText()) if self.shuffle_combo.currentText() else 1,
            'displayiters': self.display_spin.value(),
            'saveiters': self.save_epochs_spin.value(),
            'maxiters': self.max_epochs_spin.value(),
            'max_snapshots_to_keep': self.snapshots_spin.value()
        }
        
        self.worker = TrainingWorker(self.manager, config, **kwargs)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_finished(self):
        """Handle training completion"""
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Start Training")
        self.load_snapshots()
        QMessageBox.information(self, "Success", "Training completed successfully")
    
    def on_error(self, error: str):
        """Handle training error"""
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Start Training")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error}")
