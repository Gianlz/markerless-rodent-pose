"""Model Training Tab"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)

        # --- Configuration ---
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(16, 24, 16, 16)

        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        self.config_input.textChanged.connect(self.on_config_changed)

        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.clicked.connect(self.browse_config)

        config_layout.addWidget(QLabel("Config:"))
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # --- Info ---
        info_group = QGroupBox("Network Info")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(16, 24, 16, 16)

        self.info_label = QLabel("Load a config file to see training info")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- Parameters ---
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        params_layout.setContentsMargins(16, 24, 16, 16)
        params_layout.setSpacing(12)

        # Shuffle
        self.shuffle_combo = QComboBox()
        params_layout.addRow("Shuffle Index:", self.shuffle_combo)

        # Iterations
        self.display_spin = QSpinBox()
        self.display_spin.setRange(10, 100000)
        self.display_spin.setSingleStep(100)
        self.display_spin.setValue(1000)
        self.display_spin.setSuffix(" iters")
        params_layout.addRow("Display Iterations:", self.display_spin)

        self.save_epochs_spin = QSpinBox()
        self.save_epochs_spin.setRange(100, 1000000)
        self.save_epochs_spin.setSingleStep(1000)
        self.save_epochs_spin.setValue(50000)
        self.save_epochs_spin.setSuffix(" iters")
        params_layout.addRow("Save Checkpoint Every:", self.save_epochs_spin)

        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(100, 10000000)
        self.max_epochs_spin.setSingleStep(1000)
        self.max_epochs_spin.setValue(200000)
        self.max_epochs_spin.setSuffix(" iters")
        params_layout.addRow("Max Iterations:", self.max_epochs_spin)

        self.snapshots_spin = QSpinBox()
        self.snapshots_spin.setRange(1, 20)
        self.snapshots_spin.setValue(5)
        params_layout.addRow("Keep Last N Snapshots:", self.snapshots_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Resume Training ---
        resume_group = QGroupBox("Resume Training (Optional)")
        resume_layout = QHBoxLayout()
        resume_layout.setContentsMargins(16, 24, 16, 16)

        self.snapshot_combo = QComboBox()
        self.snapshot_combo.addItem("Start from scratch")
        resume_layout.addWidget(QLabel("Resume from:"))
        resume_layout.addWidget(self.snapshot_combo, 1)

        resume_group.setLayout(resume_layout)
        layout.addWidget(resume_group)

        # --- Actions ---
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        layout.addStretch()

    def browse_config(self):
        """Browse for config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "YAML Files (*.yaml *.yml)"
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

            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

            net_type = cfg.get("net_type", "resnet_50")
            init_weights = cfg.get("init_weights", "imagenet")
            multianimal = cfg.get("multianimalproject", False)

            info = self.manager.get_training_info(config)

            text = f"Network: <b>{net_type}</b> (PyTorch)<br>"
            text += f"Init Weights: <b>{init_weights}</b><br>"
            text += f"Project Type: <b>{'Multi-animal' if multianimal else 'Single-animal'}</b><br>"
            created_text = '<span style="color:green">✓ Created</span>'
            not_created_text = '<span style="color:red">✗ Not created</span>'
            status = (
                created_text if info["training_dataset_exists"] else not_created_text
            )
            text += f"Training Dataset: {status}"

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
            self.shuffle_combo.currentTextChanged.connect(self.load_snapshots)
        except Exception:
            pass

    def load_snapshots(self):
        """Load available snapshots for selected shuffle"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        if not valid:
            return

        try:
            shuffle = (
                int(self.shuffle_combo.currentText())
                if self.shuffle_combo.currentText()
                else 1
            )
            snapshots = self.manager.get_available_snapshots(config, shuffle)

            self.snapshot_combo.clear()
            self.snapshot_combo.addItem("Start from scratch")
            self.snapshot_combo.addItems(snapshots)
        except Exception:
            pass

    def start_training(self):
        """Start model training"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)

        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        info = self.manager.get_training_info(config)
        if not info["training_dataset_exists"]:
            QMessageBox.warning(
                self, "Warning", "Training dataset not found. Please create it first."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm",
            "Start training? This may take several hours.\n\n"
            "The training will run in the background. You can monitor progress in the terminal.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.set_busy(True)
        self.status_label.setText("Training in progress...")

        kwargs = {
            "shuffle": int(self.shuffle_combo.currentText())
            if self.shuffle_combo.currentText()
            else 1,
            "displayiters": self.display_spin.value(),
            "saveiters": self.save_epochs_spin.value(),
            "maxiters": self.max_epochs_spin.value(),
            "max_snapshots_to_keep": self.snapshots_spin.value(),
        }

        self.worker = TrainingWorker(self.manager, config, **kwargs)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        """Handle training completion"""
        self.set_busy(False)
        self.status_label.setText("Training completed.")
        self.load_snapshots()
        QMessageBox.information(self, "Success", "Training completed successfully")

    def on_error(self, error: str):
        """Handle training error"""
        self.set_busy(False)
        self.status_label.setText("Training failed.")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error}")

    def set_busy(self, busy: bool):
        """Toggle UI"""
        self.train_btn.setEnabled(not busy)
