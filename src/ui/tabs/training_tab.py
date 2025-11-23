"""Training Dataset Creation Tab"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QTextEdit,
)
from PySide6.QtCore import QThread, Signal, QSettings

from ...core.training_manager import TrainingManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class TrainingDatasetWorker(QThread):
    """Worker thread for training dataset creation"""

    finished = Signal()
    error = Signal(str)

    def __init__(
        self, manager: TrainingManager, config: str, is_multianimal: bool, **kwargs
    ):
        super().__init__()
        self.manager = manager
        self.config = config
        self.is_multianimal = is_multianimal
        self.kwargs = kwargs

    def run(self):
        try:
            if self.is_multianimal:
                self.manager.create_multianimal_training_dataset(
                    self.config, **self.kwargs
                )
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
        self.settings = QSettings("DeepLabCut", "FrameExtractor")
        self.init_ui()
        self.load_settings()

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

        # --- Dataset Settings ---
        settings_group = QGroupBox("Dataset Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)

        # Shuffles
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(1, 10)
        self.shuffle_spin.setValue(1)
        settings_layout.addRow("Shuffles:", self.shuffle_spin)

        # Network
        self.net_combo = QComboBox()
        settings_layout.addRow("Network Architecture:", self.net_combo)

        # Augmentation
        self.aug_combo = QComboBox()
        self.aug_combo.addItems(self.manager.get_available_augmenters())
        self.aug_combo.currentTextChanged.connect(self.save_settings)
        settings_layout.addRow("Augmentation:", self.aug_combo)

        # Weight Init
        self.weight_combo = QComboBox()
        self.weight_combo.addItems(self.manager.get_available_weight_init())
        self.weight_combo.currentTextChanged.connect(self.save_settings)
        settings_layout.addRow("Weight Initialization:", self.weight_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Status & Actions ---
        status_group = QGroupBox("Dataset Status")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(16, 24, 16, 16)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("Dataset status will appear here...")
        status_layout.addWidget(self.status_text)

        btn_layout = QHBoxLayout()
        self.check_status_btn = QPushButton("Check Status")
        self.check_status_btn.setObjectName(SECONDARY_BUTTON)
        self.check_status_btn.clicked.connect(self.check_status)

        self.create_btn = QPushButton("Create Training Dataset")
        self.create_btn.setMinimumHeight(40)
        self.create_btn.clicked.connect(self.create_dataset)

        btn_layout.addWidget(self.check_status_btn)
        btn_layout.addWidget(self.create_btn)
        status_layout.addLayout(btn_layout)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

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
            self.net_combo.setCurrentText("dlcrnet_ms5")
        else:
            self.net_combo.setCurrentText("resnet_50")

    def check_status(self):
        """Check training dataset status"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)

        if not valid:
            self.status_text.setPlainText(f"Error: {error}")
            return

        try:
            status = self.manager.check_training_dataset_exists(config)

            if status["exists"]:
                text = "✓ Training dataset exists\n"
                text += f"Iterations: {status.get('iterations', 'N/A')}\n"
                text += f"Latest: {status.get('latest_iteration', 'N/A')}\n"
                text += f"Train folders: {status.get('train_folders', 'N/A')}"
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

        self.set_busy(True)
        self.status_text.setPlainText("Creating training dataset...")

        weight_choice = self.weight_combo.currentText()

        if "SuperAnimal" in weight_choice:
            init_weights = "superanimal"
        elif "Random" in weight_choice:
            init_weights = "random"
        else:
            init_weights = "imagenet"

        kwargs = {
            "num_shuffles": self.shuffle_spin.value(),
            "net_type": self.net_combo.currentText(),
            "augmenter_type": self.aug_combo.currentText(),
            "init_weights": init_weights,
        }

        self.worker = TrainingDatasetWorker(
            self.manager, config, self.is_multianimal, **kwargs
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        """Handle dataset creation completion"""
        self.set_busy(False)
        self.check_status()
        QMessageBox.information(
            self, "Success", "Training dataset created successfully"
        )

    def on_error(self, error: str):
        """Handle dataset creation error"""
        self.set_busy(False)
        self.status_text.setPlainText(f"Error: {error}")
        QMessageBox.critical(
            self, "Error", f"Failed to create training dataset:\n{error}"
        )

    def set_busy(self, busy: bool):
        """Toggle UI"""
        self.create_btn.setEnabled(not busy)
        self.check_status_btn.setEnabled(not busy)

    def save_settings(self):
        """Save current settings"""
        self.settings.setValue("network", self.net_combo.currentText())
        self.settings.setValue("augmentation", self.aug_combo.currentText())
        self.settings.setValue("weight_init", self.weight_combo.currentText())

    def load_settings(self):
        """Load saved settings"""
        augmentation = self.settings.value("augmentation", "default")
        if augmentation:
            index = self.aug_combo.findText(augmentation)
            if index >= 0:
                self.aug_combo.setCurrentIndex(index)

        weight_init = self.settings.value(
            "weight_init", "Transfer Learning - SuperAnimal TopViewMouse"
        )
        if weight_init:
            index = self.weight_combo.findText(weight_init)
            if index >= 0:
                self.weight_combo.setCurrentIndex(index)
