"""Model Evaluation Tab"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.evaluation_manager import EvaluationManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class EvaluationWorker(QThread):
    """Worker thread for model evaluation"""

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        manager: EvaluationManager,
        config: str,
        shuffle: int,
        plotting: bool,
    ):
        super().__init__()
        self.manager = manager
        self.config = config
        self.shuffle = shuffle
        self.plotting = plotting

    def run(self):
        try:
            metrics = self.manager.evaluate_network(
                self.config,
                shuffle=self.shuffle,
                plotting=self.plotting,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))


class EvaluationTab(QWidget):
    """Model evaluation and metrics tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = EvaluationManager()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

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

        # --- Settings ---
        settings_group = QGroupBox("Evaluation Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)

        self.shuffle_combo = QComboBox()
        settings_layout.addRow("Shuffle:", self.shuffle_combo)

        self.plotting_check = QCheckBox("Generate Plots")
        self.plotting_check.setChecked(False)
        self.plotting_check.setToolTip("Generate evaluation plots (slower)")
        settings_layout.addRow("", self.plotting_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Results ---
        results_group = QGroupBox("Evaluation Metrics")
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(16, 24, 16, 16)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMinimumHeight(250)
        self.metrics_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.metrics_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # --- Actions ---
        action_layout = QHBoxLayout()

        self.evaluate_btn = QPushButton("Evaluate Model")
        self.evaluate_btn.setMinimumHeight(40)
        self.evaluate_btn.clicked.connect(self.run_evaluation)
        action_layout.addWidget(self.evaluate_btn)

        layout.addLayout(action_layout)
        layout.addStretch()

    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)

    def set_config_path(self, path: str):
        self.config_input.setText(path)

    def on_config_changed(self):
        config = self.config_input.text()
        if config:
            valid, _ = validate_config_path(config)
            if valid:
                self.load_shuffles()

    def load_shuffles(self):
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        if not valid:
            return

        try:
            from ...core.train_manager import TrainManager

            train_manager = TrainManager()
            shuffles = train_manager.get_available_shuffles(config)
            self.shuffle_combo.clear()
            self.shuffle_combo.addItems([str(s) for s in shuffles])
        except Exception:
            self.shuffle_combo.clear()
            self.shuffle_combo.addItem("1")

    def run_evaluation(self):
        config = self.config_input.text()
        valid, error = validate_config_path(config)

        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        shuffle = (
            int(self.shuffle_combo.currentText())
            if self.shuffle_combo.currentText()
            else 1
        )

        self.set_busy(True)
        self.status_label.setText("Starting evaluation...")
        self.metrics_table.setRowCount(0)

        self.worker = EvaluationWorker(
            self.manager,
            config,
            shuffle,
            self.plotting_check.isChecked(),
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.start()

    def on_finished(self, metrics: dict):
        self.set_busy(False)
        self.status_label.setText("Evaluation complete!")
        self.display_metrics(metrics)

    def display_metrics(self, metrics: dict):
        """Display metrics in the table"""
        display_names = {
            "precision": "Precision",
            "recall": "Recall (Sensibilidade)",
            "f1_score": "F1-Score",
            "iou": "IoU (Intersection over Union)",
            "train_error": "Train Error (pixels)",
            "test_error": "Test Error (pixels)",
            "pcutoff": "P-cutoff",
        }

        rows = []
        for key, display_name in display_names.items():
            value = metrics.get(key)
            if value is not None:
                if isinstance(value, float):
                    if key in ["precision", "recall", "f1_score", "iou"]:
                        formatted = f"{value:.4f} ({value * 100:.2f}%)"
                    else:
                        formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                rows.append((display_name, formatted))

        self.metrics_table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

        self.metrics_table.resizeColumnsToContents()

    def on_error(self, error: str):
        self.set_busy(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"Evaluation failed:\n{error}")

    def set_busy(self, busy: bool):
        self.evaluate_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
