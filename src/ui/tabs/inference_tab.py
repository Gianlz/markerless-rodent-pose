"""Video Inference Tab"""

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.inference_manager import InferenceManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class InferenceWorker(QThread):
    """Worker thread for video analysis and labeled video creation"""

    finished = Signal()
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        manager: InferenceManager,
        config: str,
        videos: list[str],
        analyze: bool,
        create_video: bool,
        **kwargs,
    ):
        super().__init__()
        self.manager = manager
        self.config = config
        self.videos = videos
        self.analyze = analyze
        self.create_video = create_video
        self.kwargs = kwargs

    def run(self):
        try:
            if self.analyze:
                self.progress.emit("Analyzing videos...")
                # Only pass analyze-specific kwargs
                analyze_kwargs = {
                    k: v
                    for k, v in self.kwargs.items()
                    if k
                    in [
                        "shuffle",
                        "trainingsetindex",
                        "gputouse",
                        "save_as_csv",
                        "destfolder",
                    ]
                }
                self.manager.analyze_videos(self.config, self.videos, **analyze_kwargs)

            if self.create_video:
                self.progress.emit("Creating labeled videos...")
                # Only pass video-specific kwargs
                video_kwargs = {
                    k: v
                    for k, v in self.kwargs.items()
                    if k
                    in [
                        "shuffle",
                        "trainingsetindex",
                        "filtered",
                        "draw_skeleton",
                        "trailpoints",
                        "displayedbodyparts",
                        "destfolder",
                    ]
                }
                self.manager.create_labeled_video(
                    self.config, self.videos, **video_kwargs
                )

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class InferenceTab(QWidget):
    """Video inference and labeled video creation tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = InferenceManager()
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

        # --- Videos ---
        video_group = QGroupBox("Videos to Analyze")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(16, 24, 16, 16)
        video_layout.setSpacing(12)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
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
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)

        self.shuffle_combo = QComboBox()
        settings_layout.addRow("Shuffle Index:", self.shuffle_combo)

        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(0, 50)
        self.trail_spin.setValue(0)
        settings_layout.addRow("Trail Points:", self.trail_spin)

        self.filtered_check = QCheckBox("Use Filtered Predictions")
        self.filtered_check.setChecked(True)
        settings_layout.addRow("", self.filtered_check)

        self.skeleton_check = QCheckBox("Draw Skeleton")
        self.skeleton_check.setChecked(True)
        settings_layout.addRow("", self.skeleton_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # --- Actions ---
        action_layout = QHBoxLayout()
        action_layout.setSpacing(16)

        self.analyze_btn = QPushButton("Analyze Only")
        self.analyze_btn.setObjectName(SECONDARY_BUTTON)
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.clicked.connect(
            lambda: self.run_inference(analyze=True, create_video=False)
        )

        self.create_video_btn = QPushButton("Create Video Only")
        self.create_video_btn.setObjectName(SECONDARY_BUTTON)
        self.create_video_btn.setMinimumHeight(40)
        self.create_video_btn.clicked.connect(
            lambda: self.run_inference(analyze=False, create_video=True)
        )

        self.full_btn = QPushButton("Analyze & Create Video")
        self.full_btn.setMinimumHeight(40)
        self.full_btn.clicked.connect(
            lambda: self.run_inference(analyze=True, create_video=True)
        )

        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.create_video_btn)
        action_layout.addWidget(self.full_btn)

        layout.addLayout(action_layout)
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
                self.load_shuffles()

    def load_shuffles(self):
        """Load available shuffles"""
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

    def add_videos(self):
        """Add videos to list"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        existing = [
            self.video_list.item(i).text() for i in range(self.video_list.count())
        ]
        for path in file_paths:
            if path not in existing:
                self.video_list.addItem(path)

    def remove_video(self):
        """Remove selected videos"""
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def clear_videos(self):
        """Clear all videos"""
        self.video_list.clear()

    def run_inference(self, analyze: bool, create_video: bool):
        """Run video analysis and/or labeled video creation"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)

        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "Please add videos to analyze")
            return

        videos = [
            self.video_list.item(i).text() for i in range(self.video_list.count())
        ]

        # Check model
        best_snapshot = self.manager.get_best_snapshot(config)
        if not best_snapshot:
            QMessageBox.warning(
                self, "No Model", "No trained model found. Please train a model first."
            )
            return

        if create_video and not analyze:
            unanalyzed = []
            for video in videos:
                if not self.manager.check_analysis_exists(video, config):
                    unanalyzed.append(Path(video).name)

            if unanalyzed:
                reply = QMessageBox.question(
                    self,
                    "Videos Not Analyzed",
                    f"{len(unanalyzed)} video(s) have not been analyzed yet. Analyze them first?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    analyze = True
                else:
                    return

        self.set_busy(True)
        self.status_label.setText("Starting processing...")

        shuffle = (
            int(self.shuffle_combo.currentText())
            if self.shuffle_combo.currentText()
            else 1
        )

        kwargs = {
            "shuffle": shuffle,
            "trainingsetindex": 0,
            "save_as_csv": True,
            "filtered": self.filtered_check.isChecked(),
            "draw_skeleton": self.skeleton_check.isChecked(),
            "trailpoints": self.trail_spin.value(),
        }

        self.worker = InferenceWorker(
            self.manager, config, videos, analyze, create_video, **kwargs
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, message: str):
        """Handle progress update"""
        self.status_label.setText(message)

    def on_finished(self):
        """Handle inference completion"""
        self.set_busy(False)
        self.status_label.setText("Completed!")
        QMessageBox.information(
            self, "Success", "Video processing completed successfully"
        )

    def on_error(self, error: str):
        """Handle inference error"""
        self.set_busy(False)
        self.status_label.setText("Error occurred.")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error}")

    def set_busy(self, busy: bool):
        """Toggle UI"""
        self.analyze_btn.setEnabled(not busy)
        self.create_video_btn.setEnabled(not busy)
        self.full_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
