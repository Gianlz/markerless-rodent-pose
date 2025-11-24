"""Freezing Test Tab"""

import cv2
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QGroupBox,
    QListWidget,
    QCheckBox,
)

from ...core.freezing_manager import FreezingManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class AnalysisWorker(QThread):
    """Worker thread for freezing analysis"""

    finished = Signal(str)  # Returns path to saved Excel file
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        video_path: str,
        config_path: str,
        line_points: list[QPoint],
        force_analysis: bool = False,
        create_video: bool = False,
    ):
        super().__init__()
        self.video_path = video_path
        self.config_path = config_path
        self.line_points = line_points
        self.force_analysis = force_analysis
        self.create_video = create_video
        self.manager = FreezingManager()

    def run(self):
        try:
            # Convert QPoints to tuples
            points = [(p.x(), p.y()) for p in self.line_points]

            output_path = self.manager.run_freezing_analysis(
                video_path=self.video_path,
                config_path=self.config_path,
                line_points=points,
                force_analysis=self.force_analysis,
                create_video=self.create_video,
                progress_callback=self.progress.emit,
            )

            msg = f"Analysis saved to:\n{output_path}"
            if self.create_video:
                video_path = Path(self.video_path)
                output_video_path = video_path.parent / f"{video_path.stem}_labeled.mp4"
                msg += f"\n\nLabeled video saved to:\n{output_video_path}"

            self.finished.emit(msg)

        except Exception as e:
            self.error.emit(str(e))


class VideoLabel(QLabel):
    """Custom Label to handle drawing"""

    def __init__(self):
        super().__init__()
        self.show_line = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid #444; background-color: #222;")

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_line:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))

            # Draw vertical line in the center
            x = self.width() // 2
            painter.drawLine(x, 0, x, self.height())


class FreezingTab(QWidget):
    """Freezing Test Analysis Tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.current_video_path = None
        self.analysis_points = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- Top Controls ---
        top_layout = QHBoxLayout()

        # Config
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        config_btn = QPushButton("Browse Config")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.clicked.connect(self.browse_config)

        top_layout.addWidget(QLabel("Config:"))
        top_layout.addWidget(self.config_input)
        top_layout.addWidget(config_btn)

        layout.addLayout(top_layout)

        # --- Video Selection ---
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()

        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(100)
        self.video_list.itemClicked.connect(self.on_video_selected)
        video_layout.addWidget(self.video_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Video")
        add_btn.clicked.connect(self.add_video)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName(SECONDARY_BUTTON)
        clear_btn.clicked.connect(self.video_list.clear)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        video_layout.addLayout(btn_layout)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # --- Video Preview ---
        preview_group = QGroupBox("Video Preview (Auto-Split Center)")
        preview_layout = QVBoxLayout()

        self.video_label = VideoLabel()
        preview_layout.addWidget(self.video_label)

        instr_label = QLabel(
            "The video will be automatically split in the center for analysis."
        )
        instr_label.setStyleSheet("color: #888;")
        preview_layout.addWidget(instr_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # --- Settings ---
        settings_layout = QHBoxLayout()

        self.force_analysis_check = QCheckBox("Force Re-analysis (DeepLabCut)")
        self.force_analysis_check.setToolTip(
            "If checked, will run DeepLabCut analysis even if results already exist."
        )
        settings_layout.addWidget(self.force_analysis_check)

        self.create_video_check = QCheckBox("Create Labeled Video")
        self.create_video_check.setToolTip(
            "Create a new video with side labels and skeleton overlay."
        )
        self.create_video_check.setChecked(True)
        settings_layout.addWidget(self.create_video_check)

        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # --- Actions ---
        self.analyze_btn = QPushButton("Run Freezing Test")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analyze_btn)

    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)

    def set_config_path(self, path: str):
        self.config_input.setText(path)

    def add_video(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        for path in file_paths:
            self.video_list.addItem(path)

    def on_video_selected(self, item):
        self.current_video_path = item.text()
        self.load_video_frame(self.current_video_path)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Ready to analyze.")

    def load_video_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )

            # Scale to fit label while keeping aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.show_line = True

            # Calculate analysis points (Center Split)
            center_x = w // 2
            self.analysis_points = [QPoint(center_x, 0), QPoint(center_x, h)]

        else:
            QMessageBox.warning(self, "Error", "Could not read video frame")

    def run_analysis(self):
        if not self.current_video_path or not self.analysis_points:
            return

        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        self.set_busy(True)
        self.worker = AnalysisWorker(
            self.current_video_path,
            config,
            self.analysis_points,
            self.force_analysis_check.isChecked(),
            self.create_video_check.isChecked(),
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.start()

    def on_finished(self, output_path):
        self.set_busy(False)
        self.status_label.setText("Completed!")
        QMessageBox.information(self, "Success", output_path)

    def on_error(self, error):
        self.set_busy(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error}")

    def set_busy(self, busy):
        self.analyze_btn.setEnabled(not busy)
        self.video_list.setEnabled(not busy)
        self.force_analysis_check.setEnabled(not busy)
        self.create_video_check.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
