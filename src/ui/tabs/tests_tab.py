"""Custom Tests Tab"""

from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QListWidget,
    QAbstractItemView,
    QDialog,
    QSlider,
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen

import cv2

from ...core.tests_manager import TestsManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON, INFO_LABEL

class VideoPreviewDialog(QDialog):
    """Dialog to preview video and set a vertical line"""
    
    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Box Center Line")
        self.video_path = video_path
        self.line_x = 0
        self.max_w = 100
        self.init_ui()
        self.load_frame()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Image Display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Slider for X coordinate
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Center Line:"))
        
        self.x_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(100)
        self.x_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.x_slider)
        
        self.val_label = QLabel("0 px")
        slider_layout.addWidget(self.val_label)
        layout.addLayout(slider_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Accept")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)
        
    def load_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
            
        ret, self.frame = cap.read()
        cap.release()
        
        if ret:
            self.max_w = self.frame.shape[1]
            self.x_slider.setMaximum(self.max_w)
            self.line_x = self.max_w // 2
            self.x_slider.setValue(self.line_x)
            self.update_image()
            
    def on_slider_changed(self, val):
        self.line_x = val
        self.val_label.setText(f"{val} px")
        self.update_image()
        
    def update_image(self):
        if not hasattr(self, 'frame'):
            return
            
        # Draw line on frame
        display_frame = self.frame.copy()
        cv2.line(display_frame, (self.line_x, 0), (self.line_x, display_frame.shape[0]), (0, 0, 255), 2)
        
        # Convert to QImage
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        # OpenCV uses BGR
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


class TestWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, manager: TestsManager, config: str, video: str, line_x: int):
        super().__init__()
        self.manager = manager
        self.config = config
        self.video = video
        self.line_x = line_x

    def run(self):
        try:
            self.progress.emit(f"Checking for analysis: {Path(self.video).name}")
            h5_file = self.manager.analyze_video_if_needed(self.config, self.video)
            
            self.progress.emit(f"Getting video properties...")
            fps, w, h = self.manager.get_video_info(self.video)
            
            self.progress.emit(f"Calculating time in sides A and B...")
            results = self.manager.calculate_box_sides_time(h5_file, self.line_x, fps)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class TestsTab(QWidget):
    """Custom tests tab for box side experiments"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = TestsManager()
        self.worker = None
        self.line_x = None
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

        # --- Video Selection ---
        video_group = QGroupBox("Tests: Two Sides Box")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(16, 24, 16, 16)
        video_layout.setSpacing(12)

        info_label = QLabel(
            "Select a video to test, set the vertical center line to divide Side A (Left) and Side B (Right), "
            "then run the test to calculate the time the rat spent on each side. "
            "The rat is only considered on a side if ALL body parts are on that side."
        )
        info_label.setWordWrap(True)
        video_layout.addWidget(info_label)

        video_param_layout = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Select video for the test...")
        self.video_input.setReadOnly(True)
        
        video_btn = QPushButton("Select Video")
        video_btn.setObjectName(SECONDARY_BUTTON)
        video_btn.clicked.connect(self.select_video)
        
        video_param_layout.addWidget(self.video_input)
        video_param_layout.addWidget(video_btn)
        video_layout.addLayout(video_param_layout)

        setup_layout = QHBoxLayout()
        self.line_label = QLabel("Center Line: Not Set")
        
        self.set_line_btn = QPushButton("Set Center Line")
        self.set_line_btn.setObjectName(SECONDARY_BUTTON)
        self.set_line_btn.clicked.connect(self.set_center_line)
        self.set_line_btn.setEnabled(False) # Enables after video is selected
        
        setup_layout.addWidget(self.line_label)
        setup_layout.addWidget(self.set_line_btn)
        setup_layout.addStretch()
        video_layout.addLayout(setup_layout)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # --- Progress & Execution ---
        exec_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        exec_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        exec_layout.addWidget(self.status_label)

        self.run_tests_btn = QPushButton("Run Custom Test")
        self.run_tests_btn.setMinimumHeight(40)
        self.run_tests_btn.clicked.connect(self.run_custom_tests)
        exec_layout.addWidget(self.run_tests_btn)
        
        layout.addLayout(exec_layout)

        # --- Results ---
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        
        self.result_a_lbl = QLabel("-")
        self.result_b_lbl = QLabel("-")
        self.result_total_lbl = QLabel("-")
        
        results_layout.addRow("Time in Side A (Left):", self.result_a_lbl)
        results_layout.addRow("Time in Side B (Right):", self.result_b_lbl)
        results_layout.addRow("Total Time Evaluated:", self.result_total_lbl)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)

    def set_config_path(self, path: str):
        self.config_input.setText(path)
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_input.setText(file_path)
            self.set_line_btn.setEnabled(True)
            self.line_x = None
            self.line_label.setText("Center Line: Not Set")

    def set_center_line(self):
        video_path = self.video_input.text()
        if not video_path:
            return
            
        dialog = VideoPreviewDialog(video_path, self)
        if dialog.exec():
            self.line_x = dialog.line_x
            self.line_label.setText(f"Center Line: {self.line_x} px")

    def run_custom_tests(self):
        config = self.config_input.text()
        video = self.video_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
            
        if not video:
            QMessageBox.warning(self, "Validation Error", "Please select a video.")
            return
            
        if self.line_x is None:
            QMessageBox.warning(self, "Validation Error", "Please set the center line first.")
            return

        self.set_busy(True)
        self.status_label.setText("Initializing test...")
        
        self.worker = TestWorker(self.manager, config, video, self.line_x)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
        
    def on_progress(self, msg: str):
        self.status_label.setText(msg)
        
    def on_finished(self, results: dict):
        self.set_busy(False)
        self.status_label.setText("Test completed successfully.")
        
        self.result_a_lbl.setText(f"{results['time_A_sec']:.2f} seconds ({results['frames_A']} frames)")
        self.result_b_lbl.setText(f"{results['time_B_sec']:.2f} seconds ({results['frames_B']} frames)")
        
        total_sec = results['total_frames'] / results['fps'] if results['fps'] > 0 else 0
        self.result_total_lbl.setText(f"{total_sec:.2f} seconds ({results['total_frames']} frames @ {results['fps']:.1f} fps)")
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Test completed successfully!\n\nResults saved to:\n{results['saved_file']}"
        )

    def on_error(self, error: str):
        self.set_busy(False)
        self.status_label.setText("Error during calculation.")
        QMessageBox.critical(self, "Test Error", f"An error occurred:\n{error}")
        
    def set_busy(self, busy: bool):
        self.run_tests_btn.setEnabled(not busy)
        self.set_line_btn.setEnabled(not busy)
        self.config_input.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

