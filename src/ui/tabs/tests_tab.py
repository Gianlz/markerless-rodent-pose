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
    QDialog,
    QSlider,
    QTabWidget,
    QSpinBox,
    QComboBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer

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
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
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
            
        display_frame = self.frame.copy()
        cv2.line(display_frame, (self.line_x, 0), (self.line_x, display_frame.shape[0]), (0, 0, 255), 2)
        
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


class GridPreviewDialog(QDialog):
    """Dialog to preview video and set a square ROI for 3x3 grid"""
    
    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set 3x3 Grid Region of Interest")
        self.video_path = video_path
        self.roi_x = 0
        self.roi_y = 0
        self.roi_size = 300
        self.max_w = 100
        self.max_h = 100
        self.init_ui()
        self.load_frame()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        # X Slider
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Position:"))
        self.x_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_slider.valueChanged.connect(self.on_x_changed)
        x_layout.addWidget(self.x_slider)
        self.x_val_label = QLabel("0 px")
        x_layout.addWidget(self.x_val_label)
        layout.addLayout(x_layout)

        # Y Slider
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Position:"))
        self.y_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_slider.valueChanged.connect(self.on_y_changed)
        y_layout.addWidget(self.y_slider)
        self.y_val_label = QLabel("0 px")
        y_layout.addWidget(self.y_val_label)
        layout.addLayout(y_layout)

        # Size Slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Grid Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(50)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_slider)
        self.size_val_label = QLabel("300 px")
        size_layout.addWidget(self.size_val_label)
        layout.addLayout(size_layout)
        
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
            self.max_h = self.frame.shape[0]
            
            self.x_slider.setMaximum(self.max_w - 50)
            self.y_slider.setMaximum(self.max_h - 50)
            self.size_slider.setMaximum(min(self.max_w, self.max_h))
            
            self.roi_x = self.max_w // 4
            self.roi_y = self.max_h // 4
            self.roi_size = min(self.max_w, self.max_h) // 2
            
            self.x_slider.setValue(self.roi_x)
            self.y_slider.setValue(self.roi_y)
            self.size_slider.setValue(self.roi_size)
            
            self.update_image()
            
    def on_x_changed(self, val):
        self.roi_x = val
        self.x_val_label.setText(f"{val} px")
        self.update_image()
        
    def on_y_changed(self, val):
        self.roi_y = val
        self.y_val_label.setText(f"{val} px")
        self.update_image()
        
    def on_size_changed(self, val):
        self.roi_size = val
        self.size_val_label.setText(f"{val} px")
        self.update_image()
        
    def update_image(self):
        if not hasattr(self, 'frame'):
            return
            
        display_frame = self.frame.copy()
        cv2.rectangle(display_frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_size, self.roi_y + self.roi_size), (0, 0, 255), 2)
        cell_size = self.roi_size // 3
        for i in range(1, 4):
            line_pos_x = self.roi_x + int(i * cell_size)
            if i < 3:
                cv2.line(display_frame, (line_pos_x, self.roi_y), (line_pos_x, self.roi_y + self.roi_size), (0, 255, 0), 1)
            line_pos_y = self.roi_y + int(i * cell_size)
            if i < 3:
                cv2.line(display_frame, (self.roi_x, line_pos_y), (self.roi_x + self.roi_size, line_pos_y), (0, 255, 0), 1)

        
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
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
            
            self.progress.emit("Getting video properties...")
            fps, _, _ = self.manager.get_video_info(self.video)
            
            self.progress.emit("Calculating time in sides A and B...")
            results = self.manager.calculate_box_sides_time(h5_file, self.video, self.line_x, fps)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class GridWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, manager: TestsManager, config: str, video: str, roi: tuple[int, int, int]):
        super().__init__()
        self.manager = manager
        self.config = config
        self.video = video
        self.roi = roi

    def run(self):
        try:
            self.progress.emit(f"Checking for analysis: {Path(self.video).name}")
            h5_file = self.manager.analyze_video_if_needed(self.config, self.video)
            
            self.progress.emit("Getting video properties...")
            fps, _, _ = self.manager.get_video_info(self.video)
            
            self.progress.emit("Calculating grid interactions...")
            results = self.manager.calculate_grid_test(h5_file, self.video, self.roi, fps)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class PlaybackDialog(QDialog):
    """Dialog to playback the realtime trace or detection using PySide6 rendering"""
    def __init__(self, manager: TestsManager, h5_file: str, video_path: str, trail_length: int, mode: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Realtime {mode.capitalize()} Playback")
        self.setMinimumSize(800, 600)
        
        self.manager = manager
        self.h5_file = h5_file
        self.video_path = video_path
        self.trail_length = trail_length
        self.mode = mode
        self.params = {'pcutoff': 0.6}
        
        self.generator = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Precision Slider (p-cutoff):"))
        self.pcutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.pcutoff_slider.setRange(0, 100)
        self.pcutoff_slider.setValue(60)
        self.pcutoff_slider.valueChanged.connect(self.on_pcutoff_changed)
        controls_layout.addWidget(self.pcutoff_slider)
        
        self.pcutoff_val_label = QLabel("0.60")
        controls_layout.addWidget(self.pcutoff_val_label)
        layout.addLayout(controls_layout)
        
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
    def on_pcutoff_changed(self, val):
        pcutoff = val / 100.0
        self.pcutoff_val_label.setText(f"{pcutoff:.2f}")
        self.params['pcutoff'] = pcutoff

    def showEvent(self, event):
        super().showEvent(event)
        if self.mode == 'trace':
            self.generator = self.manager.play_realtime_trace(
                self.h5_file, self.video_path, self.trail_length, self.params
            )
        else:
            self.generator = self.manager.play_realtime_detection(
                self.h5_file, self.video_path, self.params
            )
        self.timer.start(33) # ~30 fps
        
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(33)
            self.play_btn.setText("Pause")
            
    def next_frame(self):
        try:
            frame = next(self.generator)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        except StopIteration:
            self.timer.stop()
            self.play_btn.setEnabled(False)
            self.play_btn.setText("Finished")
            

class PrepWorker(QThread):
    finished = Signal(str, str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, manager: TestsManager, config: str, video: str, mode: str):
        super().__init__()
        self.manager = manager
        self.config = config
        self.video = video
        self.mode = mode

    def run(self):
        try:
            self.progress.emit(f"Checking for analysis: {Path(self.video).name}")
            h5_file = self.manager.analyze_video_if_needed(self.config, self.video)
            self.finished.emit(h5_file, self.mode)
        except Exception as e:
            self.error.emit(str(e))


class TestsTab(QWidget):
    """Custom tests tab for box side experiments and real-time trace"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = TestsManager()
        self.box_worker = None
        self.viewer_worker = None
        self.grid_worker = None
        self.line_x = None
        self.roi = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- Configuration & Video Mapping ---
        globals_group = QGroupBox("Target Selection")
        globals_layout = QVBoxLayout()
        globals_layout.setContentsMargins(16, 24, 16, 16)
        
        # Config
        config_layout = QHBoxLayout()
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.clicked.connect(self.browse_config)
        config_layout.addWidget(QLabel("Config:"))
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        globals_layout.addLayout(config_layout)

        # Video
        video_layout = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Select video to test...")
        self.video_input.setReadOnly(True)
        video_btn = QPushButton("Select Video")
        video_btn.setObjectName(SECONDARY_BUTTON)
        video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.video_input)
        video_layout.addWidget(video_btn)
        globals_layout.addLayout(video_layout)

        globals_group.setLayout(globals_layout)
        layout.addWidget(globals_group)

        # --- Tests Tabs ---
        self.tests_tabs = QTabWidget()
        
        # 1. Two Sides Box Tab
        self.box_tab = QWidget()
        box_layout = QVBoxLayout(self.box_tab)
        
        box_info = QLabel(
            "Set the vertical center line to divide Side A (Left) and Side B (Right), "
            "then run the test to calculate the time the rat spent on each side. "
            "The rat is only considered on a side if ALL body parts are on that side."
        )
        box_info.setWordWrap(True)
        box_layout.addWidget(box_info)
        
        setup_layout = QHBoxLayout()
        self.line_label = QLabel("Center Line: Not Set")
        self.set_line_btn = QPushButton("Set Center Line")
        self.set_line_btn.setObjectName(SECONDARY_BUTTON)
        self.set_line_btn.clicked.connect(self.set_center_line)
        self.set_line_btn.setEnabled(False) 
        
        setup_layout.addWidget(self.line_label)
        setup_layout.addWidget(self.set_line_btn)
        setup_layout.addStretch()
        box_layout.addLayout(setup_layout)
        
        self.run_box_btn = QPushButton("Run Box Test")
        self.run_box_btn.setMinimumHeight(40)
        self.run_box_btn.clicked.connect(self.run_box_test)
        box_layout.addWidget(self.run_box_btn)
        
        self.results_group = QGroupBox("Box Test Results")
        res_layout = QFormLayout()
        self.result_a_lbl = QLabel("-")
        self.result_b_lbl = QLabel("-")
        self.result_total_lbl = QLabel("-")
        res_layout.addRow("Time in Side A (Left):", self.result_a_lbl)
        res_layout.addRow("Time in Side B (Right):", self.result_b_lbl)
        res_layout.addRow("Total Time Evaluated:", self.result_total_lbl)
        self.results_group.setLayout(res_layout)
        box_layout.addWidget(self.results_group)
        box_layout.addStretch()
        
        self.tests_tabs.addTab(self.box_tab, "Two-Sides Box Test")
        
        # 2. Realtime Viewer Tab
        self.viewer_tab = QWidget()
        viewer_layout = QVBoxLayout(self.viewer_tab)
        
        viewer_info = QLabel(
            "Plays the selected video in real-time overlaid with the tracking results.\n"
            "Choose 'Detection Mode' to see exact points detected and their body part labels. "
            "Choose 'Trace Mode' to see movement paths visualizing the tracking history."
        )
        viewer_info.setWordWrap(True)
        viewer_layout.addWidget(viewer_info)
        
        viewer_settings = QFormLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Detection", "Trace"])
        viewer_settings.addRow("Visualization Mode:", self.mode_combo)
        
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(1, 500)
        self.trail_spin.setValue(60)
        viewer_settings.addRow("Trail Length (Frames):", self.trail_spin)
        
        self.mode_combo.currentTextChanged.connect(
            lambda text: self.trail_spin.setEnabled(text == "Trace")
        )
        self.trail_spin.setEnabled(False) # Default is Detection

        viewer_layout.addLayout(viewer_settings)
        
        self.run_viewer_btn = QPushButton("Play Realtime Visualization")
        self.run_viewer_btn.setMinimumHeight(40)
        self.run_viewer_btn.clicked.connect(self.run_viewer_test)
        viewer_layout.addWidget(self.run_viewer_btn)
        viewer_layout.addStretch()
        
        self.tests_tabs.addTab(self.viewer_tab, "Realtime Viewer")

        # 3. 3x3 Grid Tab
        self.grid_tab = QWidget()
        grid_layout = QVBoxLayout(self.grid_tab)
        
        grid_info = QLabel(
            "Select a Region of Interest (ROI) spanning a 3x3 grid where all 4 paws "
            "need to be present in the same square simultaneously to log an entry. "
            "This will output entry statistics, a path visualization, and a heatmap limit."
        )
        grid_info.setWordWrap(True)
        grid_layout.addWidget(grid_info)
        
        setup_grid_layout = QHBoxLayout()
        self.grid_roi_label = QLabel("Grid ROI: Not Set")
        self.set_grid_btn = QPushButton("Select ROI Grid")
        self.set_grid_btn.setObjectName(SECONDARY_BUTTON)
        self.set_grid_btn.clicked.connect(self.set_grid_roi)
        self.set_grid_btn.setEnabled(False) 
        
        setup_grid_layout.addWidget(self.grid_roi_label)
        setup_grid_layout.addWidget(self.set_grid_btn)
        setup_grid_layout.addStretch()
        grid_layout.addLayout(setup_grid_layout)
        
        self.run_grid_btn = QPushButton("Run Grid Test")
        self.run_grid_btn.setMinimumHeight(40)
        self.run_grid_btn.clicked.connect(self.run_grid_test)
        grid_layout.addWidget(self.run_grid_btn)
        
        self.grid_results = QGroupBox("Grid Test Results")
        grid_res_layout = QFormLayout()
        self.grid_total_entries_lbl = QLabel("-")
        self.grid_total_time_lbl = QLabel("-")
        grid_res_layout.addRow("Total Entries:", self.grid_total_entries_lbl)
        grid_res_layout.addRow("Total Time Evaluated:", self.grid_total_time_lbl)
        self.grid_results.setLayout(grid_res_layout)
        grid_layout.addWidget(self.grid_results)
        grid_layout.addStretch()
        
        self.tests_tabs.addTab(self.grid_tab, "3x3 Grid ROI Test")

        layout.addWidget(self.tests_tabs)

        # --- Progress & Execution ---
        exec_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        exec_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        exec_layout.addWidget(self.status_label)
        layout.addLayout(exec_layout)

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
            self.set_grid_btn.setEnabled(True)
            self.line_x = None
            self.roi = None
            self.line_label.setText("Center Line: Not Set")
            self.grid_roi_label.setText("Grid ROI: Not Set")

    def set_center_line(self):
        video_path = self.video_input.text()
        if not video_path:
            return
            
        dialog = VideoPreviewDialog(video_path, self)
        if dialog.exec():
            self.line_x = dialog.line_x
            self.line_label.setText(f"Center Line: {self.line_x} px")

    def set_grid_roi(self):
        video_path = self.video_input.text()
        if not video_path:
            return
            
        dialog = GridPreviewDialog(video_path, self)
        if dialog.exec():
            self.roi = (dialog.roi_x, dialog.roi_y, dialog.roi_size)
            self.grid_roi_label.setText(f"ROI: x={self.roi[0]}, y={self.roi[1]}, size={self.roi[2]}")

    # --- Box Test Execution ---
    def run_box_test(self):
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
        self.status_label.setText("Initializing box test...")
        
        self.box_worker = TestWorker(self.manager, config, video, self.line_x)
        self.box_worker.finished.connect(self.on_box_finished)
        self.box_worker.error.connect(self.on_error)
        self.box_worker.progress.connect(self.on_progress)
        self.box_worker.start()

    def on_box_finished(self, results: dict):
        self.set_busy(False)
        self.status_label.setText("Box test completed successfully.")
        
        self.result_a_lbl.setText(f"{results['time_A_sec']:.2f} seconds ({results['frames_A']} frames)")
        self.result_b_lbl.setText(f"{results['time_B_sec']:.2f} seconds ({results['frames_B']} frames)")
        
        total_sec = results['total_frames'] / results['fps'] if results['fps'] > 0 else 0
        self.result_total_lbl.setText(f"{total_sec:.2f} seconds ({results['total_frames']} frames @ {results['fps']:.1f} fps)")
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Box test completed successfully!\n\nReport saved to:\n{results['saved_file']}\n\nVideo generated at:\n{results['saved_video']}"
        )

    # --- Grid Test Execution ---
    def run_grid_test(self):
        config = self.config_input.text()
        video = self.video_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        if not video:
            QMessageBox.warning(self, "Validation Error", "Please select a video.")
            return
        if self.roi is None:
            QMessageBox.warning(self, "Validation Error", "Please set the Grid ROI first.")
            return

        self.set_busy(True)
        self.status_label.setText("Initializing grid test...")
        
        self.grid_worker = GridWorker(self.manager, config, video, self.roi)
        self.grid_worker.finished.connect(self.on_grid_finished)
        self.grid_worker.error.connect(self.on_error)
        self.grid_worker.progress.connect(self.on_progress)
        self.grid_worker.start()

    def on_grid_finished(self, results: dict):
        self.set_busy(False)
        self.status_label.setText("Grid test completed successfully.")
        
        self.grid_total_entries_lbl.setText(f"{results['total_entries']} entries")
        total_time = sum(results['time_per_square'].values())
        self.grid_total_time_lbl.setText(f"{total_time:.2f} seconds")
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Grid test completed successfully!\n\nReport saved to:\n{results['saved_file']}\n\n"
            f"Trajectory: {results['trajectory_image']}\n"
            f"Heatmap: {results['heatmap_image']}"
        )

    # --- Viewer Test Execution ---
    def run_viewer_test(self):
        config = self.config_input.text()
        video = self.video_input.text()
        
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        if not video:
            QMessageBox.warning(self, "Validation Error", "Please select a video.")
            return

        self.set_busy(True)
        self.status_label.setText("Preparing real-time visualization data...")
        
        mode = self.mode_combo.currentText().lower()
        self.viewer_worker = PrepWorker(self.manager, config, video, mode)
        self.viewer_worker.finished.connect(self.on_viewer_prepared)
        self.viewer_worker.error.connect(self.on_error)
        self.viewer_worker.progress.connect(self.on_progress)
        self.viewer_worker.start()

    def on_viewer_prepared(self, h5_file: str, mode: str):
        self.status_label.setText("Starting playback dialog...")
        try:
            # We open an independent Dialog so we can safely kill it
            dialog = PlaybackDialog(
                self.manager,
                h5_file,
                self.video_input.text(),
                self.trail_spin.value(),
                mode,
                self
            )
            dialog.exec()
            self.status_label.setText("Playback closed.")
        except Exception as e:
            self.on_error(str(e))
        finally:
            self.set_busy(False)

    # --- Shared Slots ---
    def on_progress(self, msg: str):
        self.status_label.setText(msg)

    def on_error(self, error: str):
        self.set_busy(False)
        self.status_label.setText("Error occurred during test execution.")
        QMessageBox.critical(self, "Test Error", f"An error occurred:\n{error}")
        
    def set_busy(self, busy: bool):
        self.run_box_btn.setEnabled(not busy)
        self.run_viewer_btn.setEnabled(not busy)
        self.run_grid_btn.setEnabled(not busy)
        self.set_line_btn.setEnabled(not busy)
        self.set_grid_btn.setEnabled(not busy)
        self.config_input.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
