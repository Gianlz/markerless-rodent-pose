"""Freezing Test Tab"""

import cv2
import pandas as pd
import numpy as np
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

from ...core.inference_manager import InferenceManager
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
        self.manager = InferenceManager()

    def run(self):
        try:
            self.progress.emit("Checking analysis status...")
            video_path = Path(self.video_path)

            # Check if analysis is needed
            analysis_exists = self.manager.check_analysis_exists(
                str(video_path), self.config_path
            )

            if self.force_analysis or not analysis_exists:
                self.progress.emit(
                    "Running DeepLabCut analysis (this may take time)..."
                )
                self.manager.analyze_videos(
                    self.config_path, [str(video_path)], save_as_csv=True
                )

            self.progress.emit("Loading data...")

            # Find analysis file
            h5_files = list(video_path.parent.glob(f"{video_path.stem}DLC*.h5"))
            if not h5_files:
                raise FileNotFoundError(
                    "No analysis file found even after running analysis."
                )

            # Use the most recent one if multiple
            h5_file = sorted(h5_files)[-1]

            df = pd.read_hdf(h5_file)

            # Get scorer and bodyparts
            scorer = df.columns.levels[0][0]
            bodyparts = df.columns.levels[1]

            # Check for required bodyparts
            required_paws = ["FR_paw", "FL_paw", "BR_paw", "BL_paw"]
            for paw in required_paws:
                if paw not in bodyparts:
                    raise ValueError(
                        f"Missing bodypart: {paw}. Model must have: {required_paws}"
                    )

            self.progress.emit("Analyzing frames...")

            # Define line
            p1 = np.array([self.line_points[0].x(), self.line_points[0].y()])
            p2 = np.array([self.line_points[1].x(), self.line_points[1].y()])

            # Vector representing the line
            line_vec = p2 - p1

            # Normal vector to the line (for determining side)
            # Rotate 90 degrees
            normal_vec = np.array([-line_vec[1], line_vec[0]])

            def get_side(point):
                # Dot product with normal vector determines side
                # point is (x, y)
                vec_to_point = point - p1
                return np.sign(np.dot(vec_to_point, normal_vec))

            # Open Video Capture
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not fps or np.isnan(fps):
                fps = 30.0  # Fallback

            # Setup Video Writer if requested
            writer = None
            if self.create_video:
                output_video_path = video_path.parent / f"{video_path.stem}_labeled.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, fps, (width, height)
                )

            total_frames = len(df)
            side_a_frames = 0
            side_b_frames = 0

            # Event logging
            events = []
            current_state = None  # 'A', 'B', or None
            state_start_frame = 0

            # Iterate through frames
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                paws_on_side_a = 0
                paws_on_side_b = 0

                # Draw center line on frame
                if writer:
                    cv2.line(
                        frame,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        (0, 0, 255),  # Red
                        2,
                    )

                for paw in required_paws:
                    x = df.iloc[i][(scorer, paw, "x")]
                    y = df.iloc[i][(scorer, paw, "y")]
                    likelihood = df.iloc[i][(scorer, paw, "likelihood")]

                    if likelihood < 0.1:
                        continue

                    side = get_side(np.array([x, y]))
                    if side > 0:
                        paws_on_side_a += 1
                    else:
                        paws_on_side_b += 1

                    # Draw paw points
                    if writer:
                        color = (
                            (0, 255, 0) if side > 0 else (255, 0, 0)
                        )  # Green for A, Blue for B
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

                # Determine state for this frame
                frame_state = None
                if paws_on_side_a == 4:
                    frame_state = "A"
                elif paws_on_side_b == 4:
                    frame_state = "B"

                # Check for state change
                if frame_state != current_state:
                    if current_state is not None:
                        # End previous event
                        duration_frames = i - state_start_frame
                        events.append(
                            {
                                "Start Time (s)": state_start_frame / fps,
                                "End Time (s)": i / fps,
                                "Duration (s)": duration_frames / fps,
                                "Side": current_state,
                            }
                        )

                    # Start new event
                    current_state = frame_state
                    state_start_frame = i

                # Write frame with overlay
                if writer:
                    # Draw Side Label
                    label_text = f"Side: {frame_state if frame_state else 'None'}"
                    cv2.putText(
                        frame,
                        label_text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),  # Yellow
                        3,
                    )

                    # Draw Stats
                    stats_text = f"A: {side_a_frames} | B: {side_b_frames}"
                    cv2.putText(
                        frame,
                        stats_text,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    writer.write(frame)

                # Update progress every 100 frames
                if i % 100 == 0:
                    self.progress.emit(f"Processing frame {i}/{total_frames}...")

            cap.release()
            if writer:
                writer.release()

            # Close last event
            if current_state is not None:
                duration_frames = total_frames - state_start_frame
                events.append(
                    {
                        "Start Time (s)": state_start_frame / fps,
                        "End Time (s)": total_frames / fps,
                        "Duration (s)": duration_frames / fps,
                        "Side": current_state,
                    }
                )

            side_a_time = side_a_frames / fps
            side_b_time = side_b_frames / fps

            # Prepare output
            output_path = video_path.parent / f"{video_path.stem}_freezing_test.xlsx"

            with pd.ExcelWriter(output_path) as writer:
                # Summary Sheet
                summary_data = {
                    "Video": [video_path.name],
                    "Side A Total Time (s)": [side_a_time],
                    "Side B Total Time (s)": [side_b_time],
                    "Total Video Time (s)": [total_frames / fps],
                    "FPS": [fps],
                }
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name="Summary", index=False
                )

                # Events Sheet
                if events:
                    pd.DataFrame(events).to_excel(
                        writer, sheet_name="Events", index=False
                    )
                else:
                    pd.DataFrame({"Message": ["No events detected"]}).to_excel(
                        writer, sheet_name="Events", index=False
                    )

            msg = f"Analysis saved to:\n{output_path}"
            if self.create_video:
                msg += f"\n\nLabeled video saved to:\n{output_video_path}"

            self.finished.emit(msg)

        except Exception as e:
            if "cap" in locals():
                cap.release()
            if "writer" in locals() and writer:
                writer.release()
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
