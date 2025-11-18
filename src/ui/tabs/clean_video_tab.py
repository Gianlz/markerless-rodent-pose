"""Video Cleaning Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QListWidget, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QAbstractItemView, QListWidgetItem
)
from PySide6.QtCore import QThread, Signal, Qt
from pathlib import Path

from ...utils.video_utils import reencode_video, check_video_integrity
from ..styles import SECONDARY_BUTTON, INFO_LABEL


class VideoCleanWorker(QThread):
    """Worker thread for video re-encoding"""
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, int, str)  # current, total, filename
    
    def __init__(self, videos: list[str], output_folder: str, codec: str, crf: int, preset: str):
        super().__init__()
        self.videos = videos
        self.output_folder = output_folder
        self.codec = codec
        self.crf = crf
        self.preset = preset
    
    def run(self):
        try:
            output_path = Path(self.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, video in enumerate(self.videos):
                video_path = Path(video)
                output_file = output_path / f"{video_path.stem}_clean{video_path.suffix}"
                
                self.progress.emit(i + 1, len(self.videos), video_path.name)
                
                reencode_video(
                    video,
                    str(output_file),
                    codec=self.codec,
                    crf=self.crf,
                    preset=self.preset
                )
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class CleanVideoTab(QWidget):
    """Video cleaning and re-encoding tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)
        
        # Info Label
        info_frame = QGroupBox()
        info_layout = QVBoxLayout()
        info_label = QLabel(
            "Re-encode videos to fix corruption issues and ensure compatibility with DeepLabCut.\n"
            "This creates clean copies with optimal encoding settings."
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_frame.setLayout(info_layout)
        info_frame.setStyleSheet("QGroupBox { border: none; background: transparent; margin-top: 0; padding: 0; }")
        layout.addWidget(info_frame)
        
        # --- Videos to Clean ---
        video_group = QGroupBox("Videos to Clean")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(16, 24, 16, 16)
        video_layout.setSpacing(12)
        
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.video_list.setAlternatingRowColors(True)
        self.video_list.setMinimumHeight(150)
        video_layout.addWidget(self.video_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Videos")
        add_btn.clicked.connect(self.add_videos)
        
        folder_btn = QPushButton("Add Folder")
        folder_btn.setObjectName(SECONDARY_BUTTON)
        folder_btn.clicked.connect(self.add_folder)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setObjectName(SECONDARY_BUTTON)
        remove_btn.clicked.connect(self.remove_video)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName(SECONDARY_BUTTON)
        clear_btn.clicked.connect(self.clear_videos)
        
        check_btn = QPushButton("Check Integrity")
        check_btn.setObjectName(SECONDARY_BUTTON)
        check_btn.clicked.connect(self.check_integrity)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(folder_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(check_btn)
        btn_layout.addStretch()
        
        video_layout.addLayout(btn_layout)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # --- Output Settings ---
        settings_group = QGroupBox("Output Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(16, 24, 16, 16)
        settings_layout.setSpacing(12)
        
        # Output Folder
        out_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select destination folder...")
        browse_out_btn = QPushButton("Browse")
        browse_out_btn.setObjectName(SECONDARY_BUTTON)
        browse_out_btn.clicked.connect(self.browse_output)
        out_layout.addWidget(self.output_input)
        out_layout.addWidget(browse_out_btn)
        settings_layout.addRow("Output Folder:", out_layout)
        
        # Codec
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(['libx264', 'libx265', 'h264_nvenc'])
        self.codec_combo.setCurrentText('libx264')
        settings_layout.addRow("Codec:", self.codec_combo)
        
        # CRF
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        self.crf_spin.setToolTip("Lower = better quality (18 recommended, 23 default)")
        settings_layout.addRow("Quality (CRF):", self.crf_spin)
        
        # Preset
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'])
        self.preset_combo.setCurrentText('medium')
        settings_layout.addRow("Speed Preset:", self.preset_combo)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Clean Button
        self.clean_btn = QPushButton("Start Cleaning Process")
        self.clean_btn.setMinimumHeight(40)
        self.clean_btn.clicked.connect(self.clean_videos)
        layout.addWidget(self.clean_btn)
        
        layout.addStretch()
    
    def add_videos(self):
        """Add videos to list"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
        )
        if file_paths:
            existing = [self.video_list.item(i).text() for i in range(self.video_list.count())]
            for path in file_paths:
                if path not in existing:
                    self.video_list.addItem(path)
    
    def add_folder(self):
        """Add all videos from a folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Videos")
        if folder:
            folder_path = Path(folder)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            existing = [self.video_list.item(i).text() for i in range(self.video_list.count())]
            
            for ext in video_extensions:
                for video in folder_path.glob(f'*{ext}'):
                    video_str = str(video)
                    if video_str not in existing:
                        self.video_list.addItem(video_str)
    
    def remove_video(self):
        """Remove selected videos"""
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
    
    def clear_videos(self):
        """Clear all videos"""
        self.video_list.clear()
    
    def browse_output(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_input.setText(folder)
    
    def check_integrity(self):
        """Check integrity of selected videos"""
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "Please add videos first")
            return
        
        self.status_label.setText("Checking integrity...")
        self.set_busy(True)
        
        # This could be threaded but fast enough for small lists usually
        results = []
        for i in range(self.video_list.count()):
            video_path = self.video_list.item(i).text()
            video_name = Path(video_path).name
            
            info = check_video_integrity(video_path)
            if info:
                results.append(f"✓ {video_name}: {info['width']}x{info['height']}, {info['fps']:.2f} fps")
            else:
                results.append(f"✗ {video_name}: Could not read metadata (possibly corrupted)")
        
        self.set_busy(False)
        self.status_label.setText("Integrity check complete.")
        
        QMessageBox.information(
            self,
            "Video Integrity Check",
            "\n".join(results)
        )
    
    def clean_videos(self):
        """Start video cleaning process"""
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "Please add videos to clean")
            return
        
        output_folder = self.output_input.text()
        if not output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder")
            return
        
        videos = [self.video_list.item(i).text() for i in range(self.video_list.count())]
        
        reply = QMessageBox.question(
            self,
            "Confirm",
            f"Clean {len(videos)} video(s)?\n\n"
            f"This will create re-encoded copies in:\n{output_folder}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.set_busy(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting cleaning process...")
        
        self.worker = VideoCleanWorker(
            videos,
            output_folder,
            self.codec_combo.currentText(),
            self.crf_spin.value(),
            self.preset_combo.currentText()
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
    
    def on_progress(self, current: int, total: int, filename: str):
        """Handle progress update"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processing {current}/{total}: {filename}")
    
    def on_finished(self):
        """Handle cleaning completion"""
        self.set_busy(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Completed!")
        QMessageBox.information(
            self,
            "Success",
            f"Successfully cleaned {self.video_list.count()} video(s)!\n\n"
            f"Output folder: {self.output_input.text()}"
        )
    
    def on_error(self, error: str):
        """Handle cleaning error"""
        self.set_busy(False)
        self.status_label.setText("Error occurred during cleaning.")
        QMessageBox.critical(self, "Error", f"Video cleaning failed:\n{error}")
        
    def set_busy(self, busy: bool):
        """Toggle UI state"""
        self.clean_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
