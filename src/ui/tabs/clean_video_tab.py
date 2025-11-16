"""Video Cleaning Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QListWidget, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import QThread, Signal, Qt
from pathlib import Path

from ...utils.video_utils import reencode_video, check_video_integrity
from ..styles import SECONDARY_BUTTON


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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Info label
        info_label = QLabel(
            "Re-encode videos to fix corruption issues and ensure compatibility with DeepLabCut.\n"
            "This process creates clean copies of your videos with optimal encoding settings."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; padding: 8px;")
        layout.addWidget(info_label)
        
        # Video selection
        video_group = QGroupBox("Videos to Clean")
        video_layout = QVBoxLayout()
        video_layout.setSpacing(6)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(150)
        video_layout.addWidget(self.video_list)
        
        video_btn_layout = QHBoxLayout()
        video_btn_layout.setSpacing(4)
        
        self.add_videos_btn = QPushButton("Add Videos")
        self.add_videos_btn.setObjectName(SECONDARY_BUTTON)
        self.add_videos_btn.clicked.connect(self.add_videos)
        
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.setObjectName(SECONDARY_BUTTON)
        self.add_folder_btn.clicked.connect(self.add_folder)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setObjectName(SECONDARY_BUTTON)
        self.remove_btn.clicked.connect(self.remove_video)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setObjectName(SECONDARY_BUTTON)
        self.clear_btn.clicked.connect(self.clear_videos)
        
        self.check_btn = QPushButton("Check Integrity")
        self.check_btn.setObjectName(SECONDARY_BUTTON)
        self.check_btn.clicked.connect(self.check_integrity)
        
        video_btn_layout.addWidget(self.add_videos_btn)
        video_btn_layout.addWidget(self.add_folder_btn)
        video_btn_layout.addWidget(self.remove_btn)
        video_btn_layout.addWidget(self.clear_btn)
        video_btn_layout.addWidget(self.check_btn)
        video_layout.addLayout(video_btn_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(6)
        output_layout.setContentsMargins(8, 8, 8, 8)
        
        # Output folder
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(4)
        
        folder_label = QLabel("Output Folder:")
        folder_label.setFixedWidth(100)
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select output folder for cleaned videos")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setObjectName(SECONDARY_BUTTON)
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self.browse_output)
        
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.output_input)
        folder_layout.addWidget(output_browse_btn)
        output_layout.addLayout(folder_layout)
        
        # Encoding settings
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        
        # Codec
        codec_label = QLabel("Codec:")
        codec_label.setFixedWidth(100)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(['libx264', 'libx265', 'h264_nvenc'])
        self.codec_combo.setCurrentText('libx264')
        settings_layout.addWidget(codec_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.codec_combo, 0, 1)
        
        # Quality (CRF)
        crf_label = QLabel("Quality (CRF):")
        crf_label.setFixedWidth(100)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        self.crf_spin.setToolTip("Lower = better quality (18 recommended, 23 default)")
        settings_layout.addWidget(crf_label, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.crf_spin, 0, 3)
        
        # Preset
        preset_label = QLabel("Speed Preset:")
        preset_label.setFixedWidth(100)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'])
        self.preset_combo.setCurrentText('medium')
        self.preset_combo.setToolTip("Slower = better compression")
        settings_layout.addWidget(preset_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.preset_combo, 1, 1, 1, 3)
        
        output_layout.addLayout(settings_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(6)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Clean button
        self.clean_btn = QPushButton("Clean Videos")
        self.clean_btn.setFixedHeight(32)
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
        for path in file_paths:
            if path not in [self.video_list.item(i).text() 
                           for i in range(self.video_list.count())]:
                self.video_list.addItem(path)
    
    def add_folder(self):
        """Add all videos from a folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Videos"
        )
        if folder:
            folder_path = Path(folder)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            
            for ext in video_extensions:
                for video in folder_path.glob(f'*{ext}'):
                    video_str = str(video)
                    if video_str not in [self.video_list.item(i).text() 
                                        for i in range(self.video_list.count())]:
                        self.video_list.addItem(video_str)
    
    def remove_video(self):
        """Remove selected video"""
        current = self.video_list.currentRow()
        if current >= 0:
            self.video_list.takeItem(current)
    
    def clear_videos(self):
        """Clear all videos"""
        self.video_list.clear()
    
    def browse_output(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder"
        )
        if folder:
            self.output_input.setText(folder)
    
    def check_integrity(self):
        """Check integrity of selected videos"""
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "Please add videos first")
            return
        
        results = []
        for i in range(self.video_list.count()):
            video_path = self.video_list.item(i).text()
            video_name = Path(video_path).name
            
            info = check_video_integrity(video_path)
            if info:
                results.append(f"✓ {video_name}: {info['width']}x{info['height']}, {info['fps']:.2f} fps")
            else:
                results.append(f"✗ {video_name}: Could not read metadata (possibly corrupted)")
        
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
        
        self.clean_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")
        
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
        self.progress_label.setText(f"Processing {current}/{total}: {filename}")
    
    def on_finished(self):
        """Handle cleaning completion"""
        self.clean_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Completed!")
        QMessageBox.information(
            self,
            "Success",
            f"Successfully cleaned {self.video_list.count()} video(s)!\n\n"
            f"Output folder: {self.output_input.text()}"
        )
    
    def on_error(self, error: str):
        """Handle cleaning error"""
        self.clean_btn.setEnabled(True)
        self.progress_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"Video cleaning failed:\n{error}")
