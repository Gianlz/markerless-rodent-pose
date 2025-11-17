"""Video Inference Tab"""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox, QListWidget,
    QGroupBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Qt

from ...core.inference_manager import InferenceManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class InferenceWorker(QThread):
    """Worker thread for video analysis and labeled video creation"""
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, manager: InferenceManager, config: str, videos: list[str], 
                 analyze: bool, create_video: bool, **kwargs):
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
                analyze_kwargs = {k: v for k, v in self.kwargs.items() 
                                 if k in ['shuffle', 'trainingsetindex', 'gputouse', 
                                         'save_as_csv', 'destfolder']}
                self.manager.analyze_videos(self.config, self.videos, **analyze_kwargs)
            
            if self.create_video:
                self.progress.emit("Creating labeled videos...")
                # Only pass video-specific kwargs
                video_kwargs = {k: v for k, v in self.kwargs.items() 
                               if k in ['shuffle', 'trainingsetindex', 'filtered', 
                                       'draw_skeleton', 'trailpoints', 'displayedbodyparts', 'destfolder']}
                self.manager.create_labeled_video(self.config, self.videos, **video_kwargs)
            
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Config file
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setSpacing(4)
        config_layout.setContentsMargins(8, 8, 8, 8)
        
        config_label = QLabel("Config:")
        config_label.setFixedWidth(60)
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        self.config_input.textChanged.connect(self.on_config_changed)
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.setFixedWidth(80)
        config_btn.clicked.connect(self.browse_config)
        
        config_layout.addWidget(config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Video selection
        video_group = QGroupBox("Videos")
        video_layout = QVBoxLayout()
        video_layout.setSpacing(6)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(100)
        video_layout.addWidget(self.video_list)
        
        video_btn_layout = QHBoxLayout()
        video_btn_layout.setSpacing(4)
        
        self.add_video_btn = QPushButton("Add Videos")
        self.add_video_btn.setObjectName(SECONDARY_BUTTON)
        self.add_video_btn.clicked.connect(self.add_videos)
        
        self.remove_video_btn = QPushButton("Remove")
        self.remove_video_btn.setObjectName(SECONDARY_BUTTON)
        self.remove_video_btn.clicked.connect(self.remove_video)
        
        self.clear_videos_btn = QPushButton("Clear All")
        self.clear_videos_btn.setObjectName(SECONDARY_BUTTON)
        self.clear_videos_btn.clicked.connect(self.clear_videos)
        
        video_btn_layout.addWidget(self.add_video_btn)
        video_btn_layout.addWidget(self.remove_video_btn)
        video_btn_layout.addWidget(self.clear_videos_btn)
        video_layout.addLayout(video_btn_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setColumnStretch(1, 1)
        
        # Shuffle
        shuffle_label = QLabel("Shuffle:")
        shuffle_label.setFixedWidth(100)
        self.shuffle_combo = QComboBox()
        settings_layout.addWidget(shuffle_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.shuffle_combo, 0, 1)
        
        # Trail points
        trail_label = QLabel("Trail Points:")
        trail_label.setFixedWidth(100)
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(0, 50)
        self.trail_spin.setValue(0)
        settings_layout.addWidget(trail_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.addWidget(self.trail_spin, 1, 1)
        
        # Checkboxes
        self.filtered_check = QCheckBox("Use Filtered Predictions")
        self.filtered_check.setChecked(True)
        settings_layout.addWidget(self.filtered_check, 2, 0, 1, 2)
        
        self.skeleton_check = QCheckBox("Draw Skeleton")
        self.skeleton_check.setChecked(True)
        settings_layout.addWidget(self.skeleton_check, 3, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.analyze_btn = QPushButton("Analyze Videos")
        self.analyze_btn.setFixedHeight(32)
        self.analyze_btn.clicked.connect(lambda: self.run_inference(analyze=True, create_video=False))
        
        self.create_video_btn = QPushButton("Create Labeled Videos")
        self.create_video_btn.setFixedHeight(32)
        self.create_video_btn.clicked.connect(lambda: self.run_inference(analyze=False, create_video=True))
        
        self.analyze_and_create_btn = QPushButton("Analyze + Create Videos")
        self.analyze_and_create_btn.setFixedHeight(32)
        self.analyze_and_create_btn.clicked.connect(lambda: self.run_inference(analyze=True, create_video=True))
        
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.create_video_btn)
        btn_layout.addWidget(self.analyze_and_create_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
    
    def browse_config(self):
        """Browse for config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config File",
            "",
            "YAML Files (*.yaml *.yml)"
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
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        for path in file_paths:
            if path not in [self.video_list.item(i).text() 
                           for i in range(self.video_list.count())]:
                self.video_list.addItem(path)
    
    def remove_video(self):
        """Remove selected video"""
        current = self.video_list.currentRow()
        if current >= 0:
            self.video_list.takeItem(current)
    
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
        
        videos = [self.video_list.item(i).text() for i in range(self.video_list.count())]
        
        # Check if best snapshot exists
        best_snapshot = self.manager.get_best_snapshot(config)
        if not best_snapshot:
            QMessageBox.warning(
                self, "No Model",
                "No trained model found. Please train a model first."
            )
            return
        
        if create_video and not analyze:
            unanalyzed = []
            for video in videos:
                if not self.manager.check_analysis_exists(video, config):
                    unanalyzed.append(Path(video).name)
            
            if unanalyzed:
                reply = QMessageBox.question(
                    self, "Videos Not Analyzed",
                    f"{len(unanalyzed)} video(s) have not been analyzed yet:\n\n" +
                    "\n".join(unanalyzed[:5]) +
                    ("\n..." if len(unanalyzed) > 5 else "") +
                    "\n\nDo you want to analyze them first?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    analyze = True
                else:
                    return
        
        self.analyze_btn.setEnabled(False)
        self.create_video_btn.setEnabled(False)
        self.analyze_and_create_btn.setEnabled(False)
        
        shuffle = int(self.shuffle_combo.currentText()) if self.shuffle_combo.currentText() else 1
        
        analyze_kwargs = {
            'shuffle': shuffle,
            'trainingsetindex': 0,
            'save_as_csv': True
        }
        
        video_kwargs = {
            'shuffle': shuffle,
            'trainingsetindex': 0,
            'filtered': self.filtered_check.isChecked(),
            'draw_skeleton': self.skeleton_check.isChecked(),
            'trailpoints': self.trail_spin.value()
        }
        
        kwargs = {**analyze_kwargs, **video_kwargs}
        
        self.worker = InferenceWorker(
            self.manager, config, videos, analyze, create_video, **kwargs
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()
    
    def on_progress(self, message: str):
        """Handle progress update"""
        pass
    
    def on_finished(self):
        """Handle inference completion"""
        self.analyze_btn.setEnabled(True)
        self.create_video_btn.setEnabled(True)
        self.analyze_and_create_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Video processing completed successfully")
    
    def on_error(self, error: str):
        """Handle inference error"""
        self.analyze_btn.setEnabled(True)
        self.create_video_btn.setEnabled(True)
        self.analyze_and_create_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error}")
