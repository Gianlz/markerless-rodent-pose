"""Main application window"""
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt

from .tabs import (
    CleanVideoTab, ProjectTab, ExtractTab, OutlierTab, 
    LabelTab, TrainingTab, TrainTab, InferenceTab, SystemInfoTab
)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("DeepLabCut Frame Extractor")
        self.setMinimumSize(750, 550)
        self.resize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Create tabs
        tabs = QTabWidget()
        
        self.clean_video_tab = CleanVideoTab()
        self.project_tab = ProjectTab()
        self.extract_tab = ExtractTab()
        self.label_tab = LabelTab()
        self.training_tab = TrainingTab()
        self.train_tab = TrainTab()
        self.inference_tab = InferenceTab()
        self.outlier_tab = OutlierTab()
        self.system_info_tab = SystemInfoTab()
        
        tabs.addTab(self.clean_video_tab, "Clean Videos")
        tabs.addTab(self.project_tab, "Project Manager")
        tabs.addTab(self.extract_tab, "Extract Frames")
        tabs.addTab(self.label_tab, "Label Frames")
        tabs.addTab(self.training_tab, "Create Training Dataset")
        tabs.addTab(self.train_tab, "Train Network")
        tabs.addTab(self.inference_tab, "Analyze Videos")
        tabs.addTab(self.outlier_tab, "Extract Outliers")
        tabs.addTab(self.system_info_tab, "System Info")
        
        # Connect project creation to auto-fill config
        tabs.currentChanged.connect(self.on_tab_changed)
        
        layout.addWidget(tabs)
    
    def on_tab_changed(self, index: int):
        """Handle tab change to sync config paths"""
        config_path = self.project_tab.get_config_path()
        if config_path:
            self.extract_tab.set_config_path(config_path)
            self.label_tab.set_config_path(config_path)
            self.training_tab.set_config_path(config_path)
            self.train_tab.set_config_path(config_path)
            self.inference_tab.set_config_path(config_path)
            self.outlier_tab.set_config_path(config_path)
