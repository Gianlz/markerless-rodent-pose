"""Main application window"""

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar

from .tabs import (
    CleanVideoTab,
    ProjectTab,
    ExtractTab,
    OutlierTab,
    LabelTab,
    TrainingTab,
    TrainTab,
    InferenceTab,
    SystemInfoTab,
)
from .widgets import ResponsiveTabPage


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("DeepLabCut Frame Extractor")
        self.resize(1280, 860)
        self.setMinimumSize(720, 540)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with padding
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # cleaner look on some platforms
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)

        # Initialize tabs
        self.project_tab = ProjectTab()
        self.clean_video_tab = CleanVideoTab()
        self.extract_tab = ExtractTab()
        self.label_tab = LabelTab()
        self.training_tab = TrainingTab()
        self.train_tab = TrainTab()
        self.inference_tab = InferenceTab()
        self.outlier_tab = OutlierTab()
        self.system_info_tab = SystemInfoTab()

        # Add tabs in logical order
        self.tabs.addTab(ResponsiveTabPage(self.project_tab, 820), "Project Manager")
        self.tabs.addTab(ResponsiveTabPage(self.clean_video_tab, 980), "Clean Videos")
        self.tabs.addTab(ResponsiveTabPage(self.extract_tab, 760), "Extract Frames")
        self.tabs.addTab(ResponsiveTabPage(self.label_tab, 940), "Label Frames")
        self.tabs.addTab(ResponsiveTabPage(self.training_tab, 780), "Create Dataset")
        self.tabs.addTab(ResponsiveTabPage(self.train_tab, 780), "Train Network")
        self.tabs.addTab(ResponsiveTabPage(self.inference_tab, 920), "Analyze Videos")
        self.tabs.addTab(ResponsiveTabPage(self.outlier_tab, 820), "Extract Outliers")
        self.tabs.addTab(self.system_info_tab, "System Info")

        # Connect project creation to auto-fill config
        self.tabs.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tabs)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

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
