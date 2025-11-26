"""Main application window"""

from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget

from .tabs import (
    CleanVideoTab,
    EvaluationTab,
    ExtractTab,
    FreezingTab,
    InferenceTab,
    LabelTab,
    OutlierTab,
    ProjectTab,
    SystemInfoTab,
    TrainingTab,
    TrainTab,
)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("DeepLabCut Frame Extractor")
        self.setMinimumSize(900, 700)  # Slightly larger for modern spacing

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with padding
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # cleaner look on some platforms

        # Initialize tabs
        self.project_tab = ProjectTab()
        self.clean_video_tab = CleanVideoTab()
        self.extract_tab = ExtractTab()
        self.label_tab = LabelTab()
        self.training_tab = TrainingTab()
        self.train_tab = TrainTab()
        self.inference_tab = InferenceTab()
        self.evaluation_tab = EvaluationTab()
        self.outlier_tab = OutlierTab()
        self.freezing_tab = FreezingTab()
        self.system_info_tab = SystemInfoTab()

        # Add tabs in logical order
        self.tabs.addTab(self.project_tab, "Project Manager")
        self.tabs.addTab(self.clean_video_tab, "Clean Videos")
        self.tabs.addTab(self.extract_tab, "Extract Frames")
        self.tabs.addTab(self.label_tab, "Label Frames")
        self.tabs.addTab(self.training_tab, "Create Dataset")
        self.tabs.addTab(self.train_tab, "Train Network")
        self.tabs.addTab(self.evaluation_tab, "Evaluate Model")
        self.tabs.addTab(self.inference_tab, "Analyze Videos")
        self.tabs.addTab(self.outlier_tab, "Extract Outliers")
        self.tabs.addTab(self.freezing_tab, "Test")
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
            self.evaluation_tab.set_config_path(config_path)
            self.outlier_tab.set_config_path(config_path)
            self.freezing_tab.set_config_path(config_path)
