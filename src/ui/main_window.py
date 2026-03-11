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
    TestsTab,
)
from .widgets import ResponsiveTabPage


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self._session_config_path: str = ""
        self._syncing = False
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
        self.tests_tab = TestsTab()
        self.system_info_tab = SystemInfoTab()

        # Tabs that own a config_input
        self._config_tabs = [
            self.extract_tab,
            self.label_tab,
            self.training_tab,
            self.train_tab,
            self.inference_tab,
            self.outlier_tab,
            self.tests_tab,
        ]

        # Connect each tab's config_input.textChanged → shared sync
        for tab in self._config_tabs:
            tab.config_input.textChanged.connect(self._on_config_input_changed)

        # Connect project creation to auto-fill config
        self.project_tab.config_created_signal = None  # handled below

        # Add tabs in logical order
        self.tabs.addTab(ResponsiveTabPage(self.project_tab, 820), "Project Manager")
        self.tabs.addTab(ResponsiveTabPage(self.clean_video_tab, 980), "Clean Videos")
        self.tabs.addTab(ResponsiveTabPage(self.extract_tab, 760), "Extract Frames")
        self.tabs.addTab(ResponsiveTabPage(self.label_tab, 940), "Label Frames")
        self.tabs.addTab(ResponsiveTabPage(self.training_tab, 780), "Create Dataset")
        self.tabs.addTab(ResponsiveTabPage(self.train_tab, 780), "Train Network")
        self.tabs.addTab(ResponsiveTabPage(self.inference_tab, 920), "Analyze Videos")
        self.tabs.addTab(ResponsiveTabPage(self.outlier_tab, 820), "Extract Outliers")
        self.tabs.addTab(ResponsiveTabPage(self.tests_tab, 820), "Custom Tests")
        self.tabs.addTab(self.system_info_tab, "System Info")

        # Sync config when switching tabs (handles project creation)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ------------------------------------------------------------------
    # Config path synchronization
    # ------------------------------------------------------------------

    def _broadcast_config(self, path: str) -> None:
        """Push *path* into every config tab, guarded against recursion."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._session_config_path = path
            for tab in self._config_tabs:
                if tab.config_input.text() != path:
                    tab.set_config_path(path)
        finally:
            self._syncing = False

    def _on_config_input_changed(self, text: str) -> None:
        """Called whenever *any* tab's config_input text changes."""
        if self._syncing:
            return
        # Only broadcast non-empty, valid-looking paths
        if text and text.endswith((".yaml", ".yml")):
            self._broadcast_config(text)

    def _on_tab_changed(self, _index: int) -> None:
        """Sync config when the user switches tabs."""
        # 1. If the project tab just created a config, adopt it
        project_config = self.project_tab.get_config_path()
        if project_config and project_config != self._session_config_path:
            self._broadcast_config(project_config)
            return

        # 2. Otherwise re-apply the session config to the newly-visible tab
        if self._session_config_path:
            self._broadcast_config(self._session_config_path)
