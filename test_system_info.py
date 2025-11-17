"""Quick test for System Info tab"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from src.ui.main_window import MainWindow
from src.ui.styles import load_stylesheet


def test_system_info():
    """Test system info tab"""
    app = QApplication(sys.argv)
    
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)
    
    window = MainWindow()
    window.show()
    
    # Switch to System Info tab (last tab, index 8)
    def show_system_info():
        from PySide6.QtWidgets import QTabWidget
        tabs = window.findChild(QTabWidget)
        if tabs:
            tabs.setCurrentIndex(8)
            print("\n" + "="*60)
            print("SYSTEM INFO TAB TEST")
            print("="*60)
            print("\nSystem Info tab loaded successfully!")
            print("Check the GUI window to see system information.")
            print("\nClose the window to exit.")
    
    QTimer.singleShot(500, show_system_info)
    
    # Auto-close after 5 seconds for automated testing
    QTimer.singleShot(5000, app.quit)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    test_system_info()
