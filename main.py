"""DeepLabCut Frame Extraction GUI Application"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from src.ui.main_window import MainWindow
from src.ui.styles import load_stylesheet


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DeepLabCut Frame Extractor")
    
    # Set default font
    font = QFont("Roboto", 12)
    if not font.exactMatch():
        font = QFont("Segoe UI", 12)
    app.setFont(font)
    
    # Load and apply stylesheet
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
