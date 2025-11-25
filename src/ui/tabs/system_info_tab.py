"""System Information Tab"""

import platform
import sys

from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..styles import SECONDARY_BUTTON


class SystemInfoTab(QWidget):
    """System information and diagnostics tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_system_info()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(24)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Info Groups
        self.system_text = self.create_info_group(content_layout, "System Information")
        self.python_text = self.create_info_group(content_layout, "Python Environment")
        self.gpu_text = self.create_info_group(content_layout, "GPU & Acceleration")
        self.deps_text = self.create_info_group(content_layout, "Key Dependencies")

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Actions
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh System Info")
        refresh_btn.setObjectName(SECONDARY_BUTTON)
        refresh_btn.setMinimumHeight(40)
        refresh_btn.clicked.connect(self.load_system_info)

        copy_btn = QPushButton("Copy All to Clipboard")
        copy_btn.setMinimumHeight(40)
        copy_btn.clicked.connect(self.copy_all)

        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(copy_btn)
        layout.addLayout(btn_layout)

    def create_info_group(self, layout, title):
        group = QGroupBox(title)
        glayout = QVBoxLayout()
        glayout.setContentsMargins(16, 24, 16, 16)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(150)
        text_edit.setStyleSheet(
            "font-family: Consolas, Monaco, monospace; background-color: #f9fafb;"
        )

        glayout.addWidget(text_edit)
        group.setLayout(glayout)
        layout.addWidget(group)
        return text_edit

    def load_system_info(self):
        """Load and display system information"""
        self.system_text.setPlainText(self.get_system_info())
        self.python_text.setPlainText(self.get_python_info())
        self.gpu_text.setPlainText(self.get_gpu_info())
        self.deps_text.setPlainText(self.get_dependencies_info())

    def copy_all(self):
        """Copy all info to clipboard"""
        full_text = f"=== System Info ===\n{self.system_text.toPlainText()}\n\n"
        full_text += f"=== Python Info ===\n{self.python_text.toPlainText()}\n\n"
        full_text += f"=== GPU Info ===\n{self.gpu_text.toPlainText()}\n\n"
        full_text += f"=== Dependencies ===\n{self.deps_text.toPlainText()}"

        clipboard = QApplication.clipboard() if QApplication.instance() else None
        if clipboard:
            clipboard.setText(full_text)
        else:
            # Fallback if we can't access clipboard directly (rare in PySide app)
            pass

    def get_system_info(self) -> str:
        """Get system information"""
        info = []

        info.append(f"OS: {platform.system()} {platform.release()}")
        info.append(f"Platform: {platform.platform()}")
        info.append(f"Architecture: {platform.machine()}")
        info.append(f"Processor: {platform.processor()}")

        try:
            import psutil

            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            info.append(f"CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")

            mem = psutil.virtual_memory()
            info.append(
                f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available"
            )
        except ImportError:
            info.append("CPU/RAM: Install psutil for detailed info")

        return "\n".join(info)

    def get_python_info(self) -> str:
        """Get Python environment information"""
        info = []

        info.append(f"Python: {sys.version.split()[0]}")
        info.append(f"Executable: {sys.executable}")
        info.append(f"Virtual Env: {sys.prefix}")

        return "\n".join(info)

    def get_gpu_info(self) -> str:
        """Get GPU and acceleration information"""
        info = []

        # PyTorch
        try:
            import torch

            info.append(f"PyTorch: {torch.__version__}")

            if torch.cuda.is_available():
                info.append("CUDA Available: YES")
                info.append(f"CUDA Version: {torch.version.cuda}")
                info.append(f"cuDNN Version: {torch.backends.cudnn.version()}")
                info.append(f"GPU Count: {torch.cuda.device_count()}")

                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    try:
                        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (
                            1024**3
                        )
                        info.append(f"GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
                    except Exception:
                        info.append(f"GPU {i}: {gpu_name}")

                info.append(f"Current Device: {torch.cuda.current_device()}")
            else:
                info.append("CUDA Available: NO")

            if torch.backends.mps.is_available():
                info.append("MPS (Apple Silicon): Available")
            else:
                info.append("MPS (Apple Silicon): Not available")

        except ImportError:
            info.append("PyTorch: Not installed")

        # FAISS
        try:
            import faiss

            info.append(
                f"\nFAISS: {getattr(faiss, '__version__', 'Installed (unknown version)')}"
            )

            try:
                faiss.StandardGpuResources()
                info.append("FAISS GPU (CUDA): Available")
            except Exception:
                if torch.backends.mps.is_available():
                    info.append("FAISS: CPU mode (MPS not supported by FAISS)")
                else:
                    info.append("FAISS GPU: Not available (CPU only)")
        except ImportError:
            info.append("\nFAISS: Not installed")

        return "\n".join(info)

    def get_dependencies_info(self) -> str:
        """Get key dependencies information"""
        info = []

        dependencies = [
            ("deeplabcut", "DeepLabCut"),
            ("PySide6", "PySide6"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("yaml", "PyYAML"),
            ("tables", "PyTables"),
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
        ]

        for module_name, display_name in dependencies:
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "Unknown")
                info.append(f"{display_name}: {version}")
            except ImportError:
                info.append(f"{display_name}: Not installed")

        return "\n".join(info)
