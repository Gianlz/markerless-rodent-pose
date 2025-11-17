"""System Information Tab"""
import platform
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTextEdit, QPushButton, QScrollArea
)
from PySide6.QtCore import Qt
from ..styles import SECONDARY_BUTTON


class SystemInfoTab(QWidget):
    """System information and diagnostics tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_system_info()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # System info
        system_group = QGroupBox("System Information")
        system_layout = QVBoxLayout()
        system_layout.setSpacing(4)
        system_layout.setContentsMargins(8, 8, 8, 8)
        
        self.system_text = QTextEdit()
        self.system_text.setReadOnly(True)
        self.system_text.setMaximumHeight(150)
        system_layout.addWidget(self.system_text)
        
        system_group.setLayout(system_layout)
        content_layout.addWidget(system_group)
        
        # Python info
        python_group = QGroupBox("Python Environment")
        python_layout = QVBoxLayout()
        python_layout.setSpacing(4)
        python_layout.setContentsMargins(8, 8, 8, 8)
        
        self.python_text = QTextEdit()
        self.python_text.setReadOnly(True)
        self.python_text.setMaximumHeight(100)
        python_layout.addWidget(self.python_text)
        
        python_group.setLayout(python_layout)
        content_layout.addWidget(python_group)
        
        # GPU/Acceleration info
        gpu_group = QGroupBox("GPU & Acceleration")
        gpu_layout = QVBoxLayout()
        gpu_layout.setSpacing(4)
        gpu_layout.setContentsMargins(8, 8, 8, 8)
        
        self.gpu_text = QTextEdit()
        self.gpu_text.setReadOnly(True)
        self.gpu_text.setMaximumHeight(200)
        gpu_layout.addWidget(self.gpu_text)
        
        gpu_group.setLayout(gpu_layout)
        content_layout.addWidget(gpu_group)
        
        # Dependencies info
        deps_group = QGroupBox("Key Dependencies")
        deps_layout = QVBoxLayout()
        deps_layout.setSpacing(4)
        deps_layout.setContentsMargins(8, 8, 8, 8)
        
        self.deps_text = QTextEdit()
        self.deps_text.setReadOnly(True)
        self.deps_text.setMaximumHeight(200)
        deps_layout.addWidget(self.deps_text)
        
        deps_group.setLayout(deps_layout)
        content_layout.addWidget(deps_group)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh System Info")
        refresh_btn.setObjectName(SECONDARY_BUTTON)
        refresh_btn.setFixedHeight(32)
        refresh_btn.clicked.connect(self.load_system_info)
        layout.addWidget(refresh_btn)
    
    def load_system_info(self):
        """Load and display system information"""
        self.system_text.setPlainText(self.get_system_info())
        self.python_text.setPlainText(self.get_python_info())
        self.gpu_text.setPlainText(self.get_gpu_info())
        self.deps_text.setPlainText(self.get_dependencies_info())
    
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
            info.append(f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
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
                info.append(f"CUDA Available: ✓ Yes")
                info.append(f"CUDA Version: {torch.version.cuda}")
                info.append(f"cuDNN Version: {torch.backends.cudnn.version()}")
                info.append(f"GPU Count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    info.append(f"GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
                
                info.append(f"Current Device: {torch.cuda.current_device()}")
            else:
                info.append("CUDA Available: ✗ No")
            
            if torch.backends.mps.is_available():
                info.append("MPS (Apple Silicon): ✓ Available")
            else:
                info.append("MPS (Apple Silicon): ✗ Not available")
            
            info.append(f"Default Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            
        except ImportError:
            info.append("PyTorch: Not installed")
        
        # FAISS
        try:
            import faiss
            info.append(f"\nFAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'Installed'}")
            
            try:
                # Test GPU availability
                res = faiss.StandardGpuResources()
                info.append("FAISS GPU: ✓ Available")
            except:
                info.append("FAISS GPU: ✗ Not available (CPU only)")
        except ImportError:
            info.append("\nFAISS: Not installed")
        
        return "\n".join(info)
    
    def get_dependencies_info(self) -> str:
        """Get key dependencies information"""
        info = []
        
        dependencies = [
            ('deeplabcut', 'DeepLabCut'),
            ('PySide6', 'PySide6'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('yaml', 'PyYAML'),
            ('tables', 'PyTables'),
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('faiss', 'FAISS'),
        ]
        
        for module_name, display_name in dependencies:
            try:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'Unknown')
                info.append(f"{display_name}: {version}")
            except ImportError:
                info.append(f"{display_name}: ✗ Not installed")
        
        return "\n".join(info)
