"""Frame Labeling Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton,
    QLabel, QLineEdit, QListWidget, QGroupBox, QFileDialog,
    QMessageBox, QInputDialog, QComboBox, QAbstractItemView
)
from PySide6.QtCore import Qt

from ...core.label_manager import LabelManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON, INFO_LABEL, DANGER_BUTTON


class LabelTab(QWidget):
    """Frame labeling tab with keypoint CRUD and skeleton building"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_manager = LabelManager()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)
        
        # --- Configuration ---
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(16, 24, 16, 16)
        
        self.config_input = QLineEdit()
        self.config_input.setPlaceholderText("Path to config.yaml")
        self.config_input.textChanged.connect(self.on_config_changed)
        
        config_btn = QPushButton("Browse")
        config_btn.setObjectName(SECONDARY_BUTTON)
        config_btn.clicked.connect(self.browse_config)
        
        config_layout.addWidget(QLabel("Config:"))
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # --- Labeling Actions ---
        action_group = QGroupBox("Labeling Actions")
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(16, 24, 16, 16)
        action_layout.setSpacing(16)
        
        self.label_btn = QPushButton("Launch Labeling GUI")
        self.label_btn.setMinimumHeight(40)
        self.label_btn.clicked.connect(self.launch_labeling)
        
        self.check_labels_btn = QPushButton("Check Labels Status")
        self.check_labels_btn.setObjectName(SECONDARY_BUTTON)
        self.check_labels_btn.setMinimumHeight(40)
        self.check_labels_btn.clicked.connect(self.check_labels)
        
        action_layout.addWidget(self.label_btn)
        action_layout.addWidget(self.check_labels_btn)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # --- Keypoints & Skeleton ---
        # We'll use a horizontal layout to split keypoints and skeleton side-by-side
        split_layout = QHBoxLayout()
        split_layout.setSpacing(16)
        
        # 1. Keypoints Group
        kp_group = QGroupBox("Keypoints Management")
        kp_layout = QVBoxLayout()
        kp_layout.setContentsMargins(16, 24, 16, 16)
        kp_layout.setSpacing(12)
        
        self.keypoints_list = QListWidget()
        self.keypoints_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.keypoints_list.setAlternatingRowColors(True)
        kp_layout.addWidget(self.keypoints_list)
        
        kp_btns = QHBoxLayout()
        kp_add = QPushButton("Add")
        kp_add.setObjectName(SECONDARY_BUTTON)
        kp_add.clicked.connect(self.add_keypoint)
        
        kp_edit = QPushButton("Edit")
        kp_edit.setObjectName(SECONDARY_BUTTON)
        kp_edit.clicked.connect(self.edit_keypoint)
        
        kp_del = QPushButton("Remove")
        kp_del.setObjectName(DANGER_BUTTON)
        kp_del.clicked.connect(self.remove_keypoint)
        
        kp_btns.addWidget(kp_add)
        kp_btns.addWidget(kp_edit)
        kp_btns.addWidget(kp_del)
        kp_layout.addLayout(kp_btns)
        
        kp_group.setLayout(kp_layout)
        split_layout.addWidget(kp_group)
        
        # 2. Skeleton Group
        sk_group = QGroupBox("Skeleton Builder")
        sk_layout = QVBoxLayout()
        sk_layout.setContentsMargins(16, 24, 16, 16)
        sk_layout.setSpacing(12)
        
        self.skeleton_list = QListWidget()
        self.skeleton_list.setAlternatingRowColors(True)
        sk_layout.addWidget(self.skeleton_list)
        
        # Connection inputs
        conn_form = QHBoxLayout()
        self.bp1_combo = QComboBox()
        self.bp2_combo = QComboBox()
        conn_form.addWidget(self.bp1_combo, 1)
        conn_form.addWidget(QLabel("→"))
        conn_form.addWidget(self.bp2_combo, 1)
        sk_layout.addLayout(conn_form)
        
        sk_btns = QHBoxLayout()
        sk_add = QPushButton("Connect")
        sk_add.setObjectName(SECONDARY_BUTTON)
        sk_add.clicked.connect(self.add_connection)
        
        sk_del = QPushButton("Disconnect")
        sk_del.setObjectName(DANGER_BUTTON)
        sk_del.clicked.connect(self.remove_connection)
        
        sk_btns.addWidget(sk_add)
        sk_btns.addWidget(sk_del)
        sk_layout.addLayout(sk_btns)
        
        sk_group.setLayout(sk_layout)
        split_layout.addWidget(sk_group)
        
        layout.addLayout(split_layout)
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
            self.load_keypoints()
            self.load_skeleton()
    
    def load_keypoints(self):
        """Load keypoints from config"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        if not valid: return
        
        try:
            bodyparts = self.label_manager.get_bodyparts(config)
            self.keypoints_list.clear()
            self.keypoints_list.addItems(bodyparts)
            
            # Update skeleton combos
            self.bp1_combo.clear()
            self.bp2_combo.clear()
            self.bp1_combo.addItems(bodyparts)
            self.bp2_combo.addItems(bodyparts)
        except Exception:
            pass
    
    def load_skeleton(self):
        """Load skeleton from config"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        if not valid: return
        
        try:
            skeleton = self.label_manager.get_skeleton(config)
            self.skeleton_list.clear()
            for connection in skeleton:
                self.skeleton_list.addItem(f"{connection[0]} → {connection[1]}")
        except Exception:
            pass
    
    def launch_labeling(self):
        """Launch DeepLabCut labeling GUI"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        try:
            self.label_manager.label_frames(config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch labeling GUI:\n{str(e)}")
    
    def check_labels(self):
        """Check labeling status"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        try:
            status = self.label_manager.check_labels(config)
            if 'error' in status:
                QMessageBox.warning(self, "Status", f"Error: {status['error']}")
            else:
                msg = "Labeling Status:\n\n"
                for key, value in status.items():
                    msg += f"{key}: {value}\n"
                QMessageBox.information(self, "Label Status", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check labels:\n{str(e)}")
    
    def add_keypoint(self):
        """Add new keypoint"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        name, ok = QInputDialog.getText(self, "Add Keypoint", "Keypoint name:")
        if ok and name:
            try:
                self.label_manager.add_bodypart(config, name)
                self.load_keypoints()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add keypoint:\n{str(e)}")
    
    def edit_keypoint(self):
        """Edit selected keypoint"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        current = self.keypoints_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Warning", "Select a keypoint to edit")
            return
        
        old_name = current.text()
        new_name, ok = QInputDialog.getText(
            self, "Edit Keypoint", "New name:", text=old_name
        )
        
        if ok and new_name and new_name != old_name:
            try:
                self.label_manager.update_bodypart(config, old_name, new_name)
                self.load_keypoints()
                self.load_skeleton()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update keypoint:\n{str(e)}")
    
    def remove_keypoint(self):
        """Remove selected keypoint"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        current = self.keypoints_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Warning", "Select a keypoint to remove")
            return
        
        name = current.text()
        reply = QMessageBox.question(
            self, "Confirm", f"Remove keypoint '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.label_manager.remove_bodypart(config, name)
                self.load_keypoints()
                self.load_skeleton()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove keypoint:\n{str(e)}")
    
    def add_connection(self):
        """Add skeleton connection"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        bp1 = self.bp1_combo.currentText()
        bp2 = self.bp2_combo.currentText()
        
        if not bp1 or not bp2:
            QMessageBox.warning(self, "Warning", "Select both keypoints")
            return
        
        if bp1 == bp2:
            QMessageBox.warning(self, "Warning", "Cannot connect keypoint to itself")
            return
        
        try:
            self.label_manager.add_skeleton_connection(config, bp1, bp2)
            self.load_skeleton()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add connection:\n{str(e)}")
    
    def remove_connection(self):
        """Remove selected skeleton connection"""
        config = self.config_input.text()
        valid, error = validate_config_path(config)
        if not valid:
            QMessageBox.warning(self, "Validation Error", error)
            return
        
        current = self.skeleton_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Warning", "Select a connection to remove")
            return
        
        connection_text = current.text()
        bp1, bp2 = connection_text.split(" → ")
        
        reply = QMessageBox.question(
            self, "Confirm", f"Remove connection '{connection_text}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.label_manager.remove_skeleton_connection(config, bp1, bp2)
                self.load_skeleton()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove connection:\n{str(e)}")
