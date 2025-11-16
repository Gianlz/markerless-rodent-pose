"""Frame Labeling Tab"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QListWidget, QGroupBox, QFileDialog,
    QMessageBox, QInputDialog, QComboBox
)
from PySide6.QtCore import Qt

from ...core.label_manager import LabelManager
from ...utils.validators import validate_config_path
from ..styles import SECONDARY_BUTTON


class LabelTab(QWidget):
    """Frame labeling tab with keypoint CRUD and skeleton building"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_manager = LabelManager()
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
        
        # Labeling section
        label_group = QGroupBox("Frame Labeling")
        label_layout = QVBoxLayout()
        label_layout.setSpacing(6)
        label_layout.setContentsMargins(8, 8, 8, 8)
        
        self.label_btn = QPushButton("Launch Labeling GUI")
        self.label_btn.setFixedHeight(32)
        self.label_btn.clicked.connect(self.launch_labeling)
        
        self.check_labels_btn = QPushButton("Check Labels Status")
        self.check_labels_btn.setObjectName(SECONDARY_BUTTON)
        self.check_labels_btn.setFixedHeight(28)
        self.check_labels_btn.clicked.connect(self.check_labels)
        
        label_layout.addWidget(self.label_btn)
        label_layout.addWidget(self.check_labels_btn)
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
        # Keypoints CRUD
        keypoints_group = QGroupBox("Keypoints Management")
        keypoints_layout = QVBoxLayout()
        keypoints_layout.setSpacing(6)
        keypoints_layout.setContentsMargins(8, 8, 8, 8)
        
        self.keypoints_list = QListWidget()
        self.keypoints_list.setMaximumHeight(120)
        keypoints_layout.addWidget(self.keypoints_list)
        
        keypoints_btn_layout = QHBoxLayout()
        keypoints_btn_layout.setSpacing(4)
        
        self.add_keypoint_btn = QPushButton("Add")
        self.add_keypoint_btn.setObjectName(SECONDARY_BUTTON)
        self.add_keypoint_btn.clicked.connect(self.add_keypoint)
        
        self.edit_keypoint_btn = QPushButton("Edit")
        self.edit_keypoint_btn.setObjectName(SECONDARY_BUTTON)
        self.edit_keypoint_btn.clicked.connect(self.edit_keypoint)
        
        self.remove_keypoint_btn = QPushButton("Remove")
        self.remove_keypoint_btn.setObjectName(SECONDARY_BUTTON)
        self.remove_keypoint_btn.clicked.connect(self.remove_keypoint)
        
        keypoints_btn_layout.addWidget(self.add_keypoint_btn)
        keypoints_btn_layout.addWidget(self.edit_keypoint_btn)
        keypoints_btn_layout.addWidget(self.remove_keypoint_btn)
        keypoints_layout.addLayout(keypoints_btn_layout)
        
        keypoints_group.setLayout(keypoints_layout)
        layout.addWidget(keypoints_group)
        
        # Skeleton builder
        skeleton_group = QGroupBox("Skeleton Builder")
        skeleton_layout = QVBoxLayout()
        skeleton_layout.setSpacing(6)
        skeleton_layout.setContentsMargins(8, 8, 8, 8)
        
        self.skeleton_list = QListWidget()
        self.skeleton_list.setMaximumHeight(120)
        skeleton_layout.addWidget(self.skeleton_list)
        
        connection_layout = QHBoxLayout()
        connection_layout.setSpacing(4)
        
        self.bp1_combo = QComboBox()
        self.bp2_combo = QComboBox()
        
        connection_layout.addWidget(QLabel("From:"))
        connection_layout.addWidget(self.bp1_combo)
        connection_layout.addWidget(QLabel("To:"))
        connection_layout.addWidget(self.bp2_combo)
        skeleton_layout.addLayout(connection_layout)
        
        skeleton_btn_layout = QHBoxLayout()
        skeleton_btn_layout.setSpacing(4)
        
        self.add_connection_btn = QPushButton("Add Connection")
        self.add_connection_btn.setObjectName(SECONDARY_BUTTON)
        self.add_connection_btn.clicked.connect(self.add_connection)
        
        self.remove_connection_btn = QPushButton("Remove Connection")
        self.remove_connection_btn.setObjectName(SECONDARY_BUTTON)
        self.remove_connection_btn.clicked.connect(self.remove_connection)
        
        skeleton_btn_layout.addWidget(self.add_connection_btn)
        skeleton_btn_layout.addWidget(self.remove_connection_btn)
        skeleton_layout.addLayout(skeleton_btn_layout)
        
        skeleton_group.setLayout(skeleton_layout)
        layout.addWidget(skeleton_group)
        
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
        
        if not valid:
            return
        
        try:
            bodyparts = self.label_manager.get_bodyparts(config)
            self.keypoints_list.clear()
            self.keypoints_list.addItems(bodyparts)
            
            # Update skeleton combos
            self.bp1_combo.clear()
            self.bp2_combo.clear()
            self.bp1_combo.addItems(bodyparts)
            self.bp2_combo.addItems(bodyparts)
        except Exception as e:
            pass
    
    def load_skeleton(self):
        """Load skeleton from config"""
        config = self.config_input.text()
        valid, _ = validate_config_path(config)
        
        if not valid:
            return
        
        try:
            skeleton = self.label_manager.get_skeleton(config)
            self.skeleton_list.clear()
            for connection in skeleton:
                self.skeleton_list.addItem(f"{connection[0]} → {connection[1]}")
        except Exception as e:
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
                QMessageBox.information(self, "Success", f"Added keypoint: {name}")
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
                QMessageBox.information(self, "Success", f"Updated keypoint: {old_name} → {new_name}")
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
                QMessageBox.information(self, "Success", f"Removed keypoint: {name}")
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
            QMessageBox.information(self, "Success", f"Added connection: {bp1} → {bp2}")
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
                QMessageBox.information(self, "Success", f"Removed connection: {connection_text}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove connection:\n{str(e)}")
