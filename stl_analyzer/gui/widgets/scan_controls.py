"""
Scan Controls Widget Component

This widget provides:
- Directory selection for scanning
- Shrinkwrap generation settings
- Scan control buttons
- Progress tracking
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import os
import datetime

class ScanControlsPanel(QtWidgets.QGroupBox):
    """Scan controls panel widget"""
    
    def __init__(self, parent=None):
        super().__init__("Scan STL Files", parent)
        self.parent_window = parent
        
        # Initialize scan state
        self.found_files = []
        self.scan_thread = None
        self.process_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the controls for scanning directories"""
        scan_layout = QtWidgets.QVBoxLayout(self)
        
        # Directory selection
        dir_layout = QtWidgets.QHBoxLayout()
        self.dir_input = QtWidgets.QLineEdit()
        self.dir_input.setPlaceholderText("Select directory to scan...")
        dir_button = QtWidgets.QPushButton("Browse...")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_input, 3)
        dir_layout.addWidget(dir_button, 1)
        scan_layout.addLayout(dir_layout)
        
        # Shrinkwrap generation options
        shrinkwrap_group = QtWidgets.QGroupBox("Shrinkwrap Settings")
        shrinkwrap_layout = QtWidgets.QVBoxLayout(shrinkwrap_group)
        
        # Enable shrinkwrap data calculation
        self.calculate_shrinkwrap_checkbox = QtWidgets.QCheckBox("Calculate Shrinkwrap Data (Volume & Ratio)")
        self.calculate_shrinkwrap_checkbox.setChecked(True)
        self.calculate_shrinkwrap_checkbox.setToolTip(
            "Calculate shrinkwrap volume and ratio data for analysis.\n"
            "This data is essential for material optimization and pricing."
        )
        shrinkwrap_layout.addWidget(self.calculate_shrinkwrap_checkbox)
        
        # Generate shrinkwrap STL files
        self.generate_shrinkwrap_files_checkbox = QtWidgets.QCheckBox("Generate Shrinkwrap STL Files")
        self.generate_shrinkwrap_files_checkbox.setChecked(False)
        self.generate_shrinkwrap_files_checkbox.toggled.connect(self.on_file_generation_toggled)
        shrinkwrap_layout.addWidget(self.generate_shrinkwrap_files_checkbox)
        
        # Shrinkwrap offset (fixed at 4mm)
        offset_layout = QtWidgets.QHBoxLayout()
        offset_layout.addWidget(QtWidgets.QLabel("Offset (mm):"))
        self.shrinkwrap_offset_spinbox = QtWidgets.QDoubleSpinBox()
        self.shrinkwrap_offset_spinbox.setRange(4.0, 4.0)  # Fixed at 4mm
        self.shrinkwrap_offset_spinbox.setValue(4.0)
        self.shrinkwrap_offset_spinbox.setEnabled(False)  # Disabled since it's fixed
        self.shrinkwrap_offset_spinbox.setSingleStep(0.1)
        offset_layout.addWidget(self.shrinkwrap_offset_spinbox)
        offset_layout.addWidget(QtWidgets.QLabel("(Fixed at 4mm for consistency)"))
        shrinkwrap_layout.addLayout(offset_layout)
        
        # File output options (initially hidden)
        self.file_output_widget = QtWidgets.QWidget()
        file_output_layout = QtWidgets.QVBoxLayout(self.file_output_widget)
        
        # Output mode selection
        output_mode_layout = QtWidgets.QHBoxLayout()
        output_mode_layout.addWidget(QtWidgets.QLabel("Save to:"))
        self.shrinkwrap_output_mode = QtWidgets.QComboBox()
        self.shrinkwrap_output_mode.addItems([
            "Dedicated Shrinkwrap Folder",
            "Same Directory as Original",
            "Custom Directory"
        ])
        self.shrinkwrap_output_mode.currentIndexChanged.connect(self.on_shrinkwrap_output_mode_changed)
        output_mode_layout.addWidget(self.shrinkwrap_output_mode)
        file_output_layout.addLayout(output_mode_layout)
        
        # Custom directory selection (initially hidden)
        self.shrinkwrap_dir_widget = QtWidgets.QWidget()
        shrinkwrap_dir_layout = QtWidgets.QHBoxLayout(self.shrinkwrap_dir_widget)
        self.shrinkwrap_dir_input = QtWidgets.QLineEdit()
        self.shrinkwrap_dir_input.setPlaceholderText("Select custom directory...")
        shrinkwrap_dir_button = QtWidgets.QPushButton("Browse...")
        shrinkwrap_dir_button.clicked.connect(self.browse_custom_shrinkwrap_directory)
        shrinkwrap_dir_layout.addWidget(self.shrinkwrap_dir_input, 3)
        shrinkwrap_dir_layout.addWidget(shrinkwrap_dir_button, 1)
        file_output_layout.addWidget(self.shrinkwrap_dir_widget)
        
        self.shrinkwrap_dir_widget.setVisible(False)
        shrinkwrap_layout.addWidget(self.file_output_widget)
        self.file_output_widget.setVisible(False)
        
        # Status info
        self.shrinkwrap_status_label = QtWidgets.QLabel("✅ Data calculation enabled - File generation disabled (clean scan)")
        self.shrinkwrap_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        shrinkwrap_layout.addWidget(self.shrinkwrap_status_label)
        
        scan_layout.addWidget(shrinkwrap_group)
        
        # Scan button and progress
        scan_btn_layout = QtWidgets.QHBoxLayout()
        self.scan_button = QtWidgets.QPushButton("Scan Directory")
        self.scan_button.clicked.connect(self.start_scan)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        scan_btn_layout.addWidget(self.scan_button)
        scan_btn_layout.addWidget(self.stop_button)
        
        # Add import button
        self.import_button = QtWidgets.QPushButton("Import Files")
        self.import_button.clicked.connect(self.import_files)
        scan_btn_layout.addWidget(self.import_button)
        
        scan_layout.addLayout(scan_btn_layout)
        
        # Progress bar
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setValue(0)
        scan_layout.addWidget(self.scan_progress)
        
        # Status label
        self.scan_status = QtWidgets.QLabel("Ready to scan")
        scan_layout.addWidget(self.scan_status)
    
    def browse_directory(self):
        """Browse for a directory to scan"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Scan",
            self.dir_input.text() or ""
        )
        if directory:
            self.dir_input.setText(directory)
    
    def browse_custom_shrinkwrap_directory(self):
        """Browse for a custom directory to save shrinkwrap files"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Custom Shrinkwrap Output Directory",
            self.shrinkwrap_dir_input.text() or ""
        )
        if directory:
            self.shrinkwrap_dir_input.setText(directory)
    
    def on_shrinkwrap_output_mode_changed(self, index):
        """Handle shrinkwrap output mode change"""
        self.shrinkwrap_dir_widget.setVisible(index == 2)  # Custom Directory
        
    def on_file_generation_toggled(self, checked):
        """Handle toggling of shrinkwrap file generation"""
        self.file_output_widget.setVisible(checked)
        
        # Update status label
        if checked:
            if self.calculate_shrinkwrap_checkbox.isChecked():
                self.shrinkwrap_status_label.setText("✅ Data calculation + File generation enabled")
            else:
                self.shrinkwrap_status_label.setText("⚠️ File generation enabled but data calculation disabled")
        else:
            if self.calculate_shrinkwrap_checkbox.isChecked():
                self.shrinkwrap_status_label.setText("✅ Data calculation enabled - File generation disabled (clean scan)")
            else:
                self.shrinkwrap_status_label.setText("❌ Both data calculation and file generation disabled")
    
    def start_scan(self):
        """Start scanning the selected directory"""
        directory = self.dir_input.text()
        if not directory or not os.path.isdir(directory):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Directory", 
                "Please select a valid directory to scan."
            )
            return
        
        # Get database from parent window
        if hasattr(self.parent_window, 'stl_db') and self.parent_window.stl_db:
            # Check if database already has entries
            current_entries = len(self.parent_window.stl_db.get_all_entries())
            
            if current_entries > 0:
                # Ask user what to do with existing database
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Database Action",
                    f"Current database has {current_entries} entries.\n\n"
                    f"What would you like to do?\n\n"
                    f"• Yes: Add new files to existing database\n"
                    f"• No: Save current database and start new one\n"
                    f"• Cancel: Cancel scanning",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Yes
                )
                
                if reply == QtWidgets.QMessageBox.Cancel:
                    return
                elif reply == QtWidgets.QMessageBox.No:
                    # Save current database first
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = f"stl_database_backup_{timestamp}.db"
                    
                    try:
                        # Save current database
                        current_df = self.parent_window.stl_db.get_all_entries()
                        import sqlite3
                        conn = sqlite3.connect(backup_path)
                        current_df.to_sql('stl_files', conn, if_exists='replace', index=False)
                        conn.close()
                        
                        # Clear current database
                        self.parent_window.stl_db.clear()
                        if hasattr(self.parent_window, 'update_database_view'):
                            self.parent_window.update_database_view()
                        
                        if hasattr(self.parent_window, 'show_status_message'):
                            self.parent_window.show_status_message(f"Previous database saved as {backup_path}")
                        
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Save Error",
                            f"Failed to save current database:\n{str(e)}\n\nContinuing with scan..."
                        )
        
        # Clear previous scan data
        self.found_files = []
        
        # Update UI
        self.scan_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.scan_progress.setValue(0)
        self.scan_status.setText("Scanning for STL files...")
        
        # Start scanning thread
        try:
            from ..threads.scan_thread import ScanThread
            self.scan_thread = ScanThread(directory)
            self.scan_thread.progress.connect(self.update_scan_progress)
            self.scan_thread.found_file.connect(self.add_found_file)
            self.scan_thread.scan_complete.connect(self.scan_completed)
            self.scan_thread.start()
        except ImportError:
            # Fallback to inline scanning if thread import fails
            self.scan_status.setText("Import error - using fallback scanning...")
            self.fallback_scan(directory)
    
    def stop_scan(self):
        """Stop the scanning process"""
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread.wait()
            
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.process_thread.wait()
            
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.scan_status.setText("Scan stopped")
        
    def fallback_scan(self, directory):
        """Fallback scanning method if threads fail"""
        found_files = []
        
        # Find all STL files
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.stl'):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    
        self.found_files = found_files
        self.scan_completed(len(found_files))
        
    def update_scan_progress(self, value):
        """Update the scan progress bar"""
        self.scan_progress.setValue(value)
        
    def add_found_file(self, file_path):
        """Add a found file to the list"""
        self.found_files.append(file_path)
        self.scan_status.setText(f"Found: {os.path.basename(file_path)}")
        
    def scan_completed(self, count):
        """Handle scan completion"""
        self.scan_status.setText(f"Scan complete. Found {count} STL files.")
        
        if count > 0:
            # Ask user if they want to process the files
            reply = QtWidgets.QMessageBox.question(
                self,
                "Process Files",
                f"Found {count} STL files.\n\nDo you want to process them now?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.process_files()
            else:
                self.scan_button.setEnabled(True)
                self.stop_button.setEnabled(False)
        else:
            self.scan_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
    def process_files(self):
        """Process found STL files to extract features"""
        if not self.found_files:
            self.scan_status.setText("No files to process")
            self.scan_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return
            
        self.scan_status.setText(f"Processing {len(self.found_files)} STL files...")
        self.scan_progress.setValue(0)
        
        # Get shrinkwrap generation parameters
        calculate_shrinkwrap = self.calculate_shrinkwrap_checkbox.isChecked()
        generate_files = self.generate_shrinkwrap_files_checkbox.isChecked()
        shrinkwrap_offset = self.shrinkwrap_offset_spinbox.value()
        
        # Get output mode settings
        output_mode_index = self.shrinkwrap_output_mode.currentIndex()
        if output_mode_index == 0:
            output_mode = "dedicated"
        elif output_mode_index == 1:
            output_mode = "same"
        else:
            output_mode = "custom"
        
        custom_dir = self.shrinkwrap_dir_input.text() if output_mode == "custom" else ""
        
        # Start processing in a separate thread
        try:
            from ..threads.process_thread import ProcessThread
            self.process_thread = ProcessThread(
                self.found_files,
                calculate_shrinkwrap=calculate_shrinkwrap,
                generate_files=generate_files,
                shrinkwrap_offset=shrinkwrap_offset,
                output_mode=output_mode,
                custom_dir=custom_dir
            )
            self.process_thread.progress.connect(self.update_scan_progress)
            self.process_thread.processed_file.connect(self.add_processed_file)
            self.process_thread.process_complete.connect(self.processing_completed)
            self.process_thread.start()
        except ImportError:
            # Fallback to basic processing if thread import fails
            self.scan_status.setText("Processing thread import failed - using fallback")
            self.fallback_process()
            
    def fallback_process(self):
        """Fallback processing method"""
        # This is a simplified fallback - in a real implementation,
        # you would call the feature extraction functions directly
        self.scan_status.setText("Fallback processing not fully implemented")
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def add_processed_file(self, features):
        """Add processed STL data to the database"""
        if hasattr(self.parent_window, 'stl_db') and self.parent_window.stl_db:
            # Import validation function
            try:
                try:
                    from ...stl_utils import is_valid_entry
                except ImportError:
                    from stl_analyzer.stl_utils import is_valid_entry
                    
                if features and features.get("filename") and os.path.exists(features["filename"]) and is_valid_entry(features):
                    self.parent_window.stl_db.add_entry(features)
                    self.scan_status.setText(f"Processed: {os.path.basename(features['filename'])}")
                    print(f"Added to database: {features['filename']}")
                else:
                    filename = features.get('filename', 'Unknown') if features else 'Unknown'
                    self.scan_status.setText(f"Skipped invalid file: {os.path.basename(filename)}")
                    print(f"Skipped invalid: {filename}")
            except ImportError:
                # Fallback without validation
                if features and features.get("filename") and os.path.exists(features["filename"]):
                    self.parent_window.stl_db.add_entry(features)
                    self.scan_status.setText(f"Processed: {os.path.basename(features['filename'])}")
                    print(f"Added to database (no validation): {features['filename']}")
                else:
                    print("Failed to add to database - missing filename or file doesn't exist")
                    
    def processing_completed(self, count):
        """Handle processing completion"""
        self.scan_status.setText(f"Processing complete. Added {count} files to database.")
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Save database and update view
        if hasattr(self.parent_window, 'stl_db') and self.parent_window.stl_db:
            try:
                self.parent_window.stl_db.save_database()
                print(f"Database saved with {len(self.parent_window.stl_db.get_all_entries())} total entries")
            except Exception as e:
                print(f"Error saving database: {e}")
            
        if hasattr(self.parent_window, 'update_database_view'):
            try:
                self.parent_window.update_database_view()
                print("Database view updated after scan")
            except Exception as e:
                print(f"Error updating database view: {e}")
            
        # Show completion message
        if hasattr(self.parent_window, 'show_status_message'):
            self.parent_window.show_status_message(f"Scan completed! Added {count} STL files to database.")
    
    def import_files(self):
        """Import STL files without scanning"""
        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select STL Files",
            "",
            "STL Files (*.stl);;All Files (*.*)"
        )
        
        if not file_paths:
            return
        
        # Show status message
        self.scan_status.setText(f"Processing {len(file_paths)} STL files...")
        
        # Set the found files and process them
        self.found_files = file_paths
        self.process_files() 