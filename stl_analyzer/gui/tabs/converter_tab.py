"""
STEP to STL Converter Tab Component
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

# Import thread class - handle both package and direct imports
try:
    # When imported as a package
    from ..threads import StepConversionThread
except ImportError:
    try:
        # When imported from gui package
        from gui.threads import StepConversionThread
    except ImportError:
        # When run directly - should work with the thread in the main file
        pass


class ConverterTab(QtWidgets.QWidget):
    """STEP to STL converter tab widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.step_conversion_thread = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the STEP converter tab content - HORIZONTAL SPLIT LAYOUT"""
        # Create main horizontal layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # LEFT PANEL (30%): Converter controls
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(450)  # Limit left panel width
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create STEP to STL converter panel
        converter_group = QtWidgets.QGroupBox("STEP to STL Converter")
        converter_form = QtWidgets.QFormLayout(converter_group)
        
        # Input file selection
        self.step_file_input = QtWidgets.QLineEdit()
        self.step_file_input.setReadOnly(True)
        
        step_file_layout = QtWidgets.QHBoxLayout()
        step_file_layout.addWidget(self.step_file_input)
        
        step_browse_button = QtWidgets.QPushButton("Browse...")
        step_browse_button.clicked.connect(self.browse_step_file)
        step_file_layout.addWidget(step_browse_button)
        
        converter_form.addRow("Input STEP File:", step_file_layout)
        
        # Output file selection
        self.stl_file_output = QtWidgets.QLineEdit()
        self.stl_file_output.setReadOnly(True)
        
        stl_file_layout = QtWidgets.QHBoxLayout()
        stl_file_layout.addWidget(self.stl_file_output)
        
        stl_browse_button = QtWidgets.QPushButton("Browse...")
        stl_browse_button.clicked.connect(self.browse_stl_output)
        stl_file_layout.addWidget(stl_browse_button)
        
        converter_form.addRow("Output STL File:", stl_file_layout)
        
        # Resolution slider
        self.resolution_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.resolution_slider.setMinimum(1)
        self.resolution_slider.setMaximum(10)
        self.resolution_slider.setValue(5)
        self.resolution_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.resolution_slider.setTickInterval(1)
        
        resolution_layout = QtWidgets.QHBoxLayout()
        resolution_layout.addWidget(QtWidgets.QLabel("Low"))
        resolution_layout.addWidget(self.resolution_slider)
        resolution_layout.addWidget(QtWidgets.QLabel("High"))
        
        converter_form.addRow("Resolution:", resolution_layout)
        
        # Convert button
        self.convert_button = QtWidgets.QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert_step_to_stl)
        self.convert_button.setEnabled(False)  # Disabled until files are selected
        
        # Progress bar
        self.conversion_progress = QtWidgets.QProgressBar()
        self.conversion_progress.setVisible(False)
        
        # Status label
        self.conversion_status = QtWidgets.QLabel("Select input and output files")
        
        # Add to layout
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.convert_button)
        buttons_layout.addWidget(self.conversion_progress)
        
        converter_form.addRow("", buttons_layout)
        converter_form.addRow("Status:", self.conversion_status)
        
        # Add converter group to left panel
        left_layout.addWidget(converter_group)
        
        # Add conversion history or recent files (optional)
        history_group = QtWidgets.QGroupBox("Recent Conversions")
        history_layout = QtWidgets.QVBoxLayout(history_group)
        
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMaximumHeight(200)
        history_layout.addWidget(self.history_list)
        
        left_layout.addWidget(history_group)
        
        # Add spacer to push everything to the top
        left_layout.addStretch()
        
        # RIGHT PANEL (70%): 3D viewer
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add 3D viewer using modular widget
        try:
            from ..widgets import STLViewerWidget
            self.viewer_panel = STLViewerWidget(parent=self)
            right_layout.addWidget(self.viewer_panel)
            
            # Connect to automatically load converted files
            if hasattr(self.viewer_panel, 'stl_viewer'):
                self.stl_viewer = self.viewer_panel.stl_viewer
        except ImportError as e:
            print(f"STLViewerWidget not available: {e}")
            # Fallback to basic 3D viewer
            viewer_group = QtWidgets.QGroupBox("3D Preview")
            viewer_layout = QtWidgets.QVBoxLayout(viewer_group)
            
            # Try to create a basic viewer
            try:
                from ...viewer import STLViewer
                self.stl_viewer = STLViewer(self)
                viewer_layout.addWidget(self.stl_viewer)
            except ImportError:
                # Ultimate fallback
                fallback_label = QtWidgets.QLabel("3D Viewer not available")
                fallback_label.setAlignment(Qt.AlignCenter)
                fallback_label.setStyleSheet("color: #888; font-size: 14px;")
                viewer_layout.addWidget(fallback_label)
            
            right_layout.addWidget(viewer_group)
        
        # Create main horizontal splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial sizes for main splitter (30% left, 70% right)
        main_splitter.setSizes([350, 850])
        
        # Add the splitter to the main layout
        main_layout.addWidget(main_splitter)

    def browse_step_file(self):
        """Browse for STEP file to convert"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select STEP File",
            "",
            "STEP Files (*.step *.stp);;All Files (*.*)"
        )
        if file_path:
            self.step_file_input.setText(file_path)
            self.update_converter_status()

    def browse_stl_output(self):
        """Browse for STL output file location"""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save STL File As",
            "",
            "STL Files (*.stl);;All Files (*.*)"
        )
        if file_path:
            self.stl_file_output.setText(file_path)
            self.update_converter_status()

    def update_converter_status(self):
        """Update converter status and enable/disable convert button"""
        step_file = self.step_file_input.text()
        stl_file = self.stl_file_output.text()
        
        if step_file and stl_file:
            self.convert_button.setEnabled(True)
            self.conversion_status.setText("Ready to convert")
        else:
            self.convert_button.setEnabled(False)
            self.conversion_status.setText("Select input and output files")

    def convert_step_to_stl(self):
        """Convert STEP file to STL"""
        step_file = self.step_file_input.text()
        stl_file = self.stl_file_output.text()
        resolution = self.resolution_slider.value()
        
        if not step_file or not stl_file:
            return
        
        # Show progress
        self.conversion_progress.setVisible(True)
        self.conversion_progress.setValue(0)
        self.convert_button.setEnabled(False)
        self.conversion_status.setText("Converting...")
        
        # Import thread class dynamically if not already available
        if not hasattr(self, 'StepConversionThread'):
            try:
                from ..threads import StepConversionThread
                self.StepConversionThread = StepConversionThread
            except ImportError:
                try:
                    from gui.threads import StepConversionThread
                    self.StepConversionThread = StepConversionThread
                except ImportError:
                    # Fall back to importing from main module if available
                    if self.parent_window and hasattr(self.parent_window, '__class__'):
                        # Get StepConversionThread from the same module as the parent
                        parent_module = self.parent_window.__class__.__module__
                        if parent_module:
                            import sys
                            if parent_module in sys.modules:
                                module = sys.modules[parent_module]
                                if hasattr(module, 'StepConversionThread'):
                                    self.StepConversionThread = module.StepConversionThread
                                else:
                                    QtWidgets.QMessageBox.critical(self, "Error", "StepConversionThread not available")
                                    return
                            else:
                                QtWidgets.QMessageBox.critical(self, "Error", "Cannot access conversion functionality")
                                return
                        else:
                            QtWidgets.QMessageBox.critical(self, "Error", "Conversion functionality not available")
                            return
        else:
            # Use already imported class
            self.StepConversionThread = StepConversionThread
        
        # Start conversion in separate thread
        self.step_conversion_thread = self.StepConversionThread(step_file, stl_file, resolution)
        self.step_conversion_thread.progress.connect(self.conversion_progress.setValue)
        self.step_conversion_thread.conversion_complete.connect(self.on_step_conversion_complete)
        self.step_conversion_thread.start()

    def on_step_conversion_complete(self, success, message):
        """Handle STEP to STL conversion completion"""
        self.conversion_progress.setVisible(False)
        self.convert_button.setEnabled(True)
        
        if success:
            self.conversion_status.setText("Conversion completed successfully!")
            QtWidgets.QMessageBox.information(self, "Success", "STEP to STL conversion completed successfully!")
            
            # Load the converted file in the 3D viewer
            stl_file = self.stl_file_output.text()
            if hasattr(self, 'stl_viewer') and stl_file:
                try:
                    self.stl_viewer.load_stl(stl_file)
                except Exception as e:
                    print(f"Error loading converted file in viewer: {e}")
            
            # Add to history
            if hasattr(self, 'history_list'):
                import os
                filename = os.path.basename(stl_file)
                self.history_list.insertItem(0, filename)
                # Keep only last 10 items
                while self.history_list.count() > 10:
                    self.history_list.takeItem(10)
        else:
            self.conversion_status.setText(f"Conversion failed: {message}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Conversion failed: {message}") 