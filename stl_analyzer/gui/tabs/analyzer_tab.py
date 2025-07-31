"""
Analyzer Tab Component

This tab handles:
- Directory scanning for STL files
- File processing and feature extraction  
- Database view and management
- Similarity search functionality
- 3D visualization of analyzed parts
"""

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

class AnalyzerTab(QtWidgets.QWidget):
    """Analyzer tab widget with proper vertical layout"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Store reference to main window
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the analyzer tab layout based on original working structure"""
        # Create main horizontal layout (like original)
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # Create left control panel
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add scan controls to left panel
        scan_controls = self.create_scan_controls()
        left_layout.addWidget(scan_controls)
        
        # Add database view to left panel  
        db_view = self.create_database_view()
        left_layout.addWidget(db_view)
        
        # Add similarity search panel to left panel
        sim_search = self.create_similarity_search()
        left_layout.addWidget(sim_search)
        
        # Create right panel with viewer and details
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create vertical splitter for 3D viewer and details
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # Add 3D viewer to the splitter
        viewer_panel = self.create_3d_viewer()
        self.right_splitter.addWidget(viewer_panel)
        
        # Add details panel to the splitter
        details_panel = self.create_details_panel()
        self.right_splitter.addWidget(details_panel)
        
        # Set initial sizes for right panel (70% viewer, 30% details)
        self.right_splitter.setSizes([700, 300])
        
        # Add the splitter to the right panel
        right_layout.addWidget(self.right_splitter)
        
        # Add left and right panels to main layout with a splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        
        # Set initial sizes for main splitter (30% left panel, 70% right panel)
        self.main_splitter.setSizes([300, 700])
        
        # Add the splitter to the main layout
        main_layout.addWidget(self.main_splitter)
        
        print("✅ Using modular AnalyzerTab with proper vertical layout (3D viewer on right)")
    
    def create_scan_controls(self):
        """Create the scan controls panel"""
        scan_group = QtWidgets.QGroupBox("Scan STL Files")
        scan_layout = QtWidgets.QVBoxLayout(scan_group)
        
        # Directory selection
        dir_layout = QtWidgets.QHBoxLayout()
        self.dir_input = QtWidgets.QLineEdit()
        self.dir_input.setPlaceholderText("Select directory to scan...")
        dir_button = QtWidgets.QPushButton("Browse...")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(dir_button)
        scan_layout.addLayout(dir_layout)
        
        # Shrinkwrap settings group
        shrinkwrap_group = QtWidgets.QGroupBox("Shrinkwrap Settings")
        shrinkwrap_layout = QtWidgets.QVBoxLayout(shrinkwrap_group)
        
        # Calculate shrinkwrap data (always enabled for analysis)
        self.calculate_shrinkwrap_checkbox = QtWidgets.QCheckBox("Calculate Shrinkwrap Data (Volume & Ratio)")
        self.calculate_shrinkwrap_checkbox.setChecked(True)  # Always enabled for analysis
        self.calculate_shrinkwrap_checkbox.setToolTip(
            "Calculate shrinkwrap volume and ratio data for analysis.\n"
            "This data is essential for material optimization and pricing."
        )
        shrinkwrap_layout.addWidget(self.calculate_shrinkwrap_checkbox)
        
        # Generate shrinkwrap STL files (separate control)
        self.generate_shrinkwrap_files_checkbox = QtWidgets.QCheckBox("Generate Shrinkwrap STL Files")
        self.generate_shrinkwrap_files_checkbox.setChecked(False)  # Default OFF to avoid clutter
        self.generate_shrinkwrap_files_checkbox.setToolTip(
            "Generate actual shrinkwrap STL files during scan.\n"
            "Usually not needed - the volume data is calculated regardless.\n"
            "Enable only when you need the 3D mesh files for visualization."
        )
        shrinkwrap_layout.addWidget(self.generate_shrinkwrap_files_checkbox)
        
        # File output settings (only shown when file generation is enabled)
        self.file_output_widget = QtWidgets.QWidget()
        file_output_layout = QtWidgets.QVBoxLayout(self.file_output_widget)
        file_output_layout.setContentsMargins(20, 0, 0, 0)  # Indent to show it's sub-option
        
        # Output mode selection
        output_mode_layout = QtWidgets.QHBoxLayout()
        output_mode_layout.addWidget(QtWidgets.QLabel("Save Location:"))
        
        self.shrinkwrap_output_mode = QtWidgets.QComboBox()
        self.shrinkwrap_output_mode.addItems([
            "Dedicated Shrinkwrap Folder",  # Default - keeps files separate
            "Same Directory as Original",
            "Custom Directory"
        ])
        self.shrinkwrap_output_mode.setCurrentIndex(0)  # Default to dedicated folder
        self.shrinkwrap_output_mode.setToolTip(
            "Where to store shrinkwrap files:\n"
            "• Dedicated Folder: In a 'shrinkwrap' subfolder (recommended)\n"
            "• Same Directory: Next to original files (clutters directory)\n"
            "• Custom Directory: In a user-specified location"
        )
        output_mode_layout.addWidget(self.shrinkwrap_output_mode)
        output_mode_layout.addStretch()
        
        # Custom directory selection (hidden by default)
        self.shrinkwrap_dir_widget = QtWidgets.QWidget()
        self.shrinkwrap_dir_layout = QtWidgets.QHBoxLayout(self.shrinkwrap_dir_widget)
        self.shrinkwrap_dir_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shrinkwrap_dir_input = QtWidgets.QLineEdit()
        self.shrinkwrap_dir_input.setPlaceholderText("Enter custom directory path...")
        self.shrinkwrap_dir_layout.addWidget(self.shrinkwrap_dir_input)
        
        browse_shrinkwrap_btn = QtWidgets.QPushButton("Browse")
        browse_shrinkwrap_btn.clicked.connect(self.parent_window.browse_custom_shrinkwrap_directory)
        self.shrinkwrap_dir_layout.addWidget(browse_shrinkwrap_btn)
        
        file_output_layout.addLayout(output_mode_layout)
        file_output_layout.addLayout(self.shrinkwrap_dir_layout)
        
        shrinkwrap_layout.addWidget(self.file_output_widget)
        
        # Initially hide file output settings since file generation is disabled by default
        self.file_output_widget.setVisible(False)
        
        # Offset mm control
        offset_layout = QtWidgets.QHBoxLayout()
        offset_layout.addWidget(QtWidgets.QLabel("Offset (mm):"))

        self.shrinkwrap_offset_spinbox = QtWidgets.QDoubleSpinBox()
        self.shrinkwrap_offset_spinbox.setRange(0.1, 20.0)
        self.shrinkwrap_offset_spinbox.setValue(4.0)
        self.shrinkwrap_offset_spinbox.setSingleStep(0.1)
        self.shrinkwrap_offset_spinbox.setSuffix(" mm")
        self.shrinkwrap_offset_spinbox.setToolTip("Offset for shrinkwrap inflation in millimeters (0.1-20.0 mm)")
        offset_layout.addWidget(self.shrinkwrap_offset_spinbox)
        
        offset_layout.addStretch()
        shrinkwrap_layout.addLayout(offset_layout)
        
        # Connect controls
        self.shrinkwrap_output_mode.currentIndexChanged.connect(self.on_shrinkwrap_output_mode_changed)
        self.generate_shrinkwrap_files_checkbox.toggled.connect(self.on_file_generation_toggled)
        
        scan_layout.addWidget(shrinkwrap_group)
        
        # Progress bar
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setVisible(False)
        scan_layout.addWidget(self.scan_progress)
        
        # Scan button
        scan_button = QtWidgets.QPushButton("Scan Directory")
        scan_button.clicked.connect(self.start_scan)
        scan_layout.addWidget(scan_button)
        
        # Stop button
        self.stop_scan_button = QtWidgets.QPushButton("Stop Scan")
        self.stop_scan_button.clicked.connect(self.stop_scan)
        self.stop_scan_button.setEnabled(False)
        scan_layout.addWidget(self.stop_scan_button)
        
        return scan_group
    
    def create_database_view(self):
        """Create the database view panel"""
        db_group = QtWidgets.QGroupBox("STL Database")
        db_layout = QtWidgets.QVBoxLayout(db_group)
        
        # Database controls
        db_controls_layout = QtWidgets.QHBoxLayout()
        
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_database)
        db_controls_layout.addWidget(refresh_button)
        
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_database)
        db_controls_layout.addWidget(clear_button)
        
        save_button = QtWidgets.QPushButton("Save As...")
        save_button.clicked.connect(self.save_database_as)
        db_controls_layout.addWidget(save_button)
        
        db_layout.addLayout(db_controls_layout)
        
        # Database table
        self.db_table = QtWidgets.QTableView()
        self.db_table.setAlternatingRowColors(True)
        self.db_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.db_table.setSortingEnabled(True)
        self.db_table.clicked.connect(self.on_db_item_clicked)
        
        # Set minimum height for the table
        self.db_table.setMinimumHeight(200)
        db_layout.addWidget(self.db_table)
        
        return db_group
    
    def create_similarity_search(self):
        """Create the similarity search panel"""
        sim_group = QtWidgets.QGroupBox("Similarity Search")
        sim_layout = QtWidgets.QVBoxLayout(sim_group)
        
        # Load Part section
        load_part_layout = QtWidgets.QHBoxLayout()
        load_part_layout.addWidget(QtWidgets.QLabel("Comparison Part:"))
        
        self.comparison_part_input = QtWidgets.QLineEdit()
        self.comparison_part_input.setReadOnly(True)
        self.comparison_part_input.setPlaceholderText("Select a part to compare against database...")
        load_part_layout.addWidget(self.comparison_part_input)
        
        load_part_button = QtWidgets.QPushButton("Load Part")
        load_part_button.clicked.connect(self.load_comparison_part)
        load_part_button.setToolTip("Load an STL file to compare against the scanned database")
        load_part_layout.addWidget(load_part_button)
        
        sim_layout.addLayout(load_part_layout)
        
        # Search parameter selection
        param_layout = QtWidgets.QHBoxLayout()
        param_layout.addWidget(QtWidgets.QLabel("Search by:"))
        
        self.similarity_param = QtWidgets.QComboBox()
        self.similarity_param.addItems([
            "bb_volume", "volume", "surface_area", 
            "convex_hull_volume", "shrinkwrap_volume"
        ])
        self.similarity_param.setCurrentText("bb_volume")
        param_layout.addWidget(self.similarity_param)
        param_layout.addStretch()
        
        sim_layout.addLayout(param_layout)
        
        # Number of results
        results_layout = QtWidgets.QHBoxLayout()
        results_layout.addWidget(QtWidgets.QLabel("Results:"))
        
        self.similarity_count = QtWidgets.QSpinBox()
        self.similarity_count.setRange(1, 50)
        self.similarity_count.setValue(10)
        results_layout.addWidget(self.similarity_count)
        results_layout.addStretch()
        
        sim_layout.addLayout(results_layout)
        
        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        # Find similar button (for loaded part)
        find_similar_loaded_button = QtWidgets.QPushButton("Find Similar to Loaded Part")
        find_similar_loaded_button.clicked.connect(self.find_similar_to_loaded_part)
        find_similar_loaded_button.setToolTip("Find parts similar to the loaded comparison part")
        button_layout.addWidget(find_similar_loaded_button)
        
        # Find similar button (for selected part)
        find_similar_button = QtWidgets.QPushButton("Find Similar to Selected")
        find_similar_button.clicked.connect(self.find_similar_parts)
        find_similar_button.setToolTip("Find parts similar to the currently selected part in the database")
        button_layout.addWidget(find_similar_button)
        
        sim_layout.addLayout(button_layout)
        
        # Show all button layout
        show_all_layout = QtWidgets.QHBoxLayout()
        show_all_button = QtWidgets.QPushButton("Show All Parts")
        show_all_button.clicked.connect(self.show_all_database_parts)
        show_all_button.setToolTip("Show the complete database (clear similarity filter)")
        show_all_layout.addWidget(show_all_button)
        show_all_layout.addStretch()
        
        sim_layout.addLayout(show_all_layout)
        
        # Store loaded part data
        self.loaded_comparison_part = None
        
        return sim_group
    
    def create_3d_viewer(self):
        """Create the 3D viewer panel"""
        viewer_group = QtWidgets.QGroupBox("3D Model Preview")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_group)
        
        # Create 3D viewer with proper fallback - use the same pattern that works in project tab
        try:
            # Try the direct import first (same as working project tab)
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from viewer import STLViewer
            self.stl_viewer = STLViewer(self)
            viewer_layout.addWidget(self.stl_viewer)
            print("✅ STL Viewer created successfully in analyzer tab")
        except ImportError:
            try:
                from stl_analyzer.viewer import STLViewer
                self.stl_viewer = STLViewer(self)
                viewer_layout.addWidget(self.stl_viewer)
                print("✅ STL Viewer created successfully (fallback import)")
            except ImportError:
                try:
                    # Try relative import
                    from ...viewer import STLViewer
                    self.stl_viewer = STLViewer(self)
                    viewer_layout.addWidget(self.stl_viewer)
                    print("✅ STL Viewer created successfully (relative import)")
                except ImportError:
                    # Ultimate fallback
                    self.stl_viewer = QtWidgets.QLabel("3D Viewer not available")
                    self.stl_viewer.setAlignment(Qt.AlignCenter)
                    self.stl_viewer.setStyleSheet("color: #888; font-size: 14px; border: 2px dashed #ccc;")
                    self.stl_viewer.setMinimumHeight(400)
                    viewer_layout.addWidget(self.stl_viewer)
                    print("⚠️ Using 3D Viewer fallback")
        
        # Add view controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        reset_button = QtWidgets.QPushButton("Reset View")
        if hasattr(self.stl_viewer, 'reset_camera_position'):
            reset_button.clicked.connect(self.stl_viewer.reset_camera_position)
        controls_layout.addWidget(reset_button)
        
        # Grid checkbox
        grid_checkbox = QtWidgets.QCheckBox("Grid")
        grid_checkbox.setChecked(True)
        if hasattr(self.stl_viewer, 'set_grid_visible'):
            grid_checkbox.stateChanged.connect(lambda state: self.stl_viewer.set_grid_visible(state == Qt.Checked))
        controls_layout.addWidget(grid_checkbox)
        
        # Axes checkbox
        axes_checkbox = QtWidgets.QCheckBox("Axes")
        axes_checkbox.setChecked(True)
        if hasattr(self.stl_viewer, 'set_axes_visible'):
            axes_checkbox.stateChanged.connect(lambda state: self.stl_viewer.set_axes_visible(state == Qt.Checked))
        controls_layout.addWidget(axes_checkbox)
        
        viewer_layout.addLayout(controls_layout)
        
        return viewer_group
    
    def create_details_panel(self):
        """Create the details panel"""
        details_group = QtWidgets.QGroupBox("Model Details")
        details_layout = QtWidgets.QVBoxLayout(details_group)
        
        # Create form layout for details
        form_layout = QtWidgets.QFormLayout()
        
        # Detail fields
        self.detail_filename = QtWidgets.QLabel("-")
        self.detail_filename.setWordWrap(True)
        form_layout.addRow("Filename:", self.detail_filename)
        
        # Dimensions
        dims_layout = QtWidgets.QHBoxLayout()
        self.detail_dim_x = QtWidgets.QLabel("-")
        self.detail_dim_y = QtWidgets.QLabel("-")
        self.detail_dim_z = QtWidgets.QLabel("-")
        dims_layout.addWidget(QtWidgets.QLabel("X:"))
        dims_layout.addWidget(self.detail_dim_x)
        dims_layout.addWidget(QtWidgets.QLabel("Y:"))
        dims_layout.addWidget(self.detail_dim_y)
        dims_layout.addWidget(QtWidgets.QLabel("Z:"))
        dims_layout.addWidget(self.detail_dim_z)
        dims_layout.addStretch()
        form_layout.addRow("Dimensions:", dims_layout)
        
        # Measurements
        self.detail_volume = QtWidgets.QLabel("-")
        form_layout.addRow("Volume:", self.detail_volume)
        
        self.detail_surface = QtWidgets.QLabel("-")
        form_layout.addRow("Surface Area:", self.detail_surface)
        
        self.detail_bb_volume = QtWidgets.QLabel("-")
        form_layout.addRow("BB Volume:", self.detail_bb_volume)
        
        self.detail_convex_hull = QtWidgets.QLabel("-")
        form_layout.addRow("Convex Hull:", self.detail_convex_hull)
        
        self.detail_shrinkwrap = QtWidgets.QLabel("-")
        form_layout.addRow("Shrinkwrap:", self.detail_shrinkwrap)
        
        details_layout.addLayout(form_layout)
        details_layout.addStretch()
        
        return details_group
    
    # Delegate methods to parent window if available, otherwise provide basic functionality
    def browse_directory(self):
        """Browse for directory"""
        if self.parent_window and hasattr(self.parent_window, 'browse_directory'):
            self.parent_window.browse_directory()
        else:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                self.dir_input.setText(directory)
    
    def start_scan(self):
        """Start scanning"""
        if self.parent_window and hasattr(self.parent_window, 'start_scan'):
            self.parent_window.start_scan()
        else:
            directory = self.dir_input.text()
            if directory:
                print(f"Starting scan of: {directory}")
                # Basic scan implementation here if needed
    
    def stop_scan(self):
        """Stop scanning"""
        if self.parent_window and hasattr(self.parent_window, 'stop_scan'):
            self.parent_window.stop_scan()
    
    def refresh_database(self):
        """Refresh database"""
        if self.parent_window and hasattr(self.parent_window, 'update_database_view'):
            self.parent_window.update_database_view()
    
    def clear_database(self):
        """Clear database"""
        if self.parent_window and hasattr(self.parent_window, 'clear_database'):
            self.parent_window.clear_database()
    
    def save_database_as(self):
        """Save database as"""
        if self.parent_window and hasattr(self.parent_window, 'save_database_as'):
            self.parent_window.save_database_as()
    
    def on_db_item_clicked(self, index):
        """Handle database item click"""
        if self.parent_window and hasattr(self.parent_window, 'on_db_table_clicked'):
            self.parent_window.on_db_table_clicked(index)
    
    def load_comparison_part(self):
        """Load an STL file for comparison against the database"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select STL File for Comparison",
            "",
            "STL Files (*.stl);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Extract features from the loaded part
            self.parent_window.statusBar().showMessage("Analyzing comparison part...")
            
            try:
                from ..stl_utils import extract_features
            except ImportError:
                from stl_utils import extract_features
                
            features = extract_features(file_path)
            
            if features is None:
                self.parent_window.statusBar().showMessage("Failed to analyze the selected part", 3000)
                return
            
            # Store the loaded part data
            self.loaded_comparison_part = features
            
            # Update the UI
            filename = os.path.basename(file_path)
            self.comparison_part_input.setText(filename)
            
            # Show part details
            volume = features.get('volume', 0)
            bbox_vol = features.get('bb_volume', 0)
            surface_area = features.get('surface_area', 0)
            
            self.parent_window.statusBar().showMessage(
                f"Loaded: {filename} (Vol: {volume:.2f}mm³, BBox: {bbox_vol:.2f}mm³, SA: {surface_area:.2f}mm²)", 3000
            )
            
            # Load in 3D viewer if available
            if hasattr(self, 'stl_viewer') and hasattr(self.stl_viewer, 'load_stl'):
                self.stl_viewer.load_stl(file_path)
                print(f"✅ Loaded comparison part in 3D viewer: {filename}")
            
        except Exception as e:
            self.parent_window.statusBar().showMessage(f"Error loading comparison part: {e}", 5000)
            print(f"❌ Error loading comparison part: {e}")
    
    def find_similar_parts(self):
        """Find similar parts"""
        if self.parent_window and hasattr(self.parent_window, 'find_similar_parts'):
            self.parent_window.find_similar_parts()
    
    def on_similarity_result_clicked(self, item):
        """Handle similarity result click"""
        if self.parent_window and hasattr(self.parent_window, 'on_similarity_result_clicked'):
            self.parent_window.on_similarity_result_clicked(item)
    
    def on_shrinkwrap_output_mode_changed(self, index):
        """Handle shrinkwrap output mode change"""
        # Show/hide custom directory widget based on selection
        is_custom = (index == 2)  # "Custom Directory" is index 2
        self.shrinkwrap_dir_widget.setVisible(is_custom)
        
        if self.parent_window and hasattr(self.parent_window, 'on_shrinkwrap_output_mode_changed'):
            self.parent_window.on_shrinkwrap_output_mode_changed(index)
    
    def on_file_generation_toggled(self, checked):
        """Handle file generation toggle"""
        # Show/hide file output settings based on checkbox state
        self.file_output_widget.setVisible(checked)
        
        if self.parent_window and hasattr(self.parent_window, 'on_file_generation_toggled'):
            self.parent_window.on_file_generation_toggled(checked)
    
    def find_similar_to_loaded_part(self):
        """Find parts similar to the loaded comparison part"""
        if not hasattr(self, 'loaded_comparison_part') or not self.loaded_comparison_part:
            self.parent_window.statusBar().showMessage("Please load a comparison part first", 3000)
            return
            
        if self.parent_window and hasattr(self.parent_window, 'find_similar_to_loaded_part'):
            self.parent_window.find_similar_to_loaded_part(
                self.loaded_comparison_part,
                self.similarity_param.currentText(),
                self.similarity_count.value()
            )
    
    def show_all_database_parts(self):
        """Show all parts in the database (clear similarity filter)"""
        if self.parent_window and hasattr(self.parent_window, 'update_database_view'):
            self.parent_window.update_database_view()
            self.parent_window.statusBar().showMessage("Showing all database parts", 2000) 