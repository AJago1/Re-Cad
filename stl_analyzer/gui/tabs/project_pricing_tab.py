"""
Project Pricing Tab Component

This tab handles:
- Bulk project pricing for multiple parts
- Project-level pricing analysis
- Part optimization and analysis
- 3D visualization of project parts
- Project totals and statistics
"""

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

class ProjectPricingTab(QtWidgets.QWidget):
    """Project Pricing tab widget for bulk project management"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Store reference to main window
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the project pricing tab layout"""
        # Create main layout
        project_pricing_layout = QtWidgets.QVBoxLayout(self)
        
        # Top toolbar
        toolbar_layout = QtWidgets.QHBoxLayout()
        
        # Upload Parts button
        self.upload_parts_button = QtWidgets.QPushButton("Upload STL Parts")
        self.upload_parts_button.clicked.connect(self.upload_project_parts)
        toolbar_layout.addWidget(self.upload_parts_button)
        
        # Calculate All Prices button
        self.calculate_all_button = QtWidgets.QPushButton("Calculate All Prices")
        self.calculate_all_button.clicked.connect(self.calculate_all_prices)
        self.calculate_all_button.setEnabled(False)
        toolbar_layout.addWidget(self.calculate_all_button)
        
        # Clear Project button
        self.clear_project_button = QtWidgets.QPushButton("Clear Project")
        self.clear_project_button.clicked.connect(self.clear_project)
        toolbar_layout.addWidget(self.clear_project_button)
        
        # Save/Load Project buttons
        self.save_project_button = QtWidgets.QPushButton("Save Project")
        self.save_project_button.clicked.connect(self.save_project)
        self.save_project_button.setEnabled(False)
        toolbar_layout.addWidget(self.save_project_button)
        
        self.load_project_button = QtWidgets.QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.load_project)
        toolbar_layout.addWidget(self.load_project_button)
        
        # Pricing Analysis Report button
        self.pricing_analysis_button = QtWidgets.QPushButton("📊 Pricing Analysis Report")
        self.pricing_analysis_button.clicked.connect(self.show_pricing_analysis)
        self.pricing_analysis_button.setToolTip("View comprehensive pricing analysis based on the training database")
        self.pricing_analysis_button.setStyleSheet("QPushButton { font-weight: bold; color: #2E86AB; }")
        toolbar_layout.addWidget(self.pricing_analysis_button)
        
        toolbar_layout.addStretch()
        project_pricing_layout.addLayout(toolbar_layout)
        
        # Main content splitter (50/50 split)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        project_pricing_layout.addWidget(main_splitter)
        
        # Left panel - Parts table and project totals
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Parts table
        parts_group = QtWidgets.QGroupBox("Project Parts")
        parts_layout = QtWidgets.QVBoxLayout(parts_group)
        
        self.project_parts_table = QtWidgets.QTableWidget()
        self.project_parts_table.setColumnCount(7)
        self.project_parts_table.setHorizontalHeaderLabels([
            "Part Name", "Quantity", "Volume (mm³)", "Max Dim (mm)", "Unit Price", "Total Price", "Remove"
        ])
        self.project_parts_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.project_parts_table.itemClicked.connect(self.on_project_part_clicked)
        parts_layout.addWidget(self.project_parts_table)
        
        left_layout.addWidget(parts_group)
        
        # Project Totals (moved to left panel)
        totals_group = QtWidgets.QGroupBox("Project Totals")
        totals_layout = QtWidgets.QFormLayout(totals_group)
        
        self.total_parts_label = QtWidgets.QLabel("0")
        totals_layout.addRow("Total Parts:", self.total_parts_label)
        
        self.total_volume_label = QtWidgets.QLabel("0.00 mm³")
        totals_layout.addRow("Total Volume:", self.total_volume_label)
        
        self.total_surface_area_label = QtWidgets.QLabel("0.00 mm²")
        totals_layout.addRow("Total Surface Area:", self.total_surface_area_label)
        
        self.total_bb_volume_label = QtWidgets.QLabel("0.00 mm³")
        totals_layout.addRow("Total BB Volume:", self.total_bb_volume_label)
        
        self.total_convex_hull_label = QtWidgets.QLabel("0.00 mm³")
        totals_layout.addRow("Total Convex Hull:", self.total_convex_hull_label)
        
        self.total_shrinkwrap_label = QtWidgets.QLabel("0.00 mm³")
        totals_layout.addRow("Total Shrinkwrap:", self.total_shrinkwrap_label)
        
        self.total_price_label = QtWidgets.QLabel("€0.00")
        self.total_price_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #4CAF50;")
        totals_layout.addRow("Total Project Price:", self.total_price_label)
        
        left_layout.addWidget(totals_group)
        main_splitter.addWidget(left_panel)
        
        # Right panel - 3D Viewer and part details
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # 3D Viewer (main focus on right side)
        viewer_group = QtWidgets.QGroupBox("3D Preview")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_group)
        
        # Create a VTK widget for STL viewing in project tab
        try:
            from ...viewer import STLViewer
            self.project_stl_viewer = STLViewer(self)
            viewer_layout.addWidget(self.project_stl_viewer)
        except ImportError:
            try:
                from stl_analyzer.viewer import STLViewer
                self.project_stl_viewer = STLViewer(self)
                viewer_layout.addWidget(self.project_stl_viewer)
            except ImportError:
                try:
                    from viewer import STLViewer
                    self.project_stl_viewer = STLViewer(self)
                    viewer_layout.addWidget(self.project_stl_viewer)
                except ImportError:
                    # Fallback if STLViewer not available
                    self.project_stl_viewer = None
                    fallback_label = QtWidgets.QLabel("3D Viewer not available\nSTLViewer module not found")
                    fallback_label.setAlignment(QtCore.Qt.AlignCenter)
                    fallback_label.setStyleSheet("color: #888; font-size: 14px;")
                    viewer_layout.addWidget(fallback_label)
        
        # Only add controls if we have a working viewer
        if self.project_stl_viewer:
            
            # Add viewer controls
            controls_layout = QtWidgets.QHBoxLayout()
            
            self.project_toggle_edges_btn = QtWidgets.QPushButton("Edges")
            self.project_toggle_edges_btn.setCheckable(True)
            self.project_toggle_edges_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_edges_visible(checked))
            controls_layout.addWidget(self.project_toggle_edges_btn)
            
            self.project_toggle_axes_btn = QtWidgets.QPushButton("Axes")
            self.project_toggle_axes_btn.setCheckable(True)
            self.project_toggle_axes_btn.setChecked(True)
            self.project_toggle_axes_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_axes_visible(checked))
            controls_layout.addWidget(self.project_toggle_axes_btn)
            
            self.project_toggle_bb_btn = QtWidgets.QPushButton("Bounding Box")
            self.project_toggle_bb_btn.setCheckable(True)
            self.project_toggle_bb_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_bounding_box_visible(checked))
            controls_layout.addWidget(self.project_toggle_bb_btn)
            
            self.project_toggle_ch_btn = QtWidgets.QPushButton("Convex Hull")
            self.project_toggle_ch_btn.setCheckable(True)
            self.project_toggle_ch_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_convex_hull_visible(checked))
            controls_layout.addWidget(self.project_toggle_ch_btn)
            
            self.project_toggle_sw_btn = QtWidgets.QPushButton("Shrinkwrap")
            self.project_toggle_sw_btn.setCheckable(True)
            self.project_toggle_sw_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_shrinkwrap_visible(checked))
            controls_layout.addWidget(self.project_toggle_sw_btn)
            
            self.project_toggle_bvh_btn = QtWidgets.QPushButton("8-Leaf BVH")
            self.project_toggle_bvh_btn.setCheckable(True)
            self.project_toggle_bvh_btn.clicked.connect(lambda checked: self.project_stl_viewer.set_bvh_8leaf_visible(checked))
            self.project_toggle_bvh_btn.setToolTip("Show 8-leaf Bounding Volume Hierarchy for collision detection")
            controls_layout.addWidget(self.project_toggle_bvh_btn)
            
            controls_layout.addStretch()
            viewer_layout.addLayout(controls_layout)
        
        right_layout.addWidget(viewer_group)
        
        # Model Analysis & Actions
        analysis_group = QtWidgets.QGroupBox("Model Analysis & Actions")
        analysis_layout = QtWidgets.QVBoxLayout(analysis_group)
        
        # Optimization buttons
        optimization_layout = QtWidgets.QHBoxLayout()
        
        self.optimize_bbox_btn = QtWidgets.QPushButton("Optimize Bounding Box")
        self.optimize_bbox_btn.setToolTip("Find optimal orientation to minimize bounding box volume")
        self.optimize_bbox_btn.clicked.connect(self.optimize_selected_part_bbox)
        self.optimize_bbox_btn.setEnabled(False)
        optimization_layout.addWidget(self.optimize_bbox_btn)
        
        self.recalc_shrinkwrap_btn = QtWidgets.QPushButton("Recalc Shrinkwrap")
        self.recalc_shrinkwrap_btn.setToolTip("Recalculate shrinkwrap volume for better accuracy")
        self.recalc_shrinkwrap_btn.clicked.connect(self.recalculate_selected_shrinkwrap)
        self.recalc_shrinkwrap_btn.setEnabled(False)
        optimization_layout.addWidget(self.recalc_shrinkwrap_btn)
        
        analysis_layout.addLayout(optimization_layout)
        right_layout.addWidget(analysis_group)
        
        # Detailed Part Information
        details_group = QtWidgets.QGroupBox("Selected Part Details")
        details_layout = QtWidgets.QFormLayout(details_group)
        
        # Filename
        self.project_detail_filename = QtWidgets.QLabel("-")
        self.project_detail_filename.setWordWrap(True)
        self.project_detail_filename.setStyleSheet("font-weight: bold; color: #4CAF50;")
        details_layout.addRow("Filename:", self.project_detail_filename)
        
        # Dimensions with better formatting
        dims_layout = QtWidgets.QHBoxLayout()
        self.project_detail_dim_x = QtWidgets.QLabel("-")
        self.project_detail_dim_y = QtWidgets.QLabel("-")
        self.project_detail_dim_z = QtWidgets.QLabel("-")
        dims_layout.addWidget(QtWidgets.QLabel("X:"))
        dims_layout.addWidget(self.project_detail_dim_x)
        dims_layout.addWidget(QtWidgets.QLabel("Y:"))
        dims_layout.addWidget(self.project_detail_dim_y)
        dims_layout.addWidget(QtWidgets.QLabel("Z:"))
        dims_layout.addWidget(self.project_detail_dim_z)
        dims_layout.addStretch()
        details_layout.addRow("Dimensions (mm):", dims_layout)
        
        # Volume information
        self.project_detail_volume = QtWidgets.QLabel("-")
        details_layout.addRow("Volume:", self.project_detail_volume)
        
        self.project_detail_surface = QtWidgets.QLabel("-")
        details_layout.addRow("Surface Area:", self.project_detail_surface)
        
        self.project_detail_bb_volume = QtWidgets.QLabel("-")
        details_layout.addRow("Bounding Box Vol:", self.project_detail_bb_volume)
        
        self.project_detail_waste = QtWidgets.QLabel("-")
        details_layout.addRow("Waste (BB-Vol):", self.project_detail_waste)
        
        # Advanced volumes
        self.project_detail_convex_hull_volume = QtWidgets.QLabel("-")
        details_layout.addRow("Convex Hull Vol:", self.project_detail_convex_hull_volume)
        
        self.project_detail_convexity_ratio = QtWidgets.QLabel("-")
        details_layout.addRow("Convexity Ratio:", self.project_detail_convexity_ratio)
        
        self.project_detail_shrinkwrap_volume = QtWidgets.QLabel("-")
        details_layout.addRow("Shrinkwrap Vol:", self.project_detail_shrinkwrap_volume)
        
        self.project_detail_shrinkwrap_ratio = QtWidgets.QLabel("-")
        details_layout.addRow("Shrinkwrap Ratio:", self.project_detail_shrinkwrap_ratio)
        
        # Surface to volume ratio
        self.project_detail_sa_vol_ratio = QtWidgets.QLabel("-")
        details_layout.addRow("SA/Vol Ratio:", self.project_detail_sa_vol_ratio)
        
        # Pricing information
        price_layout = QtWidgets.QHBoxLayout()
        self.project_detail_quantity = QtWidgets.QLabel("-")
        self.project_detail_unit_price = QtWidgets.QLabel("-")
        self.project_detail_total_price = QtWidgets.QLabel("-")
        price_layout.addWidget(QtWidgets.QLabel("Qty:"))
        price_layout.addWidget(self.project_detail_quantity)
        price_layout.addWidget(QtWidgets.QLabel("Unit:"))
        price_layout.addWidget(self.project_detail_unit_price)
        price_layout.addWidget(QtWidgets.QLabel("Total:"))
        price_layout.addWidget(self.project_detail_total_price)
        price_layout.addStretch()
        details_layout.addRow("Pricing:", price_layout)
        
        # Project discount info
        self.project_detail_discount = QtWidgets.QLabel("-")
        details_layout.addRow("Project Discount:", self.project_detail_discount)
        
        right_layout.addWidget(details_group)
        main_splitter.addWidget(right_panel)
        
        # Set 50/50 split
        main_splitter.setSizes([500, 500])
        
        # Initialize project parts list
        self.project_parts = []
        
    # Delegate methods to parent window
    def upload_project_parts(self):
        """Upload STL files to project"""
        if self.parent_window and hasattr(self.parent_window, 'upload_project_parts'):
            return self.parent_window.upload_project_parts()
        else:
            # Fallback implementation
            file_dialog = QtWidgets.QFileDialog()
            file_paths, _ = file_dialog.getOpenFileNames(
                self, "Select STL Files", "", "STL Files (*.stl);;All Files (*)"
            )
            
            if not file_paths:
                return
            
            QtWidgets.QMessageBox.information(
                self, "Upload Parts", 
                f"Selected {len(file_paths)} files. Feature extraction would be implemented here."
            )
    
    def calculate_all_prices(self):
        if self.parent_window and hasattr(self.parent_window, 'calculate_all_prices'):
            return self.parent_window.calculate_all_prices()
    
    def clear_project(self):
        if self.parent_window and hasattr(self.parent_window, 'clear_project'):
            return self.parent_window.clear_project()
    
    def save_project(self):
        if self.parent_window and hasattr(self.parent_window, 'save_project'):
            return self.parent_window.save_project()
    
    def load_project(self):
        if self.parent_window and hasattr(self.parent_window, 'load_project'):
            return self.parent_window.load_project()
    
    def show_pricing_analysis(self):
        if self.parent_window and hasattr(self.parent_window, 'show_pricing_analysis'):
            return self.parent_window.show_pricing_analysis()
    
    def on_project_part_clicked(self, item):
        if self.parent_window and hasattr(self.parent_window, 'on_project_part_clicked'):
            return self.parent_window.on_project_part_clicked(item)
    
    def optimize_selected_part_bbox(self):
        if self.parent_window and hasattr(self.parent_window, 'optimize_selected_part_bbox'):
            return self.parent_window.optimize_selected_part_bbox()
    
    def recalculate_selected_shrinkwrap(self):
        if self.parent_window and hasattr(self.parent_window, 'recalculate_selected_shrinkwrap'):
            return self.parent_window.recalculate_selected_shrinkwrap()
    
    def update_project_parts_table(self):
        """Update the project parts table display"""
        if self.parent_window and hasattr(self.parent_window, 'update_project_parts_display'):
            return self.parent_window.update_project_parts_display()
    
    def update_project_totals(self):
        """Update the project totals display"""
        if self.parent_window and hasattr(self.parent_window, 'update_project_totals_display'):
            return self.parent_window.update_project_totals_display() 