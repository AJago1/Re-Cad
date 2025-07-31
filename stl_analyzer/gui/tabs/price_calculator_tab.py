"""
Price Calculator Tab Component

This tab handles:
- Individual part pricing calculations  
- AI model pricing
- Cost breakdown analysis
- 3D visualization with pricing features
- Project summary and part management
"""

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

class PriceCalculatorTab(QtWidgets.QWidget):
    """Price Calculator tab widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Store reference to main window
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the price calculator tab layout based on original working structure"""
        # Create main layout
        price_calculator_layout = QtWidgets.QVBoxLayout(self)
        
        # Create top button toolbar
        top_button_layout = QtWidgets.QHBoxLayout()
        
        # Left side buttons
        left_buttons_layout = QtWidgets.QHBoxLayout()
        
        # Add Part button
        self.add_part_button = QtWidgets.QPushButton("Add Part")
        self.add_part_button.clicked.connect(self.add_price_calc_part)
        left_buttons_layout.addWidget(self.add_part_button)
        
        # Remove Part button
        self.remove_part_button = QtWidgets.QPushButton("Remove Part")
        self.remove_part_button.clicked.connect(self.remove_price_calc_part)
        self.remove_part_button.setEnabled(False)  # Disabled until a part is selected
        left_buttons_layout.addWidget(self.remove_part_button)
        
        # Add spacer to separate the groups
        left_buttons_layout.addStretch()
        
        # Right side buttons
        right_buttons_layout = QtWidgets.QHBoxLayout()
        
        # Configure Presets button
        self.configure_presets_button = QtWidgets.QPushButton("Configure Presets")
        self.configure_presets_button.clicked.connect(self.show_pricing_presets_dialog)
        right_buttons_layout.addWidget(self.configure_presets_button)
        
        # Load Project button
        self.load_project_button = QtWidgets.QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.load_price_project)
        right_buttons_layout.addWidget(self.load_project_button)
        
        # Recalculate All button
        self.recalculate_all_button = QtWidgets.QPushButton("Recalculate All")
        self.recalculate_all_button.clicked.connect(self.recalculate_all_parts)
        self.recalculate_all_button.setEnabled(False)  # Disabled until parts are added
        right_buttons_layout.addWidget(self.recalculate_all_button)
        
        # Save Project button
        self.save_to_project_button = QtWidgets.QPushButton("Save Project")
        self.save_to_project_button.clicked.connect(self.save_price_to_project)
        self.save_to_project_button.setEnabled(False)  # Disabled until calculations are done
        right_buttons_layout.addWidget(self.save_to_project_button)
        
        # Combine button layouts
        top_button_layout.addLayout(left_buttons_layout)
        top_button_layout.addStretch()
        top_button_layout.addLayout(right_buttons_layout)
        
        price_calculator_layout.addLayout(top_button_layout)
        
        # Create main content splitter (horizontal) - 3 panels like original
        self.price_calc_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        price_calculator_layout.addWidget(self.price_calc_splitter)
        
        # LEFT PANEL - Part List and Project Summary
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Part List group
        part_list_group = QtWidgets.QGroupBox("Parts in Project")
        part_list_layout = QtWidgets.QVBoxLayout(part_list_group)
        
        # Parts list widget
        self.parts_list = QtWidgets.QListWidget()
        self.parts_list.currentItemChanged.connect(self.on_part_selection_changed)
        part_list_layout.addWidget(self.parts_list)
        
        # Set item checkable and connect to event
        self.parts_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        left_layout.addWidget(part_list_group, 2)  # 2:1 ratio with project summary
        
        # Project Summary
        project_summary_group = QtWidgets.QGroupBox("Project Summary")
        project_summary_layout = QtWidgets.QVBoxLayout(project_summary_group)
        
        # Project details in a table
        self.project_summary_table = QtWidgets.QTableWidget()
        self.project_summary_table.setColumnCount(3)
        self.project_summary_table.setHorizontalHeaderLabels(["Item", "Quantity", "Price"])
        self.project_summary_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.project_summary_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  # Read-only
        project_summary_layout.addWidget(self.project_summary_table)
        
        # Total project price
        total_layout = QtWidgets.QHBoxLayout()
        total_layout.addWidget(QtWidgets.QLabel("Total Project Price:"))
        self.total_project_price_label = QtWidgets.QLabel("€0.00")
        self.total_project_price_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        total_layout.addWidget(self.total_project_price_label)
        project_summary_layout.addLayout(total_layout)
        
        left_layout.addWidget(project_summary_group, 1)  # 2:1 ratio with part list
        
        self.price_calc_splitter.addWidget(left_panel)
        
        # MIDDLE PANEL - Part Details and Pricing Settings
        middle_panel = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_panel)
        
        # Selected Part Details
        self.part_details_group = QtWidgets.QGroupBox("Selected Part Details")
        part_details_layout = QtWidgets.QFormLayout(self.part_details_group)
        
        # Part detail fields
        self.part_name_label = QtWidgets.QLabel("")
        part_details_layout.addRow("Name:", self.part_name_label)
        
        self.part_volume_label = QtWidgets.QLabel("")
        part_details_layout.addRow("Volume:", self.part_volume_label)
        
        self.part_surface_area_label = QtWidgets.QLabel("")
        part_details_layout.addRow("Surface Area:", self.part_surface_area_label)
        
        self.part_bb_volume_label = QtWidgets.QLabel("")
        part_details_layout.addRow("BB Volume:", self.part_bb_volume_label)
        
        # Quantity selector
        self.part_quantity_spinner = QtWidgets.QSpinBox()
        self.part_quantity_spinner.setMinimum(1)
        self.part_quantity_spinner.setMaximum(10000)
        self.part_quantity_spinner.setValue(1)
        self.part_quantity_spinner.valueChanged.connect(self.on_quantity_changed)
        part_details_layout.addRow("Quantity:", self.part_quantity_spinner)
        
        middle_layout.addWidget(self.part_details_group)
        
        # Pricing Settings
        pricing_group = QtWidgets.QGroupBox("Pricing Settings")
        pricing_layout = QtWidgets.QFormLayout(pricing_group)
        
        # Pricing preset combo
        self.pricing_preset_combo = QtWidgets.QComboBox()
        self.pricing_preset_combo.addItems(["Default", "Premium", "Economy"])
        pricing_layout.addRow("Preset:", self.pricing_preset_combo)
        
        # Use AI Model checkbox
        self.use_ai_model_checkbox = QtWidgets.QCheckBox("Use AI Model")
        self.use_ai_model_checkbox.setChecked(True)
        pricing_layout.addRow("AI Pricing:", self.use_ai_model_checkbox)
        
        middle_layout.addWidget(pricing_group)
        
        # Cost Breakdown
        cost_breakdown_group = QtWidgets.QGroupBox("Cost Breakdown")
        cost_breakdown_layout = QtWidgets.QFormLayout(cost_breakdown_group)
        
        self.material_cost_label = QtWidgets.QLabel("€0.00")
        cost_breakdown_layout.addRow("Material Cost:", self.material_cost_label)
        
        self.machine_cost_label = QtWidgets.QLabel("€0.00")
        cost_breakdown_layout.addRow("Machine Cost:", self.machine_cost_label)
        
        self.labor_cost_label = QtWidgets.QLabel("€0.00")
        cost_breakdown_layout.addRow("Labor Cost:", self.labor_cost_label)
        
        self.base_total_label = QtWidgets.QLabel("€0.00")
        cost_breakdown_layout.addRow("Base Total:", self.base_total_label)
        
        self.final_price_label = QtWidgets.QLabel("€0.00")
        self.final_price_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
        cost_breakdown_layout.addRow("Final Price:", self.final_price_label)
        
        middle_layout.addWidget(cost_breakdown_group)
        
        # Bottom buttons
        bottom_buttons_layout = QtWidgets.QHBoxLayout()
        
        # Recalculate button
        self.recalculate_button = QtWidgets.QPushButton("Recalculate")
        self.recalculate_button.clicked.connect(self.recalculate_price)
        bottom_buttons_layout.addWidget(self.recalculate_button)
        
        # Add to Project button
        self.add_to_project_button = QtWidgets.QPushButton("Add to Project")
        self.add_to_project_button.clicked.connect(self.save_price_to_project)
        self.add_to_project_button.setEnabled(False)  # Disabled until a valid calculation is made
        bottom_buttons_layout.addWidget(self.add_to_project_button)
        
        middle_layout.addLayout(bottom_buttons_layout)
        
        # Add middle panel to the main splitter
        self.price_calc_splitter.addWidget(middle_panel)
        
        # RIGHT PANEL - 3D Viewer (consistent with other tabs)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Create 3D viewer group
        viewer_group = QtWidgets.QGroupBox("3D Model Preview")
        viewer_inner_layout = QtWidgets.QVBoxLayout(viewer_group)
        
        # Create 3D viewer - Import with proper fallback
        try:
            from ...viewer import STLViewer
            self.price_calc_viewer = STLViewer(self)
            viewer_inner_layout.addWidget(self.price_calc_viewer)
        except ImportError:
            try:
                from stl_analyzer.viewer import STLViewer
                self.price_calc_viewer = STLViewer(self)
                viewer_inner_layout.addWidget(self.price_calc_viewer)
            except ImportError:
                try:
                    from viewer import STLViewer
                    self.price_calc_viewer = STLViewer(self)
                    viewer_inner_layout.addWidget(self.price_calc_viewer)
                except ImportError:
                    # Fallback if STLViewer not available
                    self.price_calc_viewer = QtWidgets.QLabel("3D Viewer not available")
                    self.price_calc_viewer.setAlignment(QtCore.Qt.AlignCenter)
                    self.price_calc_viewer.setMinimumHeight(400)
                    self.price_calc_viewer.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
                    viewer_inner_layout.addWidget(self.price_calc_viewer)
        
        # Add view controls 
        view_controls_layout = QtWidgets.QHBoxLayout()
        
        # Reset View button
        reset_view_button = QtWidgets.QPushButton("Reset View")
        if hasattr(self.price_calc_viewer, 'reset_camera_position'):
            reset_view_button.clicked.connect(self.price_calc_viewer.reset_camera_position)
        view_controls_layout.addWidget(reset_view_button)
        
        # Grid checkbox
        self.price_calc_grid_checkbox = QtWidgets.QCheckBox("Grid")
        self.price_calc_grid_checkbox.setChecked(True)
        if hasattr(self.price_calc_viewer, 'set_grid_visible'):
            self.price_calc_grid_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_grid_visible(state == Qt.Checked)
            )
        view_controls_layout.addWidget(self.price_calc_grid_checkbox)
        
        # Axes checkbox
        self.price_calc_axes_checkbox = QtWidgets.QCheckBox("Axes")
        self.price_calc_axes_checkbox.setChecked(True)
        if hasattr(self.price_calc_viewer, 'set_axes_visible'):
            self.price_calc_axes_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_axes_visible(state == Qt.Checked)
            )
        view_controls_layout.addWidget(self.price_calc_axes_checkbox)
        
        viewer_inner_layout.addLayout(view_controls_layout)
        right_layout.addWidget(viewer_group)
        
        # Add right panel to splitter
        self.price_calc_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (25% left, 35% middle, 40% right - 3D viewer gets good space)
        self.price_calc_splitter.setSizes([250, 350, 400])
    
    # Delegate methods to parent window
    def add_price_calc_part(self):
        if self.parent_window and hasattr(self.parent_window, 'add_price_calc_part'):
            return self.parent_window.add_price_calc_part()
    
    def remove_price_calc_part(self):
        if self.parent_window and hasattr(self.parent_window, 'remove_price_calc_part'):
            return self.parent_window.remove_price_calc_part()
    
    def on_part_selection_changed(self, current, previous):
        if self.parent_window and hasattr(self.parent_window, 'on_part_selection_changed'):
            return self.parent_window.on_part_selection_changed(current, previous)
    
    def on_quantity_changed(self, value):
        if self.parent_window and hasattr(self.parent_window, 'on_quantity_changed'):
            return self.parent_window.on_quantity_changed(value)
    
    def show_pricing_presets_dialog(self):
        if self.parent_window and hasattr(self.parent_window, 'show_pricing_presets_dialog'):
            return self.parent_window.show_pricing_presets_dialog()
    
    def load_price_project(self):
        if self.parent_window and hasattr(self.parent_window, 'load_price_project'):
            return self.parent_window.load_price_project()
    
    def recalculate_all_parts(self):
        if self.parent_window and hasattr(self.parent_window, 'recalculate_all_parts'):
            return self.parent_window.recalculate_all_parts()
    
    def save_price_to_project(self):
        if self.parent_window and hasattr(self.parent_window, 'save_price_to_project'):
            return self.parent_window.save_price_to_project()
    
    def recalculate_price(self):
        if self.parent_window and hasattr(self.parent_window, 'recalculate_price'):
            return self.parent_window.recalculate_price() 