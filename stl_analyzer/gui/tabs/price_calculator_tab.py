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
        
        left_layout.addWidget(part_list_group)  # Take full space
        
        # Simple Total Price at bottom (no wasteful table)
        total_layout = QtWidgets.QHBoxLayout()
        total_layout.addWidget(QtWidgets.QLabel("Total:"))
        self.total_project_price_label = QtWidgets.QLabel("€0.00")
        self.total_project_price_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2E7D32;")
        total_layout.addWidget(self.total_project_price_label)
        total_layout.addStretch()
        left_layout.addLayout(total_layout)
        
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
        
        # Printer Type dropdown (Phase 1: SLS Foundation)
        self.printer_type_combo = QtWidgets.QComboBox()
        self.printer_type_combo.addItems(["General AI", "MJF", "SLS"])
        self.printer_type_combo.currentTextChanged.connect(self.on_printer_type_changed)
        pricing_layout.addRow("Printer Type:", self.printer_type_combo)
        
        # Pricing preset combo (will be updated based on printer type)
        self.pricing_preset_combo = QtWidgets.QComboBox()
        self.pricing_preset_combo.addItems(["Default", "Premium", "Economy"])
        pricing_layout.addRow("Preset:", self.pricing_preset_combo)
        
        # Use AI Model checkbox
        self.use_ai_model_checkbox = QtWidgets.QCheckBox("Use AI Model")
        self.use_ai_model_checkbox.setChecked(True)
        pricing_layout.addRow("AI Pricing:", self.use_ai_model_checkbox)
        
        # C++ Optimization status (shown for SLS mode)
        self.cpp_optimization_label = QtWidgets.QLabel("Not available")
        self.cpp_optimization_label.setVisible(False)
        pricing_layout.addRow("C++ Optimization:", self.cpp_optimization_label)
        
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
        
        self.estimated_time_label = QtWidgets.QLabel("0.0 h")
        cost_breakdown_layout.addRow("Estimated Time:", self.estimated_time_label)
        
        self.base_total_label = QtWidgets.QLabel("€0.00")
        cost_breakdown_layout.addRow("Base Total:", self.base_total_label)
        
        self.final_price_label = QtWidgets.QLabel("€0.00")
        self.final_price_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
        cost_breakdown_layout.addRow("Final Price:", self.final_price_label)
        
        middle_layout.addWidget(cost_breakdown_group)
        
        # Pack3D group - compact horizontal organization
        pack3d_group = QtWidgets.QGroupBox("🚀 Pack3D Optimization")
        pack3d_main_layout = QtWidgets.QVBoxLayout(pack3d_group)
        pack3d_main_layout.setSpacing(5)  # Tighter spacing
        
        # Top row: Status, Printer, and Action buttons
        top_row = QtWidgets.QHBoxLayout()
        
        # Status (compact)
        self.pack3d_status_label = QtWidgets.QLabel("Ready")
        self.pack3d_status_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11px;")
        top_row.addWidget(QtWidgets.QLabel("Status:"))
        top_row.addWidget(self.pack3d_status_label)
        
        # Printer selection (compact)
        top_row.addWidget(QtWidgets.QLabel(" | Printer:"))
        self.pack3d_printer_combo = QtWidgets.QComboBox()
        self.pack3d_printer_combo.addItems(["EOS P396", "EOS P110"])
        self.pack3d_printer_combo.setMaximumWidth(100)
        top_row.addWidget(self.pack3d_printer_combo)
        
        top_row.addStretch()
        pack3d_main_layout.addLayout(top_row)
        
        # Pack3D Parameters - compact 2-column layout
        params_grid = QtWidgets.QGridLayout()
        
        # Row 1: Iterations and Temperature
        params_grid.addWidget(QtWidgets.QLabel("Iterations:"), 0, 0)
        self.pack3d_max_iterations_spin = QtWidgets.QSpinBox()
        self.pack3d_max_iterations_spin.setRange(100, 50000)
        self.pack3d_max_iterations_spin.setValue(10000)
        self.pack3d_max_iterations_spin.setMaximumWidth(80)
        params_grid.addWidget(self.pack3d_max_iterations_spin, 0, 1)
        
        params_grid.addWidget(QtWidgets.QLabel("Temp:"), 0, 2)
        self.pack3d_initial_temp_spin = QtWidgets.QDoubleSpinBox()
        self.pack3d_initial_temp_spin.setRange(100.0, 20000.0)
        self.pack3d_initial_temp_spin.setValue(1000.0)  # Reasonable starting temperature
        self.pack3d_initial_temp_spin.setMaximumWidth(80)
        self.pack3d_initial_temp_spin.setSuffix("°")
        params_grid.addWidget(self.pack3d_initial_temp_spin, 0, 3)
        
        # Row 2: Cooling and Min Temp
        params_grid.addWidget(QtWidgets.QLabel("Cooling:"), 1, 0)
        self.pack3d_cooling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.pack3d_cooling_rate_spin.setRange(0.9990, 0.99999)
        self.pack3d_cooling_rate_spin.setValue(0.99999)  # Ultra-slow cooling for natural convergence for long runs
        self.pack3d_cooling_rate_spin.setDecimals(4)
        self.pack3d_cooling_rate_spin.setMaximumWidth(80)
        params_grid.addWidget(self.pack3d_cooling_rate_spin, 1, 1)
        
        params_grid.addWidget(QtWidgets.QLabel("Min:"), 1, 2)
        self.pack3d_min_temp_spin = QtWidgets.QDoubleSpinBox()
        self.pack3d_min_temp_spin.setRange(0.0001, 1.0)
        self.pack3d_min_temp_spin.setValue(0.001)  # Much lower for proper fine-tuning
        self.pack3d_min_temp_spin.setDecimals(3)
        self.pack3d_min_temp_spin.setSuffix("°")
        self.pack3d_min_temp_spin.setMaximumWidth(80)
        params_grid.addWidget(self.pack3d_min_temp_spin, 1, 3)
        
        # Presets row - compact buttons
        presets_layout = QtWidgets.QHBoxLayout()
        
        fast_preset_btn = QtWidgets.QPushButton("🚀")
        fast_preset_btn.clicked.connect(self.set_fast_preset)
        fast_preset_btn.setToolTip("Fast: 1000 iter")
        fast_preset_btn.setMaximumWidth(35)
        presets_layout.addWidget(fast_preset_btn)
        
        balanced_preset_btn = QtWidgets.QPushButton("⚖️")
        balanced_preset_btn.clicked.connect(self.set_balanced_preset)
        balanced_preset_btn.setToolTip("Balanced: 5000 iter")
        balanced_preset_btn.setMaximumWidth(35)
        presets_layout.addWidget(balanced_preset_btn)
        
        thorough_preset_btn = QtWidgets.QPushButton("🎯")
        thorough_preset_btn.clicked.connect(self.set_thorough_preset)
        thorough_preset_btn.setToolTip("Thorough: 20000 iter")
        thorough_preset_btn.setMaximumWidth(35)
        presets_layout.addWidget(thorough_preset_btn)
        
        marathon_preset_btn = QtWidgets.QPushButton("🏃")
        marathon_preset_btn.clicked.connect(self.set_marathon_preset)
        marathon_preset_btn.setToolTip("Marathon: 50k iter, 5+ min runs")
        marathon_preset_btn.setMaximumWidth(35)
        presets_layout.addWidget(marathon_preset_btn)
        
        presets_layout.addStretch()
        params_grid.addLayout(presets_layout, 2, 0, 1, 4)
        
        pack3d_main_layout.addLayout(params_grid)
        
        # Pack3D Quantities - very compact
        quantities_header = QtWidgets.QHBoxLayout()
        quantities_header.addWidget(QtWidgets.QLabel("📦 Quantities:"))
        
        self.refresh_pack3d_quantities_button = QtWidgets.QPushButton("🔄")
        self.refresh_pack3d_quantities_button.clicked.connect(self.refresh_pack3d_quantities)
        self.refresh_pack3d_quantities_button.setToolTip("Refresh parts")
        self.refresh_pack3d_quantities_button.setMaximumWidth(30)
        quantities_header.addWidget(self.refresh_pack3d_quantities_button)
        quantities_header.addStretch()
        pack3d_main_layout.addLayout(quantities_header)
        
        self.pack3d_quantities_table = QtWidgets.QTableWidget()
        self.pack3d_quantities_table.setColumnCount(2)
        self.pack3d_quantities_table.setHorizontalHeaderLabels(["Part", "Qty"])
        self.pack3d_quantities_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.pack3d_quantities_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.pack3d_quantities_table.setMaximumHeight(60)  # Very compact
        self.pack3d_quantities_table.verticalHeader().setVisible(False)  # Hide row numbers
        pack3d_main_layout.addWidget(self.pack3d_quantities_table)
        
        # Pack3D controls and results - all in one compact row
        controls_results_layout = QtWidgets.QHBoxLayout()
        
        # Left side: Action buttons
        buttons_layout = QtWidgets.QVBoxLayout()
        
        run_clear_layout = QtWidgets.QHBoxLayout()
        self.run_pack3d_button = QtWidgets.QPushButton("🚀 Run")
        self.run_pack3d_button.clicked.connect(self.run_pack3d_optimization)
        self.run_pack3d_button.setMaximumWidth(60)
        run_clear_layout.addWidget(self.run_pack3d_button)
        
        self.stop_pack3d_button = QtWidgets.QPushButton("🛑 Stop")
        self.stop_pack3d_button.clicked.connect(self.stop_pack3d_optimization)
        self.stop_pack3d_button.setEnabled(False)
        self.stop_pack3d_button.setMaximumWidth(60)
        run_clear_layout.addWidget(self.stop_pack3d_button)
        
        self.clear_pack3d_button = QtWidgets.QPushButton("🗑️")
        self.clear_pack3d_button.clicked.connect(self.clear_pack3d_results)
        self.clear_pack3d_button.setEnabled(False)
        self.clear_pack3d_button.setMaximumWidth(35)
        run_clear_layout.addWidget(self.clear_pack3d_button)
        
        self.export_pack3d_button = QtWidgets.QPushButton("📄")
        self.export_pack3d_button.clicked.connect(self.export_pack3d_results)
        self.export_pack3d_button.setEnabled(False)
        self.export_pack3d_button.setMaximumWidth(35)
        run_clear_layout.addWidget(self.export_pack3d_button)
        
        buttons_layout.addLayout(run_clear_layout)
        controls_results_layout.addLayout(buttons_layout)
        
        # Right side: Results in compact grid
        results_grid = QtWidgets.QGridLayout()
        
        results_grid.addWidget(QtWidgets.QLabel("Height:"), 0, 0)
        self.pack3d_build_height_label = QtWidgets.QLabel("N/A")
        self.pack3d_build_height_label.setStyleSheet("font-size: 11px;")
        results_grid.addWidget(self.pack3d_build_height_label, 0, 1)
        
        results_grid.addWidget(QtWidgets.QLabel("Util:"), 0, 2)
        self.pack3d_utilization_label = QtWidgets.QLabel("N/A")
        self.pack3d_utilization_label.setStyleSheet("font-size: 11px;")
        results_grid.addWidget(self.pack3d_utilization_label, 0, 3)
        
        results_grid.addWidget(QtWidgets.QLabel("Iter:"), 1, 0)
        self.pack3d_iterations_label = QtWidgets.QLabel("N/A")
        self.pack3d_iterations_label.setStyleSheet("font-size: 11px;")
        results_grid.addWidget(self.pack3d_iterations_label, 1, 1)
        
        results_grid.addWidget(QtWidgets.QLabel("Time:"), 1, 2)
        self.pack3d_time_label = QtWidgets.QLabel("N/A")
        self.pack3d_time_label.setStyleSheet("font-size: 11px;")
        results_grid.addWidget(self.pack3d_time_label, 1, 3)
        
        controls_results_layout.addLayout(results_grid)
        controls_results_layout.addStretch()
        
        pack3d_main_layout.addLayout(controls_results_layout)
        
        middle_layout.addWidget(pack3d_group)
        
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
        
        # Add second row of view controls for visualization modes
        viz_modes_layout = QtWidgets.QHBoxLayout()
        
        # Bounding Box checkbox
        self.price_calc_bbox_checkbox = QtWidgets.QCheckBox("Bounding Box")
        if hasattr(self.price_calc_viewer, 'set_bounding_box_visible'):
            self.price_calc_bbox_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_bounding_box_visible(state == Qt.Checked)
            )
        viz_modes_layout.addWidget(self.price_calc_bbox_checkbox)
        
        # Convex Hull checkbox
        self.price_calc_convex_hull_checkbox = QtWidgets.QCheckBox("Convex Hull")
        if hasattr(self.price_calc_viewer, 'set_convex_hull_visible'):
            self.price_calc_convex_hull_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_convex_hull_visible(state == Qt.Checked)
            )
        viz_modes_layout.addWidget(self.price_calc_convex_hull_checkbox)
        
        # Shrinkwrap checkbox
        self.price_calc_shrinkwrap_checkbox = QtWidgets.QCheckBox("Shrinkwrap")
        if hasattr(self.price_calc_viewer, 'set_shrinkwrap_visible'):
            self.price_calc_shrinkwrap_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_shrinkwrap_visible(state == Qt.Checked)
            )
        viz_modes_layout.addWidget(self.price_calc_shrinkwrap_checkbox)
        
        # 8-Leaf BVH checkbox
        self.price_calc_bvh_checkbox = QtWidgets.QCheckBox("8-Leaf BVH")
        if hasattr(self.price_calc_viewer, 'set_bvh_8leaf_visible'):
            self.price_calc_bvh_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_bvh_8leaf_visible(state == Qt.Checked)
            )
        self.price_calc_bvh_checkbox.setToolTip("Show 8-leaf Bounding Volume Hierarchy for pack3d collision detection")
        viz_modes_layout.addWidget(self.price_calc_bvh_checkbox)
        
        # BVH Solid/Wireframe toggle
        self.price_calc_bvh_solid_checkbox = QtWidgets.QCheckBox("BVH Solid")
        if hasattr(self.price_calc_viewer, 'set_bvh_8leaf_solid'):
            self.price_calc_bvh_solid_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_bvh_8leaf_solid(state == Qt.Checked)
            )
        self.price_calc_bvh_solid_checkbox.setToolTip("Show BVH boxes as solid (easier to see coverage) or wireframe")
        viz_modes_layout.addWidget(self.price_calc_bvh_solid_checkbox)
        
        # Pack3D Results checkbox
        self.price_calc_pack3d_checkbox = QtWidgets.QCheckBox("Pack3D Results")
        if hasattr(self.price_calc_viewer, 'set_pack3d_visible'):
            self.price_calc_pack3d_checkbox.stateChanged.connect(
                lambda state: self.price_calc_viewer.set_pack3d_visible(state == Qt.Checked)
            )
        self.price_calc_pack3d_checkbox.setToolTip("Show Pack3D optimization results with colored parts")
        viz_modes_layout.addWidget(self.price_calc_pack3d_checkbox)
        
        viewer_inner_layout.addLayout(viz_modes_layout)
        right_layout.addWidget(viewer_group)
        
        # Add right panel to splitter
        self.price_calc_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (25% left, 45% middle, 30% right - More balanced without wasteful sections)
        self.price_calc_splitter.setSizes([250, 450, 300])
    
    # Delegate methods to parent window
    def add_price_calc_part(self):
        if self.parent_window and hasattr(self.parent_window, 'add_price_calc_part'):
            return self.parent_window.add_price_calc_part()
    
    def remove_price_calc_part(self):
        if self.parent_window and hasattr(self.parent_window, 'remove_price_calc_part'):
            return self.parent_window.remove_price_calc_part()
    
    def on_part_selection_changed(self, current, previous):
        # Enable/disable remove button based on selection
        has_selection = current is not None
        self.remove_part_button.setEnabled(has_selection)
        
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
    
    def update_button_states(self):
        """Update button states based on current parts list"""
        has_parts = self.parts_list.count() > 0
        has_selection = self.parts_list.currentItem() is not None
        
        # Enable/disable buttons based on state
        self.remove_part_button.setEnabled(has_selection)
        self.recalculate_all_button.setEnabled(has_parts)
        self.save_to_project_button.setEnabled(has_parts)
    
    def save_price_to_project(self):
        if self.parent_window and hasattr(self.parent_window, 'save_price_to_project'):
            return self.parent_window.save_price_to_project()
    
    def recalculate_price(self):
        if self.parent_window and hasattr(self.parent_window, 'recalculate_price'):
            return self.parent_window.recalculate_price()
    
    def on_printer_type_changed(self, printer_type):
        """Handle printer type change"""
        if self.parent_window and hasattr(self.parent_window, 'on_printer_type_changed'):
            return self.parent_window.on_printer_type_changed(printer_type)
    
    # Pack3D methods
    def run_pack3d_optimization(self):
        """Run Pack3D optimization"""
        if self.parent_window and hasattr(self.parent_window, 'run_pack3d_optimization'):
            return self.parent_window.run_pack3d_optimization()
    
    def stop_pack3d_optimization(self):
        """Stop Pack3D optimization"""
        if self.parent_window and hasattr(self.parent_window, 'stop_pack3d_optimization'):
            return self.parent_window.stop_pack3d_optimization()
    
    def clear_pack3d_results(self):
        """Clear Pack3D results"""
        if self.parent_window and hasattr(self.parent_window, 'clear_pack3d_results'):
            return self.parent_window.clear_pack3d_results()
    
    def export_pack3d_results(self):
        """Export Pack3D results to JSON"""
        if self.parent_window and hasattr(self.parent_window, 'export_pack3d_results'):
            return self.parent_window.export_pack3d_results()
    
    def refresh_pack3d_quantities(self):
        """Refresh Pack3D quantities table with current parts"""
        if self.parent_window and hasattr(self.parent_window, 'refresh_pack3d_quantities'):
            return self.parent_window.refresh_pack3d_quantities()
    
    def get_pack3d_parameters(self):
        """Get Pack3D optimization parameters from UI"""
        return {
            'max_iterations': self.pack3d_max_iterations_spin.value(),
            'initial_temperature': self.pack3d_initial_temp_spin.value(),
            'cooling_rate': self.pack3d_cooling_rate_spin.value(),
            'min_temperature': self.pack3d_min_temp_spin.value()
        }
    
    def get_pack3d_quantities(self):
        """Get Pack3D part quantities from the table"""
        quantities = {}
        for row in range(self.pack3d_quantities_table.rowCount()):
            part_name_item = self.pack3d_quantities_table.item(row, 0)
            quantity_widget = self.pack3d_quantities_table.cellWidget(row, 1)
            
            if part_name_item and quantity_widget:
                part_name = part_name_item.text()
                quantity = quantity_widget.value()
                quantities[part_name] = quantity
        
        return quantities
    
    # Preset methods for Pack3D parameters
    def set_fast_preset(self):
        """Set fast optimization preset"""
        self.pack3d_max_iterations_spin.setValue(5000)
        self.pack3d_initial_temp_spin.setValue(1000.0)  # Reasonable start temp
        self.pack3d_cooling_rate_spin.setValue(0.9995)  # Faster cooling
        self.pack3d_min_temp_spin.setValue(0.001)
        print("🚀 Fast preset applied: 5000 iterations, faster cooling")
    
    def set_balanced_preset(self):
        """Set balanced optimization preset (TRUE FOGLEMAN)"""
        self.pack3d_max_iterations_spin.setValue(10000)
        self.pack3d_initial_temp_spin.setValue(1000.0)
        self.pack3d_cooling_rate_spin.setValue(0.9999)  # Slow cooling for natural convergence
        self.pack3d_min_temp_spin.setValue(0.001)
        print("⚖️ Balanced preset applied: TRUE Fogleman parameters with natural convergence")
    
    def set_quality_preset(self):
        """Set high-quality optimization preset"""
        self.pack3d_max_iterations_spin.setValue(20000)
        self.pack3d_initial_temp_spin.setValue(1000.0)  # Reasonable start temp
        self.pack3d_cooling_rate_spin.setValue(0.99995)  # Very slow cooling
        self.pack3d_min_temp_spin.setValue(0.001)  # Very fine tuning
        print("💎 Quality preset applied: 20k iterations, very slow cooling")
    
    def set_marathon_preset(self):
        """Set marathon optimization preset for 5+ minute runs"""
        self.pack3d_max_iterations_spin.setValue(100000)
        self.pack3d_initial_temp_spin.setValue(1000.0)  # Start reasonable, rely on ultra-slow cooling
        self.pack3d_cooling_rate_spin.setValue(0.99999)  # Ultra-slow cooling for natural convergence
        self.pack3d_min_temp_spin.setValue(0.001)
        print("🏃 Marathon preset applied: 100k iterations, ultra-slow cooling for natural convergence")
    
    def set_thorough_preset(self):
        """Set thorough optimization preset"""
        self.pack3d_max_iterations_spin.setValue(20000)
        self.pack3d_initial_temp_spin.setValue(1000.0)
        self.pack3d_cooling_rate_spin.setValue(0.99995)  # Very slow cooling
        self.pack3d_min_temp_spin.setValue(0.001)
        print("🎯 Thorough preset applied: 20k iterations, very slow cooling") 