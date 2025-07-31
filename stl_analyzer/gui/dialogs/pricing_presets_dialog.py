"""
Pricing Presets Configuration Dialog

This dialog provides complete configuration for pricing presets including:
- Material settings (type, price, density, etc.)
- Machine settings (build volume, cost, energy, etc.)
- Process settings (heating, cooling, complexity, etc.)
- Labor settings (rates, setup time, monitoring, etc.)
- Pricing settings (margins, fees, minimums, etc.)
"""

import copy
from PyQt5 import QtWidgets, QtCore


class PricingPresetsDialog(QtWidgets.QDialog):
    """Dialog for configuring pricing presets"""
    
    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self.presets = copy.deepcopy(presets)  # Work with a copy
        self.current_preset = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Configure Pricing Presets")
        self.resize(800, 600)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Create splitter for presets list and settings
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Presets list
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Presets list widget
        self.presets_list = QtWidgets.QListWidget()
        self.presets_list.currentItemChanged.connect(self.on_preset_selection_changed)
        
        # Add presets to list
        for preset_name in self.presets.keys():
            self.presets_list.addItem(preset_name)
            
        left_layout.addWidget(self.presets_list)
        
        # Preset buttons
        preset_buttons_layout = QtWidgets.QHBoxLayout()
        
        # Add preset button
        add_button = QtWidgets.QPushButton("Add")
        add_button.clicked.connect(self.add_preset)
        preset_buttons_layout.addWidget(add_button)
        
        # Duplicate preset button
        duplicate_button = QtWidgets.QPushButton("Duplicate")
        duplicate_button.clicked.connect(self.duplicate_preset)
        preset_buttons_layout.addWidget(duplicate_button)
        
        # Delete preset button
        delete_button = QtWidgets.QPushButton("Delete")
        delete_button.clicked.connect(self.delete_preset)
        preset_buttons_layout.addWidget(delete_button)
        
        left_layout.addLayout(preset_buttons_layout)
        
        # Right panel - Preset settings
        right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Preset settings scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(scroll_area)
        
        # Settings widget
        self.settings_widget = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout(self.settings_widget)
        scroll_area.setWidget(self.settings_widget)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
        
        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Select first preset if available
        if self.presets_list.count() > 0:
            self.presets_list.setCurrentRow(0)
        else:
            self.create_settings_ui(None)
    
    def create_settings_ui(self, preset_name):
        """Create settings UI for the selected preset"""
        # Clear current settings
        while self.settings_layout.count():
            item = self.settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not preset_name or preset_name not in self.presets:
            # No preset selected
            label = QtWidgets.QLabel("No preset selected")
            self.settings_layout.addWidget(label)
            return
            
        # Get preset data
        preset = self.presets[preset_name]
        self.current_preset = preset_name
        
        # Create form layout for settings
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form_widget)
        self.settings_layout.addWidget(form_widget)
        
        # Preset name
        name_input = QtWidgets.QLineEdit(preset.get("name", ""))
        name_input.textChanged.connect(lambda text: self.update_preset_value("name", text))
        form_layout.addRow("Preset Name:", name_input)
        
        # Add groups for different setting categories
        self.settings_layout.addWidget(self.create_material_settings_group(preset))
        self.settings_layout.addWidget(self.create_machine_settings_group(preset))
        self.settings_layout.addWidget(self.create_process_settings_group(preset))
        self.settings_layout.addWidget(self.create_labor_settings_group(preset))
        self.settings_layout.addWidget(self.create_pricing_settings_group(preset))
        
        # Add stretch at the end
        self.settings_layout.addStretch()
    
    def create_material_settings_group(self, preset):
        """Create material settings group"""
        group = QtWidgets.QGroupBox("Material Settings")
        layout = QtWidgets.QFormLayout(group)
        
        # Material dropdown
        material_combo = QtWidgets.QComboBox()
        material_combo.addItems(["PA12", "PA11", "TPU", "PEEK", "PLA", "ABS", "Custom"])
        material_combo.setCurrentText(preset.get("material", "PA12"))
        material_combo.currentTextChanged.connect(
            lambda text: self.update_preset_value("material", text)
        )
        layout.addRow("Material:", material_combo)
        
        # Material price per kg
        material_price = QtWidgets.QDoubleSpinBox()
        material_price.setRange(1.0, 1000.0)
        material_price.setValue(preset.get("material_price", 85.0))
        material_price.setSingleStep(5.0)
        material_price.setPrefix("€")
        material_price.setSuffix("/kg")
        material_price.valueChanged.connect(
            lambda value: self.update_preset_value("material_price", value)
        )
        layout.addRow("Material Price:", material_price)
        
        # Material density
        density = QtWidgets.QDoubleSpinBox()
        density.setRange(0.5, 5.0)
        density.setValue(preset.get("material_density", 1.05))
        density.setSingleStep(0.01)
        density.setSuffix(" g/cm³")
        density.valueChanged.connect(
            lambda value: self.update_preset_value("material_density", value)
        )
        layout.addRow("Density:", density)
        
        # Material reuse ratio
        reuse_ratio = QtWidgets.QDoubleSpinBox()
        reuse_ratio.setRange(0.0, 1.0)
        reuse_ratio.setValue(preset.get("material_reuse_ratio", 0.5))
        reuse_ratio.setSingleStep(0.1)
        reuse_ratio.setSuffix(" (50%)")
        reuse_ratio.valueChanged.connect(
            lambda value: self.update_preset_value("material_reuse_ratio", value)
        )
        layout.addRow("Reuse Ratio:", reuse_ratio)
        
        # Waste during cleaning
        waste_cleaning = QtWidgets.QDoubleSpinBox()
        waste_cleaning.setRange(0.0, 1000.0)
        waste_cleaning.setValue(preset.get("waste_during_cleaning", 300.0))
        waste_cleaning.setSingleStep(50.0)
        waste_cleaning.setSuffix(" g")
        waste_cleaning.valueChanged.connect(
            lambda value: self.update_preset_value("waste_during_cleaning", value)
        )
        layout.addRow("Waste During Cleaning:", waste_cleaning)
        
        return group
    
    def create_machine_settings_group(self, preset):
        """Create machine settings group"""
        group = QtWidgets.QGroupBox("Machine Settings")
        layout = QtWidgets.QFormLayout(group)
        
        # Build volume dimensions
        build_volume_x = QtWidgets.QDoubleSpinBox()
        build_volume_x.setRange(100.0, 1000.0)
        build_volume_x.setValue(preset.get("build_volume_x", 380.0))
        build_volume_x.setSingleStep(10.0)
        build_volume_x.setSuffix(" mm")
        build_volume_x.valueChanged.connect(
            lambda value: self.update_preset_value("build_volume_x", value)
        )
        layout.addRow("Build Volume X:", build_volume_x)
        
        build_volume_y = QtWidgets.QDoubleSpinBox()
        build_volume_y.setRange(100.0, 1000.0)
        build_volume_y.setValue(preset.get("build_volume_y", 380.0))
        build_volume_y.setSingleStep(10.0)
        build_volume_y.setSuffix(" mm")
        build_volume_y.valueChanged.connect(
            lambda value: self.update_preset_value("build_volume_y", value)
        )
        layout.addRow("Build Volume Y:", build_volume_y)
        
        build_volume_z = QtWidgets.QDoubleSpinBox()
        build_volume_z.setRange(100.0, 1000.0)
        build_volume_z.setValue(preset.get("build_volume_z", 600.0))
        build_volume_z.setSingleStep(10.0)
        build_volume_z.setSuffix(" mm")
        build_volume_z.valueChanged.connect(
            lambda value: self.update_preset_value("build_volume_z", value)
        )
        layout.addRow("Build Volume Z:", build_volume_z)
        
        # Z-axis offsets
        start_z_offset = QtWidgets.QDoubleSpinBox()
        start_z_offset.setRange(0.0, 50.0)
        start_z_offset.setValue(preset.get("start_z_offset", 15.0))
        start_z_offset.setSingleStep(1.0)
        start_z_offset.setSuffix(" mm")
        start_z_offset.valueChanged.connect(
            lambda value: self.update_preset_value("start_z_offset", value)
        )
        layout.addRow("Start Z Offset:", start_z_offset)
        
        end_z_offset = QtWidgets.QDoubleSpinBox()
        end_z_offset.setRange(0.0, 50.0)
        end_z_offset.setValue(preset.get("end_z_offset", 15.0))
        end_z_offset.setSingleStep(1.0)
        end_z_offset.setSuffix(" mm")
        end_z_offset.valueChanged.connect(
            lambda value: self.update_preset_value("end_z_offset", value)
        )
        layout.addRow("End Z Offset:", end_z_offset)
        
        # Machine cost details
        machine_investment = QtWidgets.QDoubleSpinBox()
        machine_investment.setRange(10000.0, 1000000.0)
        machine_investment.setValue(preset.get("machine_investment", 200000.0))
        machine_investment.setSingleStep(10000.0)
        machine_investment.setPrefix("€")
        machine_investment.valueChanged.connect(
            lambda value: self.update_preset_value("machine_investment", value)
        )
        layout.addRow("Machine Investment:", machine_investment)
        
        machine_amort_years = QtWidgets.QSpinBox()
        machine_amort_years.setRange(1, 15)
        machine_amort_years.setValue(preset.get("machine_amort_years", 5))
        machine_amort_years.setSingleStep(1)
        machine_amort_years.setSuffix(" years")
        machine_amort_years.valueChanged.connect(
            lambda value: self.update_preset_value("machine_amort_years", value)
        )
        layout.addRow("Amortization Period:", machine_amort_years)
        
        annual_maintenance = QtWidgets.QDoubleSpinBox()
        annual_maintenance.setRange(0.0, 50000.0)
        annual_maintenance.setValue(preset.get("annual_maintenance", 15000.0))
        annual_maintenance.setSingleStep(1000.0)
        annual_maintenance.setPrefix("€")
        annual_maintenance.valueChanged.connect(
            lambda value: self.update_preset_value("annual_maintenance", value)
        )
        layout.addRow("Annual Maintenance:", annual_maintenance)
        
        weekly_machine_hours = QtWidgets.QDoubleSpinBox()
        weekly_machine_hours.setRange(1.0, 168.0)
        weekly_machine_hours.setValue(preset.get("weekly_machine_hours", 70.0))
        weekly_machine_hours.setSingleStep(5.0)
        weekly_machine_hours.setSuffix(" h")
        weekly_machine_hours.valueChanged.connect(
            lambda value: self.update_preset_value("weekly_machine_hours", value)
        )
        layout.addRow("Weekly Hours:", weekly_machine_hours)
        
        working_weeks_per_year = QtWidgets.QSpinBox()
        working_weeks_per_year.setRange(1, 52)
        working_weeks_per_year.setValue(preset.get("working_weeks_per_year", 48))
        working_weeks_per_year.setSingleStep(1)
        working_weeks_per_year.setSuffix(" weeks")
        working_weeks_per_year.valueChanged.connect(
            lambda value: self.update_preset_value("working_weeks_per_year", value)
        )
        layout.addRow("Working Weeks/Year:", working_weeks_per_year)
        
        # Build speed
        build_speed = QtWidgets.QDoubleSpinBox()
        build_speed.setRange(1.0, 50.0)
        build_speed.setValue(preset.get("build_speed", 15.0))
        build_speed.setSingleStep(1.0)
        build_speed.setSuffix(" mm/h")
        build_speed.valueChanged.connect(
            lambda value: self.update_preset_value("build_speed", value)
        )
        layout.addRow("Build Speed:", build_speed)
        
        # Energy settings
        energy_usage_per_hour = QtWidgets.QDoubleSpinBox()
        energy_usage_per_hour.setRange(0.1, 20.0)
        energy_usage_per_hour.setValue(preset.get("energy_usage_per_hour", 5.0))
        energy_usage_per_hour.setSingleStep(0.5)
        energy_usage_per_hour.setSuffix(" kWh")
        energy_usage_per_hour.valueChanged.connect(
            lambda value: self.update_preset_value("energy_usage_per_hour", value)
        )
        layout.addRow("Power Consumption:", energy_usage_per_hour)
        
        energy_cost = QtWidgets.QDoubleSpinBox()
        energy_cost.setRange(0.05, 1.0)
        energy_cost.setValue(preset.get("energy_cost", 0.15))
        energy_cost.setSingleStep(0.01)
        energy_cost.setPrefix("€")
        energy_cost.setSuffix("/kWh")
        energy_cost.valueChanged.connect(
            lambda value: self.update_preset_value("energy_cost", value)
        )
        layout.addRow("Energy Cost:", energy_cost)
        
        return group
    
    def create_process_settings_group(self, preset):
        """Create process settings group"""
        group = QtWidgets.QGroupBox("Process Settings")
        layout = QtWidgets.QFormLayout(group)
        
        # Heating time
        heating_time = QtWidgets.QDoubleSpinBox()
        heating_time.setRange(0.0, 10.0)
        heating_time.setValue(preset.get("heating_time", 2.5))
        heating_time.setSingleStep(0.25)
        heating_time.setSuffix(" h")
        heating_time.valueChanged.connect(
            lambda value: self.update_preset_value("heating_time", value)
        )
        layout.addRow("Heating Time:", heating_time)
        
        # Cooling time
        cooling_time = QtWidgets.QDoubleSpinBox()
        cooling_time.setRange(0.0, 10.0)
        cooling_time.setValue(preset.get("cooling_time", 3.0))
        cooling_time.setSingleStep(0.25)
        cooling_time.setSuffix(" h")
        cooling_time.valueChanged.connect(
            lambda value: self.update_preset_value("cooling_time", value)
        )
        layout.addRow("Cooling Time:", cooling_time)
        
        # Machine cleaning time
        cleaning_time = QtWidgets.QDoubleSpinBox()
        cleaning_time.setRange(0.0, 5.0)
        cleaning_time.setValue(preset.get("cleaning_time", 1.0))
        cleaning_time.setSingleStep(0.25)
        cleaning_time.setSuffix(" h")
        cleaning_time.valueChanged.connect(
            lambda value: self.update_preset_value("cleaning_time", value)
        )
        layout.addRow("Machine Cleaning Time:", cleaning_time)
        
        # Complexity weight
        complexity_weight = QtWidgets.QDoubleSpinBox()
        complexity_weight.setRange(0.0, 1.0)
        complexity_weight.setValue(preset.get("complexity_weight", 0.35))
        complexity_weight.setSingleStep(0.05)
        complexity_weight.setToolTip("How much part complexity affects print time (0-1)")
        complexity_weight.valueChanged.connect(
            lambda value: self.update_preset_value("complexity_weight", value)
        )
        layout.addRow("Complexity Weight:", complexity_weight)
        
        return group
    
    def create_labor_settings_group(self, preset):
        """Create labor settings group"""
        group = QtWidgets.QGroupBox("Labor Settings")
        layout = QtWidgets.QFormLayout(group)
        
        # Labor rate
        labor_cost = QtWidgets.QDoubleSpinBox()
        labor_cost.setRange(5.0, 100.0)
        labor_cost.setValue(preset.get("labor_cost", 25.0))
        labor_cost.setSingleStep(1.0)
        labor_cost.setPrefix("€")
        labor_cost.setSuffix("/h")
        labor_cost.valueChanged.connect(
            lambda value: self.update_preset_value("labor_cost", value)
        )
        layout.addRow("Labor Rate:", labor_cost)
        
        # Setup time
        setup_time = QtWidgets.QDoubleSpinBox()
        setup_time.setRange(0.0, 5.0)
        setup_time.setValue(preset.get("setup_time", 0.5))
        setup_time.setSingleStep(0.1)
        setup_time.setSuffix(" h")
        setup_time.valueChanged.connect(
            lambda value: self.update_preset_value("setup_time", value)
        )
        layout.addRow("Setup Time:", setup_time)
        
        # Monitoring time ratio
        monitoring_time = QtWidgets.QDoubleSpinBox()
        monitoring_time.setRange(0.0, 0.5)
        monitoring_time.setValue(preset.get("monitoring_time", 0.1))
        monitoring_time.setSingleStep(0.01)
        monitoring_time.setToolTip("Fraction of build time spent monitoring")
        monitoring_time.valueChanged.connect(
            lambda value: self.update_preset_value("monitoring_time", value)
        )
        layout.addRow("Monitoring Time:", monitoring_time)
        
        # Post-processing time
        post_processing_time = QtWidgets.QDoubleSpinBox()
        post_processing_time.setRange(0.0, 5.0)
        post_processing_time.setValue(preset.get("post_processing_time", 0.5))
        post_processing_time.setSingleStep(0.1)
        post_processing_time.setSuffix(" h")
        post_processing_time.valueChanged.connect(
            lambda value: self.update_preset_value("post_processing_time", value)
        )
        layout.addRow("Post-processing Time:", post_processing_time)
        
        # Packaging time
        packaging_time = QtWidgets.QDoubleSpinBox()
        packaging_time.setRange(0.0, 2.0)
        packaging_time.setValue(preset.get("packaging_time", 0.2))
        packaging_time.setSingleStep(0.1)
        packaging_time.setSuffix(" h")
        packaging_time.valueChanged.connect(
            lambda value: self.update_preset_value("packaging_time", value)
        )
        layout.addRow("Packaging Time:", packaging_time)
        
        return group
    
    def create_pricing_settings_group(self, preset):
        """Create pricing settings group"""
        group = QtWidgets.QGroupBox("Pricing Settings")
        layout = QtWidgets.QFormLayout(group)
        
        # Maintenance buffer
        maintenance_buffer = QtWidgets.QDoubleSpinBox()
        maintenance_buffer.setRange(0.0, 50.0)
        maintenance_buffer.setValue(preset.get("maintenance_buffer", 1.0))
        maintenance_buffer.setSingleStep(0.5)
        maintenance_buffer.setPrefix("€")
        maintenance_buffer.valueChanged.connect(
            lambda value: self.update_preset_value("maintenance_buffer", value)
        )
        layout.addRow("Maintenance Buffer:", maintenance_buffer)
        
        # Margin percentage
        margin = QtWidgets.QDoubleSpinBox()
        margin.setRange(0.0, 100.0)
        margin.setValue(preset.get("margin", 15.0))
        margin.setSingleStep(5.0)
        margin.setSuffix("%")
        margin.valueChanged.connect(
            lambda value: self.update_preset_value("margin", value)
        )
        layout.addRow("Margin:", margin)
        
        # Setup fee
        setup_fee = QtWidgets.QDoubleSpinBox()
        setup_fee.setRange(0.0, 100.0)
        setup_fee.setValue(preset.get("setup_fee", 0.0))
        setup_fee.setSingleStep(5.0)
        setup_fee.setPrefix("€")
        setup_fee.valueChanged.connect(
            lambda value: self.update_preset_value("setup_fee", value)
        )
        layout.addRow("Setup Fee:", setup_fee)
        
        # Minimum order value
        min_order = QtWidgets.QDoubleSpinBox()
        min_order.setRange(0.0, 200.0)
        min_order.setValue(preset.get("minimum_order", 0.0))
        min_order.setSingleStep(5.0)
        min_order.setPrefix("€")
        min_order.valueChanged.connect(
            lambda value: self.update_preset_value("minimum_order", value)
        )
        layout.addRow("Minimum Order:", min_order)
        
        return group
    
    def update_preset_value(self, key, value):
        """Update a value in the current preset"""
        if not self.current_preset:
            return
            
        self.presets[self.current_preset][key] = value
        
        # If name was changed, update list item text
        if key == "name" and value != self.current_preset:
            # Only update if name is different and not empty
            if value and value != self.current_preset:
                # Check if name already exists
                existing_items = self.presets_list.findItems(value, QtCore.Qt.MatchExactly)
                if existing_items:
                    # Name already exists, don't update
                    return
                    
                # Update preset key name
                preset_data = self.presets.pop(self.current_preset)
                self.presets[value] = preset_data
                
                # Update list item text
                current_item = self.presets_list.currentItem()
                if current_item:
                    current_item.setText(value)
                    
                # Update current preset name
                self.current_preset = value
    
    def on_preset_selection_changed(self, current, previous):
        """Handle preset selection change in the list"""
        if current:
            self.create_settings_ui(current.text())
    
    def add_preset(self):
        """Add a new preset"""
        # Get a unique default name
        base_name = "New Preset"
        name = base_name
        counter = 1
        
        while name in self.presets:
            name = f"{base_name} {counter}"
            counter += 1
            
        # Create new preset with default values
        self.presets[name] = {
            "name": name,
            "material": "PA12",
            "material_price": 85.0,
            "material_density": 1.05,
            "material_reuse_ratio": 0.5,
            "waste_during_cleaning": 300.0,
            "build_volume_x": 380.0,
            "build_volume_y": 380.0,
            "build_volume_z": 600.0,
            "start_z_offset": 15.0,
            "end_z_offset": 15.0,
            "machine_investment": 200000.0,
            "machine_amort_years": 5,
            "annual_maintenance": 15000.0,
            "weekly_machine_hours": 70.0,
            "working_weeks_per_year": 48,
            "build_speed": 15.0,
            "energy_usage_per_hour": 5.0,
            "energy_cost": 0.15,
            "heating_time": 2.5,
            "cooling_time": 3.0,
            "cleaning_time": 1.0,
            "complexity_weight": 0.35,
            "labor_cost": 25.0,
            "setup_time": 0.5,
            "monitoring_time": 0.1,
            "post_processing_time": 0.5,
            "packaging_time": 0.2,
            "maintenance_buffer": 1.0,
            "margin": 15.0,
            "setup_fee": 0.0,
            "minimum_order": 0.0
        }
        
        # Add to list and select
        item = QtWidgets.QListWidgetItem(name)
        self.presets_list.addItem(item)
        self.presets_list.setCurrentItem(item)
    
    def duplicate_preset(self):
        """Duplicate the current preset"""
        if not self.current_preset:
            return
            
        # Get a unique name based on current preset
        base_name = f"{self.current_preset} (Copy)"
        name = base_name
        counter = 1
        
        while name in self.presets:
            name = f"{base_name} {counter}"
            counter += 1
            
        # Create duplicate preset
        self.presets[name] = copy.deepcopy(self.presets[self.current_preset])
        self.presets[name]["name"] = name
        
        # Add to list and select
        item = QtWidgets.QListWidgetItem(name)
        self.presets_list.addItem(item)
        self.presets_list.setCurrentItem(item)
    
    def delete_preset(self):
        """Delete the current preset"""
        if not self.current_preset:
            return
            
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self, 
            'Delete Preset',
            f'Are you sure you want to delete preset "{self.current_preset}"?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.No:
            return
            
        # Don't allow deleting the last preset
        if self.presets_list.count() <= 1:
            QtWidgets.QMessageBox.warning(
                self, 
                'Cannot Delete',
                'Cannot delete the last preset.'
            )
            return
            
        # Remove from dictionary and list
        del self.presets[self.current_preset]
        
        # Remove from list widget and select another item
        row = self.presets_list.currentRow()
        self.presets_list.takeItem(row)
        
        # Select next available row
        if row >= self.presets_list.count():
            row = self.presets_list.count() - 1
            
        if row >= 0:
            self.presets_list.setCurrentRow(row)
        else:
            self.current_preset = None
            self.create_settings_ui(None)
    
    def get_presets(self):
        """Get the modified presets"""
        return self.presets 