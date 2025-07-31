"""
Main window for the STL Analyzer application
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import subprocess
import tempfile
import datetime
import json
import copy
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import pickle
import joblib

# Import our modules - handle both package and direct imports
try:
    # When imported as a package
    from ..stl_utils import extract_features, extract_features_with_shrinkwrap, is_valid_entry, find_similar, try_load_stl
    from ..database import STLDatabase
    from ..viewer import STLViewer
    from ..comprehensive_pricing_model import ComprehensivePricingModel, PricingModelManager
    from .threads import ScanThread, ProcessThread, StepConversionThread
except ImportError:
    # When run directly
    from stl_utils import extract_features, extract_features_with_shrinkwrap, is_valid_entry, find_similar, try_load_stl
    from database import STLDatabase
    from viewer import STLViewer
    from comprehensive_pricing_model import ComprehensivePricingModel, PricingModelManager

# Import modular components with fallbacks
try:
    from .tabs import AnalyzerTab, ConverterTab, PriceCalculatorTab, ProjectPricingTab
except ImportError:
    # Create placeholder classes if tabs not available
    class AnalyzerTab(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel("Analyzer tab loading..."))
    
    class ConverterTab(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel("Converter tab loading..."))
    
    class PriceCalculatorTab(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel("Price Calculator tab loading..."))
    
    class ProjectPricingTab(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel("Project Pricing tab loading..."))

# Add import for QTabWidget
from PyQt5.QtWidgets import QTabWidget

class PandasModel(QtCore.QAbstractTableModel):
    """Model for pandas DataFrame display in QTableView with improved formatting"""
    def __init__(self, data):
        super().__init__()
        self._data = data
        
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._data.columns)
    
    def data(self, index, role):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            
            # Handle none/nan values
            if pd.isna(value):
                return ""
                
            # By default just return as string - formatting is handled by delegates
            return str(value)
            
        elif role == Qt.TextAlignmentRole:
            # Center align filenames, right-align numbers
            col_name = self._data.columns[index.column()]
            if col_name == "filename":
                return Qt.AlignLeft | Qt.AlignVCenter
            else:
                return Qt.AlignRight | Qt.AlignVCenter
                
        elif role == Qt.FontRole:
            # Use slightly smaller font for numeric columns
            font = QtGui.QFont()
            col_name = self._data.columns[index.column()]
            if col_name != "filename":
                font.setPointSize(font.pointSize() - 1)
                return font
                
        return None
    
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            # Check if we have columns and the section is within bounds
            if len(self._data.columns) > 0 and 0 <= section < len(self._data.columns):
                return str(self._data.columns[section])
            return ""
            
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            # Check if we have rows and the section is within bounds
            if len(self._data) > 0 and 0 <= section < len(self._data):
                return str(section + 1)
            return ""
            
        return None
        
    def get_column_index(self, column_name):
        """Get the index of a column by name"""
        try:
            return self._data.columns.get_loc(column_name)
        except:
            return -1

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Setup UI components
        self.setWindowTitle("STL Analyzer")
        self.resize(1200, 800)
        
        # Initialize database first
        self.stl_db = STLDatabase()
        
        # Initialize variables for scanning
        self.found_files = []
        self.selected_files = []
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create tab widget for main pages
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Setup pages using modular structure
        self.setup_analyzer_tab()
        self.setup_converter_tab()
        self.setup_price_calculator_tab()
        self.setup_project_pricing_tab()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Show welcome message
        self.show_status_message("Welcome to STL Analyzer")
        
        # Initialize threads
        self.scan_thread = None
        self.process_thread = None
        self.conversion_thread = None
        
        # Initialize project pricing variables
        self.project_parts = []  # List of parts in current project
        self.pricing_model = None  # Linear regression model for pricing
        self.csv_data = None  # Training data from CSV
        self.project_aware_model = True  # Enable project-aware model by default
        
        # Initialize price calculator data structures
        self.price_calc_parts = {}  # Dictionary for price calculator parts
        self.current_pricing_config = {}  # Current pricing configuration
        self.pricing_presets = {}  # Pricing presets storage
        
        # Load existing database and update view
        self.update_database_view()
        
        # Initialize comprehensive pricing model manager
        try:
            self.pricing_model_manager = PricingModelManager()
            self.comprehensive_pricing_enabled = True
        except:
            self.comprehensive_pricing_enabled = False
        
        # Initialize pricing models in the GUI
        self.refresh_pricing_models()
        
        # Initialize AI pricing model
        self.initialize_ai_pricing_model()
        
        # Show maximized
        self.showMaximized()
        
    def apply_dark_theme(self):
        """Apply a modern dark theme to the application"""
        # Set the style to Fusion which is most customizable
        QtWidgets.QApplication.setStyle("Fusion")
        
        # Dark color palette
        dark_palette = QtGui.QPalette()
        
        # Text colors
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        
        # Button colors
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(65, 65, 65))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        
        # Highlight colors
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        
        # Link colors
        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.LinkVisited, QtGui.QColor(80, 145, 218))
        
        # Apply the palette
        app = QtWidgets.QApplication.instance()
        app.setPalette(dark_palette)
        
        # Set stylesheet for more customization
        app.setStyleSheet("""
            QToolTip { 
                color: #ffffff; 
                background-color: #2a2a2a; 
                border: 1px solid #3daee9; 
                padding: 5px;
                opacity: 200;
            }
            
            QTableView {
                gridline-color: #31363b;
                background-color: #232629;
                selection-background-color: #3daee9;
                selection-color: #eff0f1;
            }
            
            QTableView::item:selected { 
                background: #3daee9; 
            }
            
            QTableView QTableCornerButton::section {
                background-color: #31363b;
                border: 1px solid #76797c;
            }
            
            QHeaderView::section {
                background-color: #31363b;
                color: #eff0f1;
                border: 1px solid #76797c;
                padding: 4px;
            }
            
            QPushButton {
                border: 1px solid #76797c;
                border-radius: 2px;
                background-color: #31363b;
                padding: 5px 15px;
            }
            
            QPushButton:hover {
                background-color: #3daee9;
                border-color: #3daee9;
            }
            
            QPushButton:pressed {
                background-color: #2a82da;
                border-color: #2a82da;
            }
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #232629;
                border: 1px solid #76797c;
                color: #eff0f1;
                padding: 3px;
                border-radius: 2px;
            }
            
            QProgressBar {
                border: 1px solid #76797c;
                border-radius: 2px;
                text-align: center;
                background-color: #232629;
            }
            
            QProgressBar::chunk {
                background-color: #3daee9;
            }
            
            QStatusBar {
                background-color: #31363b;
                color: #eff0f1;
            }
        """)
        
    def setup_analyzer_tab(self):
        """Set up the main STL analyzer tab using modular AnalyzerTab class"""
        try:
            # Use the modular AnalyzerTab class with the new vertical layout
            from .tabs import AnalyzerTab
            self.analyzer_widget = AnalyzerTab(parent=self)
            
            # Connect the analyzer tab's UI elements to the main window
            if hasattr(self.analyzer_widget, 'stl_viewer'):
                self.stl_viewer = self.analyzer_widget.stl_viewer
            if hasattr(self.analyzer_widget, 'db_table'):
                self.db_table = self.analyzer_widget.db_table
            if hasattr(self.analyzer_widget, 'dir_input'):
                self.dir_input = self.analyzer_widget.dir_input
            if hasattr(self.analyzer_widget, 'scan_progress'):
                self.scan_progress = self.analyzer_widget.scan_progress
            if hasattr(self.analyzer_widget, 'stop_scan_button'):
                self.stop_scan_button = self.analyzer_widget.stop_scan_button
                
            # Connect detail labels for updating
            if hasattr(self.analyzer_widget, 'detail_filename'):
                self.detail_filename = self.analyzer_widget.detail_filename
            if hasattr(self.analyzer_widget, 'detail_volume'):
                self.detail_volume = self.analyzer_widget.detail_volume
            if hasattr(self.analyzer_widget, 'detail_surface'):
                self.detail_surface = self.analyzer_widget.detail_surface
            if hasattr(self.analyzer_widget, 'detail_bb_volume'):
                self.detail_bb_volume = self.analyzer_widget.detail_bb_volume
            if hasattr(self.analyzer_widget, 'detail_convex_hull'):
                self.detail_convex_hull = self.analyzer_widget.detail_convex_hull
            if hasattr(self.analyzer_widget, 'detail_shrinkwrap'):
                self.detail_shrinkwrap = self.analyzer_widget.detail_shrinkwrap
            if hasattr(self.analyzer_widget, 'detail_dim_x'):
                self.detail_dim_x = self.analyzer_widget.detail_dim_x
            if hasattr(self.analyzer_widget, 'detail_dim_y'):
                self.detail_dim_y = self.analyzer_widget.detail_dim_y
            if hasattr(self.analyzer_widget, 'detail_dim_z'):
                self.detail_dim_z = self.analyzer_widget.detail_dim_z
            
            print("✅ Using modular AnalyzerTab with UI elements connected to main window")
        except ImportError as e:
            print(f"⚠️ AnalyzerTab import failed: {e}")
            # Fallback to basic implementation
            self.analyzer_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(self.analyzer_widget)
            layout.addWidget(QtWidgets.QLabel("Analyzer tab - using fallback"))
        
        # Add to tab widget
        self.tab_widget.addTab(self.analyzer_widget, "STL Analyzer")
        
    def setup_converter_tab(self):
        """Set up the converter tab using modular components"""
        try:
            self.converter_tab = ConverterTab(parent=self)
            self.tab_widget.addTab(self.converter_tab, "STEP Converter")
        except Exception as e:
            print(f"Error setting up converter tab: {e}")
            # Fallback to a simple placeholder
            placeholder = QtWidgets.QLabel("Converter tab not available")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "STEP Converter")
        
    def setup_price_calculator_tab(self):
        """Set up the price calculator tab using modular components"""
        try:
            self.price_calculator_tab = PriceCalculatorTab(parent=self)
            self.tab_widget.addTab(self.price_calculator_tab, "Price Calculator")
            
            # Map ALL UI elements from price calculator tab for backward compatibility
            self.parts_list = self.price_calculator_tab.parts_list
            self.part_details_group = self.price_calculator_tab.part_details_group
            self.part_quantity_spinner = self.price_calculator_tab.part_quantity_spinner
            self.pricing_preset_combo = self.price_calculator_tab.pricing_preset_combo
            self.use_ai_model_checkbox = self.price_calculator_tab.use_ai_model_checkbox
            
            # Cost breakdown labels
            self.material_cost_label = getattr(self.price_calculator_tab, 'material_cost_label', None)
            self.machine_cost_label = getattr(self.price_calculator_tab, 'machine_cost_label', None)
            self.labor_cost_label = getattr(self.price_calculator_tab, 'labor_cost_label', None)
            self.base_total_label = getattr(self.price_calculator_tab, 'base_total_label', None)
            self.final_price_label = getattr(self.price_calculator_tab, 'final_price_label', None)
            
            # Part detail labels
            self.part_name_label = getattr(self.price_calculator_tab, 'part_name_label', None)
            self.part_volume_label = getattr(self.price_calculator_tab, 'part_volume_label', None)
            self.part_surface_area_label = getattr(self.price_calculator_tab, 'part_surface_area_label', None)
            self.part_bb_volume_label = getattr(self.price_calculator_tab, 'part_bb_volume_label', None)
            
            # Buttons
            self.add_part_button = getattr(self.price_calculator_tab, 'add_part_button', None)
            self.remove_part_button = getattr(self.price_calculator_tab, 'remove_part_button', None)
            self.recalculate_button = getattr(self.price_calculator_tab, 'recalculate_button', None)
            self.save_to_project_button = getattr(self.price_calculator_tab, 'save_to_project_button', None)
            self.add_to_project_button = getattr(self.price_calculator_tab, 'add_to_project_button', None)
            self.recalculate_all_button = getattr(self.price_calculator_tab, 'recalculate_all_button', None)
            
            # 3D Viewer
            self.price_calc_viewer = getattr(self.price_calculator_tab, 'price_calc_viewer', None)
            
            # Project summary elements
            self.project_summary_table = getattr(self.price_calculator_tab, 'project_summary_table', None)
            self.total_project_price_label = getattr(self.price_calculator_tab, 'total_project_price_label', None)
            
            print("✅ Price Calculator tab setup complete with all UI elements mapped")
            
        except Exception as e:
            print(f"Error setting up price calculator tab: {e}")
            # Fallback to a simple placeholder
            placeholder = QtWidgets.QLabel("Price Calculator tab not available")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "Price Calculator")
        
    def setup_project_pricing_tab(self):
        """Set up the project pricing tab using modular components"""
        try:
            self.project_pricing_tab = ProjectPricingTab(parent=self)
            self.tab_widget.addTab(self.project_pricing_tab, "Project Pricing")
            
            # Create references to tab UI elements for backward compatibility
            self.project_parts_table = self.project_pricing_tab.project_parts_table
            self.total_parts_label = self.project_pricing_tab.total_parts_label
            self.total_volume_label = self.project_pricing_tab.total_volume_label
            self.total_surface_area_label = self.project_pricing_tab.total_surface_area_label
            self.total_bb_volume_label = self.project_pricing_tab.total_bb_volume_label
            self.total_convex_hull_label = self.project_pricing_tab.total_convex_hull_label
            self.total_shrinkwrap_label = self.project_pricing_tab.total_shrinkwrap_label
            self.total_price_label = self.project_pricing_tab.total_price_label
            self.calculate_all_button = self.project_pricing_tab.calculate_all_button
            self.save_project_button = self.project_pricing_tab.save_project_button
            self.project_stl_viewer = self.project_pricing_tab.project_stl_viewer
            
            # Map part detail labels from the project pricing tab
            self.selected_part_name_label = self.project_pricing_tab.project_detail_filename
            self.selected_part_volume_label = self.project_pricing_tab.project_detail_volume
            self.selected_part_surface_label = self.project_pricing_tab.project_detail_surface
            self.selected_part_price_label = self.project_pricing_tab.project_detail_total_price
            
            # Map all the comprehensive detail elements
            self.project_detail_filename = self.project_pricing_tab.project_detail_filename
            self.project_detail_dim_x = self.project_pricing_tab.project_detail_dim_x
            self.project_detail_dim_y = self.project_pricing_tab.project_detail_dim_y
            self.project_detail_dim_z = self.project_pricing_tab.project_detail_dim_z
            self.project_detail_volume = self.project_pricing_tab.project_detail_volume
            self.project_detail_surface = self.project_pricing_tab.project_detail_surface
            self.project_detail_bb_volume = self.project_pricing_tab.project_detail_bb_volume
            self.project_detail_waste = self.project_pricing_tab.project_detail_waste
            self.project_detail_convex_hull_volume = self.project_pricing_tab.project_detail_convex_hull_volume
            self.project_detail_convexity_ratio = self.project_pricing_tab.project_detail_convexity_ratio
            self.project_detail_shrinkwrap_volume = self.project_pricing_tab.project_detail_shrinkwrap_volume
            self.project_detail_shrinkwrap_ratio = self.project_pricing_tab.project_detail_shrinkwrap_ratio
            self.project_detail_sa_vol_ratio = self.project_pricing_tab.project_detail_sa_vol_ratio
            self.project_detail_quantity = self.project_pricing_tab.project_detail_quantity
            self.project_detail_unit_price = self.project_pricing_tab.project_detail_unit_price
            self.project_detail_total_price = self.project_pricing_tab.project_detail_total_price
            self.project_detail_discount = self.project_pricing_tab.project_detail_discount
            
            # Map optimization buttons
            self.optimize_bbox_btn = self.project_pricing_tab.optimize_bbox_btn
            self.recalc_shrinkwrap_btn = self.project_pricing_tab.recalc_shrinkwrap_btn
            
        except Exception as e:
            print(f"Error setting up project pricing tab: {e}")
            # Fallback to a simple placeholder
            placeholder = QtWidgets.QLabel("Project Pricing tab not available")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "Project Pricing")
        
    def update_database_view(self):
        """Update the database view with current data"""
        import pandas as pd
        
        if not hasattr(self, 'db_table') or not self.stl_db:
            print(f"⚠️ Cannot update database view - db_table: {hasattr(self, 'db_table')}, stl_db: {bool(self.stl_db)}")
            return
            
        try:
            # Get all entries from database
            df = self.stl_db.get_all_entries()
            print(f"📊 Database has {len(df)} entries")
            
            # If database is empty, show empty table with proper structure
            if df.empty:
                # Create an empty DataFrame with the expected columns
                empty_columns = [
                    'name', 'filename', 'volume', 'surface_area', 
                    'x', 'y', 'z', 'bb_volume', 'waste',
                    'convex_hull_volume', 'convexity_ratio',
                    'shrinkwrap_volume', 'shrinkwrap_ratio',
                    'shrinkwrap_stl_path', 'shrinkwrap_offset_percent'
                ]
                empty_df = pd.DataFrame(columns=empty_columns)
                self.db_model = PandasModel(empty_df)
                self.db_table.setModel(self.db_model)
                print("📋 Set empty database model")
                return
                
            # Select and order columns for display
            columns = [
                'name', 'filename', 'volume', 'surface_area', 
                'x', 'y', 'z', 'bb_volume', 'waste',
                'convex_hull_volume', 'convexity_ratio',
                'shrinkwrap_volume', 'shrinkwrap_ratio'
            ]
            
            # Only keep columns that exist in the dataframe
            columns = [c for c in columns if c in df.columns]
            
            # Ensure we have at least some basic columns
            if not columns:
                # If no expected columns exist, use all available columns
                columns = list(df.columns)
            
            print(f"📋 Displaying columns: {columns}")
            
            # Select columns in desired order
            display_df = df[columns].copy()
            
            # Create new model
            self.db_model = PandasModel(display_df)
            self.db_table.setModel(self.db_model)
            
            # Auto-resize columns for best view
            self.db_table.resizeColumnsToContents()
            
            print(f"✅ Database view updated with {len(df)} entries")
            
        except Exception as e:
            print(f"❌ Error updating database view: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create completely empty model
            empty_df = pd.DataFrame()
            self.db_model = PandasModel(empty_df)
            self.db_table.setModel(self.db_model)
        
    def on_db_table_clicked(self, index):
        """Handle click on database table row - loads 3D model and updates details"""
        if not index.isValid():
            return
            
        try:
            # Get the filename from the selected row
            model = self.db_table.model()
            if not model:
                return
                
            # Find filename column
            filename_col = None
            for col in range(model.columnCount()):
                header = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
                if header and 'filename' in str(header).lower():
                    filename_col = col
                    break
            
            if filename_col is None:
                print("❌ Could not find filename column in database table")
                return
                
            filename = model.data(model.index(index.row(), filename_col), QtCore.Qt.DisplayRole)
            print(f"🔍 Database table clicked: {filename}")
            
            if not filename:
                return
                
            # Check if file exists
            if not os.path.isfile(filename):
                print(f"❌ File not found: {filename}")
                self.statusBar().showMessage(f"File not found: {os.path.basename(filename)}", 3000)
                return
            
            # Get all data for this file from the database
            file_data = self.stl_db.get_entry(filename)
            
            # Find STL viewer in analyzer tab
            stl_viewer = None
            analyzer_tab = None
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    if hasattr(analyzer_tab, 'stl_viewer'):
                        stl_viewer = analyzer_tab.stl_viewer
                    break
            
            # Load file in 3D viewer
            if stl_viewer and hasattr(stl_viewer, 'load_stl'):
                print(f"📊 Loading 3D model: {os.path.basename(filename)}")
                
                # Extract optimal transform if available
                optimal_transform = None
                if file_data and 'optimal_transform' in file_data and file_data['optimal_transform']:
                    try:
                        if isinstance(file_data['optimal_transform'], str):
                            # Convert from JSON string to array
                            import json
                            transform_data = json.loads(file_data['optimal_transform'])
                            if transform_data:
                                optimal_transform = np.array(transform_data)
                                print(f"✅ Using stored optimal transform")
                        elif isinstance(file_data['optimal_transform'], list):
                            # Already a list, convert to numpy array
                            optimal_transform = np.array(file_data['optimal_transform'])
                            print(f"✅ Using stored optimal transform")
            except Exception as e:
                        print(f"⚠️ Error loading optimal transform: {e}")
                        optimal_transform = None
                
                # Load the STL file with optimal transform
                success = stl_viewer.load_stl(filename, optimal_transform)
                
                if success:
                    print(f"✅ 3D model loaded successfully")
                    self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)}", 2000)
                else:
                    print(f"❌ Failed to load 3D model")
                    self.statusBar().showMessage(f"Failed to load: {os.path.basename(filename)}", 3000)
            else:
                print(f"⚠️ STL viewer not found")
            
            # Update details view
            if file_data:
                print(f"📋 Updating details view with database data")
                self.update_details_view(file_data)
            else:
                print(f"⚠️ No database data found for {filename}")
                # Fallback: extract features if no database data
                try:
                    from .stl_utils import extract_features
                    features = extract_features(filename)
                    if features:
                        self.update_details_view(features)
                except Exception as e:
                    print(f"❌ Error extracting features: {e}")
                    
        except Exception as e:
            print(f"❌ Error in database table click handler: {e}")
            import traceback
            traceback.print_exc()

    def update_details_view(self, data):
        """Update the details view with the provided data"""
        try:
            print(f"📊 Updating details view...")
            
            # Find analyzer tab
            analyzer_tab = None
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    break
            
            if not analyzer_tab:
                print("❌ Could not find analyzer tab")
                return
            
            # Check if data is a string (filepath) or dictionary
            if isinstance(data, str):
                # If it's a string, create a basic dictionary with the filename
                file_path = data
                filename = os.path.basename(file_path)
                data = {"filename": filename, "filepath": file_path}
            
            # Update all detail fields if they exist
            detail_fields = {
                'detail_filename': data.get("filename", "-"),
                'detail_volume': f"{data.get('volume', 0):.2f} mm³" if data.get('volume', 0) > 0 else "-",
                'detail_surface': f"{data.get('surface_area', 0):.2f} mm²" if data.get('surface_area', 0) > 0 else "-",
                'detail_bb_volume': f"{data.get('bb_volume', 0):.2f} mm³" if data.get('bb_volume', 0) > 0 else "-",
                'detail_convex_hull': f"{data.get('convex_hull_volume', 0):.2f} mm³" if data.get('convex_hull_volume', 0) > 0 else "-",
                'detail_shrinkwrap': f"{data.get('shrinkwrap_volume', 0):.2f} mm³" if data.get('shrinkwrap_volume', 0) > 0 else "-"
            }
            
            # Set dimensions
            x_dim = data.get("x", 0)
            y_dim = data.get("y", 0)
            z_dim = data.get("z", 0)
            
            if all(isinstance(d, (int, float)) and d > 0 for d in [x_dim, y_dim, z_dim]):
                detail_fields['detail_dim_x'] = f"{x_dim:.2f} mm"
                detail_fields['detail_dim_y'] = f"{y_dim:.2f} mm"
                detail_fields['detail_dim_z'] = f"{z_dim:.2f} mm"
            else:
                detail_fields['detail_dim_x'] = "-"
                detail_fields['detail_dim_y'] = "-"
                detail_fields['detail_dim_z'] = "-"
            
            # Update each field that exists in the analyzer tab
            updated_count = 0
            for field_name, value in detail_fields.items():
                if hasattr(analyzer_tab, field_name):
                    field_widget = getattr(analyzer_tab, field_name)
                    if hasattr(field_widget, 'setText'):
                        field_widget.setText(str(value))
                        updated_count += 1
            
            print(f"✅ Updated {updated_count} detail fields")
            
        except Exception as e:
            print(f"❌ Error updating details view: {e}")
            import traceback
            traceback.print_exc()

    # Shrinkwrap control handlers
    def on_shrinkwrap_output_mode_changed(self, index):
        """Handle shrinkwrap output mode change"""
        print(f"🔧 Shrinkwrap output mode changed to index {index}")
    
    def on_file_generation_toggled(self, checked):
        """Handle file generation toggle"""
        print(f"🔧 File generation toggled: {checked}")
    
    def browse_custom_shrinkwrap_directory(self):
        """Browse for custom shrinkwrap directory"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Custom Shrinkwrap Directory")
        if directory:
            # Find analyzer tab and update shrinkwrap directory input
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    if hasattr(analyzer_tab, 'shrinkwrap_dir_input'):
                        analyzer_tab.shrinkwrap_dir_input.setText(directory)
                        print(f"🔧 Custom shrinkwrap directory set: {directory}")
                    break

    # Similarity search methods
    def find_similar_parts(self):
        """Find parts similar to the currently selected part"""
        try:
            # Get current selection in database table
            if not hasattr(self, 'db_table') or not self.db_table.selectionModel():
                self.statusBar().showMessage("Please select a part first", 3000)
                return
                
            selection = self.db_table.selectionModel().selectedRows()
            if not selection:
                self.statusBar().showMessage("Please select a part first", 3000)
                return
                
            # Get selected part data
            row = selection[0].row()
            model = self.db_table.model()
            if not model:
                return
                
            # Get filename of selected part
            filename_col = None
            for col in range(model.columnCount()):
                header = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
                if header and 'filename' in str(header).lower():
                    filename_col = col
                    break
            
            if filename_col is None:
                self.statusBar().showMessage("Cannot find filename column", 3000)
                return
                
            filename = model.data(model.index(row, filename_col), QtCore.Qt.DisplayRole)
            
            # Get features for this part
            part_data = self.stl_db.get_entry(filename)
            if not part_data:
                self.statusBar().showMessage("Cannot find part data", 3000)
                return
                
            # Get all database entries
            df = self.stl_db.get_all_entries()
            if df.empty:
                self.statusBar().showMessage("Database is empty", 3000)
                return
                
            # Get similarity parameters from analyzer tab
            param = "bb_volume"  # Default similarity parameter
            top_n = 10  # Default number of results
            
            # Try to get parameters from analyzer tab UI
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    if hasattr(analyzer_tab, 'similarity_param'):
                        param = analyzer_tab.similarity_param.currentText()
                    if hasattr(analyzer_tab, 'similarity_count'):
                        top_n = analyzer_tab.similarity_count.value()
                    break
            
            # Simple similarity search based on parameter
            target_value = part_data.get(param, 0)
            if target_value <= 0:
                self.statusBar().showMessage(f"Invalid {param} value for selected part", 3000)
                return
            
            # Calculate similarity based on relative difference
            df['similarity_score'] = 1.0 - (abs(df[param] - target_value) / target_value)
            
            # Sort by similarity and get top N
            similar_df = df.nlargest(top_n, 'similarity_score')
            
            # Update database view with similar parts
            self.db_model = PandasModel(similar_df)
            self.db_table.setModel(self.db_model)
            if hasattr(self.db_table, 'resizeColumnsToContents'):
                self.db_table.resizeColumnsToContents()
            
            self.statusBar().showMessage(f"Found {len(similar_df)} similar parts", 3000)
            print(f"✅ Found {len(similar_df)} similar parts to {os.path.basename(filename)}")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error finding similar parts: {e}", 5000)
            print(f"❌ Error in find_similar_parts: {e}")
            import traceback
            traceback.print_exc()
    
    def find_similar_to_loaded_part(self, comparison_part, param="bb_volume", top_n=10):
        """Find parts similar to a loaded comparison part"""
        try:
            if not comparison_part:
                self.statusBar().showMessage("No comparison part loaded", 3000)
                return
                
            # Get all database entries
            df = self.stl_db.get_all_entries()
            if df.empty:
                self.statusBar().showMessage("Database is empty", 3000)
                return
                
            # Get target value for comparison
            target_value = comparison_part.get(param, 0)
            if target_value <= 0:
                self.statusBar().showMessage(f"Invalid {param} value for comparison part", 3000)
                return
            
            # Calculate similarity based on relative difference
            df['similarity_score'] = 1.0 - (abs(df[param] - target_value) / target_value)
            
            # Sort by similarity and get top N
            similar_df = df.nlargest(top_n, 'similarity_score')
            
            # Update database view with similar parts
            self.db_model = PandasModel(similar_df)
            self.db_table.setModel(self.db_model)
            if hasattr(self.db_table, 'resizeColumnsToContents'):
                self.db_table.resizeColumnsToContents()
            
            comparison_filename = comparison_part.get('filename', 'loaded part')
            self.statusBar().showMessage(f"Found {len(similar_df)} parts similar to {comparison_filename}", 3000)
            print(f"✅ Found {len(similar_df)} parts similar to {comparison_filename} by {param}")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error finding similar parts: {e}", 5000)
            print(f"❌ Error in find_similar_to_loaded_part: {e}")
            import traceback
            traceback.print_exc()
        
    def refresh_pricing_models(self):
        """Refresh pricing models - placeholder for now"""
        # This will be implemented by the PriceCalculatorTab
        pass
        
    def show_status_message(self, message, timeout=3000):
        """Show a message in the status bar with optional timeout"""
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(message, timeout)
        else:
            print(message)

    def on_database_selection_changed(self, selected, deselected):
        """Handle database selection changes"""
        try:
            if not selected.indexes():
                print("No selection indexes found")
                return
                
            # Get selected row
            row = selected.indexes()[0].row()
            print(f"Selected database row: {row}")
            
            # Get model data
            model = self.database_view.model()
            if not model:
                print("No model found in database view")
                return
                
            # Get file path from the selected row
            filename_column = model.get_column_index('filename')
            if filename_column >= 0:
                filename = model.data(model.index(row, filename_column), QtCore.Qt.DisplayRole)
                print(f"Selected file: {filename}")
                
                # Load STL file in viewer
                if self.stl_viewer and filename and os.path.exists(filename):
                    try:
                        print(f"Loading {filename} in 3D viewer...")
                        success = self.stl_viewer.load_stl(filename)
                        if success:
                            print("STL loaded successfully in viewer")
                        else:
                            print("Failed to load STL in viewer")
                    except Exception as e:
                        print(f"Error loading STL in viewer: {e}")
                elif not self.stl_viewer:
                    print("No STL viewer available")
                elif not filename:
                    print("No filename found in selection")
                elif not os.path.exists(filename):
                    print(f"File does not exist: {filename}")
            else:
                print(f"Filename column not found (index: {filename_column})")
                
        except Exception as e:
            print(f"Error in database selection handler: {e}")
            import traceback
            traceback.print_exc()
                    
    def clear_database(self):
        """Clear the database"""
        if self.stl_db:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Clear Database",
                "Are you sure you want to clear the database?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.stl_db.clear()
                self.update_database_view()
                self.show_status_message("Database cleared")
                
    def save_database(self):
        """Save the database"""
        if self.stl_db:
            self.stl_db.save_database()
            self.show_status_message("Database saved")

    def add_price_calc_part(self):
        """Add a part to the price calculator"""
        file_dialog = QtWidgets.QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, "Select STL Files", "", "STL Files (*.stl)"
        )
        
        if not file_paths:
            return
            
        # Process each selected file
        for file_path in file_paths:
            # Check if the file is already in the list
            if file_path in self.price_calc_parts:
                self.show_status_message(f"File already added: {os.path.basename(file_path)}")
                continue
                
            try:
                # Extract features from the STL file (with 4mm shrinkwrap)
                try:
                    from stl_utils import extract_features_with_shrinkwrap, is_valid_entry
                    # Always use 4mm shrinkwrap for consistency
                    features = extract_features_with_shrinkwrap(
                        file_path,
                        offset_mm=4.0,
                        offset_percent=None,
                        generate_shrinkwrap=False  # Don't generate files in GUI
                    )
                except ImportError:
                    try:
                        from stl_analyzer.stl_utils import extract_features_with_shrinkwrap, is_valid_entry
                        features = extract_features_with_shrinkwrap(
                            file_path,
                            offset_mm=4.0,
                            offset_percent=None,
                            generate_shrinkwrap=False
                        )
                    except ImportError:
                        # Fallback to basic features if shrinkwrap not available
                        try:
                            from stl_utils import extract_features, is_valid_entry
                        except ImportError:
                            from stl_analyzer.stl_utils import extract_features, is_valid_entry
                        features = extract_features(file_path)
                
                # Log shrinkwrap status
                if features:
                    shrinkwrap_vol = features.get('shrinkwrap_volume', 0)
                    if shrinkwrap_vol > 0:
                        print(f"✅ Price calc: Shrinkwrap data included: {shrinkwrap_vol:.2f}mm³")
                    else:
                        print(f"⚠️ Price calc: No shrinkwrap data for {os.path.basename(file_path)}")
                
                if not features or not is_valid_entry(features):
                    self.show_status_message(f"Failed to analyze: {os.path.basename(file_path)}")
                    continue
                    
                # Add part to the dictionary with quantity = 1
                part_data = {
                    "features": features,
                    "quantity": 1,
                    "calculated_price": None,
                    "breakdown": {}
                }
                
                # Add to parts dictionary
                self.price_calc_parts[file_path] = part_data
                
                # Add to list widget
                item = QtWidgets.QListWidgetItem(os.path.basename(file_path))
                item.setData(QtCore.Qt.UserRole, file_path)  # Store the full path
                self.parts_list.addItem(item)
                
            except Exception as e:
                print(f"Error adding part {file_path}: {e}")
                self.show_status_message(f"Error analyzing: {os.path.basename(file_path)}")
                continue
                
        # Select the first item if none is selected
        if self.parts_list.count() > 0 and self.parts_list.currentRow() == -1:
            self.parts_list.setCurrentRow(0)
            
        # Update UI state
        self.update_price_calculator_ui()
    
    def remove_price_calc_part(self):
        """Remove the selected part from the price calculator"""
        current_item = self.parts_list.currentItem()
        if not current_item:
            return
            
        file_path = current_item.data(QtCore.Qt.UserRole)
        
        # Remove from dictionary
        if file_path in self.price_calc_parts:
            del self.price_calc_parts[file_path]
            
        # Remove from list widget
        self.parts_list.takeItem(self.parts_list.row(current_item))
        
        # Update UI
        self.update_price_calculator_ui()
    
    def on_part_selection_changed(self, current, previous):
        """Handle part selection changes in the parts list"""
        if current:
            # Get file path from the item (UserRole stores file_path string)
            file_path = current.data(QtCore.Qt.UserRole)
            if file_path and file_path in self.price_calc_parts:
                # Get part data from our dictionary
                part_data = self.price_calc_parts[file_path]
                
                # Update the 3D viewer
                if self.price_calc_viewer and os.path.exists(file_path):
                    try:
                        self.price_calc_viewer.load_stl(file_path)
                        print(f"Loaded {file_path} in price calculator 3D viewer")
                    except Exception as e:
                        print(f"Error loading STL in price calc viewer: {e}")
                
                # Update part details display
                features = part_data.get('features', {})
                self.update_part_details_display(features)
                
                # Update quantity from stored data
                quantity = part_data.get('quantity', 1)
                self.part_quantity_spinner.setValue(quantity)
                
                # Recalculate price
                self.recalculate_price()
            else:
                # Clear displays if no valid data
                self.update_part_details_display(None)
                if self.price_calc_viewer:
                    self.price_calc_viewer.clear()
        else:
            # No part selected - clear everything
            self.update_part_details_display(None)
            if self.price_calc_viewer:
                self.price_calc_viewer.clear()
                
    def update_part_details_display(self, features):
        """Update the part details display - placeholder for now"""
        if not features:
            return
            
        # This would normally update detail labels
        # For now, just print the features
        print(f"Part details: volume={features.get('volume', 0):.2f}, surface_area={features.get('surface_area', 0):.2f}")
        
    def on_quantity_changed(self, quantity):
        """Handle quantity changes"""
        current_item = self.parts_list.currentItem()
        if current_item:
            file_path = current_item.data(QtCore.Qt.UserRole)
            if file_path in self.price_calc_parts:
                self.price_calc_parts[file_path]['quantity'] = quantity
                self.recalculate_price()
                
    def recalculate_price(self):
        """Recalculate the price for the selected part"""
        current_item = self.parts_list.currentItem()
        if not current_item:
            return
            
        file_path = current_item.data(QtCore.Qt.UserRole)
        if file_path not in self.price_calc_parts:
            return
            
        part_data = self.price_calc_parts[file_path]
        features = part_data.get('features', {})
        quantity = part_data.get('quantity', 1)
        
        # Basic price calculation - placeholder
        volume = features.get('volume', 0)
        surface_area = features.get('surface_area', 0)
        
        # Simple material-based pricing
        material_cost = volume * 0.001  # 0.001 EUR per mm³
        machine_cost = surface_area * 0.0001  # 0.0001 EUR per mm²
        labor_cost = 5.0  # Fixed labor cost
        
        base_price = (material_cost + machine_cost + labor_cost) * quantity
        final_price = base_price * 1.2  # 20% markup
        
        # Update display (with safe handling of missing UI elements)
        if hasattr(self, 'material_cost_label') and self.material_cost_label:
            self.material_cost_label.setText(f"€{material_cost:.2f}")
        if hasattr(self, 'machine_cost_label') and self.machine_cost_label:
            self.machine_cost_label.setText(f"€{machine_cost:.2f}")
        if hasattr(self, 'labor_cost_label') and self.labor_cost_label:
            self.labor_cost_label.setText(f"€{labor_cost:.2f}")
        if hasattr(self, 'base_total_label') and self.base_total_label:
            self.base_total_label.setText(f"€{base_price:.2f}")
        if hasattr(self, 'final_price_label') and self.final_price_label:
            self.final_price_label.setText(f"€{final_price:.2f}")
        
        # Store calculated price
        part_data['calculated_price'] = final_price
        
        # Enable save button (with safe handling)
        if hasattr(self, 'save_to_project_button') and self.save_to_project_button:
            self.save_to_project_button.setEnabled(True)
        
    def update_price_calculator_ui(self):
        """Update the price calculator UI state"""
        has_parts = self.parts_list.count() > 0
        current_item = self.parts_list.currentItem()
        
        # Enable/disable UI elements
        self.remove_part_button.setEnabled(current_item is not None)
        self.part_details_group.setEnabled(current_item is not None)
        self.recalculate_button.setEnabled(current_item is not None)
        
        # If no parts, clear details
        if not current_item:
            self.material_cost_label.setText("€0.00")
            self.estimated_time_label.setText("0.0 h")
            self.complexity_factor_label.setText("1.00×")
            self.machine_cost_label.setText("€0.00")
            self.energy_cost_label.setText("€0.00")
            self.labor_cost_label.setText("€0.00")
            self.maintenance_cost_label.setText("€0.00")
            self.ai_model_price_label.setText("€0.00")
            self.base_total_label.setText("€0.00")
            self.final_price_label.setText("€0.00")
            self.save_to_project_button.setEnabled(False)
            
    def save_part_to_project(self):
        """Save the current part to the project"""
        self.show_status_message("Save to project functionality not implemented")
        
    def show_pricing_presets_dialog(self):
        """Show pricing presets dialog"""
        self.show_status_message("Pricing presets dialog not implemented")
        
    def load_price_project(self):
        """Load a price project"""
        self.show_status_message("Load project functionality not implemented")
        
    def recalculate_all_parts(self):
        """Recalculate all parts"""
        for i in range(self.parts_list.count()):
            item = self.parts_list.item(i)
            self.parts_list.setCurrentItem(item)
            self.recalculate_price()
        self.show_status_message("Recalculated all parts")
        
    def save_price_project(self):
        """Save the current price project"""
        self.show_status_message("Save project functionality not implemented")
        
    def upload_project_parts(self):
        """Upload STL parts to the project"""
        file_dialog = QtWidgets.QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, "Select STL Files for Project", "", "STL Files (*.stl)"
        )
        
        if not file_paths:
            return
            
        # Process each selected file
        for file_path in file_paths:
            try:
                # Use extract_features_with_shrinkwrap for guaranteed shrinkwrap calculation
                print(f"🔄 Processing {os.path.basename(file_path)} with shrinkwrap calculation...")
                
                try:
                    # Try the shrinkwrap-enhanced extraction first
                    from .stl_utils import extract_features_with_shrinkwrap
                    features = extract_features_with_shrinkwrap(
                        file_path,
                        offset_mm=4.0,  # Always use 4mm for consistency
                        offset_percent=None,
                        generate_shrinkwrap=False  # Don't generate files in GUI
                    )
                    print(f"✅ Used extract_features_with_shrinkwrap for {os.path.basename(file_path)}")
                except ImportError as e:
                    print(f"⚠️ Could not import extract_features_with_shrinkwrap: {e}, trying fallback...")
                    try:
                        from stl_analyzer.stl_utils import extract_features_with_shrinkwrap
                        features = extract_features_with_shrinkwrap(
                            file_path,
                            offset_mm=4.0,
                            offset_percent=None,
                            generate_shrinkwrap=False
                        )
                        print(f"✅ Used fallback import for {os.path.basename(file_path)}")
                    except ImportError:
                        # Ultimate fallback to basic features + manual shrinkwrap
                        print(f"⚠️ Using basic extract_features + manual shrinkwrap for {os.path.basename(file_path)}")
                        try:
                            from .stl_utils import extract_features
                        except ImportError:
                            from stl_analyzer.stl_utils import extract_features
                        features = extract_features(file_path)
                
                if features:
                    # Ensure all required calculations are present
                    volume = features.get('volume', 0)
                    surface_area = features.get('surface_area', 0)
                    bb_volume = features.get('bb_volume', 0)
                    convex_hull_volume = features.get('convex_hull_volume', 0)
                    shrinkwrap_volume = features.get('shrinkwrap_volume', 0)
                    
                    # If shrinkwrap is still missing or 0, force calculation
                    if shrinkwrap_volume <= 0:
                        print(f"⚠️ Shrinkwrap volume is {shrinkwrap_volume}, attempting manual calculation...")
                        try:
                            from .stl_utils import try_load_stl, create_shrinkwrap
                            mesh = try_load_stl(file_path)
                            if mesh:
                                print(f"   🔧 Mesh loaded, calculating shrinkwrap with 4mm offset...")
                                shrinkwrap = create_shrinkwrap(mesh, offset_mm=4.0)
                                if shrinkwrap and hasattr(shrinkwrap, 'volume') and shrinkwrap.volume > 0:
                                    shrinkwrap_volume = shrinkwrap.volume
                                    features['shrinkwrap_volume'] = shrinkwrap_volume
                                    print(f"   ✅ Manual shrinkwrap calculated: {shrinkwrap_volume:.0f} mm³")
                                else:
                                    print(f"   ❌ create_shrinkwrap returned invalid result")
                            else:
                                print(f"   ❌ Failed to load mesh for manual shrinkwrap")
                        except Exception as e:
                            print(f"   ❌ Manual shrinkwrap calculation failed: {e}")
                    
                    # Calculate missing ratios if not present
                    if volume > 0:
                        # Convexity ratio
                        if convex_hull_volume > 0 and 'convexity_ratio' not in features:
                            features['convexity_ratio'] = volume / convex_hull_volume
                        
                        # Shrinkwrap ratio
                        if shrinkwrap_volume > 0 and 'shrinkwrap_ratio' not in features:
                            features['shrinkwrap_ratio'] = volume / shrinkwrap_volume
                            
                        # Surface area to volume ratio
                        if surface_area > 0 and 'sa_vol_ratio' not in features:
                            features['sa_vol_ratio'] = surface_area / volume
                    
                    # Waste calculation (BB Volume - Actual Volume)
                    if bb_volume > 0 and volume > 0 and 'waste_volume' not in features:
                        features['waste_volume'] = bb_volume - volume
                        features['waste_ratio'] = features['waste_volume'] / bb_volume
                    
                    # Create part data structure
                    part_data = {
                        'file_path': file_path,
                        'name': os.path.basename(file_path),
                        'features': features,
                        'quantity': 1,
                        'unit_price': 0,
                        'calculated_price': 0
                    }
                    
                    print(f"📊 Final features summary for {os.path.basename(file_path)}:")
                    print(f"   Volume: {volume:,.0f} mm³")
                    print(f"   Surface Area: {surface_area:,.0f} mm²") 
                    print(f"   BB Volume: {bb_volume:,.0f} mm³")
                    print(f"   Convex Hull: {convex_hull_volume:,.0f} mm³")
                    print(f"   Shrinkwrap: {shrinkwrap_volume:,.0f} mm³ {'✅' if shrinkwrap_volume > 0 else '❌'}")
                    if 'convexity_ratio' in features:
                        print(f"   Convexity Ratio: {features['convexity_ratio']:.3f}")
                    if 'shrinkwrap_ratio' in features:
                        print(f"   Shrinkwrap Ratio: {features['shrinkwrap_ratio']:.3f}")
                    if 'sa_vol_ratio' in features:
                        print(f"   SA/Vol Ratio: {features['sa_vol_ratio']:.3f}")
                
                else:
                    print(f"❌ Failed to extract any features from {file_path}")
                    continue
                
                # Log shrinkwrap status
                if features:
                    shrinkwrap_vol = features.get('shrinkwrap_volume', 0)
                    if shrinkwrap_vol > 0:
                        print(f"✅ Project: Shrinkwrap data included: {shrinkwrap_vol:.2f}mm³")
                    else:
                        print(f"⚠️ Project: No shrinkwrap data for {os.path.basename(file_path)}")
                
                if not features or not is_valid_entry(features):
                    self.show_status_message(f"Failed to analyze: {os.path.basename(file_path)}")
                    continue
                    
                # Create part data
                part_data = {
                    "file_path": file_path,
                    "features": features,
                    "quantity": 1,
                    "calculated_price": 0.0
                }
                
                # Add to project parts
                self.project_parts.append(part_data)
                
                # Auto-calculate price for the new part (triggers full project recalculation)
                part_index = len(self.project_parts) - 1
                self.calculate_single_part_price(part_index)
                
                print(f"✅ Added {os.path.basename(file_path)} with auto-pricing")
                
            except Exception as e:
                print(f"Error uploading project part {file_path}: {e}")
                self.show_status_message(f"Error analyzing: {os.path.basename(file_path)}")
                continue
                
        # Update project totals
        self.update_project_totals()
        
        # Update the comprehensive table display
        self.update_project_parts_display()
        
        # Enable calculate button if we have parts
        if self.project_parts:
            self.calculate_all_button.setEnabled(True)
            
    def on_project_part_clicked(self, item):
        """Handle clicking on a project part to display in 3D viewer"""
        if not item:
            return
            
        row = item.row()
        if row < 0 or row >= len(self.project_parts):
            return
            
        part = self.project_parts[row]
        part_name = os.path.basename(part.get('file_path', f'Part_{row+1}'))
        
        print(f"\n👆 Clicked on part: {part_name}")
        
        # Load the part in the 3D viewer
        if hasattr(self, 'project_pricing_tab') and hasattr(self.project_pricing_tab, 'project_stl_viewer'):
            viewer = self.project_pricing_tab.project_stl_viewer
            if viewer and 'file_path' in part:
                try:
                    file_path = part['file_path']
                    print(f"🎨 Loading {file_path} in 3D viewer...")
                    
                    # Check if file exists
                    if not os.path.exists(file_path):
                        print(f"❌ File not found: {file_path}")
                        return
                    
                    # Load the STL file
                    viewer.load_stl(file_path)
                    print(f"✅ Successfully loaded {part_name} in 3D viewer")
                    
                    # Capture shrinkwrap data if it was calculated during loading
                    if hasattr(viewer, 'shrinkwrap') and viewer.shrinkwrap and hasattr(viewer.shrinkwrap, 'volume'):
                        shrinkwrap_vol = viewer.shrinkwrap.volume
                        if shrinkwrap_vol > 0:
                            # Update the part features with the calculated shrinkwrap
                            part['features']['shrinkwrap_volume'] = shrinkwrap_vol
                            print(f"📊 Captured shrinkwrap volume from viewer: {shrinkwrap_vol:,.0f} mm³")
                            
                            # Force update of the part details display
                            features = part.get('features', {})
                            features['shrinkwrap_volume'] = shrinkwrap_vol
                    
                except Exception as e:
                    print(f"❌ Error loading STL in 3D viewer: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ 3D viewer not available or file path missing")
        
        # Update comprehensive part details on the right panel
        features = part.get('features', {})
        quantity = part.get('quantity', 1)
        unit_price = part.get('unit_price', 0)
        total_price = part.get('calculated_price', 0)
        
        # Update all the detail labels with proper error handling
        try:
            # Basic info
            if hasattr(self, 'project_detail_filename'):
                self.project_detail_filename.setText(part_name)
            
            # Dimensions
            if hasattr(self, 'project_detail_dim_x'):
                self.project_detail_dim_x.setText(f"{features.get('dimension_x', features.get('x', 0)):.2f}")
            if hasattr(self, 'project_detail_dim_y'):
                self.project_detail_dim_y.setText(f"{features.get('dimension_y', features.get('y', 0)):.2f}")
            if hasattr(self, 'project_detail_dim_z'):
                self.project_detail_dim_z.setText(f"{features.get('dimension_z', features.get('z', 0)):.2f}")
            
            # Volumes
            volume = features.get('volume', 0)
            surface_area = features.get('surface_area', 0)
            bb_volume = features.get('bb_volume', 0)
            convex_hull_vol = features.get('convex_hull_volume', 0)
            shrinkwrap_vol = features.get('shrinkwrap_volume', 0)
            
            if hasattr(self, 'project_detail_volume'):
                self.project_detail_volume.setText(f"{volume:,.0f} mm³")
            if hasattr(self, 'project_detail_surface'):
                self.project_detail_surface.setText(f"{surface_area:,.0f} mm²")
            if hasattr(self, 'project_detail_bb_volume'):
                self.project_detail_bb_volume.setText(f"{bb_volume:,.0f} mm³")
            
            # Waste calculation (BB Volume - Actual Volume)
            if hasattr(self, 'project_detail_waste'):
                waste = features.get('waste_volume', bb_volume - volume if bb_volume > 0 and volume > 0 else 0)
                self.project_detail_waste.setText(f"{waste:,.0f} mm³")
            
            # Convex hull
            if hasattr(self, 'project_detail_convex_hull_volume'):
                self.project_detail_convex_hull_volume.setText(f"{convex_hull_vol:,.0f} mm³")
            
            # Convexity ratio (Volume / Convex Hull Volume)
            if hasattr(self, 'project_detail_convexity_ratio'):
                convexity_ratio = features.get('convexity_ratio')
                if convexity_ratio is None and convex_hull_vol > 0 and volume > 0:
                    convexity_ratio = volume / convex_hull_vol
                if convexity_ratio:
                    self.project_detail_convexity_ratio.setText(f"{convexity_ratio:.3f}")
                else:
                    self.project_detail_convexity_ratio.setText("-")
            
            # Shrinkwrap
            if hasattr(self, 'project_detail_shrinkwrap_volume'):
                if shrinkwrap_vol > 0:
                    self.project_detail_shrinkwrap_volume.setText(f"{shrinkwrap_vol:,.0f} mm³")
                else:
                    self.project_detail_shrinkwrap_volume.setText("Not calculated")
            
            # Shrinkwrap ratio (Volume / Shrinkwrap Volume)
            if hasattr(self, 'project_detail_shrinkwrap_ratio'):
                shrinkwrap_ratio = features.get('shrinkwrap_ratio')
                if shrinkwrap_ratio is None and shrinkwrap_vol > 0 and volume > 0:
                    shrinkwrap_ratio = volume / shrinkwrap_vol
                if shrinkwrap_ratio:
                    self.project_detail_shrinkwrap_ratio.setText(f"{shrinkwrap_ratio:.3f}")
                else:
                    self.project_detail_shrinkwrap_ratio.setText("-")
            
            # Surface Area to Volume ratio
            if hasattr(self, 'project_detail_sa_vol_ratio'):
                sa_vol_ratio = features.get('sa_vol_ratio')
                if sa_vol_ratio is None and volume > 0 and surface_area > 0:
                    sa_vol_ratio = surface_area / volume
                if sa_vol_ratio:
                    self.project_detail_sa_vol_ratio.setText(f"{sa_vol_ratio:.3f}")
                else:
                    self.project_detail_sa_vol_ratio.setText("-")
            
            # Pricing information
            if hasattr(self, 'project_detail_quantity'):
                self.project_detail_quantity.setText(str(quantity))
            if hasattr(self, 'project_detail_unit_price'):
                self.project_detail_unit_price.setText(f"€{unit_price:.2f}")
            if hasattr(self, 'project_detail_total_price'):
                self.project_detail_total_price.setText(f"€{total_price:.2f}")
            
            # Project discount
            if hasattr(self, 'project_detail_discount'):
                discount = part.get('project_discount', 0)
                if discount > 0:
                    self.project_detail_discount.setText(f"{discount:.1f}%")
                else:
                    self.project_detail_discount.setText("-")
                
            print(f"📋 Updated comprehensive part details for {part_name}")
            print(f"   Volume: {volume:,.0f} mm³, Surface: {surface_area:,.0f} mm², BB: {bb_volume:,.0f} mm³")
            if shrinkwrap_vol > 0:
                print(f"   Shrinkwrap: {shrinkwrap_vol:,.0f} mm³")
            print(f"   Price: €{unit_price:.2f} x {quantity} = €{total_price:.2f}")
            
        except Exception as e:
            print(f"⚠️ Error updating part details: {e}")
        
        # Enable optimization buttons when a part is selected
        if hasattr(self, 'optimize_bbox_btn'):
            self.optimize_bbox_btn.setEnabled(True)
        if hasattr(self, 'recalc_shrinkwrap_btn'):
            self.recalc_shrinkwrap_btn.setEnabled(True)
    
    def recalculate_selected_shrinkwrap(self):
        """Recalculate shrinkwrap for the currently selected part"""
        # Get currently selected part from table
        if not hasattr(self, 'project_parts_table') or not self.project_parts_table.currentRow() >= 0:
            print("⚠️ No part selected")
            return
            
        current_row = self.project_parts_table.currentRow()
        if current_row >= len(self.project_parts):
            print("⚠️ Invalid part selection")
            return
            
        part = self.project_parts[current_row]
        file_path = part.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            return
            
        print(f"🔄 Recalculating shrinkwrap for {os.path.basename(file_path)}...")
        
        try:
            # Load mesh and recalculate shrinkwrap with high precision
            from .stl_utils import try_load_stl, create_shrinkwrap
            mesh = try_load_stl(file_path)
            
            if mesh:
                # Use multiple methods for better accuracy
                print("   📐 Attempting high-precision shrinkwrap calculation...")
                
                # Try surface offset method first (more reliable)
                try:
                    from .stl_utils import create_surface_offset_shrinkwrap
                    shrinkwrap = create_surface_offset_shrinkwrap(mesh, offset_mm=4.0)
                    if shrinkwrap and hasattr(shrinkwrap, 'volume') and shrinkwrap.volume > 0:
                        shrinkwrap_volume = shrinkwrap.volume
                        print(f"   ✅ Surface offset method: {shrinkwrap_volume:.0f} mm³")
                    else:
                        raise Exception("Surface offset failed")
                except Exception as e:
                    print(f"   ⚠️ Surface offset failed: {e}, trying box grid method...")
                    
                    # Fallback to box grid method
                    try:
                        from .stl_utils import create_fixed_box_grid_shrinkwrap
                        shrinkwrap = create_fixed_box_grid_shrinkwrap(mesh, box_size=4.0)
                        if shrinkwrap and hasattr(shrinkwrap, 'volume') and shrinkwrap.volume > 0:
                            shrinkwrap_volume = shrinkwrap.volume
                            print(f"   ✅ Box grid method: {shrinkwrap_volume:.0f} mm³")
                        else:
                            raise Exception("Box grid failed")
                    except Exception as e2:
                        print(f"   ❌ All shrinkwrap methods failed: {e2}")
                        return
                
                # Update part features with new shrinkwrap data
                features = part.get('features', {})
                features['shrinkwrap_volume'] = shrinkwrap_volume
                
                # Recalculate derived ratios
                volume = features.get('volume', 0)
                surface_area = features.get('surface_area', 0)
                bb_volume = features.get('bb_volume', 0)
                convex_hull_volume = features.get('convex_hull_volume', 0)
                
                if volume > 0:
                    # Update shrinkwrap ratio
                    features['shrinkwrap_ratio'] = volume / shrinkwrap_volume
                    
                    # Recalculate other ratios if missing
                    if convex_hull_volume > 0:
                        features['convexity_ratio'] = volume / convex_hull_volume
                    if surface_area > 0:
                        features['sa_vol_ratio'] = surface_area / volume
                    if bb_volume > 0:
                        features['waste_volume'] = bb_volume - volume
                        features['waste_ratio'] = features['waste_volume'] / bb_volume
                
                # Update part features
                part['features'] = features
                
                print(f"✅ Updated shrinkwrap volume: {shrinkwrap_volume:,.0f} mm³")
                print(f"   Shrinkwrap ratio: {features.get('shrinkwrap_ratio', 0):.3f}")
                
                # Recalculate price with updated features (shrinkwrap affects ML model)
                self.calculate_single_part_price(current_row)
                
                # Update all displays
                self.update_project_parts_display()
                self.on_project_part_clicked(self.project_parts_table.item(current_row, 0))
                self.update_project_totals()
                
                print(f"🎯 Recalculated pricing with new shrinkwrap data")
            else:
                print("❌ Failed to load mesh from file")
                
        except Exception as e:
            print(f"❌ Error recalculating shrinkwrap: {e}")
            import traceback
            traceback.print_exc()
    
    def optimize_selected_part_bbox(self):
        """Optimize bounding box for the currently selected part"""
        # Get currently selected part from table
        if not hasattr(self, 'project_parts_table') or not self.project_parts_table.currentRow() >= 0:
            print("⚠️ No part selected")
            return
            
        current_row = self.project_parts_table.currentRow()
        if current_row >= len(self.project_parts):
            print("⚠️ Invalid part selection")
            return
            
        part = self.project_parts[current_row]
        file_path = part.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            return
            
        print(f"🔧 Optimizing bounding box for {os.path.basename(file_path)}...")
        
        # Re-analyze the file with optimization
        try:
            from .stl_utils import extract_features
            optimized_features = extract_features(file_path)
            
            if optimized_features:
                # Update the part with optimized features
                part['features'].update(optimized_features)
                print(f"✅ Optimized features for {os.path.basename(file_path)}")
                
                # Recalculate price with optimized features
                self.calculate_single_part_price(current_row)
                
                # Update displays
                self.update_project_parts_display()
                self.on_project_part_clicked(self.project_parts_table.item(current_row, 0))
                self.update_project_totals()
                
                print(f"🎯 Recalculated pricing with optimized features")
            else:
                print("⚠️ Failed to optimize features")
                
        except Exception as e:
            print(f"❌ Error optimizing bounding box: {e}")
            import traceback
            traceback.print_exc()
    
    def update_project_part_details(self, part):
        """Update the project part details panel with comprehensive information"""
        if not hasattr(self, 'project_detail_filename'):
            print("⚠️ Project detail UI elements not available")
            return
            
        # Extract features from part data
        if 'features' in part:
            features = part['features']
        else:
            features = part  # Sometimes the part data IS the features
            
        # Update filename
        self.project_detail_filename.setText(os.path.basename(part.get('file_path', part.get('name', 'Unknown'))))
        
        # Dimensions with proper feature extraction
        x = features.get('x', features.get('dimension_x', 0))
        y = features.get('y', features.get('dimension_y', 0)) 
        z = features.get('z', features.get('dimension_z', 0))
        
        self.project_detail_dim_x.setText(f"{x:.1f}")
        self.project_detail_dim_y.setText(f"{y:.1f}")
        self.project_detail_dim_z.setText(f"{z:.1f}")
        
        # Basic volumes and areas
        volume = features.get('volume', 0)
        surface_area = features.get('surface_area', 0)
        bb_volume = features.get('bb_volume', 0)
        
        self.project_detail_volume.setText(f"{volume:,.0f} mm³")
        self.project_detail_surface.setText(f"{surface_area:,.0f} mm²")
        self.project_detail_bb_volume.setText(f"{bb_volume:,.0f} mm³")
        
        # Waste calculation
        waste = bb_volume - volume if bb_volume > 0 and volume > 0 else 0
        waste_percent = (waste / bb_volume * 100) if bb_volume > 0 else 0
        self.project_detail_waste.setText(f"{waste:,.0f} mm³ ({waste_percent:.1f}%)")
        
        # Advanced volumes
        convex_hull_volume = features.get('convex_hull_volume', 0)
        shrinkwrap_volume = features.get('shrinkwrap_volume', 0)
        
        self.project_detail_convex_hull_volume.setText(f"{convex_hull_volume:,.0f} mm³")
        self.project_detail_shrinkwrap_volume.setText(f"{shrinkwrap_volume:,.0f} mm³")
        
        # Ratios
        convexity_ratio = (volume / convex_hull_volume) if convex_hull_volume > 0 else 0
        self.project_detail_convexity_ratio.setText(f"{convexity_ratio:.3f}")
        
        shrinkwrap_ratio = (volume / shrinkwrap_volume) if shrinkwrap_volume > 0 else 0
        self.project_detail_shrinkwrap_ratio.setText(f"{shrinkwrap_ratio:.3f}")
        
        # Surface to volume ratio
        sa_vol_ratio = (surface_area / volume) if volume > 0 else 0
        self.project_detail_sa_vol_ratio.setText(f"{sa_vol_ratio:.3f} mm⁻¹")
        
        # Pricing info
        self.project_detail_quantity.setText(str(part.get('quantity', 1)))
        
        unit_price = part.get('unit_price', 0)
        total_price = unit_price * part.get('quantity', 1)
        self.project_detail_unit_price.setText(f"€{unit_price:.2f}")
        self.project_detail_total_price.setText(f"€{total_price:.2f}")
        
        # Project discount
        project_discount = part.get('project_discount', 0)
        base_unit_price = part.get('base_unit_price', unit_price)
        if project_discount > 0:
            self.project_detail_discount.setText(f"-{project_discount:.1f}% (€{base_unit_price:.2f} → €{unit_price:.2f})")
        elif project_discount < 0:
            self.project_detail_discount.setText(f"+{abs(project_discount):.1f}% (€{base_unit_price:.2f} → €{unit_price:.2f})")
        else:
            self.project_detail_discount.setText("None")
        
        # Enable optimization buttons when a part is selected
        if hasattr(self, 'optimize_bbox_btn'):
            self.optimize_bbox_btn.setEnabled(True)
        if hasattr(self, 'recalc_shrinkwrap_btn'):
            self.recalc_shrinkwrap_btn.setEnabled(True)
    
    def update_project_parts_display(self):
        """Update the project parts table with comprehensive data and pricing"""
        if not hasattr(self, 'project_parts_table'):
            print("⚠️ Project parts table not available")
            return
            
        # Clear the table first to prevent layout issues
        self.project_parts_table.clearContents()
        self.project_parts_table.setRowCount(len(self.project_parts))
        
        for i, part in enumerate(self.project_parts):
            # Extract features properly
            if 'features' in part:
                features = part['features']
                part_name = os.path.basename(part.get('file_path', 'Unknown'))
            else:
                features = part  # Sometimes part data IS the features
                part_name = part.get('name', os.path.basename(part.get('file_path', 'Unknown')))
            
            # Part name
            name_item = QtWidgets.QTableWidgetItem(part_name)
            self.project_parts_table.setItem(i, 0, name_item)
            
            # Quantity (editable)
            quantity_item = QtWidgets.QTableWidgetItem(str(part.get('quantity', 1)))
            quantity_item.setFlags(quantity_item.flags() | QtCore.Qt.ItemIsEditable)
            self.project_parts_table.setItem(i, 1, quantity_item)
            
            # Connect quantity changes to auto-recalculation
            def on_quantity_changed(row):
                def handler():
                    try:
                        new_quantity = int(self.project_parts_table.item(row, 1).text())
                        if new_quantity > 0:
                            self.project_parts[row]['quantity'] = new_quantity
                            print(f"🔄 Quantity changed for part {row+1}: {new_quantity}")
                            # Auto-recalculate all prices since project totals changed
                            self.calculate_single_part_price(row)
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Invalid quantity entered: {e}")
                return handler
            
            # Store the handler for this row
            quantity_item.handler = on_quantity_changed(i)
            
            # Volume
            volume = features.get('volume', 0)
            volume_item = QtWidgets.QTableWidgetItem(f"{volume:.0f}")
            self.project_parts_table.setItem(i, 2, volume_item)
            
            # Max Dimension
            x = features.get('x', features.get('dimension_x', 0))
            y = features.get('y', features.get('dimension_y', 0))
            z = features.get('z', features.get('dimension_z', 0))
            max_dim = max(x, y, z)
            max_dim_item = QtWidgets.QTableWidgetItem(f"{max_dim:.1f}")
            self.project_parts_table.setItem(i, 3, max_dim_item)
            
            # Unit price
            unit_price = part.get('unit_price', 0)
            unit_price_item = QtWidgets.QTableWidgetItem(f"€{unit_price:.2f}" if unit_price > 0 else "€-.--")
            self.project_parts_table.setItem(i, 4, unit_price_item)
            
            # Total price
            quantity = part.get('quantity', 1)
            total_price = unit_price * quantity
            total_price_item = QtWidgets.QTableWidgetItem(f"€{total_price:.2f}" if unit_price > 0 else "€-.--")
            self.project_parts_table.setItem(i, 5, total_price_item)
            
            # Remove button
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.clicked.connect(lambda _, idx=i: self.remove_project_part(idx))
            self.project_parts_table.setCellWidget(i, 6, remove_btn)
            
        # Connect cell change signal AFTER populating table
        try:
            # Disconnect previous connections to avoid duplicates
            self.project_parts_table.cellChanged.disconnect()
        except:
            pass
        
        # Connect to auto-recalculation
        def on_cell_changed(row, col):
            if col == 1:  # Quantity column
                try:
                    new_quantity = int(self.project_parts_table.item(row, 1).text())
                    if new_quantity > 0 and row < len(self.project_parts):
                        old_quantity = self.project_parts[row].get('quantity', 1)
                        if new_quantity != old_quantity:
                            self.project_parts[row]['quantity'] = new_quantity
                            print(f"🔄 Quantity changed for part {row+1}: {old_quantity} → {new_quantity}")
                            # Auto-recalculate all prices since project totals changed
                            self.calculate_single_part_price(row)
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Invalid quantity entered: {e}")
                    # Restore original value
                    if row < len(self.project_parts):
                        original_qty = self.project_parts[row].get('quantity', 1)
                        self.project_parts_table.item(row, 1).setText(str(original_qty))
        
        self.project_parts_table.cellChanged.connect(on_cell_changed)
    
    def update_part_quantity_and_price(self, index, quantity):
        """Update quantity for a specific part and recalculate pricing"""
        if 0 <= index < len(self.project_parts):
            self.project_parts[index]['quantity'] = quantity
            # Auto-calculate price for this part
            self.calculate_single_part_price(index)
            self.update_project_parts_display()
            self.update_project_totals()
    
    def calculate_single_part_price(self, index, apply_project_discount=True):
        """Calculate price for a single part using project-aware AI model and trigger recalculation of all parts"""
        if not (0 <= index < len(self.project_parts)):
            return
            
        print(f"\n🔄 Auto-calculating prices for all parts (triggered by part {index+1})...")
        
        # Since project-aware model depends on total project context,
        # we need to recalculate ALL parts when ANY part changes
        self.calculate_all_prices_internal()
    
    def predict_project_part_price(self, part, project_totals):
        """Predict price using Random Forest with full project context"""
        try:
            # Create enhanced feature set with project context
            enhanced_features = dict(part)  # Start with part features
            
            # Add project-level features
            enhanced_features.update(project_totals)
            
            # Add calculated features that the model expects
            enhanced_features['Volume * Quantity'] = part.get('volume', 0) * part.get('quantity', 1)
            enhanced_features['shrinkwrap_volume*Quantity'] = part.get('shrinkwrap_volume', 0) * part.get('quantity', 1)
            enhanced_features['Surface area*Quantity'] = part.get('surface_area', 0) * part.get('quantity', 1)
            enhanced_features['BB_Volume*Quantity'] = part.get('bb_volume', 0) * part.get('quantity', 1)
            enhanced_features['Convex_Hull_Volume*Quantity'] = part.get('convex_hull_volume', 0) * part.get('quantity', 1)
            
            # DEBUG: Show what features we're sending to the model
            print(f"   🔍 DEBUG - Project Features Added:")
            for key, value in project_totals.items():
                print(f"      {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"      {key}: {value}")
            
            print(f"   🔍 DEBUG - Enhanced Features Sample:")
            sample_keys = ['volume', 'Total_Parts_in_Project', 'Project_Total_Volume', 'Project_Total_surface_area']
            for key in sample_keys:
                if key in enhanced_features:
                    val = enhanced_features[key]
                    print(f"      {key}: {val:,.0f}" if isinstance(val, (int, float)) else f"      {key}: {val}")
            
            # Compare individual vs project-aware prediction
            print(f"   🔍 DEBUG - Comparing predictions:")
            
            # Get individual prediction first
            individual_result = self.predict_price_with_ai_model(part)
            if isinstance(individual_result, tuple):
                individual_price = individual_result[0]
            else:
                individual_price = individual_result
            print(f"      Individual price: €{individual_price:.2f}")
            
            # Get project-aware prediction using enhanced features
            project_result = self.predict_price_with_ai_model(enhanced_features)
            if isinstance(project_result, tuple):
                project_price = project_result[0]
            else:
                project_price = project_result
            print(f"      Project-aware price: €{project_price:.2f}")
            
            # Show the difference
            if individual_price and project_price:
                diff = project_price - individual_price
                diff_percent = (diff / individual_price) * 100 if individual_price > 0 else 0
                print(f"      Difference: €{diff:.2f} ({diff_percent:+.1f}%)")
            
            # Use the existing Random Forest model with enhanced features
            return self.predict_price_with_ai_model(enhanced_features)
                
        except Exception as e:
            print(f"   ⚠️ Project-level prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to individual pricing
            try:
                return self.predict_price_with_ai_model(part)
            except:
                return 10.0, {}  # Ultimate fallback
    
    def calculate_all_prices_internal(self):
        """Internal method for project-aware pricing calculation"""
        if not self.project_parts:
            return
            
        # Calculate project totals first 
        project_totals = self.calculate_project_totals()
        
        print(f"🌲 Using Project-Aware Random Forest model...")
        print(f"📊 Project Context:")
        print(f"   Total Parts: {project_totals.get('Total_Parts_in_Project', 0)}")
        print(f"   Total Volume: {project_totals.get('Project_Total_Volume', 0):,.0f} mm³")
        print(f"   Total Surface Area: {project_totals.get('Project_Total_surface_area', 0):,.0f} mm²")
        print(f"   Total BB Volume: {project_totals.get('Project_Total_bb_volume', 0):,.0f} mm³")
        print(f"   Total Shrinkwrap: {project_totals.get('Project_Total_shrinkwrap_volume', 0):,.0f} mm³")
        
        total_price = 0.0
        
        for i, part_data in enumerate(self.project_parts):
            features = part_data.get('features', {})
            quantity = part_data.get('quantity', 1)
            
            # Prepare features with quantity for project-aware model
            part_features = features.copy()
            part_features['quantity'] = quantity
            part_features['Quantity'] = quantity
            
            try:
                # Use project-aware pricing (PRIMARY method)
                if hasattr(self, 'predict_project_part_price') and project_totals:
                    unit_price, _ = self.predict_project_part_price(part_features, project_totals)
                    if unit_price <= 0:
                        raise ValueError("Project model returned 0")
                    print(f"   🌲 Part {i+1}: Project-aware price €{unit_price:.2f}")
                    
                elif hasattr(self, 'predict_price_with_ai_model'):
                    # Use standard AI model as fallback
                    unit_price, _ = self.predict_price_with_ai_model(part_features)
                    if unit_price <= 0:
                        raise ValueError("AI model returned 0")
                    print(f"   🌳 Part {i+1}: Standard AI price €{unit_price:.2f}")
                else:
                    raise ValueError("No ML model available")
                    
            except Exception as e:
                # Fallback to basic calculation
                print(f"⚠️ ML prediction failed for part {i+1}: {e}")
                volume = features.get('volume', 0)
                unit_price = max(10.0, volume * 0.00005)  # €0.05 per 1000 mm³
                print(f"   💰 Part {i+1}: Fallback price €{unit_price:.2f}")
            
            # Store the calculated prices
            part_data['unit_price'] = unit_price
            part_data['calculated_price'] = unit_price * quantity
            total_price += part_data['calculated_price']
            
            print(f"      Final: €{unit_price:.2f} x {quantity} = €{part_data['calculated_price']:.2f}")
        
        print(f"\n✅ Total project price: €{total_price:.2f}")
        
        # Update all displays
        self.update_project_parts_display()
        self.update_project_totals()
        
        self.save_project_button.setEnabled(True)
    
    def remove_project_part(self, index):
        """Remove a project part at the given index"""
        if 0 <= index < len(self.project_parts):
            removed_part = self.project_parts.pop(index)
            print(f"Removed part: {removed_part.get('name', 'Unknown')}")
            self.update_project_parts_display()
            self.update_project_totals()
            
            # Clear 3D viewer if no parts left
            if not self.project_parts and hasattr(self, 'project_stl_viewer') and self.project_stl_viewer:
                self.project_stl_viewer.clear()
        
    def update_project_totals(self):
        """Update the project totals display with proper quantity accounting"""
        if not self.project_parts:
            self.total_parts_label.setText("0")
            self.total_volume_label.setText("0.00 mm³")
            self.total_surface_area_label.setText("0.00 mm²")
            self.total_bb_volume_label.setText("0.00 mm³")
            self.total_convex_hull_label.setText("0.00 mm³")
            self.total_shrinkwrap_label.setText("0.00 mm³")
            self.total_price_label.setText("€0.00")
            return
            
        # Calculate totals with proper quantity accounting
        total_parts = sum(part.get('quantity', 1) for part in self.project_parts)
        total_volume = 0
        total_surface_area = 0
        total_bb_volume = 0
        total_convex_hull = 0
        total_shrinkwrap = 0
        total_price = sum(part.get('calculated_price', 0) for part in self.project_parts)
        
        for part in self.project_parts:
            features = part.get('features', {})
            qty = part.get('quantity', 1)
            
            total_volume += features.get('volume', 0) * qty
            total_surface_area += features.get('surface_area', 0) * qty
            total_bb_volume += features.get('bb_volume', 0) * qty
            total_convex_hull += features.get('convex_hull_volume', 0) * qty
            total_shrinkwrap += features.get('shrinkwrap_volume', 0) * qty
        
        # Update labels
        self.total_parts_label.setText(str(total_parts))
        self.total_volume_label.setText(f"{total_volume:,.0f} mm³")
        self.total_surface_area_label.setText(f"{total_surface_area:,.0f} mm²")
        self.total_bb_volume_label.setText(f"{total_bb_volume:,.0f} mm³")
        self.total_convex_hull_label.setText(f"{total_convex_hull:,.0f} mm³")
        if total_shrinkwrap > 0:
            self.total_shrinkwrap_label.setText(f"{total_shrinkwrap:,.0f} mm³")
        else:
            self.total_shrinkwrap_label.setText("Not calculated")
        self.total_price_label.setText(f"€{total_price:.2f}")
    
    def calculate_all_prices(self):
        """Calculate prices for all project parts using ML models"""
        if not self.project_parts:
            self.show_status_message("No parts to calculate prices for")
            return
            
        print(f"\nCalculating prices for {len(self.project_parts)} parts...")
        
        # Calculate project totals for context
        project_totals = self.calculate_project_totals()
        
        # Check if we have the project-aware model
        use_project_aware = hasattr(self, 'predict_project_part_price') and hasattr(self, 'project_aware_model') and self.project_aware_model
        
        if use_project_aware and project_totals:
            print("🌲 Using Project-Aware Random Forest model")
        elif hasattr(self, 'predict_price_with_ai_model'):
            print("🌳 Using Standard Random Forest model")
        else:
            print("⚠️ No ML models available, using basic calculation")
        
        total_price = 0.0
        
        for i, part_data in enumerate(self.project_parts):
            features = part_data.get('features', {})
            quantity = part_data.get('quantity', 1)
            
            # Prepare features with quantity
            part_features = features.copy()
            part_features['quantity'] = quantity
            part_features['Quantity'] = quantity
            
            try:
                if use_project_aware and project_totals:
                    # Use project-aware model
                    unit_price, _ = self.predict_project_part_price(part_features, project_totals)
                    if unit_price <= 0:
                        raise ValueError("Project model returned 0")
                elif hasattr(self, 'predict_price_with_ai_model'):
                    # Use standard AI model
                    unit_price, _ = self.predict_price_with_ai_model(part_features)
                    if unit_price <= 0:
                        raise ValueError("AI model returned 0")
                else:
                    raise ValueError("No ML model available")
                    
            except Exception as e:
                # Fallback to basic calculation
                print(f"⚠️ ML prediction failed for part {i+1}: {e}")
                volume = features.get('volume', 0)
                surface_area = features.get('surface_area', 0)
                
                material_cost = volume * 0.00005  # €0.05 per 1000 mm³
                machine_cost = surface_area * 0.00001  # €0.01 per 1000 mm²
                labor_cost = 10.0  # Fixed labor cost
                
                unit_price = (material_cost + machine_cost + labor_cost) * 1.2  # 20% markup
                unit_price = max(unit_price, 10.0)  # Minimum €10
            
            # Calculate total for this part
            part_total = unit_price * quantity
            part_data['calculated_price'] = part_total
            part_data['unit_price'] = unit_price
            
            total_price += part_total
            
            print(f"Part {i+1}: €{unit_price:.2f} x {quantity} = €{part_total:.2f}")
            
        # Update totals and UI
        self.update_project_totals()
        
        # Update the project pricing tab if it exists
        if hasattr(self, 'project_pricing_widget'):
            self.project_pricing_widget.project_parts = self.project_parts.copy()
            self.project_pricing_widget.update_project_parts_table()
            self.project_pricing_widget.update_project_totals()
        
        # Update the comprehensive table display
        self.update_project_parts_display()
        
        self.save_project_button.setEnabled(True)
        self.show_status_message(f"Calculated prices for all parts. Total: €{total_price:.2f}")
        print(f"\n✅ Total project price: €{total_price:.2f}")
    
    def calculate_project_totals(self):
        """Calculate project-level totals for ML model input"""
        if not self.project_parts:
            return {}
            
        total_parts = sum(part.get('quantity', 1) for part in self.project_parts)
        total_volume = 0
        total_surface_area = 0
        total_bb_volume = 0
        total_convex_hull = 0
        total_shrinkwrap = 0
        
        for part in self.project_parts:
            features = part.get('features', {})
            qty = part.get('quantity', 1)
            
            total_volume += features.get('volume', 0) * qty
            total_surface_area += features.get('surface_area', 0) * qty
            total_bb_volume += features.get('bb_volume', 0) * qty
            total_convex_hull += features.get('convex_hull_volume', 0) * qty
            total_shrinkwrap += features.get('shrinkwrap_volume', 0) * qty
        
        return {
            'Total_Parts_in_Project': total_parts,
            'Project_Total_Volume': total_volume,
            'Project_Total_surface_area': total_surface_area,
            'Project_Total_bb_volume': total_bb_volume,
            'Project_Total_convex_hull_volume': total_convex_hull,
            'Project_Total_shrinkwrap_volume': total_shrinkwrap
        }
        
    def clear_project(self):
        """Clear all project data"""
        self.project_parts.clear()
        if hasattr(self, 'project_parts_table'):
            self.project_parts_table.setRowCount(0)
        self.update_project_totals()
        self.save_project_button.setEnabled(False)
        print("🗑️ Project cleared")
        
    def save_project(self):
        """Save the current project"""
        self.show_status_message("Save project functionality not implemented")
        
    def load_project(self):
        """Load a project"""
        self.show_status_message("Load project functionality not implemented")
    
    def initialize_ai_pricing_model(self):
        """Initialize the Random Forest AI pricing model"""
        import os
        
        # Try to load the AI model
        model_files = [
            'random_forest_project_aware_model.pkl',
            'stl_analyzer/data/ai_pricing_model.pkl',
            'data/ai_pricing_model.pkl'
        ]
        
        self.pricing_model = None
        self.pricing_scaler = None
        self.ai_model_info = {}
        self.project_aware_model = False
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    if 'project_aware' in model_file:
                        print("🌲 Loading Project-Aware Random Forest model...")
                        self.pricing_model = joblib.load(model_file)
                        self.project_aware_model = True
                        # For project-aware model, we might not have a separate scaler
                        scaler_file = model_file.replace('_model.pkl', '_scaler.pkl')
                        if os.path.exists(scaler_file):
                            self.pricing_scaler = joblib.load(scaler_file)
                    else:
                        print("🌳 Loading Standard Random Forest model...")
                        self.pricing_model = joblib.load(model_file)
                        # Load scaler
                        scaler_file = model_file.replace('_model.pkl', '_scaler.pkl')
                        if os.path.exists(scaler_file):
                            self.pricing_scaler = joblib.load(scaler_file)
                        # Load metadata
                        metadata_file = model_file.replace('_model.pkl', '_metadata.json')
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                self.ai_model_info = json.load(f)
                    
                    print(f"✅ AI model loaded from: {model_file}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load model from {model_file}: {e}")
        
        if not self.pricing_model:
            print("⚠️ No AI pricing model found")
    
    def predict_price_with_ai_model(self, features_or_part):
        """Predict price using the AI model"""
        if not self.pricing_model:
            return 0.0, {'error': 'No AI model loaded'}
        
        try:
            # Prepare features
            if isinstance(features_or_part, dict):
                features = features_or_part
            else:
                features = features_or_part.__dict__ if hasattr(features_or_part, '__dict__') else {}
            
            # Get feature names from model or metadata
            feature_names = getattr(self.pricing_model, 'feature_names_in_', 
                                  self.ai_model_info.get('feature_names', []))
            
            if not feature_names:
                return 0.0, {'error': 'No feature names available'}
            
            # Create feature vector
            feature_vector = []
            for fname in feature_names:
                value = features.get(fname, 0.0)
                feature_vector.append(value)
            
            # Convert to numpy array
            X = np.array([feature_vector])
            
            # Scale if scaler available
            if self.pricing_scaler:
                X = self.pricing_scaler.transform(X)
            
            # Predict
            prediction = self.pricing_model.predict(X)[0]
            
            # Ensure positive price
            final_price = max(0.0, prediction)
            
            breakdown = {
                'model': 'Random Forest',
                'features_used': len(feature_names),
                'accuracy': self.ai_model_info.get('accuracy', 'N/A'),
                'prediction': final_price
            }
            
            return final_price, breakdown
            
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            return 0.0, {'error': str(e)}
    
    def predict_project_part_price(self, part_features, project_totals):
        """Predict price using project-aware model"""
        if not self.pricing_model:
            return 0.0, {'error': 'No model loaded'}
        
        # Combine part features with project totals
        combined_features = part_features.copy()
        combined_features.update(project_totals)
        
        # Add calculated features
        combined_features['Volume * Quantity'] = part_features.get('volume', 0) * part_features.get('quantity', 1)
        combined_features['shrinkwrap_volume*Quantity'] = part_features.get('shrinkwrap_volume', 0) * part_features.get('quantity', 1)
        combined_features['Surface area*Quantity'] = part_features.get('surface_area', 0) * part_features.get('quantity', 1)
        combined_features['BB_Volume*Quantity'] = part_features.get('bb_volume', 0) * part_features.get('quantity', 1)
        combined_features['Convex_Hull_Volume*Quantity'] = part_features.get('convex_hull_volume', 0) * part_features.get('quantity', 1)
        
        # Use the standard prediction method with combined features
        return self.predict_price_with_ai_model(combined_features)

    # Scanning functionality methods
    def browse_directory(self):
        """Browse for directory to scan"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
        if directory:
            print(f"🔍 Selected directory: {directory}")
            # Use direct reference to dir_input (connected in setup_analyzer_tab)
            if hasattr(self, 'dir_input'):
                self.dir_input.setText(directory)
                print(f"✅ Directory set via direct reference: {self.dir_input.text()}")
            else:
                # Fallback: Find the analyzer tab and update its directory input
                analyzer_tab = None
                for i in range(self.tab_widget.count()):
                    if self.tab_widget.tabText(i) == "STL Analyzer":
                        analyzer_tab = self.tab_widget.widget(i)
                        break
                
                if analyzer_tab and hasattr(analyzer_tab, 'dir_input'):
                    analyzer_tab.dir_input.setText(directory)
                    print(f"✅ Directory set via fallback method: {analyzer_tab.dir_input.text()}")
                else:
                    print("❌ Could not set directory - no dir_input found")
            
            self.show_status_message(f"Selected directory: {directory}")

    def start_scan(self):
        """Start scanning the selected directory"""
        # Get directory from direct reference or find analyzer tab
        directory = ""
        if hasattr(self, 'dir_input'):
            directory = self.dir_input.text()
        else:
            # Fallback: Get the analyzer tab
            analyzer_tab = None
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    break
            
            if not analyzer_tab or not hasattr(analyzer_tab, 'dir_input'):
                QtWidgets.QMessageBox.warning(self, "Error", "Could not find analyzer tab or directory input")
                return
                
            directory = analyzer_tab.dir_input.text()
        print(f"🚀 Starting scan with directory: '{directory}'")
        
        if not directory or not os.path.isdir(directory):
            print(f"❌ Invalid directory: '{directory}' (exists: {os.path.exists(directory) if directory else False})")
            QtWidgets.QMessageBox.warning(
                self, "Invalid Directory", 
                "Please select a valid directory to scan."
            )
            return
        
        # Check if database already has entries
        current_entries = len(self.stl_db.get_all_entries()) if hasattr(self, 'stl_db') and self.stl_db else 0
        
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
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"stl_database_backup_{timestamp}.db"
                
                try:
                    # Save current database
                    current_df = self.stl_db.get_all_entries()
                    import sqlite3
                    conn = sqlite3.connect(backup_path)
                    current_df.to_sql('stl_files', conn, if_exists='replace', index=False)
                    conn.close()
                    
                    # Clear current database
                    self.stl_db.clear()
                    self.update_database_view()
                    
                    self.show_status_message(f"Previous database saved as {backup_path}")
                    
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Save Error",
                        f"Failed to save current database:\n{str(e)}\n\nContinuing with scan..."
                    )
            
        # Clear previous scan data
        self.found_files = []
        
        # Update UI elements in analyzer tab
        if hasattr(self, 'scan_progress'):
            self.scan_progress.setValue(0)
            self.scan_progress.setVisible(True)
        
        # Start scanning thread
        from .threads import ScanThread
        self.scan_thread = ScanThread(directory)
        self.scan_thread.progress.connect(self.update_scan_progress)
        self.scan_thread.found_file.connect(self.add_found_file)
        self.scan_thread.scan_complete.connect(self.scan_completed)
        self.scan_thread.start()
        
        self.show_status_message("Scanning for STL files...")
        
    def stop_scan(self):
        """Stop the scanning process"""
        if hasattr(self, 'scan_thread') and self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.show_status_message("Scan stopped by user")
            
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.show_status_message("Processing stopped by user")

    def update_scan_progress(self, value):
        """Update the scan progress bar"""
        # Use direct reference to scan_progress
        if hasattr(self, 'scan_progress'):
            self.scan_progress.setValue(value)
        else:
            # Fallback: Find the analyzer tab and update its progress bar
            analyzer_tab = None
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "STL Analyzer":
                    analyzer_tab = self.tab_widget.widget(i)
                    break
            
            if analyzer_tab and hasattr(analyzer_tab, 'scan_progress'):
                analyzer_tab.scan_progress.setValue(value)

    def add_found_file(self, file_path):
        """Add a found STL file to the list"""
        if not hasattr(self, 'found_files'):
            self.found_files = []
        self.found_files.append(file_path)
        self.show_status_message(f"Found: {os.path.basename(file_path)}")

    def scan_completed(self, count):
        """Handle scan completion"""
        self.show_status_message(f"Scan complete. Found {count} STL files.")
        
        # Start processing the found files
        if hasattr(self, 'found_files') and self.found_files:
            self.process_files()
        else:
            # Hide progress bar
            if hasattr(self, 'scan_progress'):
                self.scan_progress.setVisible(False)

    def process_files(self):
        """Process found STL files to extract features"""
        files_to_process = getattr(self, 'found_files', [])
        
        if not files_to_process:
            self.show_status_message("No files to process")
            return
            
        self.show_status_message(f"Processing {len(files_to_process)} STL files...")
        
        # Create a filtered list of files that exist
        existing_files = []
        for file_path in files_to_process:
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                self.show_status_message(f"Skipped missing file: {os.path.basename(file_path)}")
        
        if not existing_files:
            self.show_status_message("No valid files found to process")
            return
        
        # Get analyzer tab for processing parameters
        analyzer_tab = None
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "STL Analyzer":
                analyzer_tab = self.tab_widget.widget(i)
                break
        
        # Default processing parameters
        calculate_shrinkwrap = True
        generate_files = False
        shrinkwrap_offset = 5.0
        
        # Get parameters from analyzer tab if available
        if analyzer_tab:
            if hasattr(analyzer_tab, 'generate_shrinkwrap_checkbox'):
                calculate_shrinkwrap = analyzer_tab.generate_shrinkwrap_checkbox.isChecked()
            if hasattr(analyzer_tab, 'generate_files_checkbox'):
                generate_files = analyzer_tab.generate_files_checkbox.isChecked()
        
        # Start processing in a separate thread
        from .threads import ProcessThread
        self.process_thread = ProcessThread(
            existing_files,
            generate_shrinkwrap=calculate_shrinkwrap,
            generate_files=generate_files,
            shrinkwrap_offset=shrinkwrap_offset,
            output_mode="dedicated",
            custom_dir=""
        )
        self.process_thread.progress.connect(self.update_scan_progress)
        self.process_thread.processed_file.connect(self.add_processed_file)
        self.process_thread.process_complete.connect(self.processing_completed)
        self.process_thread.start()

    def add_processed_file(self, features):
        """Add processed STL data to the database"""
        try:
            from ..stl_utils import is_valid_entry
        except ImportError:
            # Fallback for direct execution
            from stl_utils import is_valid_entry
        
        # Additional check to ensure file exists and features are valid
        if features and os.path.exists(features["filename"]) and is_valid_entry(features):
            if hasattr(self, 'stl_db') and self.stl_db:
                self.stl_db.add_entry(features)
                self.show_status_message(f"Processed: {os.path.basename(features['filename'])}")
        else:
            filename = features.get('filename', 'Unknown') if features else 'Unknown'
            self.show_status_message(f"Skipped invalid file: {os.path.basename(filename)}")

    def processing_completed(self, count):
        """Handle processing completion"""
        self.show_status_message(f"Processing complete. Added {count} files to database.")
        
        # Save database and update view
        if hasattr(self, 'stl_db') and self.stl_db:
            self.stl_db.save_database()
        self.update_database_view()
        
        # Hide progress bar
        if hasattr(self, 'scan_progress'):
            self.scan_progress.setVisible(False)
        
        # Show completion message
        self.show_status_message(f"Scan completed! Added {count} STL files to database.")

def run_app():
    """Run the STL Analyzer application"""
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    return app.exec_() 