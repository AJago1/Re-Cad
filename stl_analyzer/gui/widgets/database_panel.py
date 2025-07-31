"""
Database Management Panel Widget

This widget provides complete database management functionality:
- Database table view with sorting and filtering
- Database controls (fix, save, load)
- File validation and cleanup
- Database info display
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import os
import pandas as pd

class DatabasePanel(QtWidgets.QGroupBox):
    """Complete database management panel widget"""
    
    def __init__(self, parent=None):
        super().__init__("STL Database", parent)
        self.parent_window = parent
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the database view UI"""
        db_layout = QtWidgets.QVBoxLayout(self)
        
        # Database controls
        db_controls_layout = QtWidgets.QHBoxLayout()
        
        # Fix database button
        fix_db_button = QtWidgets.QPushButton("Fix Database")
        fix_db_button.clicked.connect(self.fix_database)
        fix_db_button.setToolTip("Remove invalid entries and fix database issues")
        db_controls_layout.addWidget(fix_db_button)
        
        # Save database button
        save_db_button = QtWidgets.QPushButton("Save As...")
        save_db_button.clicked.connect(self.save_database_as)
        save_db_button.setToolTip("Save database to a different file")
        db_controls_layout.addWidget(save_db_button)
        
        # Load database button
        load_db_button = QtWidgets.QPushButton("Load...")
        load_db_button.clicked.connect(self.load_database_from_file)
        load_db_button.setToolTip("Load database from file")
        db_controls_layout.addWidget(load_db_button)
        
        db_layout.addLayout(db_controls_layout)
        
        # Database table
        self.db_table = QtWidgets.QTableView()
        self.db_table.setAlternatingRowColors(True)
        self.db_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.db_table.setSortingEnabled(True)
        self.db_table.clicked.connect(self.on_db_table_clicked)
        
        # Set minimum height for the table
        self.db_table.setMinimumHeight(200)
        
        db_layout.addWidget(self.db_table)
        
        # Database info label
        self.db_info_label = QtWidgets.QLabel("Database: Ready")
        self.db_info_label.setStyleSheet("color: #666; font-style: italic;")
        db_layout.addWidget(self.db_info_label)
    
    def fix_database(self):
        """Fix database by attempting to validate entries and remove invalid ones"""
        if self.parent_window:
            self.parent_window.fix_database()
    
    def save_database_as(self):
        """Save the current database to a file"""
        if self.parent_window:
            self.parent_window.save_database_as()
    
    def load_database_from_file(self):
        """Load a database from a file"""
        if self.parent_window:
            self.parent_window.load_database_from_file()
    
    def on_db_table_clicked(self, index):
        """Handle click on database table row"""
        if self.parent_window:
            self.parent_window.on_db_table_clicked(index)
    
    def update_view(self):
        """Update the database view with current data"""
        if self.parent_window:
            self.parent_window.update_database_view() 