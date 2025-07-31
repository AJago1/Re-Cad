"""
Similarity Search Widget Component

This widget provides complete similarity search functionality:
- Part comparison loading
- Search parameter selection
- Similarity algorithms
- Results filtering and display
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import os

class SimilaritySearchPanel(QtWidgets.QGroupBox):
    """Complete similarity search panel widget"""
    
    def __init__(self, parent=None):
        super().__init__("Similarity Search", parent)
        self.parent_window = parent
        self.loaded_comparison_part = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the similarity search UI"""
        sim_layout = QtWidgets.QVBoxLayout(self)
        
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
    
    def load_comparison_part(self):
        """Load an STL file for comparison against the database"""
        if self.parent_window:
            self.parent_window.load_comparison_part()
            # Update our local reference
            self.loaded_comparison_part = getattr(self.parent_window, 'loaded_comparison_part', None)
            if hasattr(self.parent_window, 'comparison_part_input'):
                self.comparison_part_input.setText(self.parent_window.comparison_part_input.text())
    
    def find_similar_to_loaded_part(self):
        """Find parts similar to the loaded comparison part"""
        if self.parent_window:
            self.parent_window.find_similar_to_loaded_part()
    
    def find_similar_parts(self):
        """Find parts similar to the currently selected part"""
        if self.parent_window:
            self.parent_window.find_similar_parts()
    
    def show_all_database_parts(self):
        """Show all parts in the database (clear similarity filter)"""
        if self.parent_window:
            self.parent_window.show_all_database_parts()
    
    def get_similarity_param(self):
        """Get the selected similarity parameter"""
        return self.similarity_param.currentText()
    
    def get_similarity_count(self):
        """Get the number of results to return"""
        return self.similarity_count.value() 