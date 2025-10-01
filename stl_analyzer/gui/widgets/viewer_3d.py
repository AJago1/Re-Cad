"""
3D Viewer Widget Component

This widget provides a complete 3D STL viewing interface with:
- STL file visualization
- Viewer controls (edges, axes, bounding box, etc.)
- Shrinkwrap parameters and controls  
- Optimization features
- Import functionality
"""

from PyQt5 import QtWidgets, QtCore

# Import viewer module - handle both package and direct imports
try:
    # When imported as a package
    from ...viewer import STLViewer
except ImportError:
    try:
        # When imported from gui package
        from stl_analyzer.viewer import STLViewer
    except ImportError:
        try:
            # When run directly
            from viewer import STLViewer
        except ImportError:
            # Ultimate fallback - create a dummy class
            print("⚠️ STLViewer not available - creating fallback")
            class STLViewer(QtWidgets.QWidget):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    label = QtWidgets.QLabel("3D Viewer not available")
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    layout = QtWidgets.QVBoxLayout(self)
                    layout.addWidget(label)
                
                def load_stl(self, *args, **kwargs):
                    pass
                
                def clear(self):
                    pass


class STLViewerWidget(QtWidgets.QGroupBox):
    """Complete 3D viewer widget with controls and parameters"""
    
    def __init__(self, parent=None):
        super().__init__("3D Model Preview", parent)
        self.parent_window = parent
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the 3D viewer UI components"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create the STL viewer
        self.stl_viewer = STLViewer(self.parent_window)
        layout.addWidget(self.stl_viewer)
        
        # Connect signals to parent if available
        if self.parent_window:
            try:
                self.stl_viewer.file_dropped.connect(self.parent_window.on_file_dropped)
                self.stl_viewer.file_loaded.connect(self.parent_window.update_details_view)
                self.stl_viewer.shrinkwrap_updated.connect(self.parent_window.update_details_view)
                self.stl_viewer.optimization_complete.connect(self.parent_window.on_optimization_complete)
            except AttributeError:
                # Parent doesn't have these methods, that's OK
                pass
        
        # Add viewer controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Toggle buttons for visualization options
        self.toggle_edges_btn = QtWidgets.QPushButton("Edges")
        self.toggle_edges_btn.setCheckable(True)
        self.toggle_edges_btn.clicked.connect(lambda checked: self.stl_viewer.set_edges_visible(checked))
        controls_layout.addWidget(self.toggle_edges_btn)
        
        self.toggle_axes_btn = QtWidgets.QPushButton("Axes")
        self.toggle_axes_btn.setCheckable(True)
        self.toggle_axes_btn.setChecked(True)
        self.toggle_axes_btn.clicked.connect(lambda checked: self.stl_viewer.set_axes_visible(checked))
        controls_layout.addWidget(self.toggle_axes_btn)
        
        self.toggle_bb_btn = QtWidgets.QPushButton("Bounding Box")
        self.toggle_bb_btn.setCheckable(True)
        self.toggle_bb_btn.clicked.connect(lambda checked: self.stl_viewer.set_bounding_box_visible(checked))
        controls_layout.addWidget(self.toggle_bb_btn)
        
        self.toggle_ch_btn = QtWidgets.QPushButton("Convex Hull")
        self.toggle_ch_btn.setCheckable(True)
        self.toggle_ch_btn.clicked.connect(lambda checked: self.stl_viewer.set_convex_hull_visible(checked))
        controls_layout.addWidget(self.toggle_ch_btn)
        
        self.toggle_sw_btn = QtWidgets.QPushButton("Shrinkwrap")
        self.toggle_sw_btn.setCheckable(True)
        self.toggle_sw_btn.clicked.connect(lambda checked: self.stl_viewer.set_shrinkwrap_visible(checked))
        controls_layout.addWidget(self.toggle_sw_btn)
        
        self.toggle_bvh_btn = QtWidgets.QPushButton("8-Leaf BVH")
        self.toggle_bvh_btn.setCheckable(True)
        self.toggle_bvh_btn.clicked.connect(lambda checked: self.stl_viewer.set_bvh_8leaf_visible(checked))
        self.toggle_bvh_btn.setToolTip("Show 8-leaf Bounding Volume Hierarchy for collision detection")
        controls_layout.addWidget(self.toggle_bvh_btn)
        
        # Optimize orientation button
        optimize_btn = QtWidgets.QPushButton("Optimize")
        optimize_btn.clicked.connect(lambda: self.stl_viewer.optimize_orientation())
        optimize_btn.setToolTip("Find optimal orientation to minimize bounding box")
        controls_layout.addWidget(optimize_btn)
        
        layout.addLayout(controls_layout)
        
        # Add second row of controls for visualization parameters
        viz_controls_layout = QtWidgets.QHBoxLayout()
        
        # Shrinkwrap parameters
        viz_controls_layout.addWidget(QtWidgets.QLabel("Box Size (mm):"))
        
        self.box_size_slider = QtWidgets.QDoubleSpinBox()
        self.box_size_slider.setRange(0.5, 50.0)
        self.box_size_slider.setSingleStep(0.5)
        self.box_size_slider.setValue(5.0)
        self.box_size_slider.setDecimals(1)
        viz_controls_layout.addWidget(self.box_size_slider)
        
        viz_controls_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        
        self.resolution_slider = QtWidgets.QSpinBox()
        self.resolution_slider.setRange(10, 100)
        self.resolution_slider.setSingleStep(5)
        self.resolution_slider.setValue(40)
        viz_controls_layout.addWidget(self.resolution_slider)
        
        # Update shrinkwrap button
        self.update_shrinkwrap_btn = QtWidgets.QPushButton("Update Shrinkwrap")
        if self.parent_window and hasattr(self.parent_window, 'update_shrinkwrap'):
            self.update_shrinkwrap_btn.clicked.connect(self.parent_window.update_shrinkwrap)
        viz_controls_layout.addWidget(self.update_shrinkwrap_btn)
        
        # Import analysis button
        self.import_button = QtWidgets.QPushButton("Import Analysis")
        self.import_button.setToolTip("Import analysis from a directory without rescanning")
        if self.parent_window and hasattr(self.parent_window, 'import_btn_clicked'):
            self.import_button.clicked.connect(self.parent_window.import_btn_clicked)
        viz_controls_layout.addWidget(self.import_button)
        
        layout.addLayout(viz_controls_layout)
        
        # Store references for parent window compatibility
        if self.parent_window:
            self.parent_window.stl_viewer = self.stl_viewer
            self.parent_window.toggle_edges_btn = self.toggle_edges_btn
            self.parent_window.toggle_axes_btn = self.toggle_axes_btn
            self.parent_window.toggle_bb_btn = self.toggle_bb_btn
            self.parent_window.toggle_ch_btn = self.toggle_ch_btn
            self.parent_window.toggle_sw_btn = self.toggle_sw_btn
            self.parent_window.box_size_slider = self.box_size_slider
            self.parent_window.resolution_slider = self.resolution_slider
            self.parent_window.update_shrinkwrap_btn = self.update_shrinkwrap_btn
            self.parent_window.import_button = self.import_button 