"""
STL Viewer - A simple 3D viewer for STL files using VTK.

This module provides:
1. An STLViewer widget that can be embedded in other applications
2. A standalone viewer function that can be called directly
"""

import os
import sys
import numpy as np
import trimesh
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QProgressDialog
import tempfile
import time

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# For optimization
from scipy.optimize import minimize

# Import C++ extensions and STL utilities
try:
    # Import the C++ compiled extensions if available
    from .Release.cpp_stl_utils import *
    cpp_available = True
    print("C++ extensions loaded - using accelerated implementations")
except ImportError:
    try:
        from Release.cpp_stl_utils import *
        cpp_available = True
        print("C++ extensions loaded - using accelerated implementations")
    except ImportError:
        cpp_available = False
        print("⚠️ C++ extensions not available - using Python implementations")

# Import STL utilities with fallback handling
try:
    from .stl_utils import create_shrinkwrap, create_fixed_box_grid_shrinkwrap, extract_features_from_mesh, smart_minimal_bounding_box, try_load_stl
except ImportError:
    try:
        from stl_utils import create_shrinkwrap, create_fixed_box_grid_shrinkwrap, extract_features_from_mesh, smart_minimal_bounding_box, try_load_stl
    except ImportError:
        print("⚠️ STL utilities not available - some features disabled")
        def create_shrinkwrap(*args, **kwargs):
            return None, 0
        def create_fixed_box_grid_shrinkwrap(*args, **kwargs):
            return None, 0
        def extract_features_from_mesh(*args, **kwargs):
            return {}
        def smart_minimal_bounding_box(*args, **kwargs):
            return (0, 0, 0), 0, None, "fallback"
        def try_load_stl(*args, **kwargs):
            return None


class STLViewer(QtWidgets.QFrame):
    """A simple VTK-based widget for viewing STL files."""
    
    # Add signal definitions
    file_loaded = QtCore.pyqtSignal(str, dict)  # Signal to emit when file is loaded (path, features)
    file_dropped = QtCore.pyqtSignal(str)  # Signal to emit when file is dropped
    shrinkwrap_updated = QtCore.pyqtSignal(dict)
    optimization_complete = QtCore.pyqtSignal(str, dict)  # Signal to emit when optimization is complete (path, features)
    
    def __init__(self, parent=None):
        """Initialize the VTK widget for STL viewing."""
        super().__init__(parent)
        
        # Set up the layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)
        
        # Initialize VTK rendering pipeline
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.10, 0.10, 0.12)  # Dark background to match theme
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Enable trackball camera style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Initialize mesh data
        self.stl_path = None
        self.original_file_path = None  # Store original file path for temporary files
        self.actor = None
        self.mesh = None
        self.convex_hull = None
        self.convex_hull_actor = None
        self.shrinkwrap = None
        self.shrinkwrap_actor = None
        self.shrinkwrap_resolution = 64  # Default voxel resolution
        self.shrinkwrap_closing_size = 1  # Default morphological closing size
        self.shrinkwrap_inflation = 0.02  # Default inflation factor (2% of mesh scale)
        self.preserve_holes = True  # Default to try preserving holes
        self.optimal_transform = None  # Store the optimal transformation matrix
        
        # Initialize display settings
        self.show_edges = False
        self.show_axes = True
        self.show_bounding_box = True
        self.show_convex_hull = False
        self.show_shrinkwrap = False
        self.color = [0.2, 0.6, 0.9]  # Bright blue color for the model
        
        # Orientation correction angle (clockwise rotation around X-axis in degrees)
        self.correction_angle = -90
        
        # Create axes
        self.axes_actor = self.create_axes()
        self.renderer.AddActor(self.axes_actor)
        
        # Create grid
        self.grid_actor = self.create_grid()
        self.renderer.AddActor(self.grid_actor)
        
        # Initialize the interactor
        self.interactor.Initialize()
        
        # Add rendering safety flag
        self.is_rendering = False
        
        # Initial render
        self.safe_render()
        
        # Accept drops for STL files
        self.setAcceptDrops(True)
        
        # For message display
        self.message_actor = None
        self.message_timer = QtCore.QTimer(self)
        self.message_timer.timeout.connect(self.clear_message)
        
        # Display welcome message
        self.display_message("Drop an STL file to view it")
        
        # Store last clicked file
        self.last_clicked_file = None
        
    def create_axes(self):
        """Create coordinate axes."""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(40.0, 40.0, 40.0)
        axes.SetShaftType(0)  # Cylinder shaft
        axes.SetCylinderRadius(0.01)
        axes.SetConeRadius(0.2)
        
        # Add labels
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        # Create transform to position axes at the origin
        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)
        axes.SetUserTransform(transform)
        
        return axes
    
    def create_grid(self):
        """Create a grid on the XY plane."""
        # Create points for grid
        grid_size = 5
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        # Create grid lines along X axis
        for i in range(-grid_size, grid_size + 1):
            # Skip center line to avoid overlap with axes
            if i == 0:
                continue
                
            # Create a line
            p1 = [-grid_size, i, 0]
            p2 = [grid_size, i, 0]
            
            id1 = points.InsertNextPoint(p1)
            id2 = points.InsertNextPoint(p2)
            
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)
        
        # Create grid lines along Y axis
        for i in range(-grid_size, grid_size + 1):
            # Skip center line to avoid overlap with axes
            if i == 0:
                continue
                
            # Create a line
            p1 = [i, -grid_size, 0]
            p2 = [i, grid_size, 0]
            
            id1 = points.InsertNextPoint(p1)
            id2 = points.InsertNextPoint(p2)
            
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)
        
        # Create a polydata to store the grid
        grid = vtk.vtkPolyData()
        grid.SetPoints(points)
        grid.SetLines(lines)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.3, 0.3, 0.3)  # Darker gray to match dark theme
        actor.GetProperty().SetOpacity(0.5)
        
        return actor
    
    def get_rotation_correction(self):
        """Get the rotation correction matrix based on the current correction angle."""
        # Create rotation matrix for the X-axis
        return trimesh.transformations.rotation_matrix(
            angle=np.radians(self.correction_angle),
            direction=[1, 0, 0]  # Around X-axis
        )
    
    def load_stl(self, filepath, optimal_transform=None):
        """Load an STL file and prepare it for rendering.
        
        Args:
            filepath: Path to the STL file
            optimal_transform: Optional transformation matrix to apply for optimal orientation
                              (if None, will automatically calculate optimal orientation)
        """
        try:
            # Print diagnostic info
            print(f"\n=== LOADING STL: {filepath} ===")
            
            # Check if file exists
            if not os.path.exists(filepath):
                self.display_message(f"File not found: {filepath}")
                return False
                
            # Clear previous model data to prevent confusion
            self.clear_previous_model_data()
                
            # Store the file path
            self.stl_path = filepath
            
            # Load the mesh
            mesh = try_load_stl(filepath)
            if mesh is None:
                self.display_message(f"Failed to load STL file: {filepath}")
                return False
                
            # Store original mesh for reference
            self.original_mesh = mesh.copy()
            
            # Apply transformation if provided, otherwise calculate optimal orientation
            if optimal_transform is not None:
                print(f"Applying provided optimal transform")
                # Apply the provided transformation
                mesh.apply_transform(optimal_transform)
                self.mesh = mesh
                
                # Store the transform for future use
                self.optimal_transform = optimal_transform
                
                # Update display
                self.update_display()
                self.display_message(f"Loaded with stored optimal orientation: {os.path.basename(filepath)}")
                
            else:
                print(f"No optimal transform provided, calculating optimal orientation")
                # Store the mesh and calculate optimal orientation
                self.mesh = mesh
                
                # Calculate optimal orientation automatically
                self.orient_to_minimal_bounding_box()
                
                self.display_message(f"Loaded and optimized: {os.path.basename(filepath)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading STL file: {e}")
            import traceback
            traceback.print_exc()
            self.display_message(f"Error loading file: {str(e)}")
            return False
            
    def create_bounding_box_from_bounds(self, bounds, transform=None):
        """Create a bounding box actor from bounds."""
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        
        # Create a vtkCubeSource for the bounding box
        cube = vtk.vtkCubeSource()
        cube.SetBounds(min_x, max_x, min_y, max_y, min_z, max_z)
        
        # Create mapper directly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        # Create the actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.9, 0.5, 0.0)  # Orange bounding box for visibility
        actor.GetProperty().SetLineWidth(2.0)
        
        # Apply transform if provided
        if transform:
            actor.SetUserTransform(transform)
        
        return actor
    
    def create_convex_hull_visualization(self):
        """Create a visualization of the convex hull that matches the main model's orientation."""
        if not self.mesh:
            return False
            
        try:
            # Remove old convex hull actor if exists
            if self.convex_hull_actor and self.convex_hull_actor in self.renderer.GetActors():
                self.renderer.RemoveActor(self.convex_hull_actor)
            
            # Always recalculate convex hull to ensure it's current
            try:
                print("Creating convex hull for visualization...")
                self.convex_hull = self.mesh.convex_hull
                
                if self.convex_hull is None or not hasattr(self.convex_hull, 'vertices') or len(self.convex_hull.vertices) == 0:
                    print("Failed to create valid convex hull")
                    self.display_message("Cannot create convex hull for this model", 3000)
                    return False
                    
                print(f"Created convex hull with {len(self.convex_hull.vertices)} vertices and volume: {self.convex_hull.volume:.2f} mm³")
                
            except Exception as e:
                print(f"Error creating convex hull: {e}")
                self.display_message("Error creating convex hull", 3000)
                return False

            # Create a temporary file for the convex hull with better error handling
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    tmp_path = tmp.name
                    # Export the convex hull to STL
                    self.convex_hull.export(tmp_path)
                    
                # Verify the file was created and has content
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    print("Failed to export convex hull to temporary file")
                    return False
                    
            except Exception as e:
                print(f"Error exporting convex hull: {e}")
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                return False
                
            # Read it with VTK with error handling
            try:
                reader = vtk.vtkSTLReader()
                reader.SetFileName(tmp_path)
                reader.Update()
                
                # Check if VTK successfully read the file
                polydata = reader.GetOutput()
                if polydata.GetNumberOfPoints() == 0:
                    print("VTK reader produced empty polydata")
                    return False
                
            except Exception as e:
                print(f"Error reading convex hull with VTK: {e}")
                return False
            finally:
                # Clean up temporary file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            # Create mapper for the hull
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            # Create actor
            self.convex_hull_actor = vtk.vtkActor()
            self.convex_hull_actor.SetMapper(mapper)
            
            # Apply the EXACT same transform as the main model to ensure alignment
            if self.actor:
                transform = self.actor.GetUserTransform()
                if transform:
                    # Create an exact copy of the transform
                    new_transform = vtk.vtkTransform()
                    new_transform.DeepCopy(transform)
                    self.convex_hull_actor.SetUserTransform(new_transform)
            
            # Set improved wireframe properties for convex hull
            self.convex_hull_actor.GetProperty().SetColor(1.0, 0.7, 0.2)  # Bright orange color
            self.convex_hull_actor.GetProperty().SetOpacity(1.0)  # Fully opaque for visibility
            self.convex_hull_actor.GetProperty().SetRepresentationToWireframe()  # Wireframe only
            self.convex_hull_actor.GetProperty().SetLineWidth(2.0)  # Thicker lines
            
            # Add to renderer if show_convex_hull is enabled
            if self.show_convex_hull:
                self.renderer.AddActor(self.convex_hull_actor)
            
            print("Convex hull visualization created successfully")
            return True
                
        except Exception as e:
            print(f"Error creating convex hull visualization: {e}")
            import traceback
            traceback.print_exc()
            self.display_message("Failed to create convex hull visualization", 3000)
            return False
    
    def create_shrinkwrap_visualization(self):
        """Create a visualization of the shrinkwrap mesh that fills small holes."""
        if not self.mesh:
            return
        
        try:
            # Remove old shrinkwrap actor if exists
            if hasattr(self, 'shrinkwrap_actor') and self.shrinkwrap_actor and self.shrinkwrap_actor in self.renderer.GetActors():
                self.renderer.RemoveActor(self.shrinkwrap_actor)
            
            # Ensure we have a shrinkwrap - calculate if needed
            if not hasattr(self, 'shrinkwrap') or self.shrinkwrap is None:
                self.display_message("Creating shrinkwrap, please wait...", 2000)
                self.vtk_widget.GetRenderWindow().Render()  # Update display
                
                try:
                    # Use the faster Surface Offset method for visualization (same as used in analysis)
                    print("Creating shrinkwrap using Surface Offset method for fast visualization...")
                    
                    # Import the surface offset function
                    try:
                        from .stl_utils import create_surface_offset_shrinkwrap
                    except ImportError:
                        from stl_utils import create_surface_offset_shrinkwrap
                    
                    # Create shrinkwrap using Surface Offset method (faster and more reliable)
                    self.shrinkwrap = create_surface_offset_shrinkwrap(
                        self.mesh, 
                        offset_mm=4.0  # Use 4.0mm offset like in the analysis
                    )
                    
                    if self.shrinkwrap and self.shrinkwrap.volume > 0:
                        print(f"Created surface offset shrinkwrap with volume: {self.shrinkwrap.volume:.2f} mm³")
                    else:
                        print("Failed to create shrinkwrap using Surface Offset method")
                        self.display_message("Failed to create shrinkwrap", 3000)
                        return False
                except Exception as e:
                    print(f"Error creating shrinkwrap: {e}")
                    self.display_message("Error creating shrinkwrap", 3000)
                    return False
            
            # Verify the shrinkwrap is valid
            if not hasattr(self, 'shrinkwrap') or self.shrinkwrap is None:
                self.display_message("No valid shrinkwrap available", 3000)
                return False
            
            # Create a temporary file for the shrinkwrap
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_path = tmp.name
                # Export the shrinkwrap to STL
                self.shrinkwrap.export(tmp_path)
                
            # Read it with VTK
            reader = vtk.vtkSTLReader()
            reader.SetFileName(tmp_path)
            reader.Update()
            
            # Get the polydata from the reader
            polydata = reader.GetOutput()
            
            # Create mapper for the shrinkwrap
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # Create actor
            self.shrinkwrap_actor = vtk.vtkActor()
            self.shrinkwrap_actor.SetMapper(mapper)
            
            # Apply the EXACT same transform as the main model to ensure alignment
            if self.actor:
                # Get the user transform from the main actor
                transform = self.actor.GetUserTransform()
                if transform:
                    # Create an exact copy of the transform
                    new_transform = vtk.vtkTransform()
                    new_transform.DeepCopy(transform)
                    self.shrinkwrap_actor.SetUserTransform(new_transform)
            
            # Set improved wireframe properties for shrinkwrap - matching convex hull style
            self.shrinkwrap_actor.GetProperty().SetColor(0.2, 0.8, 0.4)  # Bright green color
            self.shrinkwrap_actor.GetProperty().SetOpacity(1.0)  # Fully opaque for visibility
            self.shrinkwrap_actor.GetProperty().SetRepresentationToWireframe()  # Wireframe only
            self.shrinkwrap_actor.GetProperty().SetLineWidth(2.0)  # Thicker lines like convex hull
            
            # Add to renderer if show_shrinkwrap is enabled
            if self.show_shrinkwrap:
                self.renderer.AddActor(self.shrinkwrap_actor)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            self.display_message("Shrinkwrap created", 2000)
            return True
                
        except Exception as e:
            print(f"Error creating shrinkwrap visualization: {e}")
            import traceback
            traceback.print_exc()
            self.display_message("Error visualizing shrinkwrap", 3000)
            return False
    
    def set_edges_visible(self, visible):
        """Set edges visibility."""
        self.show_edges = visible
        if self.actor:
            if visible:
                self.actor.GetProperty().EdgeVisibilityOn()
            else:
                self.actor.GetProperty().EdgeVisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
    
    def set_axes_visible(self, visible):
        """Set axes visibility."""
        self.show_axes = visible
        if visible:
            self.renderer.AddActor(self.axes_actor)
        else:
            self.renderer.RemoveActor(self.axes_actor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_grid_visible(self, visible):
        """Set grid visibility."""
        if visible:
            self.renderer.AddActor(self.grid_actor)
        else:
            self.renderer.RemoveActor(self.grid_actor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_bounding_box_visible(self, visible):
        """Set bounding box visibility."""
        if hasattr(self, 'bbox_actor') and self.bbox_actor:
            if visible:
                self.renderer.AddActor(self.bbox_actor)
            else:
                self.renderer.RemoveActor(self.bbox_actor)
            self.vtk_widget.GetRenderWindow().Render()
    
    def set_convex_hull_visible(self, visible):
        """Set convex hull visibility"""
        self.show_convex_hull = visible
        
        # If enabling convex hull and we don't have one yet, create it
        if visible and (not hasattr(self, 'convex_hull_actor') or self.convex_hull_actor is None):
            if self.mesh:  # Only create if we have a mesh loaded
                self.create_convex_hull_visualization()
        
        # Show/hide the actor if it exists
        if hasattr(self, 'convex_hull_actor') and self.convex_hull_actor:
            if visible:
                self.renderer.AddActor(self.convex_hull_actor)
            else:
                self.renderer.RemoveActor(self.convex_hull_actor)
            self.vtk_widget.GetRenderWindow().Render()
        elif visible:
            # If we tried to show the convex hull but couldn't, display a message
            self.display_message("Convex hull not available for this model", 3000)
    
    def set_shrinkwrap_visible(self, visible):
        """Set shrinkwrap visibility"""
        self.show_shrinkwrap = visible
        
        # If enabling shrinkwrap and we don't have one yet, create it
        if visible and (not hasattr(self, 'shrinkwrap_actor') or self.shrinkwrap_actor is None):
            if self.mesh:  # Only create if we have a mesh loaded
                self.create_shrinkwrap_visualization()
        
        # Show/hide the actor if it exists
        if hasattr(self, 'shrinkwrap_actor') and self.shrinkwrap_actor:
            if visible:
                self.renderer.AddActor(self.shrinkwrap_actor)
            else:
                self.renderer.RemoveActor(self.shrinkwrap_actor)
            self.vtk_widget.GetRenderWindow().Render()
        elif visible:
            # If we tried to show the shrinkwrap but couldn't, display a message
            self.display_message("Shrinkwrap not available for this model", 3000)
    
    def toggle_wireframe(self):
        """Toggle wireframe mode."""
        self.show_edges = not self.show_edges
        self.set_edges_visible(self.show_edges)
        return self.show_edges
    
    def set_view_angle(self, preset):
        """Set the view to a preset angle."""
        if not self.renderer:
            return
            
        camera = self.renderer.GetActiveCamera()
        
        if preset == "top":
            camera.SetPosition(0, 0, 10)
            camera.SetViewUp(0, 1, 0)
            camera.SetFocalPoint(0, 0, 0)
        elif preset == "front":
            camera.SetPosition(0, -10, 0)
            camera.SetViewUp(0, 0, 1)
            camera.SetFocalPoint(0, 0, 0)
        elif preset == "side":
            camera.SetPosition(10, 0, 0)
            camera.SetViewUp(0, 0, 1)
            camera.SetFocalPoint(0, 0, 0)
        elif preset == "isometric":
            # Adjust isometric view to work better with the corrected orientation
            camera.SetPosition(5, -5, 5)
            camera.SetViewUp(0, 0, 1)
            camera.SetFocalPoint(0, 0, 0)
        
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_view(self):
        """Reset the view camera."""
        if self.renderer:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera_position(self):
        """Reset the camera position (alias for reset_view for compatibility)."""
        self.reset_view()
    
    def safe_render(self):
        """Safely render the scene with protection against concurrent rendering."""
        try:
            if hasattr(self, 'is_rendering') and self.is_rendering:
                return  # Already rendering, skip
            
            if not self.vtk_widget or not self.vtk_widget.GetRenderWindow():
                return  # No render window available
            
            self.is_rendering = True
            self.vtk_widget.GetRenderWindow().Render()
            self.is_rendering = False
            
        except Exception as e:
            self.is_rendering = False
            print(f"Warning: Render error: {e}")
    
    def display_message(self, message, duration=3000):
        """Display a message in the viewer."""
        # Remove old message if exists
        if self.message_actor:
            self.renderer.RemoveActor2D(self.message_actor)
            
        # Create text actor
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(message)
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        text_actor.GetTextProperty().SetJustificationToCentered()
        text_actor.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0)
        text_actor.GetTextProperty().SetBackgroundOpacity(0.5)
        
        # Position at the bottom center
        text_actor.SetPosition(300, 20)
        
        # Add to renderer
        self.renderer.AddActor2D(text_actor)
        self.message_actor = text_actor
        
        # Render
        self.vtk_widget.GetRenderWindow().Render()
        
        # Start timer to clear message
        self.message_timer.stop()  # Stop any previous timer
        self.message_timer.start(duration)
    
    def clear_message(self):
        """Clear the displayed message."""
        if self.message_actor:
            self.renderer.RemoveActor2D(self.message_actor)
            self.message_actor = None
            self.vtk_widget.GetRenderWindow().Render()
    
    def get_current_file(self):
        """Get the currently loaded file path."""
        # Return the original file path if available (used for reoriented meshes)
        return self.original_file_path if self.original_file_path else self.stl_path
    
    def has_mesh(self):
        """Check if a mesh is currently loaded."""
        return self.mesh is not None and hasattr(self.mesh, 'vertices') and len(self.mesh.vertices) > 0
    
    def update_display(self):
        """Update the display after mesh modifications."""
        if not self.has_mesh():
            return
            
        try:
            # Re-export and reload the mesh to update the VTK display
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_path = tmp.name
                self.mesh.export(tmp_path)
            
            # Create STL reader
            reader = vtk.vtkSTLReader()
            reader.SetFileName(tmp_path)
            reader.Update()
            
            # Get the polydata from the reader
            polydata = reader.GetOutput()
            
            # Calculate the bounds of the model
            bounds = polydata.GetBounds()
            min_x, max_x, min_y, max_y, min_z, max_z = bounds
            
            # Position model with corner at origin
            corner_transform = vtk.vtkTransform()
            corner_transform.Translate(-min_x, -min_y, -min_z)
            
            # Update the existing actor OR create new one if none exists
            if self.actor:
                # Update the mapper with new data
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                self.actor.SetMapper(mapper)
                self.actor.SetUserTransform(corner_transform)
            else:
                # Create new actor for first time loading
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                
                self.actor = vtk.vtkActor()
                self.actor.SetMapper(mapper)
                self.actor.SetUserTransform(corner_transform)
                
                # Set model color and properties
                self.actor.GetProperty().SetColor(self.color)
                if self.show_edges:
                    self.actor.GetProperty().EdgeVisibilityOn()
                
                # Add to renderer
                self.renderer.AddActor(self.actor)
                print(f"Created VTK actor for {os.path.basename(self.stl_path) if self.stl_path else 'mesh'}")
            
            # Create or update bounding box
            if not hasattr(self, 'bbox_actor') or self.bbox_actor is None:
                bbox = self.create_bounding_box_from_bounds(bounds)
                self.bbox_actor = bbox
                self.bbox_actor.SetUserTransform(corner_transform)
                if self.show_bounding_box:
                    self.renderer.AddActor(self.bbox_actor)
            else:
                # Update bounding box
                if hasattr(self, 'bbox_actor') and self.bbox_actor:
                    self.renderer.RemoveActor(self.bbox_actor)
                    bbox = self.create_bounding_box_from_bounds(bounds)
                    self.bbox_actor = bbox
                    self.bbox_actor.SetUserTransform(corner_transform)
                    if self.show_bounding_box:
                        self.renderer.AddActor(self.bbox_actor)
            
            # Update convex hull if needed
            if self.show_convex_hull:
                self.create_convex_hull_visualization()
                
            # Update shrinkwrap if needed  
            if self.show_shrinkwrap:
                self.create_shrinkwrap_visualization()
            
            # Auto-fit camera to model size - CRITICAL for proper viewing
            self.renderer.ResetCamera()
            
            # Render the scene
            self.safe_render()
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()
        
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith('.stl'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.stl'):
                    # Load the first STL file found
                    success = self.load_stl(file_path)
                    
                    if not success:
                        self.display_message(f"Error loading file: {file_path}")
                        break
                    
                    # First emit file_dropped for compatibility with existing code
                    self.file_dropped.emit(file_path)
                    
                    # Calculate all details immediately for updating UI
                    if hasattr(self, 'mesh') and self.mesh:
                        try:
                            # Calculate all model details
                            details = {
                                'filename': file_path,
                                'volume': self.mesh.volume if hasattr(self.mesh, 'volume') else 0,
                                'surface_area': self.mesh.area if hasattr(self.mesh, 'area') else 0,
                                'bounding_box_volume': self.mesh.bounding_box.volume if hasattr(self.mesh, 'bounding_box') else 0,
                                'convex_hull_volume': self.convex_hull.volume if self.convex_hull else 0,
                                'shrinkwrap_volume': self.shrinkwrap.volume if self.shrinkwrap else 0
                            }
                            
                            # Calculate convexity and shrinkwrap ratios
                            if details['volume'] > 0 and details['convex_hull_volume'] > 0:
                                details['convexity_ratio'] = details['volume'] / details['convex_hull_volume']
                            else:
                                details['convexity_ratio'] = 0
                                
                            if details['volume'] > 0 and details['shrinkwrap_volume'] > 0:
                                details['shrinkwrap_ratio'] = details['volume'] / details['shrinkwrap_volume']
                            else:
                                details['shrinkwrap_ratio'] = 0
                            
                            # Add bounding box dimensions
                            if hasattr(self.mesh, 'bounding_box'):
                                extents = self.mesh.bounding_box.extents
                                details['dimensions'] = extents
                                details['x_dim'] = extents[0]
                                details['y_dim'] = extents[1]
                                details['z_dim'] = extents[2]
                            
                            # Calculate waste (BB volume - part volume)
                            if details['bounding_box_volume'] > 0 and details['volume'] > 0:
                                details['waste_volume'] = details['bounding_box_volume'] - details['volume']
                                details['waste_ratio'] = details['waste_volume'] / details['bounding_box_volume']
                            else:
                                details['waste_volume'] = 0
                                details['waste_ratio'] = 0
                            
                            # Emit signal with file path and details directly to update the details view
                            self.file_loaded.emit(file_path, details)
                            
                            self.display_message(f"Loaded: {os.path.basename(file_path)}")
                        except Exception as e:
                            print(f"Error calculating details in dropEvent: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    break
                    
        event.acceptProposedAction()

    def closeEvent(self, event):
        """Handle close event to properly clean up VTK resources."""
        try:
            # Stop any timers first
            if hasattr(self, 'message_timer') and self.message_timer:
                self.message_timer.stop()
            
            # Stop rendering and clear all actors
            if self.renderer:
                # Remove all actors from renderer
                actors_to_remove = []
                actor_collection = self.renderer.GetActors()
                actor_collection.InitTraversal()
                
                for i in range(actor_collection.GetNumberOfItems()):
                    actor = actor_collection.GetNextActor()
                    if actor:
                        actors_to_remove.append(actor)
                
                for actor in actors_to_remove:
                    self.renderer.RemoveActor(actor)
                
                # Clear renderer
                self.renderer.Clear()
            
            # Finalize render window before shutting down interactor
            if self.vtk_widget and self.vtk_widget.GetRenderWindow():
                render_window = self.vtk_widget.GetRenderWindow()
                
                # Stop rendering
                render_window.SetAbortRender(True)
                
                # Remove renderer
                if self.renderer:
                    render_window.RemoveRenderer(self.renderer)
                
                # Finalize the render window
                render_window.Finalize()
            
            # Terminate the interactor AFTER finalizing
            if self.interactor:
                try:
                    self.interactor.TerminateApp()
                except Exception:
                    pass  # Ignore errors during termination
            
            # Clear all VTK object references
            self.actor = None
            self.convex_hull_actor = None
            self.shrinkwrap_actor = None
            if hasattr(self, 'bbox_actor'):
                self.bbox_actor = None
            self.axes_actor = None
            self.grid_actor = None
            self.message_actor = None
            
            # Clear mesh data
            self.mesh = None
            self.convex_hull = None
            self.shrinkwrap = None
            
            # Clear renderer and interactor references
            self.renderer = None
            self.interactor = None
            
            # Clear the VTK widget
            self.vtk_widget = None
                    
        except Exception as e:
            # Ignore any errors during cleanup to prevent crash
            print(f"Warning: Error during VTK cleanup: {e}")
        
        # Call parent implementation
        super().closeEvent(event)

    def optimize_orientation(self):
        """Find optimal orientation to minimize bounding box volume."""
        if not self.mesh:
            print("No mesh loaded for optimization")
            return False
            
        try:
            # Store the original file path before optimization
            original_file_path = self.stl_path
            if not original_file_path:
                print("No file path available for optimization")
                return False
            
            print(f"\n=== OPTIMIZING ORIENTATION FOR: {original_file_path} ===")
            
            # Get current bounding box volume
            current_size, current_volume, current_transform, current_method = smart_minimal_bounding_box(self.mesh)
            original_volume = current_volume
            
            print(f"Current bounding box volume: {current_volume:.2f}")
            print(f"Current dimensions: {current_size[0]:.2f} x {current_size[1]:.2f} x {current_size[2]:.2f}")
            
            # Try different optimization strategies
            best_volume = current_volume
            best_transform = np.eye(4)
            best_method = current_method
            best_size = current_size
            
            # Strategy 1: Multiple PCA attempts with different sampling
            for attempt in range(3):
                test_mesh = self.mesh.copy()
                
                # Add small random perturbation to break symmetries
                if attempt > 0:
                    vertices = test_mesh.vertices
                    noise = np.random.normal(0, 0.001, vertices.shape) * np.std(vertices, axis=0)
                    test_mesh.vertices = vertices + noise
                
                size, volume, transform, method = smart_minimal_bounding_box(test_mesh)
                
                if volume < best_volume:
                    best_volume = volume
                    best_transform = transform
                    best_method = method
                    best_size = size
                    print(f"Attempt {attempt + 1}: Improved to {volume:.2f} using {method}")
            
            # Strategy 2: Try oriented bounding box as fallback
            try:
                obb = self.mesh.bounding_box_oriented
                obb_volume = np.prod(obb.extents)
                
                if obb_volume < best_volume:
                    best_volume = obb_volume
                    best_transform = obb.primitive.transform
                    best_method = "oriented_bounds"
                    # Calculate size from extents
                    best_size = np.sort(obb.extents)[::-1]  # Sort largest to smallest
                    print(f"Oriented bounds: Improved to {obb_volume:.2f}")
            except Exception as e:
                print(f"Oriented bounds strategy failed: {e}")
            
            # Calculate improvement
            volume_reduction = (original_volume - best_volume) / original_volume * 100
            
            # Apply transformation if there's any improvement
            transformation_applied = False
            if volume_reduction > 0.01:  # More than 0.01% improvement
                print(f"Applying transformation: {volume_reduction:.1f}% reduction using {best_method}")
                
                # Apply the transformation to the mesh
                self.mesh.apply_transform(best_transform)
                transformation_applied = True
                
                # Update the display with the optimized mesh
                self.update_display()
                
                # Recalculate the optimal transform for the new orientation
                # (this should now be close to identity since we've applied the optimization)
                self.optimal_transform = np.eye(4)
                
                print(f"Mesh transformed successfully")
            else:
                print(f"Minimal improvement ({volume_reduction:.1f}%), keeping original orientation")
                # Still update the optimal transform for consistency
                self.optimal_transform = best_transform
            
            # ALWAYS extract features from the current mesh state (optimized or not)
            optimized_features = extract_features_from_mesh(
                self.mesh, 
                original_file_path,
                model_name=os.path.splitext(os.path.basename(original_file_path))[0]
            )
            
            if optimized_features:
                # Add optimization metadata
                optimized_features['optimization_reduction'] = volume_reduction
                optimized_features['optimization_method'] = best_method
                optimized_features['transformation_applied'] = transformation_applied
                
                # Store the best transform found (even if not applied)
                optimized_features['optimal_transform'] = best_transform.tolist()
                
                print(f"Optimization complete:")
                print(f"  Method: {best_method}")
                print(f"  Volume reduction: {volume_reduction:.1f}%")
                print(f"  New dimensions: {optimized_features['x']:.2f} x {optimized_features['y']:.2f} x {optimized_features['z']:.2f}")
                print(f"  New BB volume: {optimized_features['bb_volume']:.2f}")
                
                # Emit the signal to update database and UI
                self.optimization_complete.emit(original_file_path, optimized_features)
                
                return True
            else:
                print("Failed to extract features from optimized mesh")
                return False
            
        except Exception as e:
            print(f"Error during orientation optimization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def adjust_rotation(self, degrees, axis='x'):
        """Adjust the rotation of the model and convex hull.
        
        Args:
            degrees: Rotation angle in degrees (positive = clockwise)
            axis: Axis to rotate around ('x', 'y', or 'z')
        """
        if not self.actor:
            return False
            
        try:
            # Get current transform or create a new one
            current_transform = self.actor.GetUserTransform()
            if not current_transform:
                current_transform = vtk.vtkTransform()
                current_transform.Identity()
            
            # Create a copy to avoid modifying the original
            transform = vtk.vtkTransform()
            transform.DeepCopy(current_transform)
            
            # Apply additional rotation
            if axis.lower() == 'x':
                transform.RotateX(degrees)
            elif axis.lower() == 'y':
                transform.RotateY(degrees)
            elif axis.lower() == 'z':
                transform.RotateZ(degrees)
            
            # Apply transform to the model
            self.actor.SetUserTransform(transform)
            
            # Apply the same transform to the convex hull if it exists
            if self.convex_hull_actor:
                self.convex_hull_actor.SetUserTransform(transform)
                
            # Apply the same transform to the bounding box if it exists
            if hasattr(self, 'bbox_actor') and self.bbox_actor:
                self.bbox_actor.SetUserTransform(transform)
            
            # Render the scene
            self.safe_render()
            
            return True
        except Exception as e:
            print(f"Error adjusting rotation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def clear(self):
        """Clear all models from the viewer."""
        if self.renderer:
            # Remove actors
            if self.actor:
                self.renderer.RemoveActor(self.actor)
                self.actor = None
                
            if self.convex_hull_actor:
                self.renderer.RemoveActor(self.convex_hull_actor)
                self.convex_hull_actor = None
                
            if hasattr(self, 'bbox_actor') and self.bbox_actor:
                self.renderer.RemoveActor(self.bbox_actor)
                self.bbox_actor = None
                
            # Reset mesh data
            self.mesh = None
            self.convex_hull = None
            self.shrinkwrap = None
            self.stl_path = None
            
            # Render the scene
            self.safe_render()

    def clear_previous_model_data(self):
        """Clear previous model data when loading a new model to prevent confusion."""
        try:
            # Stop any ongoing rendering before clearing
            if self.vtk_widget and self.vtk_widget.GetRenderWindow():
                self.vtk_widget.GetRenderWindow().SetAbortRender(True)
            
            # Remove old actors from renderer
            if self.renderer:
                if self.actor and self.actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(self.actor)
                    
                if self.convex_hull_actor and self.convex_hull_actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(self.convex_hull_actor)
                    
                if hasattr(self, 'shrinkwrap_actor') and self.shrinkwrap_actor and self.shrinkwrap_actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(self.shrinkwrap_actor)
                    
                if hasattr(self, 'bbox_actor') and self.bbox_actor and self.bbox_actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(self.bbox_actor)
            
            # Clear model-specific data
            self.mesh = None
            self.convex_hull = None
            self.shrinkwrap = None
            self.optimal_transform = None
            if hasattr(self, 'original_mesh'):
                self.original_mesh = None
            
            # Clear shrinkwrap cache to free memory
            if hasattr(self, 'shrinkwrap_cache'):
                self.shrinkwrap_cache.clear()
            
            # Clear actors but keep renderer setup
            self.actor = None
            self.convex_hull_actor = None
            self.shrinkwrap_actor = None
            if hasattr(self, 'bbox_actor'):
                self.bbox_actor = None
            
            print("Cleared previous model data")
            
        except Exception as e:
            print(f"Warning: Error clearing model data: {e}")

    def update_shrinkwrap(self, resolution=40, box_size=None, method='boxgrid'):
        """
        Update the shrinkwrap visualization with the given parameters.
        
        Args:
            resolution (int): Resolution for some shrinkwrap methods
            box_size (float): Box size in mm for box grid method
            method (str): Shrinkwrap method to use (boxgrid, bvh, voxel, etc.)
        """
        try:
            from .stl_utils import create_fixed_box_grid_shrinkwrap, create_bvh_shrinkwrap
        except ImportError:
            from stl_utils import create_fixed_box_grid_shrinkwrap, create_bvh_shrinkwrap
        
        if not hasattr(self, 'mesh') or self.mesh is None:
            print("No mesh loaded, cannot update shrinkwrap")
            return

        # Track performance
        start_time = time.time()
        
        # Force method to boxgrid for fastest processing
        print("Using Box Grid method for optimal performance")
        method = 'boxgrid'
        
        # Check if we already have the shrinkwrap with these parameters
        cache_key = f"{method}_{resolution}_{box_size}"
        if hasattr(self, 'shrinkwrap_cache') and cache_key in self.shrinkwrap_cache:
            print(f"Using cached shrinkwrap for {cache_key}")
            shrinkwrap_mesh = self.shrinkwrap_cache[cache_key]
        else:
            # Create the shrinkwrap mesh based on the selected method
            try:
                # For very small meshes, adjust box_size
                if box_size is None:
                    if hasattr(self.mesh, 'extents'):
                        # Calculate a reasonable box size based on the mesh extents
                        max_extent = np.max(self.mesh.extents)
                        box_size = max(0.5, max_extent / 40)  # Aim for about 40 boxes across max dimension
                        print(f"Automatically set box_size to {box_size:.2f}mm")
                    else:
                        box_size = 5.0  # Default
                
                # Try to load the C++ module first
                try:
                    import cpp_stl_utils
                    print("C++ extension available - using high-performance implementation")
                    shrinkwrap_mesh = cpp_stl_utils.create_fixed_box_grid_shrinkwrap(self.mesh, box_size_mm=box_size)
                    if shrinkwrap_mesh is not None and hasattr(shrinkwrap_mesh, 'vertices') and len(shrinkwrap_mesh.vertices) > 0:
                        print(f"C++ implementation created shrinkwrap with {len(shrinkwrap_mesh.vertices)} vertices")
                    else:
                        print("C++ implementation failed, falling back to Python")
                        raise ImportError("Invalid result from C++ implementation")
                except ImportError as e:
                    print(f"Using Python implementation: {e}")
                    shrinkwrap_mesh = create_fixed_box_grid_shrinkwrap(self.mesh, box_size_mm=box_size)
                
                # Initialize cache if needed
                if not hasattr(self, 'shrinkwrap_cache'):
                    self.shrinkwrap_cache = {}
                    
                # Cache the result if valid
                if shrinkwrap_mesh is not None:
                    self.shrinkwrap_cache[cache_key] = shrinkwrap_mesh
                    
            except Exception as e:
                print(f"Error creating shrinkwrap: {e}")
                import traceback
                traceback.print_exc()
                return
        
        if shrinkwrap_mesh is None:
            print("Failed to create shrinkwrap mesh")
            return
        
        # Update the VTK visualization by converting the mesh to VTK
        try:
            # Calculate volumes for analysis
            original_volume = 0
            shrinkwrap_volume = 0
            
            try:
                original_volume = self.mesh.volume
            except:
                print("Could not calculate original mesh volume")
            
            try:
                shrinkwrap_volume = shrinkwrap_mesh.volume
            except:
                print("Could not calculate shrinkwrap volume")
            
            # Print volume comparison
            if original_volume > 0 and shrinkwrap_volume > 0:
                volume_ratio = shrinkwrap_volume / original_volume
                print(f"Original volume: {original_volume:.2f}mm³")
                print(f"Shrinkwrap volume: {shrinkwrap_volume:.2f}mm³")
                print(f"Volume ratio (shrinkwrap/original): {volume_ratio:.4f}")
            
            # Convert the shrinkwrap mesh to a VTK actor
            if not hasattr(self, 'mesh_to_actor'):
                print("ERROR: mesh_to_actor method not found, cannot visualize shrinkwrap")
                return
            
            # Remove existing shrinkwrap actors
            if hasattr(self, 'shrinkwrap_actor') and self.shrinkwrap_actor is not None:
                self.renderer.RemoveActor(self.shrinkwrap_actor)
            
            # Create the new actor
            self.shrinkwrap_actor = self.mesh_to_actor(
                shrinkwrap_mesh, 
                color=(0.2, 0.6, 0.9),  # Blue color
                opacity=0.5,
                wireframe=True
            )
            
            # Add to renderer
            self.renderer.AddActor(self.shrinkwrap_actor)
            
            # Emit signal with shrinkwrap info
            if hasattr(self, 'shrinkwrap_updated') and self.shrinkwrap_updated is not None:
                details = {
                    'method': method,
                    'box_size': box_size,
                    'resolution': resolution,
                    'original_volume': original_volume,
                    'shrinkwrap_volume': shrinkwrap_volume,
                    'volume_ratio': shrinkwrap_volume / original_volume if original_volume > 0 else 0,
                    'processing_time': time.time() - start_time
                }
                self.shrinkwrap_updated.emit(details)
            
            # Refresh the view
            self.safe_render()
            
            print(f"Shrinkwrap updated in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error adding shrinkwrap to visualization: {e}")
            import traceback
            traceback.print_exc()

    def load_shrinkwrap_from_file(self, shrinkwrap_file_path):
        """Load a pre-existing shrinkwrap STL file.
        
        Args:
            shrinkwrap_file_path (str): Path to the shrinkwrap STL file
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(shrinkwrap_file_path):
            self.display_message(f"Shrinkwrap file not found: {shrinkwrap_file_path}", "error")
            return False
            
        try:
            # Remove existing shrinkwrap actor if present
            if hasattr(self, 'shrinkwrap_actor') and self.shrinkwrap_actor:
                self.renderer.RemoveActor(self.shrinkwrap_actor)
                self.shrinkwrap_actor = None
            
            # Load the shrinkwrap file
            shrinkwrap_reader = vtk.vtkSTLReader()
            shrinkwrap_reader.SetFileName(shrinkwrap_file_path)
            shrinkwrap_reader.Update()
            
            # Create mapper and actor
            shrinkwrap_mapper = vtk.vtkPolyDataMapper()
            shrinkwrap_mapper.SetInputConnection(shrinkwrap_reader.GetOutputPort())
            
            self.shrinkwrap_actor = vtk.vtkActor()
            self.shrinkwrap_actor.SetMapper(shrinkwrap_mapper)
            
            # Set appearance properties
            self.shrinkwrap_actor.GetProperty().SetColor(0.0, 0.8, 0.0)  # Green color
            self.shrinkwrap_actor.GetProperty().SetOpacity(0.3)  # Translucent
            self.shrinkwrap_actor.GetProperty().SetRepresentationToWireframe()
            
            # Add to renderer
            self.renderer.AddActor(self.shrinkwrap_actor)
            
            # Set the shrinkwrap mesh attribute
            shrinkwrap_trimesh = self.mesh_from_vtk(shrinkwrap_reader.GetOutput())
            self.shrinkwrap_mesh = shrinkwrap_trimesh
            
            # Refresh the view
            self.safe_render()
            
            self.display_message(f"Loaded shrinkwrap from {os.path.basename(shrinkwrap_file_path)}")
            return True
            
        except Exception as e:
            self.display_message(f"Error loading shrinkwrap: {str(e)}", "error")
            logger.error(f"Error in load_shrinkwrap_from_file: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def mesh_from_vtk(self, polydata):
        """Convert a VTK polydata to a trimesh.
        
        Args:
            polydata: A vtkPolyData object
            
        Returns:
            trimesh.Trimesh: A trimesh object
        """
        # Extract points
        points = polydata.GetPoints()
        vertices = np.zeros((points.GetNumberOfPoints(), 3))
        for i in range(points.GetNumberOfPoints()):
            vertices[i] = points.GetPoint(i)
        
        # Extract faces
        cells = polydata.GetPolys()
        cells.InitTraversal()
        faces = []
        ids = vtk.vtkIdList()
        while cells.GetNextCell(ids):
            if ids.GetNumberOfIds() == 3:  # Only process triangular faces
                face = [ids.GetId(j) for j in range(3)]
                faces.append(face)
        
        faces = np.array(faces)
        
        # Create trimesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def mesh_to_actor(self, mesh, color=(0.2, 0.6, 0.9), opacity=1.0, wireframe=False, lighting=True):
        """Convert a trimesh mesh to a VTK actor.
        
        Args:
            mesh (trimesh.Trimesh): The mesh to convert
            color (tuple): RGB color tuple (0-1 range)
            opacity (float): Opacity value (0-1)
            wireframe (bool): Whether to render as wireframe
            lighting (bool): Whether to apply lighting
            
        Returns:
            vtkActor: The created actor
        """
        try:
            # Create temporary STL file
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_path = tmp.name
                # Export the mesh to STL
                mesh.export(tmp_path)
            
            # Read with VTK
            reader = vtk.vtkSTLReader()
            reader.SetFileName(tmp_path)
            reader.Update()
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set properties
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)
            
            if wireframe:
                actor.GetProperty().SetRepresentationToWireframe()
            
            if not lighting:
                actor.GetProperty().SetAmbient(1.0)
                actor.GetProperty().SetDiffuse(0.0)
                actor.GetProperty().SetSpecular(0.0)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return actor
            
        except Exception as e:
            print(f"Error creating actor from mesh: {e}")
            import traceback
            traceback.print_exc()
            return None

    def orient_to_minimal_bounding_box(self):
        """Orient the mesh to its minimal bounding box orientation."""
        if not self.mesh:
            return False
            
        try:
            # Import the smart bounding box function
            try:
                from .stl_utils import smart_minimal_bounding_box
            except ImportError:
                from stl_utils import smart_minimal_bounding_box
            
            # Calculate minimal bounding box using smart method (PCA + fallback)
            start_time = time.time()
            size, bb_volume, optimal_transform, method_used = smart_minimal_bounding_box(self.mesh)
            bb_calc_time = time.time() - start_time
            
            print(f"Viewer bounding box: {method_used} method in {bb_calc_time:.3f}s")
            
            # Apply a rotation correction to fix the orientation
            correction = self.get_rotation_correction()
            optimal_transform = trimesh.transformations.concatenate_matrices(
                optimal_transform, correction
            )
            
            # Store the optimal transformation for later use
            self.optimal_transform = optimal_transform
            
            # Apply optimal transformation to the mesh
            self.mesh.apply_transform(optimal_transform)
            
            # Update the display
            self.update_display()
            
            print(f"Applied optimal transformation for minimal bounding box")
            return True
            
        except Exception as e:
            print(f"Error orienting to minimal bounding box: {e}")
            import traceback
            traceback.print_exc()
            return False


def toggle_color(viewer):
    """Toggle the model color."""
    if not viewer.actor:
        return
        
    current_color = list(viewer.actor.GetProperty().GetColor())
    
    # Define color options for cycling through
    colors = [
        [0.2, 0.6, 0.9],  # Bright blue
        [0.9, 0.4, 0.3],  # Red-orange
        [0.3, 0.8, 0.3],  # Green
        [0.7, 0.3, 0.8],  # Purple
        [0.9, 0.9, 0.2]   # Yellow
    ]
    
    # Find the closest color and cycle to the next one
    min_dist = float('inf')
    closest_idx = 0
    
    for i, color in enumerate(colors):
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(current_color, color))
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # Cycle to the next color
    next_idx = (closest_idx + 1) % len(colors)
    viewer.color = colors[next_idx]
    viewer.actor.GetProperty().SetColor(viewer.color)
    viewer.safe_render()


def open_stl_file(viewer):
    """Open a file dialog to choose an STL file."""
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        viewer.parent() or viewer,
        "Open STL File",
        "",
        "STL Files (*.stl);;All Files (*.*)"
    )
    
    if file_path:
        success = viewer.load_stl(file_path)
        
        # Update window title if in standalone mode
        if success and isinstance(viewer.parent(), QtWidgets.QMainWindow):
            viewer.parent().setWindowTitle(f"STL Viewer - {os.path.basename(file_path)}")


# Function to open a standalone viewer window
def open_stl_viewer(file_path=None):
    """Open a standalone STL viewer window."""
    try:
        print("Starting STL Viewer...")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
            
        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("STL Viewer")
        main_window.resize(800, 600)
        
        # Create central widget
        central_widget = QtWidgets.QWidget()
        main_window.setCentralWidget(central_widget)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create viewer
        viewer = STLViewer(main_window)
        layout.addWidget(viewer, 1)
        
        # Create button bar
        button_layout = QtWidgets.QHBoxLayout()
        
        # Open STL button
        open_button = QtWidgets.QPushButton("Open STL")
        open_button.clicked.connect(lambda: open_stl_file(viewer))
        
        # Reset view button
        reset_button = QtWidgets.QPushButton("Reset View")
        reset_button.clicked.connect(viewer.reset_view)
        
        # Toggle color button
        color_button = QtWidgets.QPushButton("Toggle Color")
        color_button.clicked.connect(lambda: toggle_color(viewer))
        
        # Wireframe button
        wireframe_button = QtWidgets.QPushButton("Wireframe: On")
        wireframe_button.clicked.connect(lambda: wireframe_button.setText(
            "Wireframe: On" if viewer.toggle_wireframe() else "Wireframe: Off"))
            
        # Bounding box button
        bbox_button = QtWidgets.QPushButton("Box: On")
        bbox_button.clicked.connect(lambda: toggle_bbox(viewer, bbox_button))
        
        # Optimize orientation button
        optimize_button = QtWidgets.QPushButton("Optimize Orientation")
        optimize_button.clicked.connect(viewer.optimize_orientation)
        
        # Add rotation adjustment buttons
        rotate_cw_button = QtWidgets.QPushButton("Rotate +90�")
        rotate_cw_button.setToolTip("Rotate model 90 degrees clockwise around X-axis")
        rotate_cw_button.clicked.connect(lambda: viewer.adjust_rotation(90))
        
        rotate_ccw_button = QtWidgets.QPushButton("Rotate -90�")
        rotate_ccw_button.setToolTip("Rotate model 90 degrees counter-clockwise around X-axis")
        rotate_ccw_button.clicked.connect(lambda: viewer.adjust_rotation(-90))
        
        # Add buttons to layout
        button_layout.addWidget(open_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(color_button)
        button_layout.addWidget(wireframe_button)
        button_layout.addWidget(bbox_button)
        button_layout.addWidget(optimize_button)
        button_layout.addWidget(rotate_cw_button)
        button_layout.addWidget(rotate_ccw_button)
        
        # Add button layout to main layout
        layout.addLayout(button_layout)
        
        # Show the window
        main_window.show()
        
        # Load file if provided
        if file_path and os.path.exists(file_path):
            print(f"Loading STL file: {file_path}")
            if viewer.load_stl(file_path):
                main_window.setWindowTitle(f"STL Viewer - {os.path.basename(file_path)}")
        
        # Override closeEvent for proper cleanup
        original_close = main_window.closeEvent
        def custom_close_event(event):
            if viewer:
                # Properly finalize VTK objects before closing
                try:
                    if viewer.vtk_widget:
                        viewer.vtk_widget.GetRenderWindow().Finalize()
                    if viewer.interactor:
                        viewer.interactor.TerminateApp()
                except:
                    pass
                
                # Clear references
                viewer.renderer = None
                viewer.interactor = None
            
            # Call original close event
            original_close(event)
        
        # Replace the closeEvent method
        main_window.closeEvent = custom_close_event
        
        # Start event loop
        print("Starting application event loop...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error in open_stl_viewer: {e}")
        import traceback
        traceback.print_exc()


def toggle_bbox(viewer, button):
    """Toggle bounding box visibility and update button text."""
    if hasattr(viewer, 'bbox_actor') and viewer.bbox_actor:
        visible = viewer.bbox_actor in viewer.renderer.GetActors()
        viewer.set_bounding_box_visible(not visible)
        button.setText("Box: On" if not visible else "Box: Off")


if __name__ == "__main__":
    # Test the viewer if run directly
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        open_stl_viewer(sys.argv[1])
    else:
        open_stl_viewer() 
