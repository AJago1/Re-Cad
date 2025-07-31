#!/usr/bin/env python
"""
STL Analyzer - Main Application
Analyze and view STL files with minimal bounding box orientation.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stl_analyzer')

# Enable fast mode for better performance - DISABLED to allow GUI shrinkwrap control
# try:
#     from .fast_config import enable_fast_mode
#     enable_fast_mode()
# except ImportError:
#     logger.warning("Fast mode configuration not available")

# Check if Open3D is available (optional dependency)
try:
    import open3d
    logger.info("Open3D available - advanced mesh processing enabled")
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not available - advanced mesh processing will be limited")
    OPEN3D_AVAILABLE = False

# Import PyQt5 for high DPI scaling
try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import Qt
except ImportError:
    logger.warning("PyQt5 not available, GUI will not function properly")

# Configure VTK to suppress error output
try:
    import vtk
    # Redirect VTK errors to file instead of console
    vtk_output = vtk.vtkFileOutputWindow()
    temp_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vtk_errors.log')
    vtk_output.SetFileName(temp_log)
    
    # Set the global output window
    vtk.vtkOutputWindow.SetInstance(vtk_output)
    
    # Set error display mode
    vtk_output.SetGlobalWarningDisplay(0)  # Turn off warnings
except ImportError:
    logger.warning("VTK not available, 3D visualization will be disabled")
except Exception as e:
    logger.warning(f"Failed to configure VTK output: {e}")

# This is needed to fix display issues on Windows with high DPI displays
try:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
except:
    logger.warning("Could not set high DPI scaling")

# Import gui module
try:
    # Import our modules (try different import paths)
    try:
        from stl_analyzer.gui import run_app
    except ImportError:
        from gui import run_app
except ImportError as e:
    error_msg = str(e)
    logger.error(f"Failed to import GUI module: {error_msg}")
    
    # Check for specific error messages and provide more helpful advice
    if "No module named 'open3d'" in error_msg:
        print(f"ERROR: Failed to import GUI module: {error_msg}")
        print("Open3D is missing but the application can still run with limited functionality.")
        print("You can install Open3D with: pip install open3d")
        print("The application will now try to continue...")
        
        # Try to continue by ensuring Open3D is marked as unavailable and retrying the import
        import sys
        sys.modules['open3d'] = None  # Create a dummy module to prevent further import errors
        
        try:
            # Try importing again
            try:
                from stl_analyzer.gui import run_app
            except ImportError:
                from gui import run_app
        except ImportError as e2:
            # If it still fails, then we have a more serious problem
            logger.error(f"Second attempt to import GUI module failed: {e2}")
            print(f"ERROR: Failed to import GUI module: {e2}")
            print("Make sure all required dependencies are installed (run install.py)")
            sys.exit(1)
    else:
        # For other import errors, display the standard message
        print(f"ERROR: Failed to import GUI module: {error_msg}")
        print("Make sure all required dependencies are installed (run install.py)")
        sys.exit(1)

def main(input_file=None):
    """Main application entry point."""
    logger.info("Starting STL Analyzer application")
    
    try:
        # Check if input_file parameter is supported
        import inspect
        run_app_params = inspect.signature(run_app).parameters
        if len(run_app_params) > 0:
            # New version of run_app that accepts input_file
            run_app(input_file)
        else:
            # Old version of run_app with no parameters
            run_app()
        return 0
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    # If run directly, accept an optional file path argument
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(input_file)) 