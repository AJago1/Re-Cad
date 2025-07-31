#!/usr/bin/env python3
"""
STL Analyzer Portable Launcher
This script properly sets up the Python path and launches the STL Analyzer
using the portable Python installation and local dependencies.
"""

import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# FORCE complete isolation - disable all external package sources
os.environ['PYTHONNOUSERSITE'] = '1'  # Disable user site-packages
os.environ['PYTHONHOME'] = ''  # Clear Python home
os.environ['PYTHONPATH'] = ''  # Clear any existing Python path

# Set up paths - packages are in venv, Python is in python-portable
portable_python_dir = os.path.join(script_dir, 'python-portable')
venv_site_packages = os.path.join(script_dir, 'venv', 'Lib', 'site-packages')

# Build the correct sys.path with only local directories
correct_paths = [
    script_dir,  # For stl_analyzer package
    os.path.join(script_dir, 'stl_analyzer'),  # For direct imports
    venv_site_packages,  # Local packages (in venv)
    os.path.join(portable_python_dir, 'python312.zip'),  # Standard library
    portable_python_dir,  # Python executable directory
    os.path.join(portable_python_dir, 'Lib'),  # Python standard library
]

# Filter existing sys.path to keep only local paths
filtered_paths = []
for path in sys.path:
    # Keep only paths that are within our directories
    if (path.startswith(portable_python_dir) or 
        path.startswith(script_dir) or 
        not path or  # Keep empty string (current directory)
        path == '.'):  # Keep current directory
        filtered_paths.append(path)

# Rebuild sys.path with our controlled paths
sys.path.clear()
for path in correct_paths:
    if path not in sys.path:
        sys.path.append(path)

# Add any remaining filtered paths that aren't already included
for path in filtered_paths:
    if path not in sys.path:
        sys.path.append(path)

try:
    # Try to import and run the application
    print("🚀 Starting STL Analyzer...")
    print(f"📁 Working directory: {script_dir}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔒 Isolated mode: PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE', 'not set')}")
    print(f"🛤️  Python path: {len(sys.path)} entries (local only)")
    
    # Verify we're not using external packages (exclude our own venv)
    external_paths = [path for path in sys.path if 
                      ('AppData' in path or 'Program Files' in path) and 
                      not path.startswith(script_dir)]
    if external_paths:
        print(f"⚠️  WARNING: Found external paths: {external_paths}")
    else:
        print("✅ All paths are local - complete isolation achieved!")
    
    # Show first few paths for verification
    print("📦 Key paths:")
    for i, path in enumerate(sys.path[:6]):
        print(f"   {i}: {path}")
    if len(sys.path) > 6:
        print(f"   ... and {len(sys.path) - 6} more")
    print()
    
    # Import the main module
    from stl_analyzer import main
    
    # Run the application
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = main.main(input_file)
    sys.exit(exit_code)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔍 Trying alternative import method...")
    
    try:
        # Alternative: import directly from stl_analyzer directory
        sys.path.insert(0, os.path.join(script_dir, 'stl_analyzer'))
        import main as stl_main
        
        input_file = sys.argv[1] if len(sys.argv) > 1 else None
        exit_code = stl_main.main(input_file)
        sys.exit(exit_code)
        
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("🛠️  Please check that all required files are present:")
        print(f"   - {os.path.join(script_dir, 'stl_analyzer', 'main.py')}")
        print(f"   - {os.path.join(script_dir, 'stl_analyzer', 'gui.py')}")
        print(f"   - {os.path.join(script_dir, 'stl_analyzer', '__init__.py')}")
        print(f"🔍 Current sys.path:")
        for i, path in enumerate(sys.path):
            print(f"   {i}: {path}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 