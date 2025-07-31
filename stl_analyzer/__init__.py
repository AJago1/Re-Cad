"""
STL Analyzer package
A comprehensive tool for analyzing, measuring, and manipulating STL files.
"""

import os
import sys

# Add the current directory to the path for loading C++ extensions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import C++ extensions
try:
    import cpp_stl_utils
    print("C++ extensions loaded successfully!")
    HAS_CPP_EXTENSIONS = True
except ImportError:
    HAS_CPP_EXTENSIONS = False

__version__ = '1.0.0'
__author__ = 'STL Analyzer Team' 