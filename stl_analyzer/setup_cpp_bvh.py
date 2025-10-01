#!/usr/bin/env python3
"""
STL Analyzer - Simple C++ BVH Extension Builder
Alternative build approach using setup.py and pybind11
"""

import os
import sys
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# The main interface of the Python package
from setuptools import setup, Extension
import numpy as np

def main():
    """Build the C++ BVH extension using pybind11"""
    print("🔧 Building C++ BVH Extension using pybind11...")
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Define the extension module
    ext_modules = [
        Pybind11Extension(
            "cpp_stl_utils_extended",
            # Sources
            [
                str(current_dir / "cpp_stl_utils_extension.cpp"),
                str(current_dir / "cpp_bvh_core.cpp"),
            ],
            # Include directories
            include_dirs=[
                str(current_dir),
                np.get_include(),
            ],
            # Language
            language='c++',
            # C++ standard
            cxx_std=17,
        ),
    ]

    # Setup configuration
    setup(
        name="cpp_stl_utils_extended",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()