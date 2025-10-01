#!/usr/bin/env python3
"""
Build Pack3D C++ Extension for 3D Packing Optimization
"""

from distutils.core import setup, Extension
import sys
import os
import pybind11

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("🔧 Building Pack3D C++ Extension with pybind11...")

# Get pybind11 include path
pybind11_includes = pybind11.get_include()
print(f"   Using pybind11 includes: {pybind11_includes}")

# Define the extension module
pack3d_extension = Extension(
    'cpp_pack3d',
    sources=[
        'cpp_pack3d_extension.cpp',
        'cpp_pack3d_core.cpp'
    ],
    include_dirs=[
        pybind11_includes,
        pybind11.get_include(user=True)
    ],
    language='c++',
    extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    extra_link_args=[]
)

# Build the extension
setup(
    name='cpp_pack3d',
    ext_modules=[pack3d_extension],
    zip_safe=False
)

print("✅ Pack3D extension built successfully!")




