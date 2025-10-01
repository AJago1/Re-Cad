#!/usr/bin/env python3
"""
Build script for Fogleman's Pack3D BVH C++ extension
"""

from distutils.core import setup, Extension
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("🔧 Building Fogleman's Pack3D BVH Extension...")

# Define the extension module
fogleman_extension = Extension(
    'cpp_bvh_fogleman',
    sources=['cpp_bvh_fogleman.cpp'],
    language='c++',
    extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    extra_link_args=[]
)

# Build the extension
setup(
    name='cpp_bvh_fogleman',
    ext_modules=[fogleman_extension],
    zip_safe=False
)

print("✅ Fogleman BVH extension built successfully!")




