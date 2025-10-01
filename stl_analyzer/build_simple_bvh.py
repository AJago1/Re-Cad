#!/usr/bin/env python3
"""
Simple C++ BVH Builder
Direct compilation using distutils - no complex dependencies
"""

import os
import sys
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import tempfile

def build_simple_bvh():
    """Build the simple C++ BVH extension"""
    print("🔧 Building Simple C++ BVH Extension...")
    
    # Define the extension
    bvh_extension = Extension(
        'cpp_bvh_simple',
        sources=['cpp_bvh_simple.cpp'],
        language='c++',
        extra_compile_args=['/std:c++17', '/O2'] if sys.platform == 'win32' else ['-std=c++17', '-O3'],
        define_macros=[('VERSION_INFO', '"dev"')],
    )
    
    # Setup configuration
    setup(
        name='cpp_bvh_simple',
        ext_modules=[bvh_extension],
        cmdclass={'build_ext': build_ext},
        script_name='build_simple_bvh.py',
        script_args=['build_ext', '--inplace'],
    )

if __name__ == "__main__":
    build_simple_bvh()