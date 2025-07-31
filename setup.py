#!/usr/bin/env python
"""
STL Analyzer - 3D Model Analysis and Pricing Tool
Setup script for installation
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "STL Analyzer - 3D Model Analysis and Pricing Tool"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="stl-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="3D Model Analysis and Pricing Tool for STL files",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stl-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-qt>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "optional": [
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "stl-analyzer=stl_analyzer.main:main",
        ],
        "gui_scripts": [
            "stl-analyzer-gui=stl_analyzer.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stl_analyzer": [
            "data/*.json",
            "gui/*.py",
            "gui/tabs/*.py", 
            "gui/widgets/*.py",
            "gui/threads/*.py",
            "gui/dialogs/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "stl", "3d-printing", "geometry-analysis", "pricing", 
        "machine-learning", "cad", "manufacturing", "mesh-processing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stl-analyzer/issues",
        "Source": "https://github.com/yourusername/stl-analyzer",
        "Documentation": "https://github.com/yourusername/stl-analyzer/wiki",
    },
) 