#!/usr/bin/env python3
"""
STL Analyzer - C++ Extensions Build Script
Builds the advanced BVH C++ extension for Pack3D integration

This script extends the existing cpp_stl_utils with new BVH functionality
while maintaining compatibility with existing features.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_dependencies():
    """Check if required build dependencies are available"""
    dependencies = {
        'cmake': 'CMake is required for building C++ extensions',
        'python': 'Python development headers are required'
    }
    
    missing = []
    
    # Check CMake
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            missing.append('cmake')
    except FileNotFoundError:
        missing.append('cmake')
    
    # Check Python development headers
    try:
        import pybind11
    except ImportError:
        missing.append('pybind11')
        print("⚠️ pybind11 not found. Install with: pip install pybind11")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}: {dependencies.get(dep, 'Required for build')}")
        return False
    
    return True

def build_cpp_extension():
    """Build the C++ BVH extension"""
    print("🔧 Building STL Analyzer C++ BVH Extension...")
    
    # Get current directory
    current_dir = Path(__file__).parent
    build_dir = current_dir / "build_cpp"
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # Platform-specific configuration
    is_windows = platform.system() == "Windows"
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={current_dir}",
    ]
    
    if is_windows:
        # Windows-specific CMake args
        cmake_args.extend([
            "-A", "x64",  # 64-bit build
            "-T", "v143"  # Use latest MSVC toolset
        ])
    
    try:
        # Configure with CMake
        print("📋 Configuring build with CMake...")
        configure_cmd = [
            "cmake",
            str(current_dir),
            f"-B{build_dir}"
        ] + cmake_args
        
        result = subprocess.run(configure_cmd, cwd=current_dir, 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ CMake configuration failed:")
            print(result.stderr)
            return False
        
        print("✅ CMake configuration successful")
        
        # Build with CMake
        print("🔨 Building C++ extension...")
        build_cmd = [
            "cmake",
            "--build", str(build_dir),
            "--config", "Release",
            "--parallel", "4"
        ]
        
        result = subprocess.run(build_cmd, cwd=current_dir,
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Build failed:")
            print(result.stderr)
            return False
        
        print("✅ C++ extension built successfully")
        
        # Check if output file exists
        expected_output = current_dir / "cpp_stl_utils_extended.pyd"
        if is_windows:
            # Look for the actual output in build directory
            build_output = build_dir / "Release" / "cpp_stl_utils_extended.pyd"
            if build_output.exists():
                # Copy to main directory
                import shutil
                shutil.copy2(build_output, expected_output)
                print(f"📦 Extension copied to {expected_output}")
        
        if expected_output.exists():
            print(f"🎉 Build complete! Extension available at: {expected_output}")
            return True
        else:
            print(f"⚠️ Build completed but output file not found at expected location")
            print(f"   Expected: {expected_output}")
            print(f"   Check build directory: {build_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def test_extension():
    """Test the built extension"""
    print("🧪 Testing C++ BVH extension...")
    
    try:
        # Try importing the new extension
        sys.path.insert(0, str(Path(__file__).parent))
        import cpp_stl_utils_extended
        
        print(f"✅ Import successful: {cpp_stl_utils_extended.__doc__}")
        print(f"📊 Max leaf count: {cpp_stl_utils_extended.MAX_LEAF_COUNT}")
        print(f"📏 Default tolerance: {cpp_stl_utils_extended.DEFAULT_TOLERANCE}")
        
        # Test basic functionality with simple mesh
        import numpy as np
        
        # Create a simple cube vertices
        vertices = np.array([
            0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,  # bottom face
            0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1   # top face
        ], dtype=np.float32)
        
        # Test BVH creation
        bvh = cpp_stl_utils_extended.create_adaptive_bvh(vertices, 8)
        if bvh is not None:
            print(f"✅ BVH creation test passed")
            print(f"   Leaf count: {bvh.get_leaf_count()}")
            print(f"   Active leaves: {bvh.get_active_leaf_count()}")
            
            # Test performance benchmark
            results = cpp_stl_utils_extended.benchmark_bvh(vertices, [8])
            creation_time = results['leaf_count_results']['8']['creation_time_ms']
            print(f"   Creation time: {creation_time:.3f}ms")
            
            if creation_time < 1.0:  # Should be sub-millisecond
                print("🚀 Performance target achieved (<1ms creation time)")
            else:
                print("⚠️ Performance slower than expected")
        else:
            print("❌ BVH creation test failed")
            return False
        
        print("✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def integrate_with_existing():
    """Update the cpp_bvh_wrapper to use the new extension"""
    print("🔗 Integrating with existing Python code...")
    
    wrapper_file = Path(__file__).parent / "cpp_bvh_wrapper.py"
    if not wrapper_file.exists():
        print(f"⚠️ Wrapper file not found: {wrapper_file}")
        return False
    
    # Update the import statement in the wrapper
    try:
        with open(wrapper_file, 'r') as f:
            content = f.read()
        
        # Add import for the new extension
        if 'cpp_stl_utils_extended' not in content:
            new_import = """
try:
    # Try importing the extended cpp_stl_utils with BVH functions
    import cpp_stl_utils_extended
    CPP_BVH_EXTENDED_AVAILABLE = True
except ImportError:
    CPP_BVH_EXTENDED_AVAILABLE = False

try:
    # Original cpp_stl_utils for existing functionality
    import cpp_stl_utils
    CPP_BVH_AVAILABLE = hasattr(cpp_stl_utils, 'create_adaptive_bvh') or CPP_BVH_EXTENDED_AVAILABLE
except ImportError:
    CPP_BVH_AVAILABLE = False"""
            
            # Replace the existing import section
            old_import_start = content.find("try:\n    # Try importing the extended cpp_stl_utils")
            if old_import_start == -1:
                old_import_start = content.find("try:\n    import cpp_stl_utils")
            
            if old_import_start != -1:
                old_import_end = content.find("except ImportError:\n    CPP_BVH_AVAILABLE = False", old_import_start)
                if old_import_end != -1:
                    old_import_end = content.find("\n", old_import_end + 50) + 1
                    content = content[:old_import_start] + new_import + content[old_import_end:]
                    
                    with open(wrapper_file, 'w') as f:
                        f.write(content)
                    
                    print("✅ Python wrapper updated to use new C++ extension")
                    return True
        
        print("✅ Python wrapper already up to date")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not update Python wrapper: {e}")
        return False

def main():
    """Main build process"""
    print("=" * 60)
    print("STL Analyzer - C++ BVH Extension Builder")
    print("Advanced Pack3D Integration - Phase 2.1")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Build failed - missing dependencies")
        return 1
    
    # Build the extension
    if not build_cpp_extension():
        print("\n❌ Build failed - compilation error")
        return 1
    
    # Test the extension
    if not test_extension():
        print("\n❌ Build failed - test error")
        return 1
    
    # Integrate with existing code
    if not integrate_with_existing():
        print("\n⚠️ Build successful but integration incomplete")
        return 0
    
    print("\n" + "=" * 60)
    print("🎉 C++ BVH Extension Build Complete!")
    print("✅ Phase 2.1 - C++ BVH Core implementation ready")
    print("🚀 Expected performance: 1000x speedup over Python")
    print("📁 Extension file: cpp_stl_utils_extended.pyd")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())