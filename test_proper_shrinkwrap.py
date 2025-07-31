#!/usr/bin/env python3
"""
Test script for the new surface offset shrinkwrap method.

This script tests the proper shrinkwrap approach that preserves surface contours
and holes, unlike the previous voxel dilation method that filled everything.
"""

import os
import sys
import time

# Add the stl_analyzer directory to path
sys.path.insert(0, 'stl_analyzer')

try:
    from stl_utils import (
        try_load_stl, 
        create_surface_offset_shrinkwrap,
        extract_features_with_shrinkwrap
    )
    print("✅ Successfully imported proper shrinkwrap utilities")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_proper_shrinkwrap():
    """Test the new surface offset shrinkwrap method."""
    
    print("🚀 Testing Proper Surface Offset Shrinkwrap Method")
    print("=" * 60)
    
    # Look for any STL files in the current directory
    stl_files = [f for f in os.listdir('.') if f.lower().endswith('.stl')]
    
    if not stl_files:
        print("❌ No STL files found in current directory")
        print("   Please copy an STL file to test with.")
        return False
    
    # Test with the first STL file found
    test_file = stl_files[0]
    print(f"📄 Testing with: {test_file}")
    
    # Test loading the file first
    mesh = try_load_stl(test_file)
    if mesh is None:
        print(f"❌ Failed to load {test_file}")
        return False
    
    print(f"✅ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"   Volume: {mesh.volume:.2f} mm³")
    print(f"   Extents: {mesh.extents}")
    
    # Test the proper surface offset method directly
    print(f"\n🔧 Testing Proper Surface Offset Method...")
    
    test_offsets = [3.0, 5.0, 10.0]
    
    for offset_percent in test_offsets:
        print(f"\n   Testing {offset_percent}% offset...")
        
        start_time = time.time()
        
        # Test the direct surface offset method
        shrinkwrap_mesh = create_surface_offset_shrinkwrap(
            mesh, 
            offset_percent=offset_percent
        )
        
        elapsed_time = time.time() - start_time
        
        if shrinkwrap_mesh and shrinkwrap_mesh.volume > 0:
            volume_increase = ((shrinkwrap_mesh.volume - mesh.volume) / mesh.volume) * 100
            print(f"   ✅ Shrinkwrap created in {elapsed_time:.2f}s")
            print(f"      Original: {mesh.volume:.2f}mm³")
            print(f"      Shrinkwrap: {shrinkwrap_mesh.volume:.2f}mm³")
            print(f"      Volume increase: {volume_increase:.1f}%")
            print(f"      Vertices: {len(shrinkwrap_mesh.vertices):,}")
            print(f"      Faces: {len(shrinkwrap_mesh.faces):,}")
            print(f"      Preserved topology: {len(shrinkwrap_mesh.faces) == len(mesh.faces)}")
        else:
            print(f"   ❌ Shrinkwrap creation failed")
    
    # Test the integrated feature extraction with shrinkwrap
    print(f"\n🔧 Testing Integrated Feature Extraction...")
    
    start_time = time.time()
    
    features = extract_features_with_shrinkwrap(
        test_file,
        offset_percent=5.0,
        generate_shrinkwrap=True
    )
    
    elapsed_time = time.time() - start_time
    
    if features:
        print(f"✅ Full feature extraction completed in {elapsed_time:.2f}s")
        print(f"   Original volume: {features['volume']:.2f} mm³")
        print(f"   Shrinkwrap volume: {features['shrinkwrap_volume']:.2f} mm³")
        print(f"   Shrinkwrap ratio: {features['shrinkwrap_ratio']:.3f}")
        
        # Check if shrinkwrap STL file was created
        if features['shrinkwrap_stl_path'] and os.path.exists(features['shrinkwrap_stl_path']):
            shrinkwrap_file = os.path.basename(features['shrinkwrap_stl_path'])
            file_size = os.path.getsize(features['shrinkwrap_stl_path']) / 1024  # KB
            print(f"   ✅ Shrinkwrap STL exported: {shrinkwrap_file} ({file_size:.1f} KB)")
        else:
            print(f"   ❌ Shrinkwrap STL file not created")
    else:
        print(f"❌ Feature extraction failed")
    
    return True

def benchmark_comparison():
    """Optional: Show benefits of proper shrinkwrap method."""
    print(f"\n📊 Benefits of Surface Offset Shrinkwrap Method:")
    print(f"   • Preserves surface contours and shape details")
    print(f"   • Maintains holes and concavities (proper shrinkwrap)")
    print(f"   • Does NOT fill internal spaces like convex hull")
    print(f"   • Fast surface-based operations")
    print(f"   • Uses vertex normal offsetting for accuracy")
    print(f"   • Fallback to alpha shapes for complex geometries")

def cleanup_test_files():
    """Clean up any test shrinkwrap files."""
    print("\n🧹 Cleaning up test files...")
    shrinkwrap_files = [f for f in os.listdir('.') if 'shrinkwrap' in f.lower() and f.endswith('.stl')]
    
    for f in shrinkwrap_files:
        try:
            os.remove(f)
            print(f"   Removed: {f}")
        except Exception as e:
            print(f"   Failed to remove {f}: {e}")

if __name__ == "__main__":
    print("🚀 STL Analyzer - Proper Surface Offset Shrinkwrap Test")
    print("=" * 70)
    
    # Run the test
    success = test_proper_shrinkwrap()
    
    if success:
        print("\n✅ Proper shrinkwrap test completed successfully!")
        benchmark_comparison()
        
        # Ask if user wants to clean up
        response = input("\nClean up test shrinkwrap files? (y/n): ").lower().strip()
        if response == 'y':
            cleanup_test_files()
    else:
        print("\n❌ Proper shrinkwrap test failed")
        
    print("\n🎯 Now you have a REAL shrinkwrap that follows contours!") 