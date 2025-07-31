#!/usr/bin/env python3
"""
Test script to verify that the rotation optimization functionality works correctly.
"""

import os
import sys
import numpy as np

# Add the stl_analyzer module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stl_analyzer'))

try:
    from stl_analyzer.stl_utils import extract_features_from_mesh, smart_minimal_bounding_box, try_load_stl
    print("✓ Successfully imported stl_utils functions")
except ImportError as e:
    print(f"✗ Failed to import stl_utils: {e}")
    sys.exit(1)

def test_feature_extraction():
    """Test that feature extraction returns consistent field names"""
    print("\n=== Testing Feature Extraction ===")
    
    # Create a simple test mesh (cube)
    try:
        import trimesh
        
        # Create a simple cube mesh
        mesh = trimesh.creation.box(extents=[10, 20, 30])
        print(f"✓ Created test mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Extract features
        features = extract_features_from_mesh(mesh, "test_cube.stl", "test_cube")
        
        if features is None:
            print("✗ Feature extraction returned None")
            return False
            
        print(f"✓ Feature extraction successful")
        
        # Check required fields
        required_fields = [
            'name', 'filename', 'x', 'y', 'z', 'volume', 'surface_area', 
            'bb_volume', 'waste', 'waste_ratio', 'sa_vol_ratio',
            'convex_hull_volume', 'convexity_ratio'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in features:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"✗ Missing required fields: {missing_fields}")
            return False
        else:
            print(f"✓ All required fields present")
        
        # Check field consistency
        print(f"  Dimensions: {features['x']:.2f} x {features['y']:.2f} x {features['z']:.2f}")
        print(f"  Volume: {features['volume']:.2f}")
        print(f"  BB Volume: {features['bb_volume']:.2f}")
        print(f"  Waste: {features['waste']:.2f} ({features['waste_ratio']:.1%})")
        print(f"  SA/Vol Ratio: {features['sa_vol_ratio']:.4f}")
        
        # Verify alternative field names exist
        alt_fields = ['bounding_box_volume', 'waste_volume', 'bbox_method']
        for field in alt_fields:
            if field in features:
                print(f"✓ Alternative field '{field}' present")
            else:
                print(f"⚠ Alternative field '{field}' missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in feature extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bounding_box_calculation():
    """Test that bounding box calculation works correctly"""
    print("\n=== Testing Bounding Box Calculation ===")
    
    try:
        import trimesh
        
        # Create a test mesh with known dimensions
        mesh = trimesh.creation.box(extents=[10, 20, 30])
        print(f"✓ Created test mesh with extents [10, 20, 30]")
        
        # Test smart bounding box calculation
        size, volume, transform, method = smart_minimal_bounding_box(mesh)
        
        print(f"✓ Smart bounding box calculation successful")
        print(f"  Method: {method}")
        print(f"  Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
        print(f"  Volume: {volume:.2f}")
        
        # Check that dimensions are sorted (largest to smallest)
        if size[0] >= size[1] >= size[2]:
            print(f"✓ Dimensions correctly sorted")
        else:
            print(f"✗ Dimensions not sorted correctly")
            return False
        
        # Check that volume is reasonable
        expected_volume = 10 * 20 * 30
        if abs(volume - expected_volume) < 1.0:  # Allow small tolerance
            print(f"✓ Volume calculation correct (expected: {expected_volume}, got: {volume:.2f})")
        else:
            print(f"✗ Volume calculation incorrect (expected: {expected_volume}, got: {volume:.2f})")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in bounding box test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("STL Analyzer Optimization Test Suite")
    print("=" * 50)
    
    tests = [
        test_feature_extraction,
        test_bounding_box_calculation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} PASSED")
            else:
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimization functionality should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 