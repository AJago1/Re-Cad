#!/usr/bin/env python
"""
Test script for PCA bounding box optimization
Compares performance and accuracy of different methods
"""

import os
import sys
import time
import numpy as np

# Add the stl_analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stl_analyzer'))

try:
    from stl_utils import (
        try_load_stl, 
        pca_minimal_bounding_box, 
        smart_minimal_bounding_box,
        convex_hull_pca_bounding_box
    )
    print("✅ Successfully imported PCA bounding box functions")
except ImportError as e:
    print(f"❌ Failed to import PCA functions: {e}")
    sys.exit(1)

def test_bounding_box_methods(stl_file):
    """Test different bounding box methods on a single file"""
    print(f"\n🔍 Testing: {os.path.basename(stl_file)}")
    print("="*60)
    
    # Load the mesh
    mesh = try_load_stl(stl_file)
    if mesh is None:
        print("❌ Failed to load mesh")
        return None
    
    face_count = len(mesh.faces)
    vertex_count = len(mesh.vertices)
    print(f"📊 Mesh info: {vertex_count} vertices, {face_count} faces")
    
    results = {}
    
    # Test 1: Original expensive method
    print("\n1️⃣ Testing original method (expensive)...")
    try:
        start_time = time.time()
        obb = mesh.bounding_box_oriented
        original_size = obb.extents
        original_size = np.sort(original_size)[::-1]
        original_time = time.time() - start_time
        original_volume = np.prod(original_size)
        
        results['original'] = {
            'size': original_size,
            'time': original_time,
            'volume': original_volume,
            'method': 'precise_iterative'
        }
        print(f"   ⏱️  Time: {original_time:.3f}s")
        print(f"   📦 Size: {original_size}")
        print(f"   📊 Volume: {original_volume:.2f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['original'] = None
    
    # Test 2: PCA method
    print("\n2️⃣ Testing PCA method...")
    try:
        pca_size, pca_transform, volume_reduction = pca_minimal_bounding_box(mesh)
        pca_volume = np.prod(pca_size)
        
        results['pca'] = {
            'size': pca_size,
            'volume': pca_volume,
            'volume_reduction': volume_reduction,
            'method': 'pca'
        }
        print(f"   📦 Size: {pca_size}")
        print(f"   📊 Volume: {pca_volume:.2f}")
        print(f"   📈 Volume reduction: {volume_reduction:.1f}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['pca'] = None
    
    # Test 3: Convex Hull PCA method
    print("\n3️⃣ Testing Convex Hull PCA method...")
    try:
        hull_size, hull_transform, hull_volume_reduction = convex_hull_pca_bounding_box(mesh)
        hull_volume = np.prod(hull_size)
        
        results['hull_pca'] = {
            'size': hull_size,
            'volume': hull_volume,
            'volume_reduction': hull_volume_reduction,
            'method': 'convex_hull_pca'
        }
        print(f"   📦 Size: {hull_size}")
        print(f"   📊 Volume: {hull_volume:.2f}")
        print(f"   📈 Volume reduction: {hull_volume_reduction:.1f}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['hull_pca'] = None
    
    # Test 4: Smart hybrid method
    print("\n4️⃣ Testing Smart Hybrid method...")
    try:
        start_time = time.time()
        smart_size, smart_transform, method_used = smart_minimal_bounding_box(mesh)
        smart_time = time.time() - start_time
        smart_volume = np.prod(smart_size)
        
        results['smart'] = {
            'size': smart_size,
            'time': smart_time,
            'volume': smart_volume,
            'method': method_used
        }
        print(f"   ⏱️  Time: {smart_time:.3f}s")
        print(f"   📦 Size: {smart_size}")
        print(f"   📊 Volume: {smart_volume:.2f}")
        print(f"   🎯 Method used: {method_used}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['smart'] = None
    
    # Performance comparison
    if results['original'] and results['smart']:
        # Avoid division by zero
        if results['smart']['time'] > 0:
            speedup = results['original']['time'] / results['smart']['time']
        else:
            speedup = float('inf') if results['original']['time'] > 0 else 1.0
            
        volume_diff = abs(results['original']['volume'] - results['smart']['volume']) / results['original']['volume'] * 100
        
        print(f"\n🚀 PERFORMANCE SUMMARY:")
        if speedup == float('inf'):
            print(f"   ⚡ Speedup: Extremely fast (smart method < 0.001s)")
        else:
            print(f"   ⚡ Speedup: {speedup:.1f}x faster")
        print(f"   🎯 Accuracy: {100-volume_diff:.1f}% (volume difference: {volume_diff:.1f}%)")
        
        if speedup > 2 or speedup == float('inf'):
            print(f"   ✅ Significant improvement!")
        elif speedup > 1.2:
            print(f"   ✅ Good improvement!")
        else:
            print(f"   ⚠️  Minimal improvement")
    
    return results

def main():
    """Main test function"""
    print("🧪 PCA Bounding Box Optimization Test")
    print("="*60)
    
    # Look for test STL files
    test_files = []
    
    # Search in common locations
    search_dirs = [
        ".",
        "..",
        "../../test_files",
        "C:/Users/andre/OneDrive/Namizje/deproma/deproma test 3d files/test files"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.lower().endswith('.stl'):
                    full_path = os.path.join(search_dir, file)
                    if os.path.getsize(full_path) < 50 * 1024 * 1024:  # Skip files > 50MB
                        test_files.append(full_path)
                        if len(test_files) >= 5:  # Limit to 5 test files
                            break
        if test_files:
            break
    
    if not test_files:
        print("❌ No STL files found for testing")
        return
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Test each file
    all_results = []
    for stl_file in test_files:
        result = test_bounding_box_methods(stl_file)
        if result:
            all_results.append(result)
    
    # Overall summary
    if all_results:
        print(f"\n🎯 OVERALL RESULTS ({len(all_results)} files tested)")
        print("="*60)
        
        speedups = []
        accuracy_scores = []
        
        for result in all_results:
            if result.get('original') and result.get('smart'):
                if result['smart']['time'] > 0:
                    speedup = result['original']['time'] / result['smart']['time']
                else:
                    speedup = 100.0  # Cap extremely fast results for averaging
                    
                volume_diff = abs(result['original']['volume'] - result['smart']['volume']) / result['original']['volume'] * 100
                accuracy = 100 - volume_diff
                
                speedups.append(speedup)
                accuracy_scores.append(accuracy)
        
        if speedups:
            avg_speedup = np.mean(speedups)
            avg_accuracy = np.mean(accuracy_scores)
            
            print(f"📈 Average speedup: {avg_speedup:.1f}x")
            print(f"🎯 Average accuracy: {avg_accuracy:.1f}%")
            print(f"⚡ Max speedup: {max(speedups):.1f}x")
            print(f"🎯 Min accuracy: {min(accuracy_scores):.1f}%")
            
            if avg_speedup > 5:
                print(f"\n🎉 EXCELLENT PERFORMANCE IMPROVEMENT!")
            elif avg_speedup > 2:
                print(f"\n✅ GOOD PERFORMANCE IMPROVEMENT!")
            else:
                print(f"\n⚠️  MODEST PERFORMANCE IMPROVEMENT")

if __name__ == "__main__":
    main() 