#!/usr/bin/env python3
"""
Fogleman's Pack3D BVH Wrapper - Triangle-centroid approach
"""

import numpy as np
import time

try:
    import cpp_bvh_fogleman
    FOGLEMAN_BVH_AVAILABLE = True
    print("✅ Fogleman's Pack3D BVH extension loaded")
except ImportError as e:
    FOGLEMAN_BVH_AVAILABLE = False
    print(f"❌ Fogleman BVH not available: {e}")

class FoglemanBVH:
    """
    Fogleman's Pack3D BVH implementation - C++ powered
    
    Key differences from our previous approach:
    - Triangle-centroid based region assignment (not vertex-based)
    - Tight-fitting boxes only where geometry exists
    - No gap filling or forced connections
    - Can produce 1-8 boxes (skips empty regions)
    - Fast and geometry-aware
    """
    
    def __init__(self, mesh, leaf_count: int = 8):
        self.mesh = mesh
        self.leaf_count = leaf_count
        self.build_time_ms = 0.0
        self.active_leaves = 0
        self.cpp_bvh_id = None
        self.use_cpp = FOGLEMAN_BVH_AVAILABLE
        self.cached_bounds = []
        
        if not self.use_cpp:
            raise ImportError("Fogleman C++ BVH extension not available")
        
        # Build C++ BVH
        self._build_fogleman_bvh()
    
    def _build_fogleman_bvh(self):
        """Build Fogleman's triangle-centroid BVH"""
        if not self.use_cpp or not hasattr(self.mesh, 'vertices'):
            return
        
        start_time = time.time()
        
        try:
            # Convert mesh to triangle vertices format expected by C++
            # Each triangle becomes 3 consecutive vertices (9 floats total)
            triangle_vertices = self.mesh.vertices[self.mesh.faces]  # Shape: (n_faces, 3, 3)
            vertices_flat = triangle_vertices.flatten().tolist()  # Flatten to 1D list
            
            print(f"📐 Mesh conversion: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces → {len(vertices_flat)//3} triangle vertices")
            
            # Create Fogleman C++ BVH
            self.cpp_bvh_id = cpp_bvh_fogleman.create_bvh(vertices_flat, self.leaf_count)
            
            if self.cpp_bvh_id is not None:
                # Get build statistics from C++
                stats = cpp_bvh_fogleman.get_bvh_stats(self.cpp_bvh_id)
                self.build_time_ms = stats['build_time_ms']
                self.active_leaves = stats['leaf_count']  # Actual number of boxes created
                
                print(f"🚀 Fogleman C++ BVH: {self.active_leaves}-box created in {self.build_time_ms:.3f}ms "
                      f"({self.active_leaves}/{self.leaf_count} regions active)")
                
                # Cache leaf bounds for visualization
                self._cache_leaf_bounds()
                
            else:
                print("❌ Fogleman C++ BVH creation failed")
                self.use_cpp = False
                
        except Exception as e:
            print(f"❌ Fogleman BVH error: {e}")
            self.use_cpp = False
            
        self.build_time_ms = (time.time() - start_time) * 1000
    
    def _cache_leaf_bounds(self):
        """Cache leaf bounds from Fogleman C++ BVH"""
        self.cached_bounds = []
        if not self.use_cpp or self.cpp_bvh_id is None: 
            return
        
        try:
            for i in range(self.active_leaves):  # Only iterate over actual boxes
                bounds = cpp_bvh_fogleman.get_leaf_bounds(self.cpp_bvh_id, i)
                if bounds and len(bounds) == 6:
                    min_coords = np.array([bounds[0], bounds[1], bounds[2]])
                    max_coords = np.array([bounds[3], bounds[4], bounds[5]])
                    self.cached_bounds.append((min_coords, max_coords))
                    
        except Exception as e:
            print(f"⚠️ Failed to get Fogleman bounds: {e}")
    
    def get_leaf_bounds(self, leaf_index: int):
        """Get bounding box for a specific leaf"""
        if leaf_index < 0 or leaf_index >= len(self.cached_bounds):
            return None, None
        
        return self.cached_bounds[leaf_index]
    
    def check_collision(self, other_bvh):
        """Check collision with another BVH (placeholder for Pack3D)"""
        # This would be implemented for Pack3D collision detection
        return False
    
    @property 
    def leaves(self):
        """Compatibility property for existing code"""
        class LeafAccessor:
            def __init__(self, cached_bounds):
                self.cached_bounds = cached_bounds
            
            def __len__(self):
                return len(self.cached_bounds)
            
            def __getitem__(self, index):
                if index < len(self.cached_bounds):
                    min_coords, max_coords = self.cached_bounds[index]
                    # Return object with min/max attributes for compatibility
                    class BoundingBox:
                        def __init__(self, min_coords, max_coords):
                            self.min = min_coords
                            self.max = max_coords
                    return BoundingBox(min_coords, max_coords)
                return None
        
        return LeafAccessor(self.cached_bounds)

def create_fogleman_bvh_from_stl(stl_path: str):
    """Create Fogleman BVH from STL file path"""
    try:
        import trimesh
        mesh = trimesh.load_mesh(stl_path)
        return FoglemanBVH(mesh)
    except Exception as e:
        print(f"❌ Failed to create Fogleman BVH from {stl_path}: {e}")
        return None
