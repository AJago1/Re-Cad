"""
8-Leaf Bounding Volume Hierarchy (BVH) Implementation for Pack3D Integration

This module implements the 8-leaf BVH algorithm inspired by Michael Fogleman's pack3d.
The BVH represents each 3D part using exactly 8 bounding boxes for fast collision detection.

Key features:
- Simplified BVH with fixed 8 leaves for consistent performance
- Fast collision detection for 3D packing optimization
- Visualization support for debugging and understanding
- Python implementation (ready for C++ optimization in Phase 2)
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
import time


class BoundingBox:
    """Simple bounding box representation for BVH leaf nodes"""
    
    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        """
        Initialize bounding box
        
        Args:
            min_point: [x_min, y_min, z_min]
            max_point: [x_max, y_max, z_max]
        """
        self.min = np.array(min_point, dtype=np.float32)
        self.max = np.array(max_point, dtype=np.float32)
        
    def overlaps(self, other: 'BoundingBox', tolerance: float = 1e-6) -> bool:
        """
        Check if this bounding box overlaps with another
        
        Args:
            other: Another BoundingBox
            tolerance: Small epsilon for floating point comparison
            
        Returns:
            True if boxes overlap, False otherwise
        """
        # Check separation on each axis - if separated on any axis, no overlap
        for i in range(3):
            if (self.max[i] < other.min[i] - tolerance or 
                self.min[i] > other.max[i] + tolerance):
                return False
        return True
        
    def volume(self) -> float:
        """Calculate volume of bounding box"""
        extents = self.max - self.min
        return np.prod(np.maximum(extents, 0))  # Avoid negative volumes
        
    def center(self) -> np.ndarray:
        """Get center point of bounding box"""
        return (self.min + self.max) / 2
        
    def extents(self) -> np.ndarray:
        """Get extents (size) of bounding box"""
        return self.max - self.min
        
    def __repr__(self):
        return f"BBox(min={self.min}, max={self.max})"


class EightLeafBVH:
    """
    8-Leaf Bounding Volume Hierarchy for fast collision detection
    
    This implementation creates exactly 8 bounding boxes that efficiently
    represent the 3D mesh for collision detection in packing optimization.
    """
    
    def __init__(self, mesh: trimesh.Trimesh, leaf_count: int = 8):
        """
        Create N-leaf BVH from a trimesh
        
        Args:
            mesh: Trimesh object to create BVH for
            leaf_count: Number of leaves (8, 16, 32 recommended)
        """
        self.mesh = mesh
        self.leaves: List[BoundingBox] = []
        self.leaf_count = leaf_count  # Configurable leaf count
        
        # Generate the N-leaf BVH
        self._build_adaptive_bvh()
        
    def _build_adaptive_bvh(self):
        """
        Build the 8-leaf BVH using adaptive geometry-based subdivision
        
        Strategy:
        1. Use PCA to find principal axes of the mesh
        2. Recursively split along the axis with maximum variance
        3. Create 8 leaves that better follow geometry distribution
        4. Fallback to octree if PCA fails
        """
        if not hasattr(self.mesh, 'vertices') or len(self.mesh.vertices) == 0:
            # Fallback: create single bounding box
            bbox = self.mesh.bounding_box
            self.leaves = [BoundingBox(bbox.bounds[0], bbox.bounds[1])]
            return
            
        print(f"🔧 Building {self.leaf_count}-leaf BVH for mesh with {len(self.mesh.vertices)} vertices")
        start_time = time.time()
        
        # Use adaptive geometry-based subdivision instead of fixed octree
        try:
            # Recursive binary splitting to create N groups based on geometry
            vertex_groups = self._recursive_split_vertices(self.mesh.vertices, target_groups=self.leaf_count)
            
            # Create tight bounding boxes for each group
            self.leaves = []
            for group_vertices in vertex_groups:
                if len(group_vertices) > 0:
                    # Create tight bounding box
                    tight_min = np.min(group_vertices, axis=0)
                    tight_max = np.max(group_vertices, axis=0)
                    
                    # Add small padding
                    padding = np.maximum((tight_max - tight_min) * 0.02, 0.1)
                    tight_min -= padding
                    tight_max += padding
                    
                    self.leaves.append(BoundingBox(tight_min, tight_max))
                else:
                    # Empty group: create minimal box
                    mesh_center = (self.mesh.bounds[0] + self.mesh.bounds[1]) / 2
                    tiny_padding = np.array([0.1, 0.1, 0.1])
                    self.leaves.append(BoundingBox(mesh_center - tiny_padding, 
                                                 mesh_center + tiny_padding))
            
            # Ensure we have exactly the target number of leaves
            while len(self.leaves) < self.leaf_count:
                # Split the largest box if we have fewer than target
                largest_idx = 0
                largest_volume = 0
                for i, leaf in enumerate(self.leaves):
                    volume = np.prod(leaf.extents())
                    if volume > largest_volume:
                        largest_volume = volume
                        largest_idx = i
                
                # Split the largest box
                largest_box = self.leaves[largest_idx]
                self.leaves.pop(largest_idx)
                
                # Split along the longest axis
                extents = largest_box.extents()
                longest_axis = np.argmax(extents)
                split_point = (largest_box.min[longest_axis] + largest_box.max[longest_axis]) / 2
                
                # Create two smaller boxes
                box1_min = largest_box.min.copy()
                box1_max = largest_box.max.copy()
                box1_max[longest_axis] = split_point
                
                box2_min = largest_box.min.copy()
                box2_max = largest_box.max.copy()
                box2_min[longest_axis] = split_point
                
                self.leaves.append(BoundingBox(box1_min, box1_max))
                self.leaves.append(BoundingBox(box2_min, box2_max))
            
            # Trim to exactly target count if we have more
            self.leaves = self.leaves[:self.leaf_count]
            
            non_empty_leaves = len([leaf for leaf in self.leaves if np.prod(leaf.extents()) > 1.0])
            
        except Exception as e:
            print(f"⚠️ Adaptive subdivision failed: {e}, falling back to octree")
            # Fallback to simple octree method
            self._build_octree_fallback()
        
        build_time = time.time() - start_time
        print(f"✅ Built {self.leaf_count}-leaf BVH in {build_time:.3f}s ({non_empty_leaves}/{self.leaf_count} leaves with geometry)")
        
        # Ensure we have exactly target number of leaves
        assert len(self.leaves) == self.leaf_count, f"Expected {self.leaf_count} leaves, got {len(self.leaves)}"
    
    def _are_octants_adjacent(self, oct1_min, oct1_max, oct2_min, oct2_max):
        """Check if two octants are adjacent (share a face, edge, or corner)"""
        # Two octants are adjacent if they touch in at least one dimension
        # and don't have a gap in any dimension
        
        touching = False
        for dim in range(3):
            # Check if they touch in this dimension
            if (abs(oct1_max[dim] - oct2_min[dim]) < 1e-6 or 
                abs(oct2_max[dim] - oct1_min[dim]) < 1e-6):
                touching = True
                break
        
        return touching
    
    def _calculate_gap_closure(self, box1, box2):
        """Calculate how to expand box1 to close gap with box2"""
        min_expansion = np.zeros(3)
        max_expansion = np.zeros(3)
        
        for dim in range(3):
            # Check for gap in each dimension
            if box1.max[dim] < box2.min[dim]:
                # Gap: box1 is "left" of box2, expand box1 max toward box2
                gap_size = box2.min[dim] - box1.max[dim]
                max_expansion[dim] = gap_size * 0.5  # Close half the gap
            elif box2.max[dim] < box1.min[dim]:
                # Gap: box2 is "left" of box1, expand box1 min toward box2  
                gap_size = box1.min[dim] - box2.max[dim]
                min_expansion[dim] = -gap_size * 0.5  # Close half the gap
        
        return [min_expansion, max_expansion]
    
    def _recursive_split_vertices(self, vertices, target_groups=8):
        """
        Recursively split vertices into groups based on geometry distribution
        
        Uses variance-based splitting along principal axes for better geometry fit
        """
        if len(vertices) == 0:
            return [np.array([])]
        
        # Start with all vertices in one group
        groups = [vertices]
        
        # Split until we have target_groups
        while len(groups) < target_groups:
            # Find the group with maximum variance to split
            best_group_idx = 0
            best_variance = 0
            
            for i, group in enumerate(groups):
                if len(group) > 1:  # Only split groups with multiple vertices
                    # Calculate variance along each axis
                    variances = np.var(group, axis=0)
                    max_variance = np.max(variances)
                    
                    if max_variance > best_variance:
                        best_variance = max_variance
                        best_group_idx = i
            
            # Split the group with maximum variance
            group_to_split = groups[best_group_idx]
            
            if len(group_to_split) <= 1:
                # Can't split further, duplicate this group
                groups.append(group_to_split.copy())
            else:
                # Find axis with maximum variance
                variances = np.var(group_to_split, axis=0)
                split_axis = np.argmax(variances)
                
                # Split at median along that axis
                median_value = np.median(group_to_split[:, split_axis])
                
                # Split vertices
                left_mask = group_to_split[:, split_axis] <= median_value
                right_mask = ~left_mask
                
                left_group = group_to_split[left_mask]
                right_group = group_to_split[right_mask]
                
                # Replace original group with two new groups
                groups.pop(best_group_idx)
                groups.append(left_group)
                groups.append(right_group)
        
        return groups[:target_groups]  # Ensure exactly target_groups
    
    def _build_octree_fallback(self):
        """Fallback octree method if adaptive subdivision fails"""
        mesh_min = self.mesh.bounds[0]
        mesh_max = self.mesh.bounds[1]
        mesh_center = (mesh_min + mesh_max) / 2
        
        self.leaves = []
        
        # Create 8 octants
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    oct_min = mesh_min.copy()
                    oct_max = mesh_max.copy()
                    
                    if x == 0:
                        oct_max[0] = mesh_center[0]
                    else:
                        oct_min[0] = mesh_center[0]
                        
                    if y == 0:
                        oct_max[1] = mesh_center[1]
                    else:
                        oct_min[1] = mesh_center[1]
                        
                    if z == 0:
                        oct_max[2] = mesh_center[2]
                    else:
                        oct_min[2] = mesh_center[2]
                    
                    # Find vertices in this octant
                    in_octant = np.all((self.mesh.vertices >= oct_min) & 
                                     (self.mesh.vertices <= oct_max), axis=1)
                    
                    if np.any(in_octant):
                        # Create tight box around vertices
                        octant_vertices = self.mesh.vertices[in_octant]
                        tight_min = np.min(octant_vertices, axis=0)
                        tight_max = np.max(octant_vertices, axis=0)
                        
                        padding = np.maximum((tight_max - tight_min) * 0.02, 0.1)
                        tight_min -= padding
                        tight_max += padding
                        
                        self.leaves.append(BoundingBox(tight_min, tight_max))
                    else:
                        # Empty octant
                        octant_center = (oct_min + oct_max) / 2
                        padding = np.array([0.1, 0.1, 0.1])
                        self.leaves.append(BoundingBox(octant_center - padding, 
                                                     octant_center + padding))
        
    def check_collision(self, other: 'EightLeafBVH', tolerance: float = 0.5) -> bool:
        """
        Check collision between this BVH and another BVH
        
        Args:
            other: Another EightLeafBVH
            tolerance: Minimum spacing between parts (mm)
            
        Returns:
            True if collision detected, False otherwise
        """
        if not other or len(other.leaves) == 0:
            return False
            
        # Check all pairs of leaves (up to 8×8 = 64 checks)
        collision_count = 0
        
        for my_leaf in self.leaves:
            for other_leaf in other.leaves:
                if my_leaf.overlaps(other_leaf, tolerance):
                    collision_count += 1
                    # Early exit on first collision for speed
                    return True
        
        return False
        
    def get_total_bounding_box(self) -> BoundingBox:
        """Get the overall bounding box that contains all leaves"""
        if not self.leaves:
            return BoundingBox(np.zeros(3), np.zeros(3))
            
        # Find min and max across all leaves
        all_mins = np.array([leaf.min for leaf in self.leaves])
        all_maxs = np.array([leaf.max for leaf in self.leaves])
        
        global_min = np.min(all_mins, axis=0)
        global_max = np.max(all_maxs, axis=0)
        
        return BoundingBox(global_min, global_max)
        
    def get_leaf_centers(self) -> List[np.ndarray]:
        """Get center points of all leaves for visualization"""
        return [leaf.center() for leaf in self.leaves]
        
    def get_leaf_volumes(self) -> List[float]:
        """Get volumes of all leaves for analysis"""
        return [leaf.volume() for leaf in self.leaves]
        
    def __repr__(self):
        total_volume = sum(leaf.volume() for leaf in self.leaves)
        return f"EightLeafBVH(leaves={len(self.leaves)}, total_volume={total_volume:.2f}mm³)"


def create_8_leaf_bvh_from_stl(stl_path: str) -> Optional[EightLeafBVH]:
    """
    Create 8-leaf BVH from STL file path
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        EightLeafBVH object or None if failed
    """
    try:
        mesh = trimesh.load_mesh(stl_path)
        return EightLeafBVH(mesh)
    except Exception as e:
        print(f"❌ Failed to create BVH from {stl_path}: {e}")
        return None


def benchmark_bvh_performance(stl_path: str, num_tests: int = 1000) -> dict:
    """
    Benchmark BVH collision detection performance
    
    Args:
        stl_path: Path to test STL file
        num_tests: Number of collision tests to perform
        
    Returns:
        Performance metrics dictionary
    """
    print(f"🚀 Benchmarking BVH performance with {num_tests} collision tests...")
    
    # Create BVH
    bvh = create_8_leaf_bvh_from_stl(stl_path)
    if not bvh:
        return {"error": "Failed to create BVH"}
    
    # Collision detection benchmark
    start_time = time.time()
    collision_count = 0
    
    for i in range(num_tests):
        # Test collision with itself (should always be true)
        if bvh.check_collision(bvh):
            collision_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_tests) * 1000
    
    metrics = {
        "total_tests": num_tests,
        "total_time_seconds": total_time,
        "average_time_ms": avg_time_ms,
        "collisions_detected": collision_count,
        "tests_per_second": num_tests / total_time,
        "leaf_count": len(bvh.leaves),
        "total_bvh_volume": sum(bvh.get_leaf_volumes())
    }
    
    print(f"📊 BVH Performance Results:")
    print(f"   Average collision check: {avg_time_ms:.3f} ms")
    print(f"   Tests per second: {metrics['tests_per_second']:.0f}")
    print(f"   Total BVH volume: {metrics['total_bvh_volume']:.2f} mm³")
    
    return metrics


if __name__ == "__main__":
    # Test the BVH implementation
    import sys
    
    if len(sys.argv) > 1:
        stl_path = sys.argv[1]
        print(f"Testing 8-leaf BVH with: {stl_path}")
        
        # Create BVH
        bvh = create_8_leaf_bvh_from_stl(stl_path)
        if bvh:
            print(f"✅ Created BVH: {bvh}")
            
            # Run performance benchmark
            benchmark_bvh_performance(stl_path, 1000)
        else:
            print("❌ Failed to create BVH")
    else:
        print("Usage: python bvh_8leaf.py <path_to_stl_file>")