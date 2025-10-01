#pragma once
/**
 * STL Analyzer - Advanced BVH Core for Pack3D Integration
 * C++ Implementation for 1000x Performance Boost
 * 
 * Based on Michael Fogleman's pack3d algorithm design
 * Target: 8-leaf minimum, configurable to 16/32 leaves
 */

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace stl_analyzer {

/**
 * Efficient 3D Bounding Box for collision detection
 * Optimized for pack3d simulated annealing
 */
struct BoundingBox {
    float min[3];  // minimum x, y, z coordinates
    float max[3];  // maximum x, y, z coordinates
    
    BoundingBox() {
        min[0] = min[1] = min[2] = 0.0f;
        max[0] = max[1] = max[2] = 0.0f;
    }
    
    BoundingBox(float min_x, float min_y, float min_z, 
                float max_x, float max_y, float max_z) {
        min[0] = min_x; min[1] = min_y; min[2] = min_z;
        max[0] = max_x; max[1] = max_y; max[2] = max_z;
    }
    
    BoundingBox(const float* min_coords, const float* max_coords) {
        for (int i = 0; i < 3; i++) {
            min[i] = min_coords[i];
            max[i] = max_coords[i];
        }
    }
    
    // Get center point of the bounding box
    void center(float* out_center) const {
        for (int i = 0; i < 3; i++) {
            out_center[i] = (min[i] + max[i]) * 0.5f;
        }
    }
    
    // Get extents (size) of the bounding box
    void extents(float* out_extents) const {
        for (int i = 0; i < 3; i++) {
            out_extents[i] = max[i] - min[i];
        }
    }
    
    // Calculate volume of the bounding box
    float volume() const {
        float dx = max[0] - min[0];
        float dy = max[1] - min[1];
        float dz = max[2] - min[2];
        return std::max(0.0f, dx * dy * dz);
    }
    
    // Check if this box overlaps with another (with tolerance for spacing)
    bool overlaps(const BoundingBox& other, float tolerance = 0.5f) const {
        // Two boxes overlap if they intersect in all 3 dimensions
        // tolerance is minimum spacing required between parts
        for (int i = 0; i < 3; i++) {
            if (max[i] + tolerance < other.min[i] || 
                min[i] - tolerance > other.max[i]) {
                return false;  // Separated in this dimension
            }
        }
        return true;  // Overlapping in all dimensions
    }
    
    // Expand the bounding box by a margin
    void expand(float margin) {
        for (int i = 0; i < 3; i++) {
            min[i] -= margin;
            max[i] += margin;
        }
    }
};

/**
 * Adaptive BVH for efficient 3D collision detection
 * Implements variance-based subdivision from our Python prototype
 * Configurable leaf count: 8, 16, 32 for different accuracy/speed trade-offs
 */
class AdaptiveBVH {
private:
    static const int MAX_LEAVES = 32;
    BoundingBox leaves[MAX_LEAVES];
    int leaf_count;
    int active_leaves;  // Number of leaves with actual geometry
    
    // Vertex group for recursive subdivision
    struct VertexGroup {
        std::vector<int> indices;  // indices into original vertex array
        BoundingBox bounds;
        
        VertexGroup() {}
        VertexGroup(const std::vector<int>& idx) : indices(idx) {}
    };
    
    // Calculate variance along each axis for a group of vertices
    void calculate_variance(const float* vertices, const std::vector<int>& indices,
                          float* variance_out) const;
    
    // Split a vertex group along the axis with maximum variance
    void split_vertex_group(const float* vertices, const VertexGroup& group,
                           VertexGroup& left_group, VertexGroup& right_group) const;
    
    // Create tight bounding box around a group of vertices
    BoundingBox create_tight_bbox(const float* vertices, 
                                 const std::vector<int>& indices,
                                 float padding = 0.1f) const;
    
    // Recursive binary splitting to create target number of groups
    void recursive_split_vertices(const float* vertices, int vertex_count,
                                 std::vector<VertexGroup>& groups) const;

public:
    AdaptiveBVH(int target_leaf_count = 8) 
        : leaf_count(std::min(target_leaf_count, MAX_LEAVES)), active_leaves(0) {}
    
    // Build BVH from vertex array (x,y,z,x,y,z,...)
    bool build_from_vertices(const float* vertices, int vertex_count);
    
    // Check collision with another BVH
    bool check_collision(const AdaptiveBVH& other, float tolerance = 0.5f) const;
    
    // Get number of leaves with geometry
    int get_active_leaf_count() const { return active_leaves; }
    
    // Get leaf count
    int get_leaf_count() const { return leaf_count; }
    
    // Get bounding box for specific leaf (for visualization)
    const BoundingBox& get_leaf(int index) const { 
        return leaves[std::min(index, leaf_count - 1)]; 
    }
    
    // Get overall bounding box
    BoundingBox get_overall_bounds() const;
    
    // Performance statistics
    struct BuildStats {
        double build_time_ms;
        int vertices_processed;
        int leaves_with_geometry;
        int subdivision_depth;
    };
    
    BuildStats last_build_stats;
};

/**
 * Factory functions for Python interface
 */
extern "C" {
    // Create BVH from vertex array
    AdaptiveBVH* create_adaptive_bvh(const float* vertices, int vertex_count, int leaf_count = 8);
    
    // Check collision between two BVHs
    bool bvh_collision_check(const AdaptiveBVH* bvh1, const AdaptiveBVH* bvh2, float tolerance = 0.5f);
    
    // Get leaf bounding box for visualization
    void bvh_get_leaf_bounds(const AdaptiveBVH* bvh, int leaf_index, float* min_coords, float* max_coords);
    
    // Get build statistics
    void bvh_get_build_stats(const AdaptiveBVH* bvh, double* build_time, int* active_leaves);
    
    // Clean up BVH
    void destroy_adaptive_bvh(AdaptiveBVH* bvh);
}

} // namespace stl_analyzer