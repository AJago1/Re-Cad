/**
 * STL Analyzer - Advanced BVH Core Implementation
 * C++ Implementation for Pack3D Integration
 * 
 * Performance Target: 1000x speedup over Python implementation
 * Based on adaptive variance-based subdivision algorithm
 */

#include "cpp_bvh_core.hpp"
#include <chrono>
#include <numeric>
#include <cstring>

namespace stl_analyzer {

void AdaptiveBVH::calculate_variance(const float* vertices, const std::vector<int>& indices,
                                   float* variance_out) const {
    if (indices.empty()) {
        variance_out[0] = variance_out[1] = variance_out[2] = 0.0f;
        return;
    }
    
    // Calculate mean for each axis
    float mean[3] = {0.0f, 0.0f, 0.0f};
    for (int idx : indices) {
        int vertex_offset = idx * 3;
        for (int axis = 0; axis < 3; axis++) {
            mean[axis] += vertices[vertex_offset + axis];
        }
    }
    
    float inv_count = 1.0f / indices.size();
    for (int axis = 0; axis < 3; axis++) {
        mean[axis] *= inv_count;
    }
    
    // Calculate variance for each axis
    for (int axis = 0; axis < 3; axis++) {
        float sum_squared_diff = 0.0f;
        for (int idx : indices) {
            int vertex_offset = idx * 3;
            float diff = vertices[vertex_offset + axis] - mean[axis];
            sum_squared_diff += diff * diff;
        }
        variance_out[axis] = sum_squared_diff * inv_count;
    }
}

void AdaptiveBVH::split_vertex_group(const float* vertices, const VertexGroup& group,
                                   VertexGroup& left_group, VertexGroup& right_group) const {
    if (group.indices.size() <= 1) {
        // Can't split single vertex or empty group
        left_group.indices = group.indices;
        right_group.indices.clear();
        return;
    }
    
    // Find axis with maximum variance
    float variance[3];
    calculate_variance(vertices, group.indices, variance);
    
    int split_axis = 0;
    float max_variance = variance[0];
    for (int axis = 1; axis < 3; axis++) {
        if (variance[axis] > max_variance) {
            max_variance = variance[axis];
            split_axis = axis;
        }
    }
    
    // Calculate median value along split axis
    std::vector<float> axis_values;
    axis_values.reserve(group.indices.size());
    for (int idx : group.indices) {
        axis_values.push_back(vertices[idx * 3 + split_axis]);
    }
    
    std::sort(axis_values.begin(), axis_values.end());
    float median_value = axis_values[axis_values.size() / 2];
    
    // Split vertices based on median
    left_group.indices.clear();
    right_group.indices.clear();
    left_group.indices.reserve(group.indices.size() / 2);
    right_group.indices.reserve(group.indices.size() / 2);
    
    for (int idx : group.indices) {
        if (vertices[idx * 3 + split_axis] <= median_value) {
            left_group.indices.push_back(idx);
        } else {
            right_group.indices.push_back(idx);
        }
    }
    
    // Ensure both groups have at least one vertex
    if (left_group.indices.empty() && !right_group.indices.empty()) {
        left_group.indices.push_back(right_group.indices.back());
        right_group.indices.pop_back();
    } else if (right_group.indices.empty() && !left_group.indices.empty()) {
        right_group.indices.push_back(left_group.indices.back());
        left_group.indices.pop_back();
    }
}

BoundingBox AdaptiveBVH::create_tight_bbox(const float* vertices, 
                                         const std::vector<int>& indices,
                                         float padding) const {
    if (indices.empty()) {
        return BoundingBox();
    }
    
    // Initialize with first vertex
    int first_vertex = indices[0] * 3;
    float min_coords[3] = {vertices[first_vertex], vertices[first_vertex + 1], vertices[first_vertex + 2]};
    float max_coords[3] = {vertices[first_vertex], vertices[first_vertex + 1], vertices[first_vertex + 2]};
    
    // Find actual min/max coordinates
    for (int idx : indices) {
        int vertex_offset = idx * 3;
        for (int axis = 0; axis < 3; axis++) {
            float coord = vertices[vertex_offset + axis];
            min_coords[axis] = std::min(min_coords[axis], coord);
            max_coords[axis] = std::max(max_coords[axis], coord);
        }
    }
    
    // Add padding to avoid zero-volume boxes
    for (int axis = 0; axis < 3; axis++) {
        float extent = max_coords[axis] - min_coords[axis];
        float dynamic_padding = std::max(padding, extent * 0.02f);  // 2% padding or minimum
        min_coords[axis] -= dynamic_padding;
        max_coords[axis] += dynamic_padding;
    }
    
    return BoundingBox(min_coords, max_coords);
}

void AdaptiveBVH::recursive_split_vertices(const float* vertices, int vertex_count,
                                         std::vector<VertexGroup>& groups) const {
    // Start with all vertices in one group
    std::vector<int> all_indices(vertex_count);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    groups.clear();
    groups.emplace_back(all_indices);
    
    // Split until we have target number of groups
    while (groups.size() < static_cast<size_t>(leaf_count)) {
        // Find group with maximum variance to split
        int best_group_idx = -1;
        float best_variance = 0.0f;
        
        for (size_t i = 0; i < groups.size(); i++) {
            if (groups[i].indices.size() > 1) {  // Only split groups with multiple vertices
                float variance[3];
                calculate_variance(vertices, groups[i].indices, variance);
                float max_variance = std::max({variance[0], variance[1], variance[2]});
                
                if (max_variance > best_variance) {
                    best_variance = max_variance;
                    best_group_idx = static_cast<int>(i);
                }
            }
        }
        
        if (best_group_idx == -1) {
            // No more groups can be split
            break;
        }
        
        // Split the group with maximum variance
        VertexGroup left_group, right_group;
        split_vertex_group(vertices, groups[best_group_idx], left_group, right_group);
        
        // Replace original group with two new groups
        groups[best_group_idx] = left_group;
        groups.push_back(right_group);
    }
    
    // Ensure we have exactly leaf_count groups
    groups.resize(leaf_count);
}

bool AdaptiveBVH::build_from_vertices(const float* vertices, int vertex_count) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!vertices || vertex_count <= 0) {
        active_leaves = 0;
        return false;
    }
    
    try {
        // Use adaptive geometry-based subdivision
        std::vector<VertexGroup> groups;
        recursive_split_vertices(vertices, vertex_count, groups);
        
        // Create tight bounding boxes for each group
        active_leaves = 0;
        for (int i = 0; i < leaf_count && i < static_cast<int>(groups.size()); i++) {
            if (!groups[i].indices.empty()) {
                leaves[i] = create_tight_bbox(vertices, groups[i].indices);
                if (leaves[i].volume() > 0.001f) {  // Only count boxes with reasonable volume
                    active_leaves++;
                }
            } else {
                // Empty group: create minimal placeholder
                leaves[i] = BoundingBox(-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, 0.1f);
            }
        }
        
        // Fill remaining leaves with empty boxes if needed
        for (int i = static_cast<int>(groups.size()); i < leaf_count; i++) {
            leaves[i] = BoundingBox();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update build statistics
        last_build_stats.build_time_ms = duration.count() / 1000.0;
        last_build_stats.vertices_processed = vertex_count;
        last_build_stats.leaves_with_geometry = active_leaves;
        last_build_stats.subdivision_depth = static_cast<int>(std::log2(leaf_count)) + 1;
        
        return true;
        
    } catch (const std::exception& e) {
        active_leaves = 0;
        return false;
    }
}

bool AdaptiveBVH::check_collision(const AdaptiveBVH& other, float tolerance) const {
    // Fast early exit if no geometry in either BVH
    if (active_leaves == 0 || other.active_leaves == 0) {
        return false;
    }
    
    // Check all pairs of leaves for collision
    // This is O(leaf_count^2) but with small leaf counts (8-32) it's very fast
    for (int i = 0; i < leaf_count; i++) {
        for (int j = 0; j < other.leaf_count; j++) {
            if (leaves[i].overlaps(other.leaves[j], tolerance)) {
                return true;  // Early exit on first collision
            }
        }
    }
    
    return false;  // No collision detected
}

BoundingBox AdaptiveBVH::get_overall_bounds() const {
    if (active_leaves == 0) {
        return BoundingBox();
    }
    
    BoundingBox overall = leaves[0];
    for (int i = 1; i < leaf_count; i++) {
        if (leaves[i].volume() > 0.001f) {  // Only include non-empty leaves
            for (int axis = 0; axis < 3; axis++) {
                overall.min[axis] = std::min(overall.min[axis], leaves[i].min[axis]);
                overall.max[axis] = std::max(overall.max[axis], leaves[i].max[axis]);
            }
        }
    }
    
    return overall;
}

// C Interface Functions for Python Integration

extern "C" {

AdaptiveBVH* create_adaptive_bvh(const float* vertices, int vertex_count, int leaf_count) {
    AdaptiveBVH* bvh = new AdaptiveBVH(leaf_count);
    if (bvh->build_from_vertices(vertices, vertex_count)) {
        return bvh;
    } else {
        delete bvh;
        return nullptr;
    }
}

bool bvh_collision_check(const AdaptiveBVH* bvh1, const AdaptiveBVH* bvh2, float tolerance) {
    if (!bvh1 || !bvh2) {
        return false;
    }
    return bvh1->check_collision(*bvh2, tolerance);
}

void bvh_get_leaf_bounds(const AdaptiveBVH* bvh, int leaf_index, float* min_coords, float* max_coords) {
    if (!bvh || leaf_index < 0 || leaf_index >= bvh->get_leaf_count()) {
        return;
    }
    
    const BoundingBox& leaf = bvh->get_leaf(leaf_index);
    std::memcpy(min_coords, leaf.min, 3 * sizeof(float));
    std::memcpy(max_coords, leaf.max, 3 * sizeof(float));
}

void bvh_get_build_stats(const AdaptiveBVH* bvh, double* build_time, int* active_leaves) {
    if (!bvh) {
        return;
    }
    
    if (build_time) {
        *build_time = bvh->last_build_stats.build_time_ms;
    }
    if (active_leaves) {
        *active_leaves = bvh->get_active_leaf_count();
    }
}

void destroy_adaptive_bvh(AdaptiveBVH* bvh) {
    delete bvh;
}

} // extern "C"

} // namespace stl_analyzer