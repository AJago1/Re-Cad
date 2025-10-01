/**
 * STL Analyzer - C++ Utils Extension with Advanced BVH
 * Extends existing cpp_stl_utils module with Pack3D BVH functionality
 * 
 * This file creates a new module that can be imported alongside the existing
 * cpp_stl_utils.pyd file, adding BVH functions without breaking compatibility
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp_bvh_core.hpp"

namespace py = pybind11;
using namespace stl_analyzer;

// Custom deleter for AdaptiveBVH to ensure proper cleanup
struct BVHDeleter {
    void operator()(AdaptiveBVH* bvh) {
        delete bvh;
    }
};

// Python wrapper for AdaptiveBVH with automatic memory management
class PyAdaptiveBVH {
private:
    std::unique_ptr<AdaptiveBVH, BVHDeleter> bvh;
    
public:
    PyAdaptiveBVH(int leaf_count = 8) 
        : bvh(std::make_unique<AdaptiveBVH>(leaf_count)) {}
    
    bool build_from_vertices(py::array_t<float> vertices_array) {
        py::buffer_info buf = vertices_array.request();
        
        if (buf.ndim != 1) {
            throw std::runtime_error("Vertices must be 1D array (x,y,z,x,y,z,...)");
        }
        
        if (buf.size % 3 != 0) {
            throw std::runtime_error("Vertices array size must be multiple of 3");
        }
        
        float* vertices = static_cast<float*>(buf.ptr);
        int vertex_count = static_cast<int>(buf.size / 3);
        
        return bvh->build_from_vertices(vertices, vertex_count);
    }
    
    bool check_collision(const PyAdaptiveBVH& other, float tolerance = 0.5f) const {
        return bvh->check_collision(*other.bvh, tolerance);
    }
    
    int get_leaf_count() const {
        return bvh->get_leaf_count();
    }
    
    int get_active_leaf_count() const {
        return bvh->get_active_leaf_count();
    }
    
    py::tuple get_leaf_bounds(int leaf_index) const {
        const BoundingBox& leaf = bvh->get_leaf(leaf_index);
        
        py::array_t<float> min_coords = py::array_t<float>(3);
        py::array_t<float> max_coords = py::array_t<float>(3);
        
        py::buffer_info min_buf = min_coords.request();
        py::buffer_info max_buf = max_coords.request();
        
        float* min_ptr = static_cast<float*>(min_buf.ptr);
        float* max_ptr = static_cast<float*>(max_buf.ptr);
        
        for (int i = 0; i < 3; i++) {
            min_ptr[i] = leaf.min[i];
            max_ptr[i] = leaf.max[i];
        }
        
        return py::make_tuple(min_coords, max_coords);
    }
    
    py::dict get_build_stats() const {
        auto stats = bvh->last_build_stats;
        return py::dict(
            "build_time_ms"_a=stats.build_time_ms,
            "vertices_processed"_a=stats.vertices_processed,
            "leaves_with_geometry"_a=stats.leaves_with_geometry,
            "subdivision_depth"_a=stats.subdivision_depth
        );
    }
    
    py::list get_all_leaf_bounds() const {
        py::list bounds_list;
        for (int i = 0; i < bvh->get_leaf_count(); i++) {
            bounds_list.append(get_leaf_bounds(i));
        }
        return bounds_list;
    }
};

// High-level convenience functions for direct use
py::object create_adaptive_bvh_from_mesh(py::array_t<float> vertices, int leaf_count = 8) {
    auto bvh = std::make_unique<PyAdaptiveBVH>(leaf_count);
    
    if (bvh->build_from_vertices(vertices)) {
        return py::cast(std::move(bvh));
    } else {
        return py::none();
    }
}

bool fast_collision_check(py::array_t<float> vertices1, py::array_t<float> vertices2, 
                         int leaf_count = 8, float tolerance = 0.5f) {
    PyAdaptiveBVH bvh1(leaf_count);
    PyAdaptiveBVH bvh2(leaf_count);
    
    if (!bvh1.build_from_vertices(vertices1) || !bvh2.build_from_vertices(vertices2)) {
        return false;
    }
    
    return bvh1.check_collision(bvh2, tolerance);
}

py::dict benchmark_bvh_performance(py::array_t<float> vertices, 
                                  py::list leaf_counts = py::list()) {
    py::dict results;
    
    if (leaf_counts.empty()) {
        leaf_counts.append(8);
        leaf_counts.append(16);
        leaf_counts.append(32);
    }
    
    py::buffer_info buf = vertices.request();
    int vertex_count = static_cast<int>(buf.size / 3);
    
    results["vertex_count"] = vertex_count;
    results["leaf_count_results"] = py::dict();
    
    for (auto leaf_count_obj : leaf_counts) {
        int leaf_count = leaf_count_obj.cast<int>();
        
        // Time BVH creation
        auto start = std::chrono::high_resolution_clock::now();
        PyAdaptiveBVH bvh(leaf_count);
        bool success = bvh.build_from_vertices(vertices);
        auto end = std::chrono::high_resolution_clock::now();
        
        double creation_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        py::dict leaf_results;
        leaf_results["creation_time_ms"] = creation_time;
        leaf_results["success"] = success;
        leaf_results["active_leaves"] = bvh.get_active_leaf_count();
        
        if (success) {
            // Test collision detection speed
            PyAdaptiveBVH bvh2(leaf_count);
            bvh2.build_from_vertices(vertices);
            
            start = std::chrono::high_resolution_clock::now();
            bool collision = bvh.check_collision(bvh2, 0.5f);
            end = std::chrono::high_resolution_clock::now();
            
            double collision_time = std::chrono::duration<double, std::milli>(end - start).count();
            leaf_results["collision_time_ms"] = collision_time;
            leaf_results["collision_detected"] = collision;
            leaf_results["build_stats"] = bvh.get_build_stats();
        }
        
        results["leaf_count_results"][py::str(leaf_count)] = leaf_results;
    }
    
    return results;
}

// Module definition
PYBIND11_MODULE(cpp_stl_utils_extended, m) {
    m.doc() = "STL Analyzer C++ Extensions - Advanced BVH for Pack3D Integration";
    
    // Main BVH class
    py::class_<PyAdaptiveBVH>(m, "AdaptiveBVH")
        .def(py::init<int>(), "leaf_count"_a = 8)
        .def("build_from_vertices", &PyAdaptiveBVH::build_from_vertices,
             "Build BVH from vertex array (x,y,z,x,y,z,...)", "vertices"_a)
        .def("check_collision", &PyAdaptiveBVH::check_collision,
             "Check collision with another BVH", "other"_a, "tolerance"_a = 0.5f)
        .def("get_leaf_count", &PyAdaptiveBVH::get_leaf_count)
        .def("get_active_leaf_count", &PyAdaptiveBVH::get_active_leaf_count)
        .def("get_leaf_bounds", &PyAdaptiveBVH::get_leaf_bounds,
             "Get min/max coordinates for a leaf", "leaf_index"_a)
        .def("get_all_leaf_bounds", &PyAdaptiveBVH::get_all_leaf_bounds)
        .def("get_build_stats", &PyAdaptiveBVH::get_build_stats);
    
    // Convenience functions
    m.def("create_adaptive_bvh", &create_adaptive_bvh_from_mesh,
          "Create BVH from vertex array", "vertices"_a, "leaf_count"_a = 8);
    
    m.def("fast_collision_check", &fast_collision_check,
          "Fast collision check between two vertex arrays",
          "vertices1"_a, "vertices2"_a, "leaf_count"_a = 8, "tolerance"_a = 0.5f);
    
    m.def("benchmark_bvh", &benchmark_bvh_performance,
          "Benchmark BVH performance across leaf counts",
          "vertices"_a, "leaf_counts"_a = py::list());
    
    // Module constants
    m.attr("MAX_LEAF_COUNT") = 32;
    m.attr("DEFAULT_TOLERANCE") = 0.5f;
    m.attr("VERSION") = "2.0.0-pack3d";
}