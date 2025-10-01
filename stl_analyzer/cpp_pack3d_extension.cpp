#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_pack3d_core.hpp"

namespace py = pybind11;
using namespace Pack3D;

PYBIND11_MODULE(cpp_pack3d, m) {
    m.doc() = "Pack3D Simulated Annealing for 3D Packing Optimization";
    
    // Vector3 class
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z)
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(" + std::to_string(v.x) + ", " + 
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    // Rotation3 class
    py::class_<Rotation3>(m, "Rotation3")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("roll", &Rotation3::roll)
        .def_readwrite("pitch", &Rotation3::pitch)
        .def_readwrite("yaw", &Rotation3::yaw);
    
    // Transform3D class
    py::class_<Transform3D>(m, "Transform3D")
        .def(py::init<>())
        .def(py::init<const Vector3&, const Rotation3&>())
        .def_readwrite("position", &Transform3D::position)
        .def_readwrite("rotation", &Transform3D::rotation);
    
    // BoundingBox class
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init<const Vector3&, const Vector3&>())
        .def_readwrite("min", &BoundingBox::min)
        .def_readwrite("max", &BoundingBox::max)
        .def("size", &BoundingBox::size)
        .def("volume", &BoundingBox::volume)
        .def("center", &BoundingBox::center)
        .def("intersects", py::overload_cast<const BoundingBox&>(&BoundingBox::intersects, py::const_))
        .def("intersects", py::overload_cast<const BVHLeaf&>(&BoundingBox::intersects, py::const_));
    
    // BVHLeaf class
    py::class_<BVHLeaf>(m, "BVHLeaf")
        .def(py::init<>())
        .def(py::init<const Vector3&, const Vector3&>())
        .def_readwrite("min", &BVHLeaf::min)
        .def_readwrite("max", &BVHLeaf::max)
        .def("intersects", py::overload_cast<const BoundingBox&>(&BVHLeaf::intersects, py::const_))
        .def("intersects", py::overload_cast<const BVHLeaf&>(&BVHLeaf::intersects, py::const_));
    
    // Part class
    py::class_<Part>(m, "Part")
        .def(py::init<int, const std::string&, const BoundingBox&, float>())
        .def_readwrite("id", &Part::id)
        .def_readwrite("name", &Part::name)
        .def_readwrite("transform", &Part::transform)
        .def_readwrite("original_bbox", &Part::original_bbox)
        .def_readwrite("volume", &Part::volume)
        .def_readwrite("is_placed", &Part::is_placed)
        .def("set_bvh_leaves", &Part::set_bvh_leaves)
        .def("get_current_bbox", &Part::get_current_bbox)
        .def("get_transformed_bvh_leaves", &Part::get_transformed_bvh_leaves)
        .def("collides_with", &Part::collides_with)
        .def("set_transform", &Part::set_transform);
    
    // BuildPlate class
    py::class_<BuildPlate>(m, "BuildPlate")
        .def(py::init<const Vector3&, float>(), py::arg("dimensions"), py::arg("layer_height") = 0.1f)
        .def_readwrite("dimensions", &BuildPlate::dimensions)
        .def_readwrite("z_layer_height", &BuildPlate::z_layer_height)
        .def("fits", &BuildPlate::fits)
        .def("get_utilization", &BuildPlate::get_utilization)
        .def("get_build_height", &BuildPlate::get_build_height);
    
    // Pack3DOptimizer::Results struct
    py::class_<Pack3DOptimizer::Results>(m, "OptimizationResults")
        .def_readwrite("success", &Pack3DOptimizer::Results::success)
        .def_readwrite("final_energy", &Pack3DOptimizer::Results::final_energy)
        .def_readwrite("build_height", &Pack3DOptimizer::Results::build_height)
        .def_readwrite("plate_utilization", &Pack3DOptimizer::Results::plate_utilization)
        .def_readwrite("total_iterations", &Pack3DOptimizer::Results::total_iterations)
        .def_readwrite("accepted_moves", &Pack3DOptimizer::Results::accepted_moves)
        .def_readwrite("optimization_time_ms", &Pack3DOptimizer::Results::optimization_time_ms)
        .def_readwrite("part_transforms", &Pack3DOptimizer::Results::part_transforms);
    
    // Pack3DOptimizer class
    py::class_<Pack3DOptimizer>(m, "Pack3DOptimizer")
        .def(py::init<const Vector3&, float>(), py::arg("plate_dimensions"), py::arg("layer_height") = 0.1f)
        .def("add_part", &Pack3DOptimizer::add_part)
        .def("set_parameters", &Pack3DOptimizer::set_parameters)
        .def("optimize", &Pack3DOptimizer::optimize)
        .def("get_results", &Pack3DOptimizer::get_results)
        .def("get_transforms", &Pack3DOptimizer::get_transforms)
        .def("has_collisions", &Pack3DOptimizer::has_collisions)
        .def("calculate_energy", &Pack3DOptimizer::calculate_energy);
    
    // Utility functions
    m.def("create_part_from_mesh_bounds", [](int id, const std::string& name, 
                                            py::array_t<float> bounds_array, float volume) {
        auto bounds = bounds_array.unchecked<2>();
        if (bounds.shape(0) != 2 || bounds.shape(1) != 3) {
            throw std::runtime_error("Bounds array must be 2x3 (min/max, x/y/z)");
        }
        
        Vector3 min_pt(bounds(0, 0), bounds(0, 1), bounds(0, 2));
        Vector3 max_pt(bounds(1, 0), bounds(1, 1), bounds(1, 2));
        BoundingBox bbox(min_pt, max_pt);
        
        return Part(id, name, bbox, volume);
    }, "Create a Part from mesh bounds array");
    
    // Set BVH leaves for a part from Python
    m.def("set_part_bvh_leaves", [](Part& part, const std::vector<std::vector<float>>& bvh_bounds) {
        std::vector<BVHLeaf> leaves;
        leaves.reserve(bvh_bounds.size());
        
        for (const auto& bounds : bvh_bounds) {
            if (bounds.size() == 6) {
                Vector3 min_pt(bounds[0], bounds[1], bounds[2]);
                Vector3 max_pt(bounds[3], bounds[4], bounds[5]);
                leaves.emplace_back(min_pt, max_pt);
            }
        }
        
        part.set_bvh_leaves(leaves);
    }, "Set BVH leaves for a part from Python list of bounds");
    
    // Printer presets
    m.def("get_eos_p396_plate", []() {
        return Vector3(340, 340, 600);  // EOS P396 dimensions
    });
    
    m.def("get_eos_p110_plate", []() {
        return Vector3(200, 250, 330);  // EOS P110 FORMIGA dimensions  
    });
}




