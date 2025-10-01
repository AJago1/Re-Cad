// Simple C++ BVH Extension for STL Analyzer
// Minimal approach - just the essential BVH functions needed for pack3d

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

struct BoundingBox {
    float min[3], max[3];
    
    BoundingBox() {
        min[0] = min[1] = min[2] = 0.0f;
        max[0] = max[1] = max[2] = 0.0f;
    }
    
    BoundingBox(float* min_coords, float* max_coords) {
        for (int i = 0; i < 3; i++) {
            min[i] = min_coords[i];
            max[i] = max_coords[i];
        }
    }
    
    bool overlaps(const BoundingBox& other, float tolerance) const {
        for (int i = 0; i < 3; i++) {
            if (max[i] + tolerance < other.min[i] || 
                min[i] - tolerance > other.max[i]) {
                return false;
            }
        }
        return true;
    }
    
    float volume() const {
        float dx = std::max(0.0f, max[0] - min[0]);
        float dy = std::max(0.0f, max[1] - min[1]);
        float dz = std::max(0.0f, max[2] - min[2]);
        return dx * dy * dz;
    }
};

class SimpleBVH {
public:
    std::vector<BoundingBox> leaves;
    int leaf_count;
    double build_time_ms;
    
    // Store mesh data for gap closure
    float* vertices;
    int vertex_count;
    float mesh_min[3];
    float mesh_max[3];
    
    SimpleBVH(int target_leaves = 8) : leaf_count(target_leaves), build_time_ms(0.0), vertices(nullptr), vertex_count(0) {}
    
    bool build_from_vertices(float* input_vertices, int input_vertex_count) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!input_vertices || input_vertex_count <= 0) return false;
        
        // Store mesh data for gap closure
        this->vertices = input_vertices;
        this->vertex_count = input_vertex_count;
        
        // FOGLEMAN'S APPROACH: Triangle-centroid based BVH subdivision  
        // Step 1: Compute global AABB of the mesh
        mesh_min[0] = input_vertices[0]; mesh_min[1] = input_vertices[1]; mesh_min[2] = input_vertices[2];
        mesh_max[0] = input_vertices[0]; mesh_max[1] = input_vertices[1]; mesh_max[2] = input_vertices[2];
        
        for (int i = 0; i < this->vertex_count; i++) {
            for (int axis = 0; axis < 3; axis++) {
                mesh_min[axis] = std::min(mesh_min[axis], this->vertices[i*3 + axis]);
                mesh_max[axis] = std::max(mesh_max[axis], this->vertices[i*3 + axis]);
            }
        }
        
        // Step 2: Define the 8 spatial regions (2x2x2 subdivision)
        float mesh_center[3] = {
            (mesh_min[0] + mesh_max[0]) * 0.5f,
            (mesh_min[1] + mesh_max[1]) * 0.5f,
            (mesh_min[2] + mesh_max[2]) * 0.5f
        };
        
        leaves.clear();
        leaves.reserve(8);  // Max 8 boxes, but could be fewer
        
        // Step 3: For each of the 8 spatial regions, collect triangles by centroid
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                for (int z = 0; z < 2; z++) {
                    
                    // Define this region's boundaries
                    float region_min[3], region_max[3];
                    region_min[0] = (x == 0) ? mesh_min[0] : mesh_center[0];
                    region_max[0] = (x == 0) ? mesh_center[0] : mesh_max[0];
                    region_min[1] = (y == 0) ? mesh_min[1] : mesh_center[1];
                    region_max[1] = (y == 0) ? mesh_center[1] : mesh_max[1];
                    region_min[2] = (z == 0) ? mesh_min[2] : mesh_center[2];
                    region_max[2] = (z == 0) ? mesh_center[2] : mesh_max[2];
                    
                    // Step 4: Collect triangle vertices whose "triangle centroid" falls in this region
                    // Since we only have vertices, we'll treat groups of 3 vertices as triangles
                    std::vector<int> triangles_in_region;
                    
                    // Process vertices in groups of 3 (assuming triangular mesh)
                    for (int tri = 0; tri < this->vertex_count / 3; tri++) {
                        int v1_idx = tri * 3;
                        int v2_idx = tri * 3 + 1;
                        int v3_idx = tri * 3 + 2;
                        
                        if (v3_idx >= this->vertex_count) break;  // Safety check
                        
                        // Calculate triangle centroid
                        float centroid[3] = {
                            (this->vertices[v1_idx*3 + 0] + this->vertices[v2_idx*3 + 0] + this->vertices[v3_idx*3 + 0]) / 3.0f,
                            (this->vertices[v1_idx*3 + 1] + this->vertices[v2_idx*3 + 1] + this->vertices[v3_idx*3 + 1]) / 3.0f,
                            (this->vertices[v1_idx*3 + 2] + this->vertices[v2_idx*3 + 2] + this->vertices[v3_idx*3 + 2]) / 3.0f
                        };
                        
                        // Check if triangle centroid falls in this region
                        if (centroid[0] >= region_min[0] && centroid[0] <= region_max[0] &&
                            centroid[1] >= region_min[1] && centroid[1] <= region_max[1] &&
                            centroid[2] >= region_min[2] && centroid[2] <= region_max[2]) {
                            
                            triangles_in_region.push_back(v1_idx);
                            triangles_in_region.push_back(v2_idx);
                            triangles_in_region.push_back(v3_idx);
                        }
                    }
                    
                    // Step 5: If this region contains triangles, compute tight AABB around them
                    if (!triangles_in_region.empty()) {
                        
                        // Initialize tight bounds with first vertex
                        float tight_min[3] = {
                            this->vertices[triangles_in_region[0]*3 + 0],
                            this->vertices[triangles_in_region[0]*3 + 1], 
                            this->vertices[triangles_in_region[0]*3 + 2]
                        };
                        float tight_max[3] = { tight_min[0], tight_min[1], tight_min[2] };
                        
                        // Expand bounds to include all vertices of triangles in this region
                        for (int v_idx : triangles_in_region) {
                            for (int axis = 0; axis < 3; axis++) {
                                tight_min[axis] = std::min(tight_min[axis], this->vertices[v_idx*3 + axis]);
                                tight_max[axis] = std::max(tight_max[axis], this->vertices[v_idx*3 + axis]);
                            }
                        }
                        
                        // Strategy selection based on density
                        if (density_ratio > 0.3f) {
                            // Dense geometry: Use tight-fitting with safety padding
                            for (int axis = 0; axis < 3; axis++) {
                                float extent = tight_max[axis] - tight_min[axis];
                                float padding = std::max(0.2f, extent * 0.1f);  // 10% padding minimum
                                tight_min[axis] -= padding;
                                tight_max[axis] += padding;
                            }
                        } else {
                            // Sparse geometry: Ensure full vertex coverage with smart expansion
                            for (int axis = 0; axis < 3; axis++) {
                                float extent = tight_max[axis] - tight_min[axis];
                                float octant_extent = oct_max[axis] - oct_min[axis];
                                
                                // Always start with actual vertex bounds (never shrink!)
                                float final_min = tight_min[axis];
                                float final_max = tight_max[axis];
                                
                                // Add minimum padding to ensure no vertices are on exact boundaries
                                float min_padding = 0.1f;
                                final_min -= min_padding;
                                final_max += min_padding;
                                
                                // For very sparse geometry, expand towards octant center for better gap closure
                                float min_coverage = octant_extent * 0.4f;  // Reduced from 60% to 40%
                                if (extent < min_coverage) {
                                    float needed_expansion = min_coverage - extent;
                                    float octant_center = (oct_min[axis] + oct_max[axis]) * 0.5f;
                                    float geometry_center = (tight_min[axis] + tight_max[axis]) * 0.5f;
                                    
                                    // Expand more towards octant center
                                    if (geometry_center < octant_center) {
                                        final_max += needed_expansion * 0.7f;  // 70% expansion towards center
                                        final_min -= needed_expansion * 0.3f;  // 30% expansion away
                                    } else {
                                        final_min -= needed_expansion * 0.7f;
                                        final_max += needed_expansion * 0.3f;
                                    }
                                }
                                
                                // Critical: Clamp to octant boundaries but ensure vertex coverage
                                tight_min[axis] = std::max(final_min, oct_min[axis]);
                                tight_max[axis] = std::min(final_max, oct_max[axis]);
                                
                                // Safety check: Ensure original vertices are still covered
                                for (int i = 0; i < this->vertex_count; i++) {
                                    float* v = &this->vertices[i*3];
                                    if (v[0] >= oct_min[0] && v[0] <= oct_max[0] &&
                                        v[1] >= oct_min[1] && v[1] <= oct_max[1] &&
                                        v[2] >= oct_min[2] && v[2] <= oct_max[2]) {
                                        // This vertex is in this octant - ensure it's covered
                                        if (v[axis] < tight_min[axis]) {
                                            tight_min[axis] = v[axis] - 0.05f;  // Small margin
                                        }
                                        if (v[axis] > tight_max[axis]) {
                                            tight_max[axis] = v[axis] + 0.05f;  // Small margin
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Final gap prevention: Ensure center overlap for adjacent octants
                        float overlap_margin = 0.05f;  // 0.05mm overlap at boundaries
                        
                        // X axis overlap
                        if (x == 0 && tight_max[0] < mesh_center[0]) {
                            tight_max[0] = mesh_center[0] + overlap_margin;
                        }
                        if (x == 1 && tight_min[0] > mesh_center[0]) {
                            tight_min[0] = mesh_center[0] - overlap_margin;
                        }
                        
                        // Y axis overlap
                        if (y == 0 && tight_max[1] < mesh_center[1]) {
                            tight_max[1] = mesh_center[1] + overlap_margin;
                        }
                        if (y == 1 && tight_min[1] > mesh_center[1]) {
                            tight_min[1] = mesh_center[1] - overlap_margin;
                        }
                        
                        // Z axis overlap
                        if (z == 0 && tight_max[2] < mesh_center[2]) {
                            tight_max[2] = mesh_center[2] + overlap_margin;
                        }
                        if (z == 1 && tight_min[2] > mesh_center[2]) {
                            tight_min[2] = mesh_center[2] - overlap_margin;
                        }
                        
                        leaves.emplace_back(tight_min, tight_max);
                    } else {
                        // Empty octant - create minimal box
                        float center[3] = {
                            (oct_min[0] + oct_max[0]) * 0.5f,
                            (oct_min[1] + oct_max[1]) * 0.5f,
                            (oct_min[2] + oct_max[2]) * 0.5f
                        };
                        float padding[3] = {0.1f, 0.1f, 0.1f};
                        float min_coords[3] = {center[0] - padding[0], center[1] - padding[1], center[2] - padding[2]};
                        float max_coords[3] = {center[0] + padding[0], center[1] + padding[1], center[2] + padding[2]};
                        leaves.emplace_back(min_coords, max_coords);
                    }
                }
                if (leaves.size() >= leaf_count) break;
            }
            if (leaves.size() >= leaf_count) break;
        }
        
        // Pad to exact leaf count
        while (leaves.size() < leaf_count) {
            leaves.push_back(BoundingBox());
        }
        
        // Post-processing: Apply gap closure like Python version
        apply_gap_closure();
        
        auto end = std::chrono::high_resolution_clock::now();
        build_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        return true;
    }
    
    bool check_collision(const SimpleBVH& other, float tolerance) const {
        for (const auto& leaf1 : leaves) {
            for (const auto& leaf2 : other.leaves) {
                if (leaf1.overlaps(leaf2, tolerance)) {
                    return true;
                }
            }
        }
        return false;
    }
    
private:
    void apply_gap_closure() {
        // Face-to-face geometry bridging: Only connect adjacent box faces where geometry actually bridges between
        // Strategy: Check if vertices exist in the narrow bridging region between exactly adjacent faces
        
        const float gap_threshold = 0.5f;  // Only consider very small gaps (0.5mm max)
        const float bridge_tolerance = 1.0f;  // How close vertices must be to bridge region
        
        for (int i = 0; i < leaf_count; i++) {
            for (int j = i + 1; j < leaf_count; j++) {
                if (i >= (int)leaves.size() || j >= (int)leaves.size()) continue;
                
                BoundingBox& box1 = leaves[i];
                BoundingBox& box2 = leaves[j];
                
                // Check each axis for face-to-face adjacency
                for (int axis = 0; axis < 3; axis++) {
                    float gap_size = 0.0f;
                    bool adjacent_faces = false;
                    
                    // Check if box1's max face is near box2's min face (box1 "left" of box2)
                    if (box1.max[axis] <= box2.min[axis]) {
                        gap_size = box2.min[axis] - box1.max[axis];
                        adjacent_faces = true;
                    }
                    // Check if box2's max face is near box1's min face (box2 "left" of box1)  
                    else if (box2.max[axis] <= box1.min[axis]) {
                        gap_size = box1.min[axis] - box2.max[axis];
                        adjacent_faces = true;
                    }
                    
                    // Only process small gaps between adjacent faces
                    if (!adjacent_faces || gap_size <= 0.001f || gap_size > gap_threshold) {
                        continue;
                    }
                    
                    // Define the bridge region (narrow strip between the faces)
                    float bridge_start = (box1.max[axis] <= box2.min[axis]) ? box1.max[axis] : box2.max[axis];
                    float bridge_end = (box1.max[axis] <= box2.min[axis]) ? box2.min[axis] : box1.min[axis];
                    float bridge_center = (bridge_start + bridge_end) * 0.5f;
                    
                    // Calculate overlapping region in other two dimensions (where faces could actually touch)
                    float overlap_min[3], overlap_max[3];
                    bool has_face_overlap = true;
                    
                    for (int other_axis = 0; other_axis < 3; other_axis++) {
                        if (other_axis == axis) {
                            overlap_min[other_axis] = bridge_start;
                            overlap_max[other_axis] = bridge_end;
                        } else {
                            // Find overlapping region in this dimension
                            overlap_min[other_axis] = std::max(box1.min[other_axis], box2.min[other_axis]);
                            overlap_max[other_axis] = std::min(box1.max[other_axis], box2.max[other_axis]);
                            
                            // If no overlap in this dimension, faces don't actually touch
                            if (overlap_min[other_axis] >= overlap_max[other_axis]) {
                                has_face_overlap = false;
                                break;
                            }
                        }
                    }
                    
                    // Only proceed if faces actually overlap (can touch)
                    if (!has_face_overlap) continue;
                    
                    // Check if vertices exist in the bridge region
                    bool geometry_bridges = false;
                    
                    for (int v = 0; v < this->vertex_count; v++) {
                        float* vertex = &this->vertices[v * 3];
                        
                        // Check if vertex is within the bridge region
                        bool in_bridge_region = true;
                        
                        for (int check_axis = 0; check_axis < 3; check_axis++) {
                            float coord = vertex[check_axis];
                            float min_bound = overlap_min[check_axis] - bridge_tolerance;
                            float max_bound = overlap_max[check_axis] + bridge_tolerance;
                            
                            if (coord < min_bound || coord > max_bound) {
                                in_bridge_region = false;
                                break;
                            }
                        }
                        
                        // Specifically check the gap axis - vertex must be near bridge center
                        if (in_bridge_region) {
                            float gap_axis_dist = std::abs(vertex[axis] - bridge_center);
                            if (gap_axis_dist <= bridge_tolerance) {
                                geometry_bridges = true;
                                break;
                            }
                        }
                    }
                    
                    // Only connect faces where geometry actually bridges between them
                    if (geometry_bridges) {
                        float expansion = gap_size * 0.4f;  // Close 40% of the gap conservatively
                        
                        if (box1.max[axis] <= box2.min[axis]) {
                            // Expand box1 toward box2 and box2 toward box1
                            box1.max[axis] += expansion;
                            box2.min[axis] -= expansion;
                        } else if (box2.max[axis] <= box1.min[axis]) {
                            // Expand box2 toward box1 and box1 toward box2
                            box2.max[axis] += expansion;
                            box1.min[axis] -= expansion;
                        }
                    }
                }
            }
        }
        
        // Critical: Clamp ALL boxes to mesh bounds (prevent overshooting)
        for (int i = 0; i < leaf_count; i++) {
            if (i >= (int)leaves.size()) continue;
            
            BoundingBox& box = leaves[i];
            for (int axis = 0; axis < 3; axis++) {
                // Strict clamping to mesh bounds (no overshooting allowed)
                box.min[axis] = std::max(box.min[axis], this->mesh_min[axis]);
                box.max[axis] = std::min(box.max[axis], this->mesh_max[axis]);
            }
        }
    }
};

// Global BVH storage (simple approach)
static std::vector<SimpleBVH*> g_bvhs;

// Python C API functions
static PyObject* create_bvh(PyObject* self, PyObject* args) {
    PyObject* vertices_obj;
    int leaf_count = 8;
    
    if (!PyArg_ParseTuple(args, "O|i", &vertices_obj, &leaf_count)) {
        return NULL;
    }
    
    // Convert Python array to C array
    if (!PyList_Check(vertices_obj)) {
        PyErr_SetString(PyExc_TypeError, "vertices must be a list");
        return NULL;
    }
    
    int vertex_count = PyList_Size(vertices_obj);
    if (vertex_count % 3 != 0) {
        PyErr_SetString(PyExc_ValueError, "vertices must have length divisible by 3");
        return NULL;
    }
    vertex_count /= 3;
    
    std::vector<float> vertices(PyList_Size(vertices_obj));
    for (int i = 0; i < PyList_Size(vertices_obj); i++) {
        PyObject* item = PyList_GetItem(vertices_obj, i);
        vertices[i] = (float)PyFloat_AsDouble(item);
    }
    
    // Create BVH
    SimpleBVH* bvh = new SimpleBVH(leaf_count);
    bool success = bvh->build_from_vertices(vertices.data(), vertex_count);
    
    if (success) {
        g_bvhs.push_back(bvh);
        int bvh_id = g_bvhs.size() - 1;
        return PyLong_FromLong(bvh_id);
    } else {
        delete bvh;
        Py_RETURN_NONE;
    }
}

static PyObject* check_collision(PyObject* self, PyObject* args) {
    int bvh_id1, bvh_id2;
    float tolerance = 0.5f;
    
    if (!PyArg_ParseTuple(args, "ii|f", &bvh_id1, &bvh_id2, &tolerance)) {
        return NULL;
    }
    
    if (bvh_id1 >= 0 && bvh_id1 < g_bvhs.size() && 
        bvh_id2 >= 0 && bvh_id2 < g_bvhs.size() &&
        g_bvhs[bvh_id1] && g_bvhs[bvh_id2]) {
        
        bool collision = g_bvhs[bvh_id1]->check_collision(*g_bvhs[bvh_id2], tolerance);
        return PyBool_FromLong(collision);
    }
    
    Py_RETURN_FALSE;
}

static PyObject* get_bvh_stats(PyObject* self, PyObject* args) {
    int bvh_id;
    
    if (!PyArg_ParseTuple(args, "i", &bvh_id)) {
        return NULL;
    }
    
    if (bvh_id >= 0 && bvh_id < g_bvhs.size() && g_bvhs[bvh_id]) {
        SimpleBVH* bvh = g_bvhs[bvh_id];
        
        PyObject* dict = PyDict_New();
        PyDict_SetItemString(dict, "leaf_count", PyLong_FromLong(bvh->leaf_count));
        PyDict_SetItemString(dict, "build_time_ms", PyFloat_FromDouble(bvh->build_time_ms));
        
        return dict;
    }
    
    Py_RETURN_NONE;
}

static PyObject* get_leaf_bounds(PyObject* self, PyObject* args) {
    int bvh_id, leaf_index;
    
    if (!PyArg_ParseTuple(args, "ii", &bvh_id, &leaf_index)) {
        return NULL;
    }
    
    if (bvh_id >= 0 && bvh_id < g_bvhs.size() && g_bvhs[bvh_id]) {
        SimpleBVH* bvh = g_bvhs[bvh_id];
        
        if (leaf_index >= 0 && leaf_index < bvh->leaf_count) {
            const BoundingBox& box = bvh->leaves[leaf_index];
            
            // Return as list: [min_x, min_y, min_z, max_x, max_y, max_z]
            PyObject* bounds = PyList_New(6);
            PyList_SetItem(bounds, 0, PyFloat_FromDouble(box.min[0]));
            PyList_SetItem(bounds, 1, PyFloat_FromDouble(box.min[1]));
            PyList_SetItem(bounds, 2, PyFloat_FromDouble(box.min[2]));
            PyList_SetItem(bounds, 3, PyFloat_FromDouble(box.max[0]));
            PyList_SetItem(bounds, 4, PyFloat_FromDouble(box.max[1]));
            PyList_SetItem(bounds, 5, PyFloat_FromDouble(box.max[2]));
            
            return bounds;
        }
    }
    
    Py_RETURN_NONE;
}

static PyObject* cleanup_bvhs(PyObject* self, PyObject* args) {
    for (auto* bvh : g_bvhs) {
        delete bvh;
    }
    g_bvhs.clear();
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef BVHMethods[] = {
    {"create_bvh", create_bvh, METH_VARARGS, "Create BVH from vertices"},
    {"check_collision", check_collision, METH_VARARGS, "Check collision between two BVHs"},
    {"get_bvh_stats", get_bvh_stats, METH_VARARGS, "Get BVH statistics"},
    {"get_leaf_bounds", get_leaf_bounds, METH_VARARGS, "Get leaf bounding box"},
    {"cleanup_bvhs", cleanup_bvhs, METH_VARARGS, "Clean up all BVHs"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef bvhmodule = {
    PyModuleDef_HEAD_INIT,
    "cpp_bvh_simple",
    "Simple C++ BVH for Pack3D",
    -1,
    BVHMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_cpp_bvh_simple(void) {
    return PyModule_Create(&bvhmodule);
}