/**
 * Michael Fogleman's Pack3D BVH Algorithm - C++ Implementation
 * 
 * Triangle-centroid based subdivision for fast collision detection
 * No gap filling, no forced connections, only geometry-aware boxes
 */

#include <Python.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

struct BoundingBox {
    float min[3];
    float max[3];
    
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
    
    float volume() const {
        float dx = std::max(0.0f, max[0] - min[0]);
        float dy = std::max(0.0f, max[1] - min[1]);
        float dz = std::max(0.0f, max[2] - min[2]);
        return dx * dy * dz;
    }
};

class FogleBVH {
public:
    std::vector<BoundingBox> leaves;
    double build_time_ms;
    
    // Store mesh data
    float* vertices;
    int vertex_count;
    float mesh_min[3];
    float mesh_max[3];
    
    FogleBVH() : build_time_ms(0.0), vertices(nullptr), vertex_count(0) {}
    
    bool build_from_vertices(float* input_vertices, int input_vertex_count) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!input_vertices || input_vertex_count <= 0) return false;
        
        this->vertices = input_vertices;
        this->vertex_count = input_vertex_count;
        
        // FOGLEMAN'S APPROACH: Step 1 - Compute global AABB
        mesh_min[0] = input_vertices[0]; mesh_min[1] = input_vertices[1]; mesh_min[2] = input_vertices[2];
        mesh_max[0] = input_vertices[0]; mesh_max[1] = input_vertices[1]; mesh_max[2] = input_vertices[2];
        
        for (int i = 0; i < this->vertex_count; i++) {
            for (int axis = 0; axis < 3; axis++) {
                mesh_min[axis] = std::min(mesh_min[axis], this->vertices[i*3 + axis]);
                mesh_max[axis] = std::max(mesh_max[axis], this->vertices[i*3 + axis]);
            }
        }
        
        int total_triangles = this->vertex_count / 3;
        printf("🎯 Fogleman BVH: %d vertices, %d triangles\n", this->vertex_count, total_triangles);
        printf("   Mesh bounds: [%.1f,%.1f,%.1f] to [%.1f,%.1f,%.1f]\n",
               mesh_min[0], mesh_min[1], mesh_min[2], 
               mesh_max[0], mesh_max[1], mesh_max[2]);
        
        // Step 2: Define 8 spatial regions (2x2x2 subdivision)
        float mesh_center[3] = {
            (mesh_min[0] + mesh_max[0]) * 0.5f,
            (mesh_min[1] + mesh_max[1]) * 0.5f,
            (mesh_min[2] + mesh_max[2]) * 0.5f
        };
        
        leaves.clear();
        leaves.reserve(8);  // Max 8, but could be fewer
        
        // Step 3: For each spatial region, collect triangles by centroid
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                for (int z = 0; z < 2; z++) {
                    
                    // Define region boundaries
                    float region_min[3], region_max[3];
                    region_min[0] = (x == 0) ? mesh_min[0] : mesh_center[0];
                    region_max[0] = (x == 0) ? mesh_center[0] : mesh_max[0];
                    region_min[1] = (y == 0) ? mesh_min[1] : mesh_center[1];
                    region_max[1] = (y == 0) ? mesh_center[1] : mesh_max[1];
                    region_min[2] = (z == 0) ? mesh_min[2] : mesh_center[2];
                    region_max[2] = (z == 0) ? mesh_center[2] : mesh_max[2];
                    
                    // Step 4: Collect triangles by centroid
                    std::vector<int> triangle_vertices;
                    int triangles_in_region = 0;
                    
                    // Process vertices in groups of 3 (triangular mesh)
                    // STL format: every 3 consecutive vertices form a triangle
                    for (int vert_idx = 0; vert_idx + 2 < this->vertex_count; vert_idx += 3) {
                        // Get the 3 vertices of this triangle
                        float v1[3] = {this->vertices[vert_idx*3 + 0], this->vertices[vert_idx*3 + 1], this->vertices[vert_idx*3 + 2]};
                        float v2[3] = {this->vertices[(vert_idx+1)*3 + 0], this->vertices[(vert_idx+1)*3 + 1], this->vertices[(vert_idx+1)*3 + 2]};
                        float v3[3] = {this->vertices[(vert_idx+2)*3 + 0], this->vertices[(vert_idx+2)*3 + 1], this->vertices[(vert_idx+2)*3 + 2]};
                        
                        // Calculate triangle centroid
                        float centroid[3] = {
                            (v1[0] + v2[0] + v3[0]) / 3.0f,
                            (v1[1] + v2[1] + v3[1]) / 3.0f,
                            (v1[2] + v2[2] + v3[2]) / 3.0f
                        };
                        
                        // Check if centroid falls in this region (inclusive boundaries with epsilon)
                        float epsilon = 1e-6f;  // Small epsilon for floating point precision
                        if (centroid[0] >= (region_min[0] - epsilon) && centroid[0] <= (region_max[0] + epsilon) &&
                            centroid[1] >= (region_min[1] - epsilon) && centroid[1] <= (region_max[1] + epsilon) &&
                            centroid[2] >= (region_min[2] - epsilon) && centroid[2] <= (region_max[2] + epsilon)) {
                            
                            // Add all 3 vertices of this triangle
                            triangle_vertices.push_back(vert_idx);
                            triangle_vertices.push_back(vert_idx + 1);
                            triangle_vertices.push_back(vert_idx + 2);
                            triangles_in_region++;
                        }
                    }
                    
                    // Step 5: If region has triangles, create PURE FOGLEMAN AABB
                    if (!triangle_vertices.empty()) {
                        printf("   Region [%d,%d,%d]: %d triangles, %zu vertices\n", 
                               x, y, z, triangles_in_region, triangle_vertices.size());
                        
                        // PURE FOGLEMAN: Tight AABB around only the triangles in this region
                        float tight_min[3] = {
                            this->vertices[triangle_vertices[0]*3 + 0],
                            this->vertices[triangle_vertices[0]*3 + 1],
                            this->vertices[triangle_vertices[0]*3 + 2]
                        };
                        float tight_max[3] = { tight_min[0], tight_min[1], tight_min[2] };
                        
                        // Expand to include ALL vertices of ALL triangles assigned to this region
                        for (int v_idx : triangle_vertices) {
                            for (int axis = 0; axis < 3; axis++) {
                                tight_min[axis] = std::min(tight_min[axis], this->vertices[v_idx*3 + axis]);
                                tight_max[axis] = std::max(tight_max[axis], this->vertices[v_idx*3 + axis]);
                            }
                        }
                        
                        printf("      → Pure Fogleman Box: [%.1f,%.1f,%.1f] to [%.1f,%.1f,%.1f]\n",
                               tight_min[0], tight_min[1], tight_min[2],
                               tight_max[0], tight_max[1], tight_max[2]);
                        
                        // Add small padding for numerical stability (Fogleman's approach)
                        float padding = 0.01f;  // 0.01mm
                        for (int axis = 0; axis < 3; axis++) {
                            tight_min[axis] -= padding;
                            tight_max[axis] += padding;
                        }
                        
                        // Create the bounding box for this region
                        leaves.emplace_back(tight_min, tight_max);
                    }
                    // Key insight: Skip empty regions entirely!
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        build_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        return !leaves.empty();
    }
    
    bool check_collision(const BoundingBox& other) const {
        for (const auto& leaf : leaves) {
            // Check if any leaf intersects with other box
            bool intersects = true;
            for (int axis = 0; axis < 3; axis++) {
                if (leaf.max[axis] < other.min[axis] || leaf.min[axis] > other.max[axis]) {
                    intersects = false;
                    break;
                }
            }
            if (intersects) return true;
        }
        return false;
    }
};

// Global BVH storage
static std::vector<FogleBVH*> g_fogleman_bvhs;

// Python interface functions
static PyObject* create_fogleman_bvh(PyObject* self, PyObject* args) {
    PyObject* vertex_list;
    int leaf_count;
    
    if (!PyArg_ParseTuple(args, "Oi", &vertex_list, &leaf_count)) {
        return NULL;
    }
    
    if (!PyList_Check(vertex_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected list of vertices");
        return NULL;
    }
    
    Py_ssize_t vertex_count = PyList_Size(vertex_list) / 3;
    std::vector<float> vertices(PyList_Size(vertex_list));
    
    for (Py_ssize_t i = 0; i < PyList_Size(vertex_list); i++) {
        PyObject* item = PyList_GetItem(vertex_list, i);
        vertices[i] = (float)PyFloat_AsDouble(item);
    }
    
    FogleBVH* bvh = new FogleBVH();
    if (!bvh->build_from_vertices(vertices.data(), (int)vertex_count)) {
        delete bvh;
        Py_RETURN_NONE;
    }
    
    g_fogleman_bvhs.push_back(bvh);
    return PyLong_FromLong((long)(g_fogleman_bvhs.size() - 1));
}

static PyObject* get_fogleman_bvh_stats(PyObject* self, PyObject* args) {
    int bvh_id;
    if (!PyArg_ParseTuple(args, "i", &bvh_id)) {
        return NULL;
    }
    
    if (bvh_id < 0 || bvh_id >= (int)g_fogleman_bvhs.size() || !g_fogleman_bvhs[bvh_id]) {
        Py_RETURN_NONE;
    }
    
    FogleBVH* bvh = g_fogleman_bvhs[bvh_id];
    
    PyObject* stats = PyDict_New();
    PyDict_SetItemString(stats, "build_time_ms", PyFloat_FromDouble(bvh->build_time_ms));
    PyDict_SetItemString(stats, "leaf_count", PyLong_FromLong((long)bvh->leaves.size()));
    
    return stats;
}

static PyObject* get_fogleman_leaf_bounds(PyObject* self, PyObject* args) {
    int bvh_id, leaf_index;
    if (!PyArg_ParseTuple(args, "ii", &bvh_id, &leaf_index)) {
        return NULL;
    }
    
    if (bvh_id < 0 || bvh_id >= (int)g_fogleman_bvhs.size() || !g_fogleman_bvhs[bvh_id]) {
        Py_RETURN_NONE;
    }
    
    FogleBVH* bvh = g_fogleman_bvhs[bvh_id];
    if (leaf_index < 0 || leaf_index >= (int)bvh->leaves.size()) {
        Py_RETURN_NONE;
    }
    
    const BoundingBox& box = bvh->leaves[leaf_index];
    PyObject* bounds = PyList_New(6);
    PyList_SetItem(bounds, 0, PyFloat_FromDouble(box.min[0]));
    PyList_SetItem(bounds, 1, PyFloat_FromDouble(box.min[1]));
    PyList_SetItem(bounds, 2, PyFloat_FromDouble(box.min[2]));
    PyList_SetItem(bounds, 3, PyFloat_FromDouble(box.max[0]));
    PyList_SetItem(bounds, 4, PyFloat_FromDouble(box.max[1]));
    PyList_SetItem(bounds, 5, PyFloat_FromDouble(box.max[2]));
    
    return bounds;
}

static PyObject* cleanup_bvhs(PyObject* self, PyObject* args) {
    // Clean up all BVHs
    for (auto* bvh : g_fogleman_bvhs) {
        if (bvh) {
            delete bvh;
        }
    }
    g_fogleman_bvhs.clear();
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef FogleMethods[] = {
    {"create_bvh", create_fogleman_bvh, METH_VARARGS, "Create Fogleman BVH"},
    {"get_bvh_stats", get_fogleman_bvh_stats, METH_VARARGS, "Get BVH statistics"},
    {"get_leaf_bounds", get_fogleman_leaf_bounds, METH_VARARGS, "Get leaf bounding box"},
    {"cleanup_bvhs", cleanup_bvhs, METH_VARARGS, "Clean up all BVHs"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef fogle_module = {
    PyModuleDef_HEAD_INIT,
    "cpp_bvh_fogleman",
    "Fogleman's Pack3D BVH implementation",
    -1,
    FogleMethods
};

PyMODINIT_FUNC PyInit_cpp_bvh_fogleman(void) {
    return PyModule_Create(&fogle_module);
}
