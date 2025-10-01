#pragma once

#include <vector>
#include <memory>
#include <random>
#include <chrono>

namespace Pack3D {

// Forward declarations
struct BoundingBox;

/**
 * 3D Vector/Point structure
 */
struct Vector3 {
    float x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
};

/**
 * Simple BVH leaf for collision detection (defined early to avoid forward declaration issues)
 */
struct BVHLeaf {
    Vector3 min, max;
    
    BVHLeaf() {}
    BVHLeaf(const Vector3& min_pt, const Vector3& max_pt) : min(min_pt), max(max_pt) {}
    
    // Check if this leaf intersects with another leaf
    bool intersects(const BVHLeaf& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
    
    // Check if this leaf intersects with a bounding box (defined after BoundingBox)
    bool intersects(const BoundingBox& bbox) const;
};

/**
 * 3D Rotation structure (Euler angles)
 */
struct Rotation3 {
    float roll, pitch, yaw;  // In radians
    
    Rotation3() : roll(0), pitch(0), yaw(0) {}
    Rotation3(float r, float p, float y) : roll(r), pitch(p), yaw(y) {}
};

/**
 * 3D Transformation (position + rotation)
 */
struct Transform3D {
    Vector3 position;
    Rotation3 rotation;
    
    Transform3D() {}
    Transform3D(const Vector3& pos, const Rotation3& rot) 
        : position(pos), rotation(rot) {}
};

/**
 * Bounding Box structure
 */
struct BoundingBox {
    Vector3 min, max;
    
    BoundingBox() {}
    BoundingBox(const Vector3& min_, const Vector3& max_) : min(min_), max(max_) {}
    
    Vector3 size() const {
        return max - min;
    }
    
    float volume() const {
        Vector3 s = size();
        return s.x * s.y * s.z;
    }
    
    Vector3 center() const {
        return (min + max) * 0.5f;
    }
    
    // Check if this box intersects with another
    bool intersects(const BoundingBox& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
    
    // Check if this box intersects with a BVH leaf
    bool intersects(const BVHLeaf& leaf) const {
        return (min.x <= leaf.max.x && max.x >= leaf.min.x) &&
               (min.y <= leaf.max.y && max.y >= leaf.min.y) &&
               (min.z <= leaf.max.z && max.z >= leaf.min.z);
    }
};

// Add BVHLeaf method for BoundingBox intersection (now that BoundingBox is defined)
inline bool BVHLeaf::intersects(const BoundingBox& bbox) const {
    return (min.x <= bbox.max.x && max.x >= bbox.min.x) &&
           (min.y <= bbox.max.y && max.y >= bbox.min.y) &&
           (min.z <= bbox.max.z && max.z >= bbox.min.z);
}

/**
 * A part to be packed (STL mesh with 8-leaf BVH)
 */
class Part {
public:
    int id;
    std::string name;
    Transform3D transform;
    BoundingBox original_bbox;  // Original bounding box
    float volume;
    bool is_placed;
    
    // 8-leaf BVH for collision detection (Fogleman's approach)
    std::vector<BVHLeaf> original_bvh_leaves;  // Original BVH leaves (untransformed)
    
    Part(int id_, const std::string& name_, const BoundingBox& bbox, float vol);
    ~Part();
    
    // Set BVH leaves from external source (Python)
    void set_bvh_leaves(const std::vector<BVHLeaf>& leaves);
    
    // Get current bounding box after transformation
    BoundingBox get_current_bbox() const;
    
    // Get transformed BVH leaves after current transformation
    std::vector<BVHLeaf> get_transformed_bvh_leaves() const;
    
    // Check collision with another part using BVH
    bool collides_with(const Part& other) const;
    
    // Apply a random perturbation for simulated annealing
    void apply_random_perturbation(std::mt19937& rng, float max_translation, float max_rotation);
    
    // Set specific transform
    void set_transform(const Transform3D& new_transform);
};

/**
 * Build plate constraints for SLS printers
 */
class BuildPlate {
public:
    Vector3 dimensions;  // EOS P396: 340x340x600mm, P110: 200x250x330mm
    float z_layer_height;  // Minimum layer spacing
    
    BuildPlate(const Vector3& dims, float layer_height = 0.1f);
    
    // Check if a part fits within build plate
    bool fits(const Part& part) const;
    
    // Get build plate utilization (0.0 to 1.0)
    float get_utilization(const std::vector<Part>& parts) const;
    
    // Get total height of all parts
    float get_build_height(const std::vector<Part>& parts) const;
};

/**
 * Pack3D Simulated Annealing Algorithm
 */
class Pack3DOptimizer {
private:
    BuildPlate build_plate;
    std::vector<Part> parts;
    std::mt19937 rng;
    
    // Annealing parameters
    float initial_temperature;
    float cooling_rate;
    float min_temperature;
    int max_iterations;
    int plateau_threshold;
    
    // Current state
    float current_energy;
    float best_energy;
    std::vector<Transform3D> best_transforms;
    
    // Statistics
    int total_iterations;
    int accepted_moves;
    int collision_checks;
    
public:
    Pack3DOptimizer(const Vector3& plate_dims, float layer_height = 0.1f);
    
    // Add a part to be packed
    void add_part(const Part& part);
    
    // Set annealing parameters
    void set_parameters(float init_temp, float cool_rate, float min_temp, int max_iter);
    
    // Run the simulated annealing optimization
    bool optimize();
    
    // Calculate energy (fitness) of current configuration
    float calculate_energy() const;
    
    // Check if current configuration has any collisions
    bool has_collisions() const;
    
    // Generate initial random placement
    void generate_initial_placement();
    
    // Accept or reject a move based on temperature
    bool accept_move(float energy_delta, float temperature);
    
    // Get optimization results
    struct Results {
        bool success;
        float final_energy;
        float build_height;
        float plate_utilization;
        int total_iterations;
        int accepted_moves;
        float optimization_time_ms;
        std::vector<Transform3D> part_transforms;
    };
    
    Results get_results() const;
    
    // Get current part transforms
    const std::vector<Transform3D>& get_transforms() const;
};

} // namespace Pack3D
