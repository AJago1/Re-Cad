#include "cpp_pack3d_core.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

// Define M_PI for Windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Pack3D {

// TODO: BVH integration will be added later
// For now using simple bounding box collision detection

/**
 * Part Implementation
 */
Part::Part(int id_, const std::string& name_, const BoundingBox& bbox, float vol)
    : id(id_), name(name_), original_bbox(bbox), volume(vol), is_placed(false)
{
    transform = Transform3D();  // Identity transform
    original_bvh_leaves.clear();  // Will be set later via set_bvh_leaves()
}

Part::~Part() {
    // Destructor
    original_bvh_leaves.clear();
}

BoundingBox Part::get_current_bbox() const {
    // Apply rotation to original bounding box corners, then find new AABB
    Vector3 corners[8];
    Vector3 min = original_bbox.min;
    Vector3 max = original_bbox.max;
    
    // 8 corners of original bounding box
    corners[0] = Vector3(min.x, min.y, min.z);
    corners[1] = Vector3(max.x, min.y, min.z);
    corners[2] = Vector3(min.x, max.y, min.z);
    corners[3] = Vector3(max.x, max.y, min.z);
    corners[4] = Vector3(min.x, min.y, max.z);
    corners[5] = Vector3(max.x, min.y, max.z);
    corners[6] = Vector3(min.x, max.y, max.z);
    corners[7] = Vector3(max.x, max.y, max.z);
    
    // Apply rotation to each corner (simple Euler rotation)
    Vector3 rotated_corners[8];
    float cos_roll = std::cos(transform.rotation.roll);
    float sin_roll = std::sin(transform.rotation.roll);
    float cos_pitch = std::cos(transform.rotation.pitch);
    float sin_pitch = std::sin(transform.rotation.pitch);
    float cos_yaw = std::cos(transform.rotation.yaw);
    float sin_yaw = std::sin(transform.rotation.yaw);
    
    for (int i = 0; i < 8; i++) {
        Vector3 p = corners[i];
        
        // Rotate around X (roll)
        float y1 = p.y * cos_roll - p.z * sin_roll;
        float z1 = p.y * sin_roll + p.z * cos_roll;
        p.y = y1;
        p.z = z1;
        
        // Rotate around Y (pitch)
        float x2 = p.x * cos_pitch + p.z * sin_pitch;
        float z2 = -p.x * sin_pitch + p.z * cos_pitch;
        p.x = x2;
        p.z = z2;
        
        // Rotate around Z (yaw)
        float x3 = p.x * cos_yaw - p.y * sin_yaw;
        float y3 = p.x * sin_yaw + p.y * cos_yaw;
        p.x = x3;
        p.y = y3;
        
        // Apply translation
        rotated_corners[i] = Vector3(
            p.x + transform.position.x,
            p.y + transform.position.y,
            p.z + transform.position.z
        );
    }
    
    // Find new axis-aligned bounding box from rotated corners
    Vector3 new_min = rotated_corners[0];
    Vector3 new_max = rotated_corners[0];
    
    for (int i = 1; i < 8; i++) {
        new_min.x = std::min(new_min.x, rotated_corners[i].x);
        new_min.y = std::min(new_min.y, rotated_corners[i].y);
        new_min.z = std::min(new_min.z, rotated_corners[i].z);
        new_max.x = std::max(new_max.x, rotated_corners[i].x);
        new_max.y = std::max(new_max.y, rotated_corners[i].y);
        new_max.z = std::max(new_max.z, rotated_corners[i].z);
    }
    
    return BoundingBox(new_min, new_max);
}

void Part::set_bvh_leaves(const std::vector<BVHLeaf>& leaves) {
    original_bvh_leaves = leaves;
    std::cout << "   📦 Set " << leaves.size() << " BVH leaves for part " << name << std::endl;
    
    // CRITICAL DEBUG: Log first few leaf bounds
    if (!leaves.empty()) {
        std::cout << "   📦 First leaf bounds: [" << leaves[0].min.x << "," << leaves[0].min.y << "," << leaves[0].min.z 
                  << "] to [" << leaves[0].max.x << "," << leaves[0].max.y << "," << leaves[0].max.z << "]" << std::endl;
    } else {
        std::cout << "   ⚠️ WARNING: No BVH leaves provided for part " << name << " - collision detection will fail!" << std::endl;
    }
}

std::vector<BVHLeaf> Part::get_transformed_bvh_leaves() const {
    std::vector<BVHLeaf> transformed_leaves;
    transformed_leaves.reserve(original_bvh_leaves.size());
    
    // Apply the same transformation logic as get_current_bbox()
    float cos_roll = std::cos(transform.rotation.roll);
    float sin_roll = std::sin(transform.rotation.roll);
    float cos_pitch = std::cos(transform.rotation.pitch);
    float sin_pitch = std::sin(transform.rotation.pitch);
    float cos_yaw = std::cos(transform.rotation.yaw);
    float sin_yaw = std::sin(transform.rotation.yaw);
    
    for (const auto& leaf : original_bvh_leaves) {
        // Transform all 8 corners of the BVH leaf bounding box
        Vector3 corners[8];
        corners[0] = Vector3(leaf.min.x, leaf.min.y, leaf.min.z);
        corners[1] = Vector3(leaf.max.x, leaf.min.y, leaf.min.z);
        corners[2] = Vector3(leaf.min.x, leaf.max.y, leaf.min.z);
        corners[3] = Vector3(leaf.max.x, leaf.max.y, leaf.min.z);
        corners[4] = Vector3(leaf.min.x, leaf.min.y, leaf.max.z);
        corners[5] = Vector3(leaf.max.x, leaf.min.y, leaf.max.z);
        corners[6] = Vector3(leaf.min.x, leaf.max.y, leaf.max.z);
        corners[7] = Vector3(leaf.max.x, leaf.max.y, leaf.max.z);
        
        Vector3 transformed_corners[8];
        for (int i = 0; i < 8; i++) {
            Vector3 p = corners[i];
            
            // Apply 3D rotation (roll, pitch, yaw)
            float y1 = p.y * cos_roll - p.z * sin_roll;
            float z1 = p.y * sin_roll + p.z * cos_roll;
            p.y = y1; p.z = z1;
            
            float x2 = p.x * cos_pitch + p.z * sin_pitch;
            float z2 = -p.x * sin_pitch + p.z * cos_pitch;
            p.x = x2; p.z = z2;
            
            float x3 = p.x * cos_yaw - p.y * sin_yaw;
            float y3 = p.x * sin_yaw + p.y * cos_yaw;
            p.x = x3; p.y = y3;
            
            // Apply translation
            transformed_corners[i] = Vector3(
                p.x + transform.position.x,
                p.y + transform.position.y,
                p.z + transform.position.z
            );
        }
        
        // Find new axis-aligned bounding box from transformed corners
        Vector3 new_min = transformed_corners[0];
        Vector3 new_max = transformed_corners[0];
        
        for (int i = 1; i < 8; i++) {
            new_min.x = std::min(new_min.x, transformed_corners[i].x);
            new_min.y = std::min(new_min.y, transformed_corners[i].y);
            new_min.z = std::min(new_min.z, transformed_corners[i].z);
            new_max.x = std::max(new_max.x, transformed_corners[i].x);
            new_max.y = std::max(new_max.y, transformed_corners[i].y);
            new_max.z = std::max(new_max.z, transformed_corners[i].z);
        }
        
        transformed_leaves.emplace_back(new_min, new_max);
    }
    
    return transformed_leaves;
}

bool Part::collides_with(const Part& other) const {
    // FOGLEMAN APPROACH: Use 8-leaf BVH for accurate collision detection
    // This is much more precise than bounding box collision
    
    // DEBUG: Log collision check details
    static int collision_check_count = 0;
    collision_check_count++;
    
    // Get transformed BVH leaves for both parts
    std::vector<BVHLeaf> my_leaves = get_transformed_bvh_leaves();
    std::vector<BVHLeaf> other_leaves = other.get_transformed_bvh_leaves();
    
    // CRITICAL DEBUG: Log BVH leaf counts
    if (collision_check_count % 100 == 0) {
        std::cout << "   🔍 BVH Collision Check #" << collision_check_count 
                  << " - " << name << " vs " << other.name 
                  << " (My leaves: " << my_leaves.size() << ", Other leaves: " << other_leaves.size() << ")" << std::endl;
        
        // Log first few leaf bounds for debugging
        if (!my_leaves.empty()) {
            std::cout << "   📦 My first leaf: [" << my_leaves[0].min.x << "," << my_leaves[0].min.y << "," << my_leaves[0].min.z 
                      << "] to [" << my_leaves[0].max.x << "," << my_leaves[0].max.y << "," << my_leaves[0].max.z << "]" << std::endl;
        }
        if (!other_leaves.empty()) {
            std::cout << "   📦 Other first leaf: [" << other_leaves[0].min.x << "," << other_leaves[0].min.y << "," << other_leaves[0].min.z 
                      << "] to [" << other_leaves[0].max.x << "," << other_leaves[0].max.y << "," << other_leaves[0].max.z << "]" << std::endl;
        }
    }
    
    // Check each leaf against each other leaf (8x8 = 64 collision checks)
    for (const auto& my_leaf : my_leaves) {
        for (const auto& other_leaf : other_leaves) {
            if (my_leaf.intersects(other_leaf)) {
                // CRITICAL: Add safety margin check for visual separation
                const float SAFETY_MARGIN = 5.0f;
                
                // Expand both leaves by safety margin
                BVHLeaf my_expanded = my_leaf;
                BVHLeaf other_expanded = other_leaf;
                
                my_expanded.min.x -= SAFETY_MARGIN;
                my_expanded.min.y -= SAFETY_MARGIN;
                my_expanded.min.z -= SAFETY_MARGIN;
                my_expanded.max.x += SAFETY_MARGIN;
                my_expanded.max.y += SAFETY_MARGIN;
                my_expanded.max.z += SAFETY_MARGIN;
                
                other_expanded.min.x -= SAFETY_MARGIN;
                other_expanded.min.y -= SAFETY_MARGIN;
                other_expanded.min.z -= SAFETY_MARGIN;
                other_expanded.max.x += SAFETY_MARGIN;
                other_expanded.max.y += SAFETY_MARGIN;
                other_expanded.max.z += SAFETY_MARGIN;
                
                // If expanded leaves still intersect, we have a collision
                if (my_expanded.intersects(other_expanded)) {
                    if (collision_check_count % 100 == 0) {
                        std::cout << "   ❌ COLLISION DETECTED: " << name << " vs " << other.name 
                                  << " (with " << SAFETY_MARGIN << "mm safety margin)" << std::endl;
                        std::cout << "   📍 My position: [" << transform.position.x << "," << transform.position.y << "," << transform.position.z << "]" << std::endl;
                        std::cout << "   📍 Other position: [" << other.transform.position.x << "," << other.transform.position.y << "," << other.transform.position.z << "]" << std::endl;
                    }
                    return true;
                }
            }
        }
    }
    
    return false;
}

void Part::apply_random_perturbation(std::mt19937& rng, float max_translation, float max_rotation) {
    std::uniform_real_distribution<float> trans_dist(-max_translation, max_translation);
    std::uniform_int_distribution<int> move_type_dist(0, 1); // 0 = translate, 1 = rotate
    
    if (move_type_dist(rng) == 0) {
        // Apply random translation
        transform.position.x += trans_dist(rng);
        transform.position.y += trans_dist(rng);
        transform.position.z += trans_dist(rng);
    } else {
        // Apply discrete 90-degree rotation - Fogleman's 24 orientations
        // 6 faces (±X, ±Y, ±Z up) × 4 rotations per face = 24 total orientations
        std::uniform_int_distribution<int> orientation_dist(0, 23);
        int new_orientation = orientation_dist(rng);
        
        // Convert orientation ID to discrete rotations (in multiples of π/2)
        // Face: 0=+Z up, 1=-Z up, 2=+Y up, 3=-Y up, 4=+X up, 5=-X up
        // Rotation: 0°, 90°, 180°, 270° around the up axis
        int face = new_orientation / 4;
        int rotation = new_orientation % 4;
        
        float pi_2 = static_cast<float>(M_PI) / 2.0f;
        
        switch (face) {
            case 0: // +Z up (normal orientation)
                transform.rotation.roll = 0;
                transform.rotation.pitch = 0;
                transform.rotation.yaw = rotation * pi_2;
                break;
            case 1: // -Z up (flipped)
                transform.rotation.roll = static_cast<float>(M_PI);
                transform.rotation.pitch = 0;
                transform.rotation.yaw = rotation * pi_2;
                break;
            case 2: // +Y up (front face up)
                transform.rotation.roll = 0;
                transform.rotation.pitch = -pi_2;
                transform.rotation.yaw = rotation * pi_2;
                break;
            case 3: // -Y up (back face up)
                transform.rotation.roll = 0;
                transform.rotation.pitch = pi_2;
                transform.rotation.yaw = rotation * pi_2;
                break;
            case 4: // +X up (right side up)
                transform.rotation.roll = -pi_2;
                transform.rotation.pitch = 0;
                transform.rotation.yaw = rotation * pi_2;
                break;
            case 5: // -X up (left side up)
                transform.rotation.roll = pi_2;
                transform.rotation.pitch = 0;
                transform.rotation.yaw = rotation * pi_2;
                break;
        }
    }
}

void Part::set_transform(const Transform3D& new_transform) {
    transform = new_transform;
}

/**
 * Build Plate Implementation
 */
BuildPlate::BuildPlate(const Vector3& dims, float layer_height)
    : dimensions(dims), z_layer_height(layer_height)
{
}

bool BuildPlate::fits(const Part& part) const {
    BoundingBox bbox = part.get_current_bbox();
    
    return (bbox.min.x >= 0 && bbox.max.x <= dimensions.x &&
            bbox.min.y >= 0 && bbox.max.y <= dimensions.y &&
            bbox.min.z >= 0 && bbox.max.z <= dimensions.z);
}

float BuildPlate::get_utilization(const std::vector<Part>& parts) const {
    float total_part_volume = 0;
    for (const auto& part : parts) {
        if (part.is_placed) {
            total_part_volume += part.volume;
        }
    }
    
    // Use total build plate volume (not just used height)
    float total_plate_volume = dimensions.x * dimensions.y * dimensions.z;
    return (total_plate_volume > 0) ? (total_part_volume / total_plate_volume) : 0.0f;
}

float BuildPlate::get_build_height(const std::vector<Part>& parts) const {
    float max_z = 0.0f;
    for (const auto& part : parts) {
        if (part.is_placed) {
            BoundingBox bbox = part.get_current_bbox();
            max_z = std::max(max_z, bbox.max.z);
        }
    }
    return max_z;
}

/**
 * Pack3D Optimizer Implementation
 */
Pack3DOptimizer::Pack3DOptimizer(const Vector3& plate_dims, float layer_height)
    : build_plate(plate_dims, layer_height), rng(std::random_device{}())
{
    // TRUE Fogleman parameters: Start with reasonable defaults (will be overridden by GUI)
    initial_temperature = 1000.0f;  // Reasonable default starting temperature
    cooling_rate = 0.99999f;  // CRITICAL FIX: Match GUI precision for proper cooling
    min_temperature = 0.001f;  // Very low minimum to allow fine-tuning without clamping
    max_iterations = 10000;
    plateau_threshold = 1000;
    
    // Initialize state
    current_energy = std::numeric_limits<float>::max();
    best_energy = std::numeric_limits<float>::max();
    total_iterations = 0;
    accepted_moves = 0;
    collision_checks = 0;
}

void Pack3DOptimizer::add_part(const Part& part) {
    parts.push_back(part);
    best_transforms.push_back(Transform3D());
}

void Pack3DOptimizer::set_parameters(float init_temp, float cool_rate, float min_temp, int max_iter) {
    // CRITICAL DEBUG: Log exact values received
    std::cout << "🔍 C++ DEBUG: Received parameters:" << std::endl;
    std::cout << "   init_temp: " << std::fixed << std::setprecision(6) << init_temp << std::endl;
    std::cout << "   cool_rate: " << std::fixed << std::setprecision(6) << cool_rate << std::endl;
    std::cout << "   min_temp: " << std::fixed << std::setprecision(6) << min_temp << std::endl;
    std::cout << "   max_iter: " << max_iter << std::endl;
    
    initial_temperature = init_temp;
    cooling_rate = cool_rate;
    min_temperature = min_temp;
    max_iterations = max_iter;
    
    // CRITICAL DEBUG: Log what was actually set
    std::cout << "🔍 C++ DEBUG: Parameters set to:" << std::endl;
    std::cout << "   initial_temperature: " << std::fixed << std::setprecision(6) << initial_temperature << std::endl;
    std::cout << "   cooling_rate: " << std::fixed << std::setprecision(6) << cooling_rate << std::endl;
    std::cout << "   min_temperature: " << std::fixed << std::setprecision(6) << min_temperature << std::endl;
    std::cout << "   max_iterations: " << max_iterations << std::endl;
}

bool Pack3DOptimizer::optimize() {
    if (parts.empty()) {
        std::cout << "❌ No parts to optimize" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🚀 Starting Pack3D Optimization..." << std::endl;
    std::cout << "   Parts: " << parts.size() << std::endl;
    std::cout << "   Build plate: " << build_plate.dimensions.x << "x" 
              << build_plate.dimensions.y << "x" << build_plate.dimensions.z << "mm" << std::endl;
    
    // Generate initial placement
    generate_initial_placement();
    current_energy = calculate_energy();
    best_energy = current_energy;
    
    // Save initial state as best
    for (size_t i = 0; i < parts.size(); i++) {
        best_transforms[i] = parts[i].transform;
    }
    
    std::cout << "   Initial energy: " << current_energy << std::endl;
    
    // Simulated annealing loop - Fogleman style (runs indefinitely until plateau)
    float temperature = initial_temperature;
    int iterations_without_improvement = 0;
    int iteration = 0;
    
    // Run indefinitely like Fogleman - no max_iterations limit
    while (true) {
        iteration++;
        total_iterations++;
        
        // Choose random part to move
        int part_index = std::uniform_int_distribution<int>(0, static_cast<int>(parts.size()) - 1)(rng);
        Part& selected_part = parts[part_index];
        
        // Save current transform
        Transform3D old_transform = selected_part.transform;
        
        // Apply random perturbation - Fogleman approach: meaningful moves at all temperatures
        float temp_factor = temperature / initial_temperature;  // Ratio from 1.0 to 0.0
        float max_translation = 10.0f + (temp_factor * 40.0f);  // 10-50mm moves (always meaningful)
        float max_rotation = 0.1f + (temp_factor * 0.4f);       // 0.1-0.5 radians (~6-30 degrees)
        selected_part.apply_random_perturbation(rng, max_translation, max_rotation);
        
        // Check if part is still within build plate bounds
        if (!build_plate.fits(selected_part)) {
            // Restore old transform and skip this iteration
            selected_part.set_transform(old_transform);
            continue;
        }
        
        // FOGLEMAN'S APPROACH: Check for collisions and IMMEDIATELY REJECT
        bool has_collision = false;
        for (size_t i = 0; i < parts.size() && !has_collision; i++) {
            if (i == static_cast<size_t>(part_index)) continue; // Skip self
            if (selected_part.collides_with(parts[i])) {
                has_collision = true;
                // DEBUG: Log collision details
                if (total_iterations % 1000 == 0) {
                    std::cout << "   💥 COLLISION REJECTED: " << selected_part.name << " vs " << parts[i].name << std::endl;
                }
            }
        }
        
        if (has_collision) {
            // REJECT move immediately - no energy calculation needed
            selected_part.set_transform(old_transform);
            continue;
        }
        
        // Calculate new energy (only for collision-free moves)
        float new_energy = calculate_energy();
        float energy_delta = new_energy - current_energy;
        
        // Accept or reject move based on energy (Fogleman: pure volume minimization)
        if (accept_move(energy_delta, temperature)) {
            current_energy = new_energy;
            accepted_moves++;
            
            // Update best solution if improved
            if (current_energy < best_energy) {
                best_energy = current_energy;
                for (size_t i = 0; i < parts.size(); i++) {
                    best_transforms[i] = parts[i].transform;
                }
                iterations_without_improvement = 0;
                
                std::cout << "   🎯 New best energy: " << best_energy 
                          << " (iteration " << iteration << ")" << std::endl;
                
                // Check if this is collision-free for real-time updates
                bool is_collision_free = true;
                for (size_t i = 0; i < parts.size() && is_collision_free; i++) {
                    for (size_t j = i + 1; j < parts.size(); j++) {
                        if (parts[i].collides_with(parts[j])) {
                            is_collision_free = false;
                            break;
                        }
                    }
                }
                
                if (is_collision_free) {
                    std::cout << "   ✅ COLLISION-FREE solution found! Energy: " << best_energy << std::endl;
                    // Real-time update flag - will be picked up by Python wrapper
                }
            }
        } else {
            // Reject move - restore old transform
            selected_part.set_transform(old_transform);
            iterations_without_improvement++;
        }
        
        // Cool down temperature - Fogleman's exponential cooling
        float old_temp = temperature;
        temperature *= cooling_rate;
        
        // Ensure temperature doesn't go below minimum
        temperature = std::max(temperature, min_temperature);
        
        // DEBUG: Log temperature changes
        if (iteration % 10000 == 0) {
            std::cout << "   🌡️ Temperature: " << std::fixed << std::setprecision(6) << old_temp 
                      << " → " << std::setprecision(6) << temperature 
                      << " (cooling_rate=" << cooling_rate << ", min=" << min_temperature << ")" << std::endl;
        }
        
        // Progress reporting
        if (iteration % 1000 == 0 && iteration > 0) {
            float utilization = build_plate.get_utilization(parts);
            float build_height = build_plate.get_build_height(parts);
            
            std::cout << "   Iteration " << iteration << ": "
                      << "T=" << std::fixed << std::setprecision(3) << temperature
                      << ", E=" << std::setprecision(1) << current_energy
                      << ", Util=" << std::setprecision(2) << utilization * 100 << "%"
                      << ", Height=" << std::setprecision(1) << build_height << "mm" << std::endl;
        }
        
        // Fogleman-style termination - only plateau detection
        // Adaptive plateau threshold based on temperature
        // Fogleman-style: MUCH longer plateau tolerance for deep exploration
        int adaptive_plateau_threshold = static_cast<int>(200000 + (temperature / initial_temperature) * 800000);
        adaptive_plateau_threshold = std::min(adaptive_plateau_threshold, 1000000); // Max 1M iterations without improvement
        
        if (iterations_without_improvement > adaptive_plateau_threshold) {
            std::cout << "   📈 Plateau reached: " << iterations_without_improvement << " iterations without improvement (threshold: " << adaptive_plateau_threshold << ")" << std::endl;
            break;
        }
        
        // Optional safety limit to prevent infinite runs
        if (iteration > 50000000) { // 50M iteration safety limit - allow much longer exploration like Fogleman
            std::cout << "   🛑 Safety limit reached: 50M iterations" << std::endl;
            break;
        }
    }
    
    // Restore best solution
    for (size_t i = 0; i < parts.size(); i++) {
        parts[i].set_transform(best_transforms[i]);
        parts[i].is_placed = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "🎉 Pack3D Optimization Complete!" << std::endl;
    std::cout << "   Final energy: " << best_energy << std::endl;
    std::cout << "   Total iterations: " << total_iterations << std::endl;
    std::cout << "   Accepted moves: " << accepted_moves << " (" 
              << (100.0f * accepted_moves / total_iterations) << "%)" << std::endl;
    std::cout << "   Optimization time: " << duration.count() << "ms" << std::endl;
    std::cout << "   Build height: " << build_plate.get_build_height(parts) << "mm" << std::endl;
    std::cout << "   Plate utilization: " << (build_plate.get_utilization(parts) * 100) << "%" << std::endl;
    
    return !has_collisions();
}

float Pack3DOptimizer::calculate_energy() const {
    // AGGRESSIVE PACKING: Multi-objective function for tight clustering
    if (parts.empty()) return 0.0f;
    
    // Calculate the overall bounding box of all parts
    Vector3 min_coords(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vector3 max_coords(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    
    Vector3 center_of_mass(0, 0, 0);
    int placed_parts = 0;
    
    for (const auto& part : parts) {
        if (part.is_placed) {
            BoundingBox bbox = part.get_current_bbox();
            min_coords.x = std::min(min_coords.x, bbox.min.x);
            min_coords.y = std::min(min_coords.y, bbox.min.y);
            min_coords.z = std::min(min_coords.z, bbox.min.z);
            max_coords.x = std::max(max_coords.x, bbox.max.x);
            max_coords.y = std::max(max_coords.y, bbox.max.y);
            max_coords.z = std::max(max_coords.z, bbox.max.z);
            
            // Calculate center of mass for clustering
            Vector3 part_center = (bbox.min + bbox.max) * 0.5f;
            center_of_mass = center_of_mass + part_center;
            placed_parts++;
        }
    }
    
    if (placed_parts == 0) return 0.0f;
    
    center_of_mass = center_of_mass * (1.0f / placed_parts);
    
    // 1. PRIMARY: Bounding volume (Fogleman's core metric)
    Vector3 size = max_coords - min_coords;
    float bounding_volume = size.x * size.y * size.z;
    
    // 2. AGGRESSIVE: Heavy penalty for Z height (prioritize flat packing)
    float height_penalty = size.z * size.z * 10.0f;  // Quadratic penalty for height
    
    // 3. VERY AGGRESSIVE: Strong clustering penalty - heavily penalize parts far from center of mass
    float clustering_penalty = 0.0f;
    for (const auto& part : parts) {
        if (part.is_placed) {
            BoundingBox bbox = part.get_current_bbox();
            Vector3 part_center = (bbox.min + bbox.max) * 0.5f;
            Vector3 distance_vec = part_center - center_of_mass;
            float distance = std::sqrt(distance_vec.x * distance_vec.x + 
                                     distance_vec.y * distance_vec.y + 
                                     distance_vec.z * distance_vec.z);
            clustering_penalty += distance * distance * distance;  // CUBIC penalty for strong clustering force
        }
    }
    
    // 4. AGGRESSIVE: XY spread penalty (force tight XY clustering)
    float xy_spread_penalty = (size.x * size.y) * 5.0f;
    
    // Combine all factors with very aggressive weighting
    float total_energy = (bounding_volume / 1000.0f) +           // Base Fogleman metric
                        (height_penalty / 100.0f) +              // Heavy height penalty  
                        (clustering_penalty / 20.0f) +           // VERY HEAVY clustering penalty (cubic)
                        (xy_spread_penalty / 200.0f);            // XY spread penalty
    
    return total_energy;
}

bool Pack3DOptimizer::has_collisions() const {
    for (size_t i = 0; i < parts.size(); i++) {
        for (size_t j = i + 1; j < parts.size(); j++) {
            if (parts[i].collides_with(parts[j])) {
                return true;
            }
        }
    }
    return false;
}

void Pack3DOptimizer::generate_initial_placement() {
    std::cout << "   🎲 Generating compact initial placement (Fogleman approach)..." << std::endl;
    
    // COMPACT INITIAL PLACEMENT: Start parts close together in center
    // This gives better starting volume for optimization
    
    float center_x = build_plate.dimensions.x / 2.0f;
    float center_y = build_plate.dimensions.y / 2.0f;
    
    // Place parts in an extremely tight initial region (30x30mm) around center
    float initial_region_size = 30.0f;
    float region_min_x = center_x - initial_region_size / 2.0f;
    float region_max_x = center_x + initial_region_size / 2.0f;
    float region_min_y = center_y - initial_region_size / 2.0f;
    float region_max_y = center_y + initial_region_size / 2.0f;
    
    for (size_t i = 0; i < parts.size(); i++) {
        auto& part = parts[i];
        Vector3 part_size = part.original_bbox.size();
        
        bool placed = false;
        int attempts = 0;
        const int max_attempts = 50;
        
        while (!placed && attempts < max_attempts) {
            attempts++;
            
            // Random position in compact initial region
            std::uniform_real_distribution<float> x_dist(region_min_x, region_max_x);
            std::uniform_real_distribution<float> y_dist(region_min_y, region_max_y);
            std::uniform_real_distribution<float> z_dist(0.0f, 10.0f); // Keep Z extremely low initially
            
            part.transform.position = Vector3(x_dist(rng), y_dist(rng), z_dist(rng));
            part.transform.rotation = Rotation3(0, 0, 0); // Start without rotation
            
            // Check bounds and collisions with already placed parts
            if (build_plate.fits(part)) {
                bool has_collision = false;
                for (size_t j = 0; j < i; j++) {
                    if (part.collides_with(parts[j])) {
                        has_collision = true;
                        break;
                    }
                }
                
                if (!has_collision) {
                    placed = true;
                    part.is_placed = true;
                    std::cout << "   ✅ Part " << part.name << " placed collision-free at (" 
                              << part.transform.position.x << ", " << part.transform.position.y << ", " << part.transform.position.z 
                              << ") after " << attempts << " attempts" << std::endl;
                }
            }
        }
        
        if (!placed) {
            // Fallback: Place in a grid pattern with spacing
            int grid_x = i % 4;
            int grid_y = (i / 4) % 4;
            int grid_z = i / 16;
            
            float spacing = 30.0f; // 30mm spacing between parts
            part.transform.position = Vector3(
                center_x - 60.0f + grid_x * spacing,
                center_y - 60.0f + grid_y * spacing,
                grid_z * spacing
            );
            part.transform.rotation = Rotation3(0, 0, 0);
            part.is_placed = true;
            
            std::cout << "   ⚠️ Part " << part.name << " placed in grid fallback at (" 
                      << part.transform.position.x << ", " << part.transform.position.y << ", " << part.transform.position.z 
                      << ")" << std::endl;
        }
    }
    
    std::cout << "   📦 Compact initial placement complete - all parts positioned close together" << std::endl;
}

bool Pack3DOptimizer::accept_move(float energy_delta, float temperature) {
    // Always accept improvements
    if (energy_delta <= 0) {
        return true;
    }
    
    // Accept worse moves with probability based on temperature
    float probability = exp(-energy_delta / temperature);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    return prob_dist(rng) < probability;
}

Pack3DOptimizer::Results Pack3DOptimizer::get_results() const {
    Results results;
    results.success = !has_collisions();
    results.final_energy = best_energy;
    results.build_height = build_plate.get_build_height(parts);
    results.plate_utilization = build_plate.get_utilization(parts);
    results.total_iterations = total_iterations;
    results.accepted_moves = accepted_moves;
    results.optimization_time_ms = 0.0f;  // TODO: Calculate actual time
    results.part_transforms = best_transforms;
    
    return results;
}

const std::vector<Transform3D>& Pack3DOptimizer::get_transforms() const {
    return best_transforms;
}

} // namespace Pack3D
