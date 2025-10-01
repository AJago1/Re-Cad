#!/usr/bin/env python3
"""
Pack3D Python Wrapper - Integration with STL Analyzer
"""

import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import trimesh

# Try to import C++ Pack3D extension
PACK3D_AVAILABLE = False
try:
    import cpp_pack3d
    PACK3D_AVAILABLE = True
    print("✅ Pack3D C++ extension loaded")
except ImportError:
    print("⚠️ Pack3D C++ extension not available")

class Pack3DManager:
    """
    High-level Python interface for Pack3D optimization
    """
    
    def __init__(self, printer_type="EOS_P396"):
        """
        Initialize Pack3D manager
        
        Args:
            printer_type: "EOS_P396" or "EOS_P110"
        """
        self.printer_type = printer_type
        self.parts = []
        self.optimization_results = None
        self.use_cpp = PACK3D_AVAILABLE
        
        # Set printer dimensions
        if printer_type == "EOS_P396":
            self.plate_dimensions = (340, 340, 600)  # mm
            self.layer_height = 0.1
        elif printer_type == "EOS_P110":
            self.plate_dimensions = (200, 250, 330)  # mm  
            self.layer_height = 0.1
        else:
            raise ValueError(f"Unknown printer type: {printer_type}")
        
        if self.use_cpp:
            plate_dims = cpp_pack3d.Vector3(*self.plate_dimensions)
            self.optimizer = cpp_pack3d.Pack3DOptimizer(plate_dims, self.layer_height)
        
        print(f"🖨️ Pack3D initialized for {printer_type}")
        print(f"   Build plate: {self.plate_dimensions[0]}x{self.plate_dimensions[1]}x{self.plate_dimensions[2]}mm")
    
    def add_part_from_mesh(self, mesh: trimesh.Trimesh, name: str, part_id: Optional[int] = None) -> bool:
        """
        Add a part from a trimesh mesh with 8-leaf BVH collision detection
        
        Args:
            mesh: Trimesh object
            name: Part name
            part_id: Optional part ID (auto-generated if None)
            
        Returns:
            True if successful
        """
        if not self.use_cpp:
            print("❌ C++ Pack3D not available")
            return False
        
        try:
            # Generate part ID if not provided
            if part_id is None:
                part_id = len(self.parts)
            
            # Get mesh bounds and volume
            bounds = mesh.bounds  # 2x3 array: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            volume = float(mesh.volume)
            
            # Convert to numpy array for C++
            bounds_array = np.array(bounds, dtype=np.float32)
            
            # Create C++ Part object
            cpp_part = cpp_pack3d.create_part_from_mesh_bounds(part_id, name, bounds_array, volume)
            
            # 🎯 Generate 8-leaf BVH for collision detection (Fogleman's approach)
            bvh_leaves = self._generate_fogleman_bvh(mesh)
            if bvh_leaves:
                # Pass BVH leaves to C++ Part
                cpp_pack3d.set_part_bvh_leaves(cpp_part, bvh_leaves)
                print(f"   🔧 Generated {len(bvh_leaves)} BVH leaves using Fogleman's approach")
            else:
                print("   ⚠️ Failed to generate BVH, using bounding box collision fallback")
            
            # Store part info
            part_info = {
                'id': part_id,
                'name': name,
                'mesh': mesh,
                'cpp_part': cpp_part,
                'bounds': bounds,
                'volume': volume,
                'bvh_leaves': bvh_leaves
            }
            
            self.parts.append(part_info)
            
            # CRITICAL FIX: Set BVH leaves on the C++ part for accurate collision detection
            if bvh_leaves:
                print(f"   📦 Setting {len(bvh_leaves)} BVH leaves on C++ part for accurate collision detection")
                cpp_pack3d.set_part_bvh_leaves(cpp_part, bvh_leaves)
            else:
                print(f"   ⚠️ WARNING: No BVH leaves available - collision detection will be inaccurate!")
            
            self.optimizer.add_part(cpp_part)
            
            print(f"✅ Added part '{name}' (ID: {part_id})")
            print(f"   Volume: {volume:.1f}mm³")
            print(f"   Bounds: {bounds[0]} to {bounds[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add part '{name}': {e}")
            return False
    
    def _generate_fogleman_bvh(self, mesh: trimesh.Trimesh) -> Optional[list]:
        """
        Generate 8-leaf BVH using Fogleman's triangle-centroid approach
        
        Args:
            mesh: Trimesh object
            
        Returns:
            List of BVH leaf bounds [(min_x, min_y, min_z, max_x, max_y, max_z), ...] or None
        """
        try:
            # Try Fogleman's C++ BVH first (most accurate)
            try:
                import cpp_bvh_fogleman
                
                # Convert mesh to vertex list
                vertices = mesh.vertices.flatten().astype(np.float32).tolist()
                
                # Create Fogleman BVH
                bvh_id = cpp_bvh_fogleman.create_bvh(vertices, 8)
                if bvh_id is None:
                    raise ImportError("Failed to create Fogleman BVH")
                
                # Get BVH statistics
                stats = cpp_bvh_fogleman.get_bvh_stats(bvh_id)
                if not stats:
                    raise ImportError("Failed to get BVH stats")
                
                leaf_count = stats["leaf_count"]
                
                # Extract leaf bounds
                bvh_leaves = []
                for i in range(leaf_count):
                    bounds = cpp_bvh_fogleman.get_leaf_bounds(bvh_id, i)
                    if bounds and len(bounds) == 6:
                        # Convert to (min_x, min_y, min_z, max_x, max_y, max_z)
                        bvh_leaves.append(bounds)
                
                # Clean up
                cpp_bvh_fogleman.cleanup_bvhs()
                
                return bvh_leaves if bvh_leaves else None
                
            except ImportError:
                pass
            
            # Fallback to Python BVH implementation
            try:
                from .bvh_8leaf import EightLeafBVH
                
                # Create Python BVH
                python_bvh = EightLeafBVH(mesh, leaf_count=8)
                
                # Extract leaf bounds
                bvh_leaves = []
                for leaf in python_bvh.leaves:
                    # Convert BoundingBox to tuple format
                    bounds = (leaf.min[0], leaf.min[1], leaf.min[2], 
                             leaf.max[0], leaf.max[1], leaf.max[2])
                    bvh_leaves.append(bounds)
                
                return bvh_leaves if bvh_leaves else None
                
            except ImportError:
                pass
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ BVH generation failed: {e}")
            return None
    
    def add_part_from_file(self, stl_path: str, name: Optional[str] = None) -> bool:
        """
        Add a part from STL file
        
        Args:
            stl_path: Path to STL file
            name: Part name (uses filename if None)
            
        Returns:
            True if successful
        """
        try:
            # Load mesh
            mesh = trimesh.load(stl_path)
            
            # Use filename as name if not provided
            if name is None:
                import os
                name = os.path.splitext(os.path.basename(stl_path))[0]
            
            return self.add_part_from_mesh(mesh, name)
            
        except Exception as e:
            print(f"❌ Failed to load STL '{stl_path}': {e}")
            return False
    
    def set_optimization_parameters(self, initial_temp: float = 1000.0, 
                                  cooling_rate: float = 0.95,
                                  min_temp: float = 0.1,
                                  max_iterations: int = 10000):
        """
        Set simulated annealing parameters
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor (0.9-0.99)
            min_temp: Minimum temperature 
            max_iterations: Maximum iterations
        """
        adjusted_cooling_rate = cooling_rate  # Default
        
        if self.use_cpp:
            # FOGLEMAN PARAMETERS: Use exact values from GUI, no workarounds needed
            print(f"🎯 Using TRUE Fogleman parameters:")
            print(f"   Initial temp: {initial_temp}°")
            print(f"   Cooling rate: {cooling_rate} (slow exponential)")
            print(f"   Min temp: {min_temp}°")
            print(f"   Max iterations: {max_iterations} (runs indefinitely until plateau)")
            
            # CRITICAL DEBUG: Log exact values being passed to C++
            print(f"🔍 DEBUG: Passing to C++ optimizer:")
            print(f"   initial_temp: {initial_temp} (type: {type(initial_temp)})")
            print(f"   cooling_rate: {cooling_rate} (type: {type(cooling_rate)})")
            print(f"   min_temp: {min_temp} (type: {type(min_temp)})")
            print(f"   max_iterations: {max_iterations} (type: {type(max_iterations)})")
            
            self.optimizer.set_parameters(initial_temp, cooling_rate, min_temp, max_iterations)
            
        print(f"🔧 Pack3D parameters set:")
        print(f"   Initial temperature: {initial_temp}")
        print(f"   Cooling rate: {cooling_rate} (adjusted: {adjusted_cooling_rate:.5f} if C++)")
        print(f"   Min temperature: {min_temp}")
        print(f"   Max iterations: {max_iterations}")
    
    def optimize(self) -> bool:
        """
        Run Pack3D optimization
        
        Returns:
            True if optimization completed successfully
        """
        if not self.use_cpp:
            print("❌ C++ Pack3D not available")
            return False
        
        if not self.parts:
            print("❌ No parts to optimize")
            return False
        
        print(f"🚀 Starting Pack3D optimization with {len(self.parts)} parts...")
        
        start_time = time.time()
        
        # Run C++ optimization
        success = self.optimizer.optimize()
        
        # Get results
        self.optimization_results = self.optimizer.get_results()
        
        optimization_time = time.time() - start_time
        
        # Update optimization time in results (convert to milliseconds)
        self.optimization_results.optimization_time_ms = optimization_time * 1000
        
        print(f"🎉 Pack3D optimization completed in {optimization_time:.2f}s")
        print(f"   Success: {success}")
        print(f"   Final energy: {self.optimization_results.final_energy:.1f}")
        print(f"   Build height: {self.optimization_results.build_height:.1f}mm")
        print(f"   Plate utilization: {self.optimization_results.plate_utilization*100:.1f}%")
        print(f"   Iterations: {self.optimization_results.total_iterations}")
        print(f"   Accepted moves: {self.optimization_results.accepted_moves}")
        
        return success
    
    def optimize_with_realtime_updates(self, update_callback: Optional[Callable] = None, update_interval: float = 2.0) -> bool:
        """
        Run Pack3D optimization with real-time updates (Fogleman style - runs indefinitely)
        
        Args:
            update_callback: Function to call periodically with current state
            update_interval: Time in seconds between updates
            
        Returns:
            True if optimization completed successfully
        """
        if not self.use_cpp:
            print("❌ C++ Pack3D not available")
            return False
        
        if not self.parts:
            print("❌ No parts to optimize")
            return False
        
        print(f"🚀 Starting indefinite Pack3D optimization (Fogleman style) with {len(self.parts)} parts...")
        print(f"   Real-time updates every {update_interval:.1f}s")
        
        self.optimization_running = True
        self.optimization_success = False
        
        def run_optimization():
            try:
                start_time = time.time()
                
                # Run C++ optimization (now runs indefinitely until plateau)
                success = self.optimizer.optimize()
                
                # Get final results
                self.optimization_results = self.optimizer.get_results()
                
                optimization_time = time.time() - start_time
                self.optimization_results.optimization_time_ms = optimization_time * 1000
                
                print(f"🎉 Pack3D optimization completed in {optimization_time:.2f}s")
                print(f"   Success: {success}")
                print(f"   Final energy: {self.optimization_results.final_energy:.1f}")
                print(f"   Build height: {self.optimization_results.build_height:.1f}mm")
                print(f"   Plate utilization: {self.optimization_results.plate_utilization*100:.1f}%")
                print(f"   Iterations: {self.optimization_results.total_iterations}")
                print(f"   Accepted moves: {self.optimization_results.accepted_moves}")
                
                self.optimization_success = success
                
            except Exception as e:
                print(f"❌ Pack3D optimization error: {e}")
                self.optimization_success = False
            finally:
                self.optimization_running = False
        
        def update_loop():
            """Periodic updates while optimization is running"""
            while self.optimization_running:
                time.sleep(update_interval)
                if self.optimization_running and update_callback:
                    try:
                        # Get current state from optimizer
                        current_results = self.optimizer.get_results()
                        update_callback(current_results)
                    except Exception as e:
                        print(f"⚠️ Update callback error: {e}")
        
        # Start optimization in background thread
        optimization_thread = threading.Thread(target=run_optimization, daemon=True)
        optimization_thread.start()
        
        # Start update loop if callback provided
        if update_callback:
            update_thread = threading.Thread(target=update_loop, daemon=True)
            update_thread.start()
        
        return True  # Return immediately, optimization runs in background
    
    def stop_optimization(self):
        """Stop the running optimization"""
        self.optimization_running = False
        # Note: C++ optimizer doesn't have stop mechanism yet, will stop at next plateau
        print("🛑 Optimization stop requested (will finish at next plateau)")
    
    def is_optimization_running(self) -> bool:
        """Check if optimization is currently running"""
        return getattr(self, 'optimization_running', False)
    
    def get_part_transforms(self) -> List[Dict]:
        """
        Get optimized part transforms
        
        Returns:
            List of transform dictionaries with position and rotation
        """
        if not self.optimization_results:
            return []
        
        transforms = []
        for i, transform in enumerate(self.optimization_results.part_transforms):
            if i < len(self.parts):
                part_info = self.parts[i]
                transform_dict = {
                    'part_id': part_info['id'],
                    'part_name': part_info['name'],
                    'position': [transform.position.x, transform.position.y, transform.position.z],
                    'rotation': [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw],
                    'bounds': part_info['bounds'],
                    'volume': part_info['volume'],
                    'mesh': part_info['mesh']  # Include mesh for visualization
                }
                transforms.append(transform_dict)
        
        return transforms
    
    def get_optimization_summary(self) -> Dict:
        """
        Get optimization summary statistics
        
        Returns:
            Dictionary with optimization metrics
        """
        if not self.optimization_results:
            return {}
        
        results = self.optimization_results
        
        return {
            'success': results.success,
            'final_energy': results.final_energy,
            'build_height_mm': results.build_height,
            'plate_utilization_percent': results.plate_utilization * 100,
            'total_iterations': results.total_iterations,
            'accepted_moves': results.accepted_moves,
            'acceptance_rate_percent': (results.accepted_moves / results.total_iterations * 100) if results.total_iterations > 0 else 0,
            'optimization_time_ms': results.optimization_time_ms,
            'num_parts': len(self.parts),
            'total_volume_mm3': sum(part['volume'] for part in self.parts),
            'printer_type': self.printer_type,
            'plate_dimensions_mm': self.plate_dimensions
        }
    
    def clear_parts(self):
        """Clear all parts and reset optimization"""
        self.parts.clear()
        self.optimization_results = None
        
        if self.use_cpp:
            # Recreate optimizer
            plate_dims = cpp_pack3d.Vector3(*self.plate_dimensions)
            self.optimizer = cpp_pack3d.Pack3DOptimizer(plate_dims, self.layer_height)
        
        print("🗑️ All parts cleared")
    
    def export_results_to_json(self, filepath: str) -> bool:
        """
        Export optimization results to JSON file
        
        Args:
            filepath: Output JSON file path
            
        Returns:
            True if successful
        """
        try:
            import json
            
            export_data = {
                'summary': self.get_optimization_summary(),
                'part_transforms': self.get_part_transforms(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pack3d_version': '1.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📄 Results exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export results: {e}")
            return False

# Utility functions
def get_printer_dimensions(printer_type: str) -> Tuple[float, float, float]:
    """Get build plate dimensions for a printer type"""
    if printer_type == "EOS_P396":
        return (340, 340, 600)
    elif printer_type == "EOS_P110":
        return (200, 250, 330)
    else:
        raise ValueError(f"Unknown printer type: {printer_type}")

def is_pack3d_available() -> bool:
    """Check if Pack3D C++ extension is available"""
    return PACK3D_AVAILABLE
