import numpy as np
import trimesh

def create_ball_pivoting(points, radius_factor=2.0):
    """
    A simplified ball pivoting algorithm implementation that works without Open3D.
    
    This is a simplified version that uses a convex hull approximation with some 
    modifications to try to preserve more of the original shape detail. It's not 
    a true ball pivoting algorithm, but provides a reasonable fallback when Open3D is not available.
    
    Args:
        points (numpy.ndarray): Points to create mesh from
        radius_factor (float): Factor to adjust the ball radius (higher = more holes filled)
        
    Returns:
        trimesh.Trimesh: A mesh created from the points
    """
    try:
        print("Using simplified ball pivoting implementation (without Open3D)")
        
        # First create a convex hull as our starting point
        hull = trimesh.convex.convex_hull(points)
        if hull is None:
            print("Failed to create initial hull, returning None")
            return None
        
        # Now we'll try to refine it by creating a more detailed mesh using a multi-step approach:
        # 1. Create multiple smaller convex hulls from subsets of points
        # 2. Combine them to get a more detailed shape
        
        # Split points into clusters
        num_clusters = min(10, max(3, len(points) // 500))
        
        # Use a simple spatial subdivision approach
        # First get the bounding box
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        dimensions = max_bound - min_bound
        
        # Find the dimension with the largest extent
        largest_dim = np.argmax(dimensions)
        
        # Sort points along this dimension
        sorted_indices = np.argsort(points[:, largest_dim])
        points_per_cluster = len(points) // num_clusters
        
        # Create hulls for each cluster
        hulls = []
        for i in range(num_clusters):
            start_idx = i * points_per_cluster
            end_idx = min(start_idx + points_per_cluster, len(points))
            cluster_indices = sorted_indices[start_idx:end_idx]
            
            if len(cluster_indices) < 4:
                continue  # Skip clusters that are too small
                
            cluster_points = points[cluster_indices]
            try:
                cluster_hull = trimesh.convex.convex_hull(cluster_points)
                if cluster_hull is not None and len(cluster_hull.vertices) > 0:
                    hulls.append(cluster_hull)
            except Exception as e:
                print(f"Error creating cluster hull: {e}")
        
        if not hulls:
            print("Failed to create any valid cluster hulls, returning original hull")
            return hull
            
        # Combine all hulls
        combined = trimesh.util.concatenate(hulls)
        
        # Clean up the mesh by removing duplicate vertices
        final_mesh = combined.split(only_watertight=False)
        
        # If we got multiple meshes, pick the largest one
        if isinstance(final_mesh, list) and len(final_mesh) > 0:
            volumes = [m.volume if hasattr(m, 'volume') and m.volume > 0 else 0 for m in final_mesh]
            largest_idx = np.argmax(volumes)
            final_mesh = final_mesh[largest_idx]
        
        # Make sure the result is a valid mesh
        if hasattr(final_mesh, 'vertices') and len(final_mesh.vertices) > 0:
            print(f"Created simplified ball pivoting mesh with {len(final_mesh.vertices)} vertices")
            return final_mesh
        else:
            # Fall back to original hull
            print("Failed to create valid simplified ball pivoting mesh, returning original hull")
            return hull
            
    except Exception as e:
        print(f"Error in simplified ball pivoting: {e}")
        # Fall back to convex hull
        try:
            return trimesh.convex.convex_hull(points)
        except:
            return None 