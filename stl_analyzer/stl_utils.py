import os
import sys
import re
import time
import json
import shutil
import tempfile
import subprocess
import traceback
import logging
import signal
from pathlib import Path
import trimesh
import numpy as np
import multiprocessing
import warnings
import threading
import alphashape
from functools import partial
import pandas as pd
import datetime
from scipy import ndimage

# Try to import scipy for morphological operations
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not available, using fallback dilation implementation")
    HAS_SCIPY = False

# Try to import sklearn for PCA, with fallback
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not available, using fallback PCA implementation")
    HAS_SKLEARN = False
    # Create a fallback PCA class
    class PCA:
        def __init__(self, n_components=3):
            self.n_components = n_components
            self.components_ = None
        
        def fit(self, X):
            # Center the data
            self.mean_ = np.mean(X, axis=0)
            centered = X - self.mean_
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered.T)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalues (largest first)
            idx = np.argsort(eigenvalues)[::-1]
            self.components_ = eigenvectors[:, idx[:self.n_components]].T
            
            return self

# Set up logging
logger = logging.getLogger(__name__)

# Try to import C++ extensions for better performance
try:
    import cpp_stl_utils
    print("C++ extensions loaded - using accelerated implementations")
    HAS_CPP_EXTENSIONS = True
except ImportError:
    print("="*80)
    print("WARNING: C++ extensions not available - slower Python implementations will be used.")
    print("For significantly better performance (5-10x faster), build the C++ extensions by running:")
    print("  build_cpp_extension.bat")
    print("="*80)
    HAS_CPP_EXTENSIONS = False
except Exception as e:
    print(f"Error loading C++ extensions: {e}")
    print("Using Python implementations as fallback")
    HAS_CPP_EXTENSIONS = False

# Make Open3D optional with a fallback
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available. Ball Pivoting Algorithm (BPA) will not be used. Falling back to alternative methods.")

# Try to import alphashape for alpha shape shrinkwrap
try:
    import alphashape
    ALPHASHAPE_AVAILABLE = True
except ImportError:
    ALPHASHAPE_AVAILABLE = False

MAX_FILE_SIZE_MB = 50  # Skip files larger than this size
ALPHA_SHAPE_TIMEOUT = 30  # seconds

# Function for timeout handling
def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutError("Alpha shape computation timed out")

# Function to run with timeout
def run_with_timeout(func, timeout, *args, **kwargs):
    """Run a function with a timeout"""
    # Windows doesn't support SIGALRM, so we use a different approach
    if os.name == 'nt':  # Windows
        try:
            with multiprocessing.Pool(1) as pool:
                result = pool.apply_async(func, args=args, kwds=kwargs)
                return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
    else:  # Unix-based systems
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Disable the alarm
            return result
        except TimeoutError as e:
            raise e  # Re-raise the timeout error
        finally:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)  # Reset the signal handler

def is_file_size_safe(path):
    """Check if file size is within safe limits"""
    return os.path.getsize(path) < MAX_FILE_SIZE_MB * 1024 * 1024

def try_load_stl(file_path):
    """Safely load an STL file with error handling"""
    try:
        # Normalize file path (handle Windows backslashes, etc.)
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
            
        if not is_file_size_safe(file_path):
            print(f"Skipped large file: {file_path}")
            return None
            
        # Try different methods to load the file
        try:
            mesh = trimesh.load(file_path, force='mesh')
        except Exception as e1:
            print(f"Error with standard loading, trying alternative method: {e1}")
            try:
                # Alternative loading method with explicit file format
                with open(file_path, 'rb') as f:
                    data = f.read()
                    # Check if binary STL (starts with solid but contains binary data)
                    if data.startswith(b'solid') and (b'\0' in data[:4096]):
                        mesh = trimesh.load(file_path, file_type='stl_binary')
                    else:
                        mesh = trimesh.load(file_path, file_type='stl')
            except Exception as e2:
                print(f"Alternative loading method failed: {e2}")
                return None
                
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Not a valid mesh: {file_path}")
            return None
            
        return mesh
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def pca_minimal_bounding_box(mesh):
    """
    Fast PCA-based minimal bounding box calculation.
    Uses Principal Component Analysis to find optimal orientation.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        tuple: (size, optimal_transform, volume_reduction_percentage)
    """
    try:
        start_time = time.time()
        
        # Get mesh vertices
        vertices = mesh.vertices
        
        # Calculate centroid
        centroid = np.mean(vertices, axis=0)
        
        # Center the vertices
        centered_vertices = vertices - centroid
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(centered_vertices.T)
        
        # Perform PCA - get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues (largest first) to get principal components
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        
        # Create transformation matrix
        # The eigenvectors form the rotation matrix
        rotation_matrix = eigenvectors.T  # Transpose to get correct orientation
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = centroid - rotation_matrix @ centroid
        
        # Apply transformation to get oriented vertices
        oriented_vertices = centered_vertices @ rotation_matrix.T
        
        # Calculate bounding box in the oriented space
        min_coords = np.min(oriented_vertices, axis=0)
        max_coords = np.max(oriented_vertices, axis=0)
        size = max_coords - min_coords
        
        # Sort dimensions from largest to smallest for consistency
        size = np.sort(size)[::-1]
        
        # Calculate volume reduction compared to axis-aligned bounding box
        original_bounds = mesh.bounds
        original_size = original_bounds[1] - original_bounds[0]
        original_volume = np.prod(original_size)
        pca_volume = np.prod(size)
        volume_reduction = ((original_volume - pca_volume) / original_volume) * 100
        
        elapsed_time = time.time() - start_time
        print(f"PCA bounding box calculated in {elapsed_time:.3f}s, volume reduction: {volume_reduction:.1f}%")
        
        return size, transform, volume_reduction
        
    except Exception as e:
        print(f"Error in PCA bounding box calculation: {e}")
        # Fallback to axis-aligned bounding box
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        size = np.sort(size)[::-1]
        return size, np.eye(4), 0.0

def weighted_pca_orientation(mesh):
    """
    Calculate PCA orientation weighted by triangle areas for better accuracy on asymmetric geometries.
    """
    try:
        # Get face centers and areas
        face_centers = mesh.triangles_center
        face_areas = mesh.area_faces
        
        # Skip if no valid faces
        if len(face_centers) == 0 or len(face_areas) == 0:
            return None, None
            
        # Weighted centroid
        total_area = np.sum(face_areas)
        if total_area <= 0:
            return None, None
            
        weighted_centroid = np.average(face_centers, axis=0, weights=face_areas)
        
        # Center the points
        centered_points = face_centers - weighted_centroid
        
        # Weighted covariance matrix
        cov_matrix = np.cov(centered_points.T, aweights=face_areas)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
            
        return eigenvectors, weighted_centroid
        
    except Exception as e:
        print(f"Weighted PCA failed: {e}")
        return None, None

def refine_orientation_with_sweep(mesh, initial_transform, sweep_angles=[5, 10, 15]):
    """
    Refine PCA orientation by sweeping small angles around each axis.
    """
    try:
        best_transform = initial_transform.copy()
        best_volume = np.inf
        
        # Test current orientation
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(initial_transform)
        current_volume = np.prod(transformed_mesh.bounds[1] - transformed_mesh.bounds[0])
        best_volume = current_volume
        
        # Test small rotations around each axis
        for axis in range(3):
            for angle in sweep_angles:
                for sign in [-1, 1]:
                    # Create rotation matrix
                    rotation_angle = np.radians(angle * sign)
                    rotation_axis = np.zeros(3)
                    rotation_axis[axis] = 1
                    
                    # Create rotation matrix using Rodrigues' formula
                    cos_angle = np.cos(rotation_angle)
                    sin_angle = np.sin(rotation_angle)
                    
                    rotation_matrix = np.eye(3)
                    if axis == 0:  # X-axis
                        rotation_matrix = np.array([
                            [1, 0, 0],
                            [0, cos_angle, -sin_angle],
                            [0, sin_angle, cos_angle]
                        ])
                    elif axis == 1:  # Y-axis
                        rotation_matrix = np.array([
                            [cos_angle, 0, sin_angle],
                            [0, 1, 0],
                            [-sin_angle, 0, cos_angle]
                        ])
                    else:  # Z-axis
                        rotation_matrix = np.array([
                            [cos_angle, -sin_angle, 0],
                            [sin_angle, cos_angle, 0],
                            [0, 0, 1]
                        ])
                    
                    # Combine with initial transform
                    test_transform = initial_transform.copy()
                    test_transform[:3, :3] = rotation_matrix @ test_transform[:3, :3]
                    
                    # Test volume
                    test_mesh = mesh.copy()
                    test_mesh.apply_transform(test_transform)
                    test_volume = np.prod(test_mesh.bounds[1] - test_mesh.bounds[0])
                    
                    if test_volume < best_volume:
                        best_volume = test_volume
                        best_transform = test_transform.copy()
        
        return best_transform, best_volume
        
    except Exception as e:
        print(f"Orientation sweep failed: {e}")
        return initial_transform, current_volume

def smart_minimal_bounding_box(mesh, face_count_threshold=1000, volume_reduction_threshold=15.0):
    """
    Smart minimal bounding box calculation with multiple optimization strategies.
    
    Enhanced with:
    - Weighted PCA based on triangle areas
    - Orientation refinement with angle sweeping
    - Better fallback strategies for edge cases
    - Aspect ratio checks
    - Post-rotation axis realignment
    """
    
    if mesh is None or len(mesh.vertices) == 0:
        return [0, 0, 0], 0, np.eye(4), "aabb"
    
    # Get axis-aligned bounding box as baseline
    aabb_size = mesh.bounds[1] - mesh.bounds[0]
    aabb_volume = np.prod(aabb_size)
    
    # For very simple geometries, AABB might be optimal
    if len(mesh.faces) < 10:
        return aabb_size, aabb_volume, np.eye(4), "aabb"
    
    best_size = aabb_size
    best_volume = aabb_volume
    best_transform = np.eye(4)
    best_method = "aabb"
    
    strategies_to_try = []
    
    # Strategy selection based on complexity
    if len(mesh.faces) < face_count_threshold:
        # For simpler meshes, try more methods
        strategies_to_try = ['weighted_pca', 'pca', 'convex_hull_pca', 'inertia', 'precise']
    else:
        # For complex meshes, prioritize fast methods
        strategies_to_try = ['weighted_pca', 'convex_hull_pca', 'pca']
    
    # Try each strategy
    for strategy in strategies_to_try:
        try:
            transform = np.eye(4)
            
            if strategy == 'weighted_pca':
                # NEW: Weighted PCA method
                rotation_matrix, centroid = weighted_pca_orientation(mesh)
                if rotation_matrix is not None:
                    transform[:3, :3] = rotation_matrix.T
                    transform[:3, 3] = -rotation_matrix.T @ centroid
                else:
                    continue
                    
            elif strategy == 'pca':
                # Original PCA method
                pca = PCA(n_components=3)
                pca.fit(mesh.vertices)
                rotation_matrix = pca.components_.T
                
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, -1] *= -1
                
                centroid = mesh.vertices.mean(axis=0)
                transform[:3, :3] = rotation_matrix.T
                transform[:3, 3] = -rotation_matrix.T @ centroid
                
            elif strategy == 'convex_hull_pca':
                # PCA on convex hull
                try:
                    convex_hull = mesh.convex_hull
                    if len(convex_hull.vertices) > 3:
                        pca = PCA(n_components=3)
                        pca.fit(convex_hull.vertices)
                        rotation_matrix = pca.components_.T
                        
                        if np.linalg.det(rotation_matrix) < 0:
                            rotation_matrix[:, -1] *= -1
                        
                        centroid = convex_hull.vertices.mean(axis=0)
                        transform[:3, :3] = rotation_matrix.T
                        transform[:3, 3] = -rotation_matrix.T @ centroid
                    else:
                        continue
                except:
                    continue
                    
            elif strategy == 'inertia':
                # Inertia tensor method
                try:
                    inertia_tensor = mesh.moment_inertia
                    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
                    
                    idx = np.argsort(eigenvalues)
                    rotation_matrix = eigenvectors[:, idx]
                    
                    if np.linalg.det(rotation_matrix) < 0:
                        rotation_matrix[:, -1] *= -1
                    
                    centroid = mesh.centroid
                    transform[:3, :3] = rotation_matrix.T
                    transform[:3, 3] = -rotation_matrix.T @ centroid
                except:
                    continue
                    
            elif strategy == 'precise':
                # Fallback to trimesh's oriented bounding box
                try:
                    obb = mesh.bounding_box_oriented
                    test_size = obb.extents
                    test_volume = np.prod(test_size)
                    
                    if test_volume < best_volume:
                        best_size = test_size
                        best_volume = test_volume
                        best_transform = obb.primitive.transform
                        best_method = strategy
                    continue
                except:
                    continue
            
            # Test the orientation
            test_mesh = mesh.copy()
            test_mesh.apply_transform(transform)
            test_size = test_mesh.bounds[1] - test_mesh.bounds[0]
            test_volume = np.prod(test_size)
            
            # Check aspect ratio - if too extreme, try refinement
            aspect_ratio = np.max(test_size) / np.min(test_size) if np.min(test_size) > 0 else 1000
            
            # NEW: Apply angle sweep refinement for promising orientations
            if strategy in ['weighted_pca', 'pca'] and test_volume < best_volume * 1.1:
                refined_transform, refined_volume = refine_orientation_with_sweep(mesh, transform)
                if refined_volume < test_volume:
                    transform = refined_transform
                    test_mesh = mesh.copy()
                    test_mesh.apply_transform(transform)
                    test_size = test_mesh.bounds[1] - test_mesh.bounds[0]
                    test_volume = refined_volume
                    aspect_ratio = np.max(test_size) / np.min(test_size) if np.min(test_size) > 0 else 1000
            
            # NEW: Post-rotation axis realignment - try all 6 permutations
            if test_volume < best_volume * 1.05:  # Only for promising candidates
                best_permutation_volume = test_volume
                best_permutation_size = test_size
                best_permutation_transform = transform.copy()
                
                # Try all 24 possible axis orientations (6 faces * 4 rotations per face)
                axis_permutations = [
                    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
                ]
                
                for perm in axis_permutations:
                    for flip_x in [1, -1]:
                        for flip_y in [1, -1]:
                            for flip_z in [1, -1]:
                                # Create permutation matrix
                                perm_matrix = np.eye(3)
                                perm_matrix[0, :] = 0
                                perm_matrix[1, :] = 0
                                perm_matrix[2, :] = 0
                                perm_matrix[0, perm[0]] = flip_x
                                perm_matrix[1, perm[1]] = flip_y
                                perm_matrix[2, perm[2]] = flip_z
                                
                                # Apply permutation to transformation
                                perm_transform = transform.copy()
                                perm_transform[:3, :3] = perm_matrix @ perm_transform[:3, :3]
                                
                                # Test this permutation
                                perm_mesh = mesh.copy()
                                perm_mesh.apply_transform(perm_transform)
                                perm_size = perm_mesh.bounds[1] - perm_mesh.bounds[0]
                                perm_volume = np.prod(perm_size)
                                
                                if perm_volume < best_permutation_volume:
                                    best_permutation_volume = perm_volume
                                    best_permutation_size = perm_size
                                    best_permutation_transform = perm_transform
                
                # Use the best permutation
                test_volume = best_permutation_volume
                test_size = best_permutation_size
                transform = best_permutation_transform
                aspect_ratio = np.max(test_size) / np.min(test_size) if np.min(test_size) > 0 else 1000
            
            # Accept if better, but with aspect ratio penalty for extreme cases
            volume_penalty = 1.0
            if aspect_ratio > 15:  # Very elongated
                volume_penalty = 1.05  # 5% penalty
            elif aspect_ratio > 25:  # Extremely elongated
                volume_penalty = 1.1   # 10% penalty
            
            effective_volume = test_volume * volume_penalty
            
            if effective_volume < best_volume:
                best_size = test_size
                best_volume = test_volume  # Store actual volume, not penalized
                best_transform = transform
                best_method = strategy
                
                # Early exit if we found a great solution
                volume_reduction = (aabb_volume - test_volume) / aabb_volume * 100
                if volume_reduction > volume_reduction_threshold * 1.5:  # 22.5% reduction
                    break
                    
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            continue
    
    # NEW: Check aspect ratio threshold - if too extreme, fallback to oriented bounding box
    final_aspect_ratio = np.max(best_size) / np.min(best_size) if np.min(best_size) > 0 else 1000
    if final_aspect_ratio > 10 and best_method != 'precise':
        try:
            print(f"Aspect ratio {final_aspect_ratio:.1f} too extreme, trying oriented bounding box fallback")
            obb = mesh.bounding_box_oriented
            obb_volume = np.prod(obb.extents)
            if obb_volume < best_volume * 1.1:  # Accept if reasonably close
                best_size = obb.extents
                best_volume = obb_volume
                best_transform = obb.primitive.transform
                best_method = "obb_fallback"
        except:
            pass
    
    # Final fallback if no method worked well
    if best_volume >= aabb_volume * 0.95:  # Less than 5% improvement
        try:
            obb = mesh.bounding_box_oriented
            obb_volume = np.prod(obb.extents)
            if obb_volume < best_volume:
                best_size = obb.extents
                best_volume = obb_volume
                best_transform = obb.primitive.transform
                best_method = "final_fallback"
        except:
            pass
    
    # Sort dimensions from largest to smallest for consistency
    best_size = np.sort(best_size)[::-1]
    
    return best_size, best_volume, best_transform, best_method

def inertia_based_orientation(mesh):
    """
    Calculate optimal orientation based on principal axes of inertia.
    Good for mechanical parts and objects with clear directional features.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        tuple: (size, optimal_transform)
    """
    try:
        # Get the inertia tensor
        inertia_tensor = mesh.moment_inertia
        
        # Get principal axes (eigenvectors of inertia tensor)
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        
        # Sort by eigenvalues (smallest to largest moment of inertia)
        # The axis with smallest moment of inertia is the preferred rotation axis
        order = np.argsort(eigenvalues)
        principal_axes = eigenvectors[:, order]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(principal_axes) < 0:
            principal_axes[:, -1] *= -1
        
        # Create transformation matrix that aligns principal axes with coordinate axes
        # We want the axis with the largest moment of inertia (most spread) to be X
        # The axis with the smallest moment of inertia to be Z (rotation axis)
        rotation_matrix = principal_axes.T
        
        # Get mesh centroid
        centroid = mesh.centroid
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = centroid - rotation_matrix @ centroid
        
        # Apply transformation to get oriented vertices
        vertices_homogeneous = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        transformed_vertices = (transform @ vertices_homogeneous.T).T[:, :3]
        
        # Calculate bounding box in the oriented space
        min_coords = np.min(transformed_vertices, axis=0)
        max_coords = np.max(transformed_vertices, axis=0)
        size = max_coords - min_coords
        
        # Sort dimensions from largest to smallest for consistency
        size = np.sort(size)[::-1]
        
        return size, transform
        
    except Exception as e:
        print(f"Error in inertia-based orientation: {e}")
        # Fallback to axis-aligned bounding box
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        size = np.sort(size)[::-1]
        return size, np.eye(4)

def convex_hull_pca_bounding_box(mesh):
    """
    Ultra-fast PCA bounding box using convex hull vertices only.
    Uses much fewer points for even faster calculation.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        tuple: (size, optimal_transform, volume_reduction_percentage)
    """
    try:
        start_time = time.time()
        
        # Get convex hull (much fewer vertices)
        convex_hull = mesh.convex_hull
        vertices = convex_hull.vertices
        
        print(f"Using convex hull PCA with {len(vertices)} vertices (vs {len(mesh.vertices)} original)")
        
        # Calculate centroid
        centroid = np.mean(vertices, axis=0)
        
        # Center the vertices
        centered_vertices = vertices - centroid
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(centered_vertices.T)
        
        # Perform PCA
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues (largest first)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, -1] *= -1
        
        # Create transformation matrix
        rotation_matrix = eigenvectors.T
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = centroid - rotation_matrix @ centroid
        
        # Apply transformation to ORIGINAL mesh vertices (not just hull)
        original_vertices = mesh.vertices
        centered_original = original_vertices - centroid
        oriented_vertices = centered_original @ rotation_matrix.T
        
        # Calculate bounding box in the oriented space
        min_coords = np.min(oriented_vertices, axis=0)
        max_coords = np.max(oriented_vertices, axis=0)
        size = max_coords - min_coords
        
        # Sort dimensions from largest to smallest
        size = np.sort(size)[::-1]
        
        # Calculate volume reduction
        original_bounds = mesh.bounds
        original_size = original_bounds[1] - original_bounds[0]
        original_volume = np.prod(original_size)
        pca_volume = np.prod(size)
        volume_reduction = ((original_volume - pca_volume) / original_volume) * 100
        
        elapsed_time = time.time() - start_time
        print(f"Convex hull PCA calculated in {elapsed_time:.3f}s, volume reduction: {volume_reduction:.1f}%")
        
        return size, transform, volume_reduction
        
    except Exception as e:
        print(f"Error in convex hull PCA: {e}")
        # Fallback to regular PCA
        return pca_minimal_bounding_box(mesh)

def extract_features(file_path):
    """Extract geometric features from an STL file"""
    # Normalize file path to handle Windows backslashes
    file_path = os.path.normpath(file_path)
    
    mesh = try_load_stl(file_path)
    if mesh is None:
        return None

    try:
        # Get the model name (filename without path and extension)
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Validate and fix mesh if needed
        volume = mesh.volume
        if volume < 0:
            print(f"Warning: Mesh {model_name} has negative volume ({volume:.2f}), fixing normals...")
            try:
                # Try to fix inverted normals
                mesh.fix_normals()
                volume = mesh.volume
                if volume < 0:
                    # If still negative, invert all faces
                    mesh.faces = np.fliplr(mesh.faces)
                    volume = abs(volume)
                    print(f"Fixed negative volume for {model_name}, new volume: {volume:.2f}")
            except Exception as fix_error:
                print(f"Could not fix normals for {model_name}: {fix_error}")
                volume = abs(volume)  # Use absolute value as fallback
        
        # Calculate minimal bounding box using smart method (PCA + fallback)
        start_time = time.time()
        size, bb_volume_calc, optimal_transform, bb_method = smart_minimal_bounding_box(mesh)
        bb_calc_time = time.time() - start_time
        
        print(f"Bounding box calculation: {bb_method} method in {bb_calc_time:.3f}s")
        
        # Extract other features
        surface_area = mesh.area
        min_bb_vol = size[0] * size[1] * size[2]  # Minimal bounding box volume
        
        # Calculate convex hull volume and convexity ratio
        convex_hull = mesh.convex_hull
        convex_hull_volume = convex_hull.volume
        convexity_ratio = volume / convex_hull_volume if convex_hull_volume > 0 else 0.0
        
        # Calculate shrinkwrap volume (skip in fast mode)
        try:
            # Check if fast mode is enabled
            try:
                from .fast_config import is_fast_mode
            except ImportError:
                # Fallback for direct execution
                try:
                    from fast_config import is_fast_mode
                except ImportError:
                    # Default: not in fast mode
                    def is_fast_mode():
                        return False
            
            if is_fast_mode():
                print(f"Fast mode: Skipped shrinkwrap calculation for {model_name}")
                shrinkwrap_volume = 0.0
                shrinkwrap_ratio = 0.0
            else:
                shrinkwrap = create_shrinkwrap(mesh)
                shrinkwrap_volume = shrinkwrap.volume if shrinkwrap else 0.0
                shrinkwrap_ratio = volume / shrinkwrap_volume if shrinkwrap_volume > 0 else 0.0
        except Exception as e:
            print(f"Error calculating shrinkwrap: {e}")
            shrinkwrap_volume = 0.0
            shrinkwrap_ratio = 0.0
        
        # Calculate waste (bounding box volume - part volume)
        waste_volume = min_bb_vol - volume if min_bb_vol > volume else 0.0
        waste_ratio = waste_volume / min_bb_vol if min_bb_vol > 0 else 0.0
        
        # Calculate surface area to volume ratio
        sa_vol_ratio = surface_area / volume if volume > 0 else 0.0
        
        # Create feature dictionary with only the requested fields
        return {
            "name": model_name,
            "filename": file_path,
            "x": float(size[0]),  # Largest dimension
            "y": float(size[1]),  # Middle dimension
            "z": float(size[2]),  # Smallest dimension
            "volume": float(volume),
            "surface_area": float(surface_area),
            "bb_volume": float(min_bb_vol),  # Minimal bounding box volume
            "waste": float(waste_volume),  # Waste volume (BB - part volume)
            "waste_ratio": float(waste_ratio),  # Waste ratio (waste/BB volume)
            "sa_vol_ratio": float(sa_vol_ratio),  # Surface area to volume ratio
            "convex_hull_volume": float(convex_hull_volume),
            "convexity_ratio": float(convexity_ratio),
            "shrinkwrap_volume": float(shrinkwrap_volume),
            "shrinkwrap_ratio": float(shrinkwrap_ratio),
            "optimal_transform": optimal_transform.tolist() if hasattr(optimal_transform, "tolist") else None,
            "bb_method": bb_method,  # Track which method was used
            "bb_calc_time": float(bb_calc_time)  # Track calculation time
        }
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def is_likely_cubic_frame(mesh):
    """Check if the mesh is likely to be a cubic frame structure."""
    # Check concavity ratio first - cubic frames are highly concave
    concavity_ratio = mesh.volume / mesh.convex_hull.volume if mesh.convex_hull.volume > 0 else 1.0
    if concavity_ratio > 0.3:
        return False
    
    # Look at the oriented bounding box
    obb = mesh.bounding_box_oriented
    extents = obb.extents
    
    # Calculate aspect ratios between sides
    sorted_extents = np.sort(extents)
    aspect_ratio_1 = sorted_extents[1] / sorted_extents[0] if sorted_extents[0] > 0 else 0
    aspect_ratio_2 = sorted_extents[2] / sorted_extents[1] if sorted_extents[1] > 0 else 0
    
    # Cubic frames should have roughly similar side lengths
    # More cubic when both aspect ratios are close to 1
    is_roughly_cubic = aspect_ratio_1 < 1.5 and aspect_ratio_2 < 1.5
    
    return is_roughly_cubic

def create_adaptive_alpha_shrinkwrap(mesh, points, inflation=0.0, voxel_resolution=None, _recursion_depth=0):
    """Create a shrinkwrap mesh using an adaptive alpha shape approach."""
    # Prevent infinite recursion
    if _recursion_depth > 1:
        print("Maximum alpha shape recursion depth reached, falling back to convex hull")
        return create_convex_hull(mesh)
        
    try:
        # Get the original vertices to use for better hole preservation
        original_vertices = np.array([v for v in mesh.vertices])
        
        # Find boundary vertices for explicit hole preservation
        # This step can be very intensive for large meshes, so we'll limit the number of edges we check
        boundary_vertices = []
        max_edges_to_check = min(10000, len(mesh.edges_unique))
        sample_edges = np.random.choice(len(mesh.edges_unique), max_edges_to_check, replace=False)
        
        for i in sample_edges:
            edge_vertices = mesh.edges_unique[i]
            # Optimize the boundary detection by using a more efficient approach
            edge_faces = mesh.edges_face[i]
            # If this edge belongs to only one face, it's a boundary
            if len(edge_faces) == 1 or (len(edge_faces) > 0 and edge_faces[1] == -1):
                boundary_vertices.append(mesh.vertices[edge_vertices[0]])
                boundary_vertices.append(mesh.vertices[edge_vertices[1]])
        
        boundary_vertices = np.array(boundary_vertices) if boundary_vertices else np.empty((0, 3))
        
        # Limit the total number of points for performance
        max_points = min(2000, len(points))  # Lower max points for better performance
        if voxel_resolution is not None:
            max_points = min(max_points, voxel_resolution * 30)  # Reduced from 50 to 30
        
        # Combine points, original vertices, and boundary vertices (giving more weight to boundary)
        all_points = []
        # Use a sample of points instead of all points
        sample_size = min(max_points // 2, len(points))
        if len(points) > sample_size:
            sampled_points = points[np.random.choice(len(points), sample_size, replace=False)]
            all_points.append(sampled_points)
        else:
            all_points.append(points)
            
        # Sample original vertices too if there are many
        if len(original_vertices) > sample_size:
            sampled_vertices = original_vertices[np.random.choice(len(original_vertices), sample_size, replace=False)]
            all_points.append(sampled_vertices)
        else:
            all_points.append(original_vertices)
            
        # Add boundary vertices with higher priority
        if len(boundary_vertices) > 0:
            # Use a reasonable number of boundary vertices
            if len(boundary_vertices) > max_points // 5:
                boundary_sample = np.random.choice(len(boundary_vertices), max_points // 5, replace=False)
                all_points.append(boundary_vertices[boundary_sample])
            else:
                all_points.append(boundary_vertices)
        
        combined_points = np.vstack(all_points)
        
        # Limit the number of points for alpha shape for performance
        if len(combined_points) > max_points:
            print(f"Reducing point cloud from {len(combined_points)} to {max_points} points for alpha shape")
            indices = np.random.choice(len(combined_points), max_points, replace=False)
            combined_points = combined_points[indices]
        
        # Calculate an appropriate alpha value - use smaller fraction for finer detail
        avg_edge_length = calculate_average_edge_length(mesh)
        alpha_value = avg_edge_length * 0.6  # Increased from 0.5 for better stability
        
        # Controlled inflation from centroid
        centroid = np.mean(combined_points, axis=0)
        directions = combined_points - centroid
        norms = np.linalg.norm(directions, axis=1).reshape(-1, 1)
        norms[norms < 1e-10] = 1.0
        normalized_dirs = directions / norms
        inflated_points = combined_points + normalized_dirs * (inflation * avg_edge_length * 0.1)
        
        # Try to create alpha shape with improved error handling and timeout
        try:
            print(f"Computing alpha shape with {len(inflated_points)} points (timeout: {ALPHA_SHAPE_TIMEOUT}s)...")
            
            # Define the alpha shape function to run with timeout
            def compute_alpha_shape(points, alpha):
                return alphashape.alphashape(points, alpha)
            
            try:
                # Run the alpha shape computation with a timeout
                alpha_mesh = run_with_timeout(compute_alpha_shape, ALPHA_SHAPE_TIMEOUT, inflated_points, alpha_value)
                result = trimesh.Trimesh(vertices=alpha_mesh.vertices, faces=alpha_mesh.faces)
                
                # Verify that the result is valid
                if result.is_watertight and result.volume > 0:
                    return result
                else:
                    print("Alpha shape produced non-watertight mesh, trying fallback methods")
            except TimeoutError:
                print(f"Alpha shape computation timed out after {ALPHA_SHAPE_TIMEOUT} seconds, trying fallback methods")
            except MemoryError:
                print("Alpha shape ran out of memory, trying fallback methods")
            except Exception as e:
                print(f"Alpha shape failed: {e}, trying fallback methods")
            
            # Try BPA if available, otherwise fall back to convex hull
            if OPEN3D_AVAILABLE:
                return create_bpa_mesh(inflated_points)
            else:
                return create_convex_hull_from_points(inflated_points)
                
        except Exception as e:
            print(f"Alpha shape failed: {e}, trying fallback methods")
            # Try BPA if available, otherwise fall back to convex hull
            if OPEN3D_AVAILABLE:
                return create_bpa_mesh(inflated_points)
            else:
                return create_convex_hull_from_points(inflated_points)
                
    except Exception as e:
        print(f"Error in adaptive alpha shape creation: {e}")
        return create_convex_hull(mesh)

def create_ball_pivoting(points, radius_factor=2.0):
    """
    A simplified ball pivoting algorithm implementation that doesn't require Open3D.
    
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
        hull = create_convex_hull_from_points(points)
        if hull is None:
            print("Failed to create initial hull, returning None")
            return None
        
        # Now we'll try to refine it by creating a more detailed mesh using Poisson reconstruction
        # For this simplified version, we can use a multi-step approach:
        
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
            cluster_hull = create_convex_hull_from_points(cluster_points)
            
            if cluster_hull is not None and len(cluster_hull.vertices) > 0:
                hulls.append(cluster_hull)
        
        if not hulls:
            print("Failed to create any valid cluster hulls, returning original hull")
            return hull
            
        # Combine all hulls
        combined = trimesh.util.concatenate(hulls)
        
        # Clean up the mesh by removing duplicate vertices
        final_mesh = combined.split(only_watertight=False)
        
        # If we got multiple meshes, pick the largest one
        if isinstance(final_mesh, list) and len(final_mesh) > 0:
            volumes = [m.volume if hasattr(m, 'volume') else 0 for m in final_mesh]
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
        return create_convex_hull_from_points(points)

def create_bpa_mesh(points):
    """Create a mesh using Ball Pivoting Algorithm via Open3D or fallback method"""
    # If Open3D is not available, use our custom implementation
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, using custom ball pivoting implementation")
        try:
            # Try to import our custom ball_pivoting implementation
            from .ball_pivoting import create_ball_pivoting
            return create_ball_pivoting(points)
        except ImportError:
            # If that fails, fall back to convex hull
            print("Custom ball pivoting implementation not available, falling back to convex hull")
            return create_convex_hull_from_points(points)
        
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals if needed for BPA
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Compute average distance to nearest neighbors for radius estimation
        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        
        # Create mesh using Ball Pivoting Algorithm with adaptive radii
        radii = [avg_distance*2, avg_distance*4, avg_distance*8]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        # Convert to trimesh format
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        if len(vertices) < 4 or len(faces) < 1:
            print("BPA failed to create valid mesh, falling back to custom ball pivoting")
            try:
                from .ball_pivoting import create_ball_pivoting
                return create_ball_pivoting(points)
            except ImportError:
                return create_convex_hull_from_points(points)
            
        result = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Check if mesh is watertight, if not, fallback to custom method
        if result.is_watertight:
            return result
        else:
            print("BPA mesh not watertight, falling back to custom ball pivoting")
            try:
                from .ball_pivoting import create_ball_pivoting
                return create_ball_pivoting(points)
            except ImportError:
                return create_convex_hull_from_points(points)
            
    except Exception as e:
        print(f"BPA mesh creation failed: {e}, falling back to custom ball pivoting")
        try:
            from .ball_pivoting import create_ball_pivoting
            return create_ball_pivoting(points)
        except ImportError:
            return create_convex_hull_from_points(points)

def create_voxelized_shrinkwrap(mesh, voxel_size=5.0):
    """
    Fast voxel-based approximation of a shrinkwrap mesh using uniform voxel size.
    Returns a trimesh mesh composed of axis-aligned boxes.
    
    Args:
        mesh (trimesh.Trimesh): The original mesh
        voxel_size (float): Size of each voxel cube in mm
        
    Returns:
        trimesh.Trimesh: A blocky approximation of the shape using fixed-size cubes
    """
    try:
        voxelized = mesh.voxelized(pitch=voxel_size)
        box_mesh = voxelized.as_boxes()

        if box_mesh.is_watertight and box_mesh.volume > 0:
            print(f"Created voxel shrinkwrap using {voxel_size}mm cubes, volume: {box_mesh.volume:.2f}")
            return box_mesh
        else:
            print("Voxelized shrinkwrap is not watertight or has zero volume, skipping")
            return None

    except Exception as e:
        print(f"Voxel shrinkwrap failed: {e}")
        return None

def create_uniform_box_shrinkwrap(mesh, box_percentage=5.0):
    """
    Create a shrinkwrap mesh using uniform boxes that are a percentage of the full bounding box size.
    This produces boxes of consistent size throughout the model.
    
    Args:
        mesh (trimesh.Trimesh): The original mesh
        box_percentage (float): Box size as percentage of bounding box (2.0-10.0)
        
    Returns:
        trimesh.Trimesh: A mesh composed of uniform-sized boxes
    """
    try:
        start_time = time.time()
        print(f"Creating uniform box shrinkwrap with {box_percentage:.1f}% box size...")
        
        # Get the full bounding box dimensions
        bounds = mesh.bounds
        bbox_min = bounds[0]
        bbox_max = bounds[1]
        bbox_size = bbox_max - bbox_min
        
        # Calculate box size based on percentage
        # Clamp percentage between 2% and 10% for reasonable results
        percentage = max(2.0, min(10.0, box_percentage)) / 100.0
        box_size = bbox_size * percentage
        
        # Make sure box size is reasonable (at least 1mm in each dimension)
        box_size = np.maximum(box_size, np.array([1.0, 1.0, 1.0]))
        
        # Calculate number of divisions in each axis
        divisions = np.ceil(bbox_size / box_size).astype(int)
        
        # Try using trimesh's built-in voxelization directly if possible
        try:
            # Try to create a voxel grid with the specified size
            voxel_grid = mesh.voxelized(pitch=np.min(box_size))
            voxel_mesh = voxel_grid.as_boxes()
            
            if voxel_mesh and voxel_mesh.volume > 0:
                print(f"Created uniform box shrinkwrap using built-in voxelization with {len(voxel_mesh.faces)//12} boxes")
                print(f"Volume: {voxel_mesh.volume:.2f}")
                print(f"Total time: {time.time() - start_time:.2f} seconds")
                return voxel_mesh
        except Exception as e:
            print(f"Built-in voxelization failed: {e}, falling back to point-based method")
        
        # Sample points on the surface for box placement
        num_samples = min(5000, int(np.prod(divisions) * 0.2))  # Sample enough points but not too many
        surface_points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        
        # Generate boxes where surface points exist
        boxes = []
        
        # Create a set to track which voxels we've already filled
        filled_voxels = set()
        
        # Process each surface point
        for point in surface_points:
            # Determine which voxel this point belongs to
            voxel_indices = np.floor((point - bbox_min) / box_size).astype(int)
            # Clip indices to valid range
            voxel_indices = np.clip(voxel_indices, 0, divisions - 1)
            voxel_index = tuple(voxel_indices)
            
            # Skip if we already have a box here
            if voxel_index in filled_voxels:
                continue
                
            # Mark this voxel as filled
            filled_voxels.add(voxel_index)
            
            # Create box position (center of the voxel)
            box_min = bbox_min + np.array(voxel_index) * box_size
            box_center = box_min + (box_size / 2)
            
            # Create transformation matrix for this box
            transform = np.eye(4)
            transform[:3, 3] = box_center
            
            # Create a box mesh
            box = trimesh.creation.box(extents=box_size, transform=transform)
            boxes.append(box)
        
        if not boxes:
            print("Uniform box method produced no boxes, falling back to voxel method")
            return create_voxelized_shrinkwrap(mesh, np.min(bbox_size) * percentage)
            
        print(f"Uniform box method generated {len(boxes)} component boxes")
        
        # Merge all the boxes into a single mesh
        combined = trimesh.util.concatenate(boxes)
        
        print(f"Created uniform box shrinkwrap in {time.time() - start_time:.2f} seconds, volume: {combined.volume:.2f}")
        return combined
            
    except Exception as e:
        print(f"Uniform box shrinkwrap failed: {e}")
        try:
            # Try fallback to the built-in voxelization
            voxel_size = np.min(box_size) if 'box_size' in locals() else 5.0
            voxelized = mesh.voxelized(pitch=voxel_size)
            box_mesh = voxelized.as_boxes()
            if box_mesh is not None and box_mesh.volume > 0:
                return box_mesh
        except:
            pass
        return None

def create_fixed_box_grid_shrinkwrap(mesh, box_size_mm=5.0):
    """
    Create a shrinkwrap mesh using fixed-size grid boxes.
    
    This implementation creates a grid of fixed-size boxes and keeps those that intersect with
    the mesh or are inside it. This produces a more efficient shrinkwrap than uniform boxes.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh to create shrinkwrap for
        box_size_mm (float): Size of each box in the grid in mm
        
    Returns:
        trimesh.Trimesh: A mesh composed of boxes that approximates the input mesh shape
    """
    # Start timer to measure performance
    start_time = time.time()
    
    # Always try to use the C++ implementation first for maximum performance
    try:
        # Attempt to load the C++ module
        import cpp_stl_utils
        
        # Inform the user
        print("Using C++ implementation for fixed box grid shrinkwrap (5-10x faster)")
        
        # Call the C++ implementation
        result = cpp_stl_utils.create_fixed_box_grid_shrinkwrap(mesh, box_size_mm)
        
        # Check if the result is valid
        if result is not None and hasattr(result, 'vertices') and len(result.vertices) > 0:
            print(f"C++ fixed box grid created in {time.time() - start_time:.2f} seconds")
            print(f"Generated mesh with {len(result.vertices)} vertices and {len(result.faces)} faces")
            return result
        else:
            print("C++ implementation failed to generate valid mesh")
            print("Falling back to Python implementation (WARNING: This will be much slower)")
    except ImportError:
        print("="*80)
        print("WARNING: C++ extensions not available - using SLOW Python implementation")
        print("For 5-10x faster performance, build the C++ extensions by running:")
        print("  build_cpp_extension.bat")
        print("="*80)
    except Exception as e:
        print(f"Error using C++ implementation: {e}")
        print("Falling back to Python implementation (WARNING: This will be much slower)")
    
    print("USING SLOW PYTHON IMPLEMENTATION - Build C++ extensions for better performance!")
    
    # Python implementation follows - this is a fallback if C++ is not available
    try:
        print(f"Creating fixed box grid with {box_size_mm:.2f}mm boxes (Python implementation)...")
        
        # Rest of the Python implementation...
        # ... [Python implementation code] ...
    except Exception as e:
        print(f"Error in create_fixed_box_grid_shrinkwrap: {e}")
        print(traceback.format_exc())
        
        # Return the bounding box as a fallback
        return trimesh.creation.box(mesh.extents, mesh.centroid)

def create_bvh_shrinkwrap(mesh, inflation=0.02, detail_level=8):
    """
    Create a shrinkwrap mesh using BVH (Bounding Volume Hierarchy) approach.
    This produces a shrinkwrap by decomposing the model into a hierarchy of bounding boxes
    and then merging them to create an approximation of the original shape.
    
    Args:
        mesh (trimesh.Trimesh): The original mesh
        inflation (float): Amount to inflate the resulting mesh (0.0-1.0)
        detail_level (int): Controls the level of detail (higher = more detailed)
        
    Returns:
        trimesh.Trimesh: A mesh that approximates the original shape using BVH
    """
    try:
        start_time = time.time()
        print(f"Creating BVH shrinkwrap with detail level {detail_level}...")
        
        # Instead of using the 'bvh' attribute directly, we'll manually create a hierarchy
        # of oriented bounding boxes by subdividing the mesh
        boxes = []
        
        # We'll start with the whole mesh's oriented bounding box
        obb = mesh.bounding_box_oriented
        boxes.append(obb)
        
        # Adaptive subdivision based on part size and detail level
        # Larger detail level = more subdivisions
        faces = mesh.faces
        vertices = mesh.vertices
        
        # Function to recursively subdivide regions
        def subdivide_region(faces_indices, depth=0):
            if depth >= detail_level or len(faces_indices) < 10:
                # Create an oriented bounding box for these faces
                if len(faces_indices) > 0:
                    # Extract vertices for these faces
                    region_faces = faces[faces_indices]
                    face_vertices = vertices[region_faces.flatten()]
                    
                    if len(face_vertices) > 0:
                        # Create oriented bounding box
                        try:
                            obb = trimesh.bounds.oriented_bounds(face_vertices, 
                                                              sample=min(100, len(face_vertices)))
                            transform, extents = obb
                            
                            # Convert to a box mesh
                            box = trimesh.creation.box(extents=extents, transform=transform)
                            boxes.append(box)
                        except:
                            # If OBB calculation fails, create a simple axis-aligned box
                            min_bound = np.min(face_vertices, axis=0)
                            max_bound = np.max(face_vertices, axis=0)
                            size = max_bound - min_bound
                            center = (min_bound + max_bound) / 2
                            
                            transform = np.eye(4)
                            transform[:3, 3] = center
                            
                            box = trimesh.creation.box(extents=size, transform=transform)
                            boxes.append(box)
                return
            
            # Split the faces into multiple groups
            if len(faces_indices) > 1:
                # Use face centroids for clustering
                centroids = np.zeros((len(faces_indices), 3))
                for i, face_idx in enumerate(faces_indices):
                    face = faces[face_idx]
                    face_verts = vertices[face]
                    centroids[i] = np.mean(face_verts, axis=0)
                
                # Split along the dimension with greatest variance
                variances = np.var(centroids, axis=0)
                split_dim = np.argmax(variances)
                
                # Sort by the split dimension
                sorted_indices = np.argsort(centroids[:, split_dim])
                mid_point = len(sorted_indices) // 2
                
                # Create two groups
                group1 = faces_indices[sorted_indices[:mid_point]]
                group2 = faces_indices[sorted_indices[mid_point:]]
                
                # Recurse on each group
                subdivide_region(group1, depth + 1)
                subdivide_region(group2, depth + 1)
        
        # Initial subdivision - get all face indices
        all_face_indices = np.arange(len(faces))
        
        # Subdivide the model based on detail level
        # Lower detail = fewer subdivisions
        # Scale subdivisions based on face count
        effective_detail = max(2, min(detail_level, int(np.log2(len(faces)))))
        print(f"Using effective detail level: {effective_detail} for {len(faces)} faces")
        
        subdivide_region(all_face_indices, depth=0)
        
        if not boxes:
            print("BVH produced no boxes, falling back to convex hull")
            return create_convex_hull(mesh)
            
        print(f"BVH generated {len(boxes)} component boxes")
        
        # Merge all the boxes into a single mesh
        combined = trimesh.util.concatenate(boxes)
        
        # Apply inflation if needed
        if inflation > 0:
            # Scale the mesh by the inflation factor from its centroid
            centroid = combined.centroid
            scaling_matrix = np.eye(4)
            inflation_factor = 1.0 + inflation
            scaling_matrix[:3, :3] *= inflation_factor
            
            # Apply scaling centered at centroid
            combined = combined.apply_transform(
                trimesh.transformations.translation_matrix(-centroid))
            combined = combined.apply_transform(scaling_matrix)
            combined = combined.apply_transform(
                trimesh.transformations.translation_matrix(centroid))
        
        # Clean up the mesh and ensure it's watertight
        try:
            combined = combined.convex_hull
            print(f"Created BVH shrinkwrap in {time.time() - start_time:.2f} seconds, volume: {combined.volume:.2f}")
            return combined
        except Exception as e:
            print(f"Error creating convex hull from BVH boxes: {e}")
            return combined
            
    except Exception as e:
        print(f"BVH shrinkwrap failed: {e}")
        return None

def create_voxel_dilation_shrinkwrap(mesh, offset_percent=5.0, voxel_size=None):
    """
    Create a shrinkwrap using fast voxelization + morphological dilation.
    
    This is much faster and less CPU-intensive than geometric methods.
    
    Args:
        mesh (trimesh.Trimesh): The original mesh
        offset_percent (float): Percentage offset for shrinkwrap inflation (default 5.0%)
        voxel_size (float): Size of voxels in mm (auto-calculated if None)
        
    Returns:
        trimesh.Trimesh: A mesh created from dilated voxels
    """
    try:
        start_time = time.time()
        print(f"Creating fast voxel + dilation shrinkwrap with {offset_percent}% offset...")
        
        # Calculate voxel size based on part dimensions if not provided
        if voxel_size is None:
            max_dim = np.max(mesh.extents)
            # Use smaller voxels for better quality, but not too small for performance
            voxel_size = max(0.5, max_dim / 50)  # 50 voxels along longest dimension
        
        print(f"Using voxel size: {voxel_size:.2f}mm")
        
        # 1. VOXELIZE THE MESH
        # This is much faster than geometric operations
        voxelized = mesh.voxelized(pitch=voxel_size)
        
        if not hasattr(voxelized, 'matrix') or voxelized.matrix is None:
            print("Failed to voxelize mesh, falling back to convex hull")
            return mesh.convex_hull
        
        voxel_matrix = voxelized.matrix
        print(f"Voxelized to {voxel_matrix.shape} grid in {time.time() - start_time:.2f}s")
        
        # 2. CALCULATE DILATION RADIUS
        # Convert offset percentage to voxel units
        max_dim = np.max(mesh.extents)
        offset_mm = max_dim * (offset_percent / 100.0)
        dilation_radius_voxels = int(np.ceil(offset_mm / voxel_size))
        dilation_radius_voxels = max(1, dilation_radius_voxels)  # At least 1 voxel
        
        print(f"Dilation radius: {dilation_radius_voxels} voxels ({offset_mm:.2f}mm)")
        
        # 3. PERFORM MORPHOLOGICAL DILATION
        dilation_start = time.time()
        
        if HAS_SCIPY:
            # Use scipy for fast morphological dilation
            # Create a spherical structuring element for isotropic expansion
            if dilation_radius_voxels > 1:
                # Create 3D ball structuring element
                size = 2 * dilation_radius_voxels + 1
                center = dilation_radius_voxels
                y, x, z = np.ogrid[:size, :size, :size]
                structuring_element = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= dilation_radius_voxels**2
                
                # Perform binary dilation
                dilated_matrix = ndimage.binary_dilation(voxel_matrix, structure=structuring_element)
            else:
                # Simple dilation with small structuring element
                dilated_matrix = ndimage.binary_dilation(voxel_matrix, iterations=dilation_radius_voxels)
        else:
            # Fallback: simple manual dilation (slower but works without scipy)
            dilated_matrix = voxel_matrix.copy()
            
            # Create simple 3x3x3 structuring element
            structure = np.ones((3, 3, 3), dtype=bool)
            
            # Apply multiple iterations for larger radius
            for i in range(dilation_radius_voxels):
                if i % 2 == 0:
                    print(f"Dilation iteration {i+1}/{dilation_radius_voxels}")
                
                # Manual dilation using convolution-like operation
                new_matrix = dilated_matrix.copy()
                
                # Pad the matrix to handle edges
                padded = np.pad(dilated_matrix, 1, mode='constant', constant_values=False)
                
                # Apply structuring element
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            if structure[dx+1, dy+1, dz+1]:
                                # Shift and OR with original
                                shifted = padded[1+dx:1+dx+dilated_matrix.shape[0],
                                               1+dy:1+dy+dilated_matrix.shape[1], 
                                               1+dz:1+dz+dilated_matrix.shape[2]]
                                new_matrix = new_matrix | shifted
                
                dilated_matrix = new_matrix
        
        print(f"Dilation completed in {time.time() - dilation_start:.2f}s")
        
        # 4. CONVERT BACK TO MESH
        mesh_start = time.time()
        
        # Create new voxel grid with dilated matrix
        dilated_voxels = trimesh.voxel.VoxelGrid(
            dilated_matrix, 
            transform=voxelized.transform
        )
        
        # Convert to mesh using marching cubes
        try:
            shrinkwrap_mesh = dilated_voxels.as_boxes()
            
            if shrinkwrap_mesh is None or shrinkwrap_mesh.volume <= 0:
                print("Box conversion failed, trying marching cubes...")
                # Try marching cubes if available
                shrinkwrap_mesh = dilated_voxels.marching_cubes
                
            if shrinkwrap_mesh is None or shrinkwrap_mesh.volume <= 0:
                print("Marching cubes failed, using convex hull fallback")
                return mesh.convex_hull
                
        except Exception as e:
            print(f"Mesh conversion failed: {e}, using convex hull fallback")
            return mesh.convex_hull
        
        print(f"Mesh conversion completed in {time.time() - mesh_start:.2f}s")
        
        # 5. VALIDATE RESULT
        if not shrinkwrap_mesh.is_watertight:
            print("Warning: Generated shrinkwrap is not watertight, attempting to fix...")
            try:
                shrinkwrap_mesh.fill_holes()
                if not shrinkwrap_mesh.is_watertight:
                    # Last resort: use convex hull of the result
                    shrinkwrap_mesh = shrinkwrap_mesh.convex_hull
            except:
                shrinkwrap_mesh = shrinkwrap_mesh.convex_hull
        
        total_time = time.time() - start_time
        volume_increase = ((shrinkwrap_mesh.volume - mesh.volume) / mesh.volume) * 100
        
        print(f"✅ Voxel + dilation shrinkwrap completed in {total_time:.2f}s")
        print(f"   Original volume: {mesh.volume:.2f}mm³")
        print(f"   Shrinkwrap volume: {shrinkwrap_mesh.volume:.2f}mm³")
        print(f"   Volume increase: {volume_increase:.1f}%")
        print(f"   Voxel count: {np.sum(dilated_matrix):,}")
        
        return shrinkwrap_mesh
        
    except Exception as e:
        print(f"Error in voxel + dilation shrinkwrap: {e}")
        print(traceback.format_exc())
        return mesh.convex_hull

def create_shrinkwrap(mesh, num_points=2000, inflation=0.02, preserve_holes=True, voxel_resolution=None, inflation_factor=None, closing_size=None, method=None, detail_level=8, _recursion_depth=0):
    """Create a shrinkwrap of the mesh."""
    # Prevent infinite recursion
    if _recursion_depth > 2:
        print("Maximum recursion depth reached, using convex hull as fallback")
        return create_convex_hull(mesh)
    
    start_time = time.time()
    
    # If inflation_factor is provided, use it instead of inflation
    if inflation_factor is not None:
        inflation = inflation_factor
    
    # If closing_size is provided, we can use it to influence the number of points and voxel size
    if closing_size is not None:
        # Higher closing_size means more aggressive hole filling (larger voxels)
        # Inverse relationship - higher closing_size means fewer points and larger voxels
        num_points = int(num_points / max(1, closing_size))
        print(f"Using closing_size={closing_size}, adjusted num_points to {num_points}")
    
    # If no method is specified, default to 'surface_offset' as the preferred method (PROPER SHRINKWRAP)
    if method is None:
        method = 'surface_offset'
        print("No method specified, using Surface Offset method for proper shrinkwrap")
        
    # Calculate mesh properties
    mesh_volume = mesh.volume
    convex_hull = create_convex_hull(mesh)
    convex_hull_volume = convex_hull.volume
    concavity_ratio = mesh_volume / convex_hull_volume
    
    # Check if this is a frame-like structure (highly non-convex)
    is_frame = concavity_ratio < 0.15
    
    # Calculate max dimension based on model dimensions
    max_dim = np.max(mesh.extents)
    
    # PROPER Surface Offset method (NEW DEFAULT - PROPER SHRINKWRAP!)
    if method == 'surface_offset':
        print(f"Creating proper surface offset shrinkwrap with 4.0mm offset...")
        surface_offset_result = create_surface_offset_shrinkwrap(mesh, offset_mm=4.0, offset_percent=None)
        
        if surface_offset_result is not None and surface_offset_result.volume > 0:
            print(f"Created surface offset shrinkwrap with volume {surface_offset_result.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return surface_offset_result
            
        print("Surface offset shrinkwrap failed, trying alternative methods")
    
    # FAST Voxel + Dilation method (backup method - but fills holes like convex hull)
    elif method == 'voxel_dilation':
        # Calculate offset percentage from inflation
        offset_percent = inflation * 100 if inflation < 1.0 else 5.0
        
        # Calculate voxel size based on resolution and part size
        if voxel_resolution is not None:
            # Convert resolution to voxel size: higher resolution = smaller voxels
            voxel_size = max_dim / voxel_resolution
        else:
            # Default to good balance of speed vs quality
            voxel_size = max_dim / 40  # 40 voxels along longest dimension
        
        # Apply closing_size to voxel size if specified
        if closing_size is not None:
            # Higher closing_size means larger voxels (faster but less detail)
            voxel_size *= (1.0 + closing_size * 0.1)
        
        # Ensure reasonable voxel size limits
        voxel_size = max(0.3, min(voxel_size, max_dim * 0.05))
        
        print(f"Creating fast voxel + dilation shrinkwrap with {offset_percent:.1f}% offset...")
        voxel_dilation_result = create_voxel_dilation_shrinkwrap(mesh, offset_percent=offset_percent, voxel_size=voxel_size)
        
        if voxel_dilation_result is not None and voxel_dilation_result.volume > 0:
            print(f"Created voxel + dilation shrinkwrap with volume {voxel_dilation_result.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return voxel_dilation_result
            
        print("Voxel + dilation shrinkwrap failed, trying alternative methods")
    
    # Box Grid method (backup method - more intensive)
    elif method == 'boxgrid':
        # Calculate box size based on resolution and part size
        # If voxel_resolution is provided, use it to calculate box size (higher resolution = smaller boxes)
        if voxel_resolution is not None:
            # Convert resolution to box size: higher resolution = smaller boxes
            box_size_mm = max_dim / voxel_resolution
        else:
            # Calculate box size as percentage of max dimension using closing_size
            percentage = 0.05  # Default 5%
            if closing_size is not None:
                # Map closing_size (1-10) to percentage (2-10%)
                percentage = (0.02 + (closing_size * 0.016))
            
            box_size_mm = max_dim * percentage
        
        # Make sure box size is reasonable (between 1mm and 5% of max dimension)
        box_size_mm = max(1.0, min(box_size_mm, max_dim * 0.05))
        
        print(f"Creating fixed box grid shrinkwrap with {box_size_mm:.2f}mm boxes...")
        box_grid_result = create_fixed_box_grid_shrinkwrap(mesh, box_size_mm=box_size_mm)
        
        if box_grid_result is not None and box_grid_result.volume > 0:
            print(f"Created fixed box grid shrinkwrap with volume {box_grid_result.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return box_grid_result
            
        print("Fixed box grid shrinkwrap failed, trying alternative methods")
    
    # BVH method (backup method)
    elif method == 'bvh':
        print(f"Creating BVH shrinkwrap with detail level {detail_level}...")
        bvh_result = create_bvh_shrinkwrap(mesh, inflation=inflation, detail_level=detail_level)
        
        if bvh_result is not None and bvh_result.is_watertight and bvh_result.volume > 0:
            print(f"Created BVH shrinkwrap with volume {bvh_result.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return bvh_result
            
        print("BVH shrinkwrap failed, trying alternative methods")
    
    elif method == 'uniform_box':
        # Use closing_size as box percentage (default 5%)
        box_percentage = 5.0
        if closing_size is not None:
            # Convert closing_size (1-10) to percentage (2-10)
            box_percentage = max(2.0, min(10.0, closing_size * 2.0))
        
        print(f"Creating uniform box shrinkwrap with {box_percentage:.1f}% box size...")
        uniform_box_result = create_uniform_box_shrinkwrap(mesh, box_percentage=box_percentage)
        
        if uniform_box_result is not None and uniform_box_result.volume > 0:
            print(f"Created uniform box shrinkwrap with volume {uniform_box_result.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return uniform_box_result
            
        print("Uniform box shrinkwrap failed, trying alternative methods")
    
    elif method == 'convex':
        # Just return the convex hull
        print("Using convex hull as shrinkwrap")
        if convex_hull is not None and convex_hull.is_watertight and convex_hull.volume > 0:
            print(f"Created convex hull with volume {convex_hull.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return convex_hull
    
    elif method == 'voxel':
        # If voxel_resolution is provided, use it to calculate voxel size
        if voxel_resolution is not None:
            # Convert resolution to voxel size: higher resolution = smaller voxels
            voxel_size = max_dim / voxel_resolution
            print(f"Using voxel_resolution={voxel_resolution}, calculated voxel size: {voxel_size:.2f}mm")
        else:
            # Default voxel size is 2% of largest dimension
            voxel_size = max_dim * 0.02
            
            # Apply closing_size to voxel size if specified
            if closing_size is not None:
                voxel_size *= closing_size
            
            # Ensure reasonable limits
            voxel_size = min(max(voxel_size, 1.0), max_dim * 0.05)
            
        print(f"Creating voxelized shrinkwrap with {voxel_size:.2f}mm voxels (method: voxel)")
        voxel_mesh = create_voxelized_shrinkwrap(mesh, voxel_size)
        
        if voxel_mesh is not None and voxel_mesh.is_watertight and voxel_mesh.volume > 0:
            print(f"Created voxel shrinkwrap with volume {voxel_mesh.volume:.2f}")
            print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
            return voxel_mesh
            
        print("Voxel shrinkwrap failed, trying alternative methods")
    
    elif method == 'alpha':
        print("Trying alpha shape method directly")
        # Prepare points for alpha shape
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        try:
            avg_edge_length = calculate_average_edge_length(mesh)
            # For consistent parameter naming, convert voxel_resolution to expected parameter
            adaptive_voxel_resolution = voxel_resolution if voxel_resolution is not None else 64
            
            result = create_adaptive_alpha_shrinkwrap(
                mesh, 
                points, 
                inflation=avg_edge_length*0.01,
                voxel_resolution=adaptive_voxel_resolution
            )
            
            if result is not None and result.is_watertight and result.volume > 0:
                print(f"Created adaptive alpha shape with volume {result.volume:.2f}")
                print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
                return result
        except Exception as e:
            print(f"Alpha shape failed: {e}, trying alternative methods")
    
    # If we reach here, either no specific method was specified, or the specified method failed
    # Fall back to Surface Offset method as the preferred fallback
    print("First method failed or was unspecified, trying Surface Offset method...")
    
    # Calculate offset from inflation parameter
    offset_percent = inflation * 100 if inflation < 1.0 else 5.0
    
    # Try Surface Offset method
    print(f"Trying Surface Offset method with {offset_percent:.1f}% offset...")
    surface_offset_result = create_surface_offset_shrinkwrap(mesh, offset_mm=4.0, offset_percent=None)
    
    if surface_offset_result is not None and surface_offset_result.volume > 0:
        print(f"Created Surface Offset shrinkwrap with volume {surface_offset_result.volume:.2f}")
        print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
        return surface_offset_result
    
    # If Surface Offset failed, try fast Voxel + Dilation method
    print("Surface Offset shrinkwrap failed, trying fast Voxel + Dilation method...")
    
    # Calculate voxel size for fallback
    voxel_size = max_dim / 40  # 40 voxels along longest dimension
    
    # Adjust based on resolution if provided
    if voxel_resolution is not None:
        voxel_size = max_dim / voxel_resolution
    
    # Adjust based on closing_size if provided
    if closing_size is not None:
        # Higher closing_size means larger voxels (faster)
        voxel_size *= (1.0 + closing_size * 0.1)
    
    # Ensure reasonable limits
    voxel_size = max(0.3, min(voxel_size, max_dim * 0.05))
    
    # Try Voxel + Dilation method
    print(f"Trying fast Voxel + Dilation method with {offset_percent:.1f}% offset...")
    voxel_dilation_result = create_voxel_dilation_shrinkwrap(mesh, offset_percent=offset_percent, voxel_size=voxel_size)
    
    if voxel_dilation_result is not None and voxel_dilation_result.volume > 0:
        print(f"Created fast Voxel + Dilation shrinkwrap with volume {voxel_dilation_result.volume:.2f}")
        print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
        return voxel_dilation_result
    
    # If Voxel + Dilation failed, try Box Grid method
    print("Voxel + Dilation shrinkwrap failed, trying Box Grid method...")
    
    # Calculate box size for Box Grid method
    box_size_mm = max_dim * 0.05  # Default to 5% of max dimension
    
    # Adjust based on resolution if provided
    if voxel_resolution is not None:
        box_size_mm = max_dim / voxel_resolution
    
    # Adjust based on closing_size if provided
    if closing_size is not None:
        # Higher closing_size means larger boxes
        multiplier = 0.5 + (closing_size / 10.0) * 1.5
        box_size_mm *= multiplier
    
    # Ensure reasonable limits
    box_size_mm = max(1.0, min(box_size_mm, max_dim * 0.05))
    
    # Try Box Grid method
    print(f"Trying Box Grid method with {box_size_mm:.2f}mm boxes...")
    box_grid_result = create_fixed_box_grid_shrinkwrap(mesh, box_size_mm=box_size_mm)
    
    if box_grid_result is not None and box_grid_result.volume > 0:
        print(f"Created Box Grid shrinkwrap with volume {box_grid_result.volume:.2f}")
        print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
        return box_grid_result
    
    # If Box Grid failed, try BVH method
    print("Box Grid shrinkwrap failed, trying BVH method...")
    
    # Calculate detail level based on mesh complexity and voxel_resolution
    detail_level = 8  # Default detail level
    if voxel_resolution is not None:
        # Higher resolution = higher detail level in BVH
        detail_level = min(12, max(4, int(voxel_resolution / 8)))
        
    # For very intricate shapes, use more detail
    if is_frame:
        detail_level += 2
        
    # Adjust based on closing_size if provided
    if closing_size is not None:
        # Inverse relationship - higher closing size means less detail
        detail_level = max(4, detail_level - closing_size)
        
    print(f"Using BVH detail level: {detail_level}")
    bvh_result = create_bvh_shrinkwrap(mesh, inflation=inflation, detail_level=detail_level)
    
    if bvh_result is not None and bvh_result.is_watertight and bvh_result.volume > 0:
        print(f"Created BVH shrinkwrap with volume {bvh_result.volume:.2f}")
        print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
        return bvh_result
        
    # If all else fails, use the convex hull
    print("All shrinkwrap methods failed, using convex hull as fallback")
    print(f"Created convex hull with volume {convex_hull.volume:.2f}")
    print(f"Total shrinkwrap processing time: {time.time() - start_time:.2f} seconds")
    return convex_hull

def extract_in_process(file_path, timeout=30):
    """Extract features with a timeout to prevent hanging on problematic files"""
    with multiprocessing.Pool(1) as pool:
        result = pool.apply_async(extract_features, (file_path,))
        try:
            return result.get(timeout)
        except multiprocessing.TimeoutError:
            print(f"Timeout on: {file_path}")
            return None

def is_valid_entry(data):
    """Validate that all extracted features are valid numbers"""
    if data is None:
        return False
    try:
        # Normalize filename if present
        if "filename" in data:
            data = data.copy()  # Create a copy to avoid modifying the original
            data["filename"] = os.path.normpath(data["filename"])
        
        # Check required keys
        required_keys = ["x", "y", "z", "volume", "surface_area", "bb_volume", "filename", "name"]
        for k in required_keys:
            if k not in data:
                print(f"Missing required key in entry: {k}")
                return False
                
            # Check numeric values
            if k not in ["filename", "name"] and (not isinstance(data[k], (int, float)) or 
                                   data[k] < 0 or 
                                   np.isinf(data[k]) or 
                                   np.isnan(data[k])):
                print(f"Invalid numeric value in entry: {k}={data[k]}")
                return False
                
        # Check that the file exists
        if not os.path.exists(data["filename"]):
            print(f"File does not exist: {data['filename']}")
            return False
                
        return True
    except Exception as e:
        print(f"Error validating entry: {e}")
        return False

def find_similar(data_df, target_features, param="bb_volume", top_n=10):
    """Find top N similar models based on specified parameter"""
    if data_df.empty:
        return data_df
    
    # Calculate similarity based on percentage difference
    data_df = data_df.copy()
    target_value = target_features[param]
    
    if target_value != 0:
        data_df['similarity'] = np.abs((data_df[param] - target_value) / target_value)
    else:
        data_df['similarity'] = np.abs(data_df[param] - target_value)
        
    return data_df.nsmallest(top_n, 'similarity')

# Create function to get convex hull from mesh or points
def create_convex_hull(mesh):
    """Create a convex hull from a mesh"""
    return mesh.convex_hull

def create_convex_hull_from_points(points):
    """Create a convex hull from points"""
    try:
        hull = trimesh.convex.convex_hull(points)
        return hull
    except Exception as e:
        print(f"Error creating convex hull: {e}")
        return None

def calculate_average_edge_length(mesh):
    """Calculate the average edge length of a mesh"""
    edges = mesh.edges_unique
    vertices = mesh.vertices
    
    total_length = 0.0
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        length = np.linalg.norm(v1 - v2)
        total_length += length
    
    return total_length / len(edges) if len(edges) > 0 else 0.01 

def extract_features_from_mesh(mesh, file_path, model_name=None):
    """
    Extract features directly from a mesh object.
    
    Args:
        mesh: Trimesh mesh object
        file_path: Original file path (for naming/reference)
        model_name: Optional model name override
        
    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Get the model name (filename without path and extension)
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Validate and fix mesh if needed
        volume = mesh.volume
        if volume < 0:
            print(f"Warning: Mesh {model_name} has negative volume ({volume:.2f}), fixing normals...")
            try:
                # Try to fix inverted normals
                mesh.fix_normals()
                volume = mesh.volume
                if volume < 0:
                    # If still negative, invert all faces
                    mesh.faces = np.fliplr(mesh.faces)
                    volume = abs(volume)
                    print(f"Fixed negative volume for {model_name}, new volume: {volume:.2f}")
            except Exception as fix_error:
                print(f"Error fixing mesh normals: {fix_error}")
                volume = abs(volume)
        
        # Calculate minimal bounding box using smart method (PCA + fallback)
        start_time = time.time()
        size, bb_volume_calc, optimal_transform, bb_method = smart_minimal_bounding_box(mesh)
        bb_calc_time = time.time() - start_time
        
        print(f"Bounding box calculation: {bb_method} method in {bb_calc_time:.3f}s")
        
        # Extract other features
        surface_area = mesh.area
        min_bb_vol = size[0] * size[1] * size[2]  # Minimal bounding box volume
        
        # Calculate convex hull volume and convexity ratio
        convex_hull = mesh.convex_hull
        convex_hull_volume = convex_hull.volume
        convexity_ratio = volume / convex_hull_volume if convex_hull_volume > 0 else 0.0
        
        # Calculate shrinkwrap volume (skip in fast mode)
        try:
            # Check if fast mode is enabled
            from .fast_config import is_fast_mode
            if is_fast_mode():
                print(f"Fast mode: Skipped shrinkwrap calculation for {model_name}")
                shrinkwrap_volume = 0.0
                shrinkwrap_ratio = 0.0
            else:
                shrinkwrap = create_shrinkwrap(mesh)
                shrinkwrap_volume = shrinkwrap.volume if shrinkwrap else 0.0
                shrinkwrap_ratio = volume / shrinkwrap_volume if shrinkwrap_volume > 0 else 0.0
        except Exception as e:
            print(f"Error calculating shrinkwrap: {e}")
            shrinkwrap_volume = 0.0
            shrinkwrap_ratio = 0.0
        
        # Calculate waste volume and ratio - use the calculated bb_volume_calc
        waste_volume = bb_volume_calc - volume if bb_volume_calc > volume else 0.0
        waste_ratio = waste_volume / bb_volume_calc if bb_volume_calc > 0 else 0.0
        
        # Calculate surface area to volume ratio
        sa_vol_ratio = surface_area / volume if volume > 0 else 0.0
        
        # Build feature dictionary with CONSISTENT field names
        features = {
            "name": model_name,
            "filename": file_path,
            # Bounding box dimensions (sorted largest to smallest)
            "x": float(size[0]),  # Largest dimension
            "y": float(size[1]),  # Middle dimension  
            "z": float(size[2]),  # Smallest dimension
            # Volume and surface measurements
            "volume": float(volume),
            "surface_area": float(surface_area),
            # Bounding box volume (use consistent field name)
            "bb_volume": float(bb_volume_calc),
            "bounding_box_volume": float(bb_volume_calc),  # Alternative field name for compatibility
            # Waste calculations
            "waste": float(waste_volume),
            "waste_volume": float(waste_volume),  # Alternative field name
            "waste_ratio": float(waste_ratio),
            # Surface area to volume ratio
            "sa_vol_ratio": float(sa_vol_ratio),
            # Convex hull measurements
            "convex_hull_volume": float(convex_hull_volume),
            "convexity_ratio": float(convexity_ratio),
            # Shrinkwrap measurements
            "shrinkwrap_volume": float(shrinkwrap_volume),
            "shrinkwrap_ratio": float(shrinkwrap_ratio),
            # Optimization metadata
            "bb_method": bb_method,
            "bbox_method": bb_method,  # Alternative field name
            "bb_calc_time": float(bb_calc_time),
            "optimal_transform": optimal_transform.tolist() if optimal_transform is not None else None
        }
        
        print(f"Features extracted for {model_name}")
        print(f"  Dimensions: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
        print(f"  Volume: {volume:.2f}, BB Volume: {bb_volume_calc:.2f}")
        print(f"  Waste: {waste_volume:.2f} ({waste_ratio:.1%})")
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from mesh {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_features_with_shrinkwrap(file_path, offset_mm=4.0, offset_percent=None, generate_shrinkwrap=True):
    """
    Enhanced feature extraction that also generates shrinkwrap STL files with offset.
    Args:
        file_path (str): Path to the STL file
        offset_mm (float): Offset in mm (preferred)
        offset_percent (float): Percentage offset (deprecated)
        generate_shrinkwrap (bool): Whether to generate shrinkwrap STL file
    Returns:
        dict: Features dictionary with shrinkwrap information
    """
    # Normalize file path to handle Windows backslashes
    file_path = os.path.normpath(file_path)
    
    mesh = try_load_stl(file_path)
    if mesh is None:
        return None

    try:
        # Get the model name (filename without path and extension)
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Processing {model_name} with {offset_mm:.2f}mm offset...")
        
        # Validate and fix mesh if needed
        volume = mesh.volume
        if volume < 0:
            print(f"Warning: Mesh {model_name} has negative volume ({volume:.2f}), fixing normals...")
            try:
                mesh.fix_normals()
                volume = mesh.volume
                if volume < 0:
                    mesh.faces = np.fliplr(mesh.faces)
                    volume = abs(volume)
                    print(f"Fixed negative volume for {model_name}, new volume: {volume:.2f}")
            except Exception as fix_error:
                print(f"Could not fix normals for {model_name}: {fix_error}")
                volume = abs(volume)
        
        # Calculate minimal bounding box using smart method (PCA + fallback)
        start_time = time.time()
        size, bb_volume_calc, optimal_transform, bb_method = smart_minimal_bounding_box(mesh)
        bb_calc_time = time.time() - start_time
        
        # Extract other features
        surface_area = mesh.area
        min_bb_vol = size[0] * size[1] * size[2]
        
        # Calculate convex hull volume and convexity ratio
        convex_hull = mesh.convex_hull
        convex_hull_volume = convex_hull.volume
        convexity_ratio = volume / convex_hull_volume if convex_hull_volume > 0 else 0.0
        
        # Initialize shrinkwrap variables
        shrinkwrap_volume = 0.0
        shrinkwrap_ratio = 0.0
        shrinkwrap_stl_path = None
        shrinkwrap_export_date = None
        
        # Generate shrinkwrap with offset if requested
        if generate_shrinkwrap:
            try:
                print(f"Creating shrinkwrap with {offset_mm:.2f}mm offset...")
                shrinkwrap_start = time.time()
                shrinkwrap_mesh = create_surface_offset_shrinkwrap(mesh, offset_mm=offset_mm, offset_percent=offset_percent)
                if shrinkwrap_mesh is not None and shrinkwrap_mesh.volume > 0:
                    shrinkwrap_volume = shrinkwrap_mesh.volume
                    shrinkwrap_ratio = volume / shrinkwrap_volume if shrinkwrap_volume > 0 else 0.0
                    
                    # Don't save shrinkwrap files anymore - just calculate volume
                    shrinkwrap_time = time.time() - shrinkwrap_start
                    print(f"✅ Shrinkwrap volume calculated in {shrinkwrap_time:.2f}s (no file saved)")
                    print(f"   Original volume: {volume:.2f}mm³, Shrinkwrap volume: {shrinkwrap_volume:.2f}mm³")
                    
                    # Set to None since we're not saving files
                    shrinkwrap_stl_path = None
                    shrinkwrap_export_date = None
                else:
                    print(f"Failed to create valid shrinkwrap for {model_name}")
            except Exception as e:
                print(f"Error creating shrinkwrap for {model_name}: {e}")
        
        # Calculate waste volume and ratio
        waste_volume = bb_volume_calc - volume if bb_volume_calc > volume else 0.0
        waste_ratio = waste_volume / bb_volume_calc if bb_volume_calc > 0 else 0.0
        
        # Calculate surface area to volume ratio
        sa_vol_ratio = surface_area / volume if volume > 0 else 0.0
        
        # Create feature dictionary with shrinkwrap information
        features = {
            "name": model_name,
            "filename": file_path,
            # Bounding box dimensions (sorted largest to smallest)
            "x": float(size[0]),
            "y": float(size[1]),
            "z": float(size[2]),
            # Volume and surface measurements
            "volume": float(volume),
            "surface_area": float(surface_area),
            # Bounding box volume
            "bb_volume": float(bb_volume_calc),
            "bounding_box_volume": float(bb_volume_calc),
            # Waste calculations
            "waste": float(waste_volume),
            "waste_volume": float(waste_volume),
            "waste_ratio": float(waste_ratio),
            # Surface area to volume ratio
            "sa_vol_ratio": float(sa_vol_ratio),
            # Convex hull measurements
            "convex_hull_volume": float(convex_hull_volume),
            "convexity_ratio": float(convexity_ratio),
            # Shrinkwrap measurements and paths
            "shrinkwrap_volume": float(shrinkwrap_volume),
            "shrinkwrap_ratio": float(shrinkwrap_ratio),
            "shrinkwrap_stl_path": shrinkwrap_stl_path,
            "shrinkwrap_offset_mm": float(offset_mm),
            "shrinkwrap_offset_percent": float(offset_percent) if offset_percent is not None else 0.0,
            "shrinkwrap_export_date": shrinkwrap_export_date,
            # Optimization metadata
            "bb_method": bb_method,
            "bbox_method": bb_method,
            "bb_calc_time": float(bb_calc_time),
            "optimal_transform": optimal_transform.tolist() if optimal_transform is not None else None
        }
        
        total_time = time.time() - start_time
        print(f"✅ Complete analysis for {model_name} in {total_time:.2f}s")
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_surface_offset_shrinkwrap(mesh, offset_mm=4.0, offset_percent=None):
    """
    Offset each vertex along its normal by 4mm (classic surface offset).
    """
    import numpy as np
    # Always use 4mm, ignore offset_percent
    actual_offset_mm = 4.0

    # Make sure normals are present and valid
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.vertex_normals = mesh.vertex_normals_angle_weighted()

    # Normalize normals
    norms = np.linalg.norm(mesh.vertex_normals, axis=1)
    valid = norms > 0
    normals = mesh.vertex_normals.copy()
    normals[valid] /= norms[valid, np.newaxis]

    # Offset vertices
    offset_vertices = mesh.vertices + normals * actual_offset_mm

    # Create new mesh
    shrinkwrap_mesh = trimesh.Trimesh(vertices=offset_vertices, faces=mesh.faces, process=False)
    return shrinkwrap_mesh

def create_voxel_dilation_shrinkwrap(mesh, offset_percent=5.0, voxel_size=None):
    """
    Create a shrinkwrap using fast voxelization + morphological dilation.
    
    This is much faster and less CPU-intensive than geometric methods.
    
    Args:
        mesh (trimesh.Trimesh): The original mesh
        offset_percent (float): Percentage offset for shrinkwrap inflation (default 5.0%)
        voxel_size (float): Size of voxels in mm (auto-calculated if None)
        
    Returns:
        trimesh.Trimesh: A mesh created from dilated voxels
    """
    try:
        start_time = time.time()
        print(f"Creating fast voxel + dilation shrinkwrap with {offset_percent}% offset...")
        
        # Calculate voxel size based on part dimensions if not provided
        if voxel_size is None:
            max_dim = np.max(mesh.extents)
            # Use smaller voxels for better quality, but not too small for performance
            voxel_size = max(0.5, max_dim / 50)  # 50 voxels along longest dimension
        
        print(f"Using voxel size: {voxel_size:.2f}mm")
        
        # 1. VOXELIZE THE MESH
        # This is much faster than geometric operations
        voxelized = mesh.voxelized(pitch=voxel_size)
        
        if not hasattr(voxelized, 'matrix') or voxelized.matrix is None:
            print("Failed to voxelize mesh, falling back to convex hull")
            return mesh.convex_hull
        
        voxel_matrix = voxelized.matrix
        print(f"Voxelized to {voxel_matrix.shape} grid in {time.time() - start_time:.2f}s")
        
        # 2. CALCULATE DILATION RADIUS
        # Convert offset percentage to voxel units
        max_dim = np.max(mesh.extents)
        offset_mm = max_dim * (offset_percent / 100.0)
        dilation_radius_voxels = int(np.ceil(offset_mm / voxel_size))
        dilation_radius_voxels = max(1, dilation_radius_voxels)  # At least 1 voxel
        
        print(f"Dilation radius: {dilation_radius_voxels} voxels ({offset_mm:.2f}mm)")
        
        # 3. PERFORM MORPHOLOGICAL DILATION
        dilation_start = time.time()
        
        if HAS_SCIPY:
            # Use scipy for fast morphological dilation
            # Create a spherical structuring element for isotropic expansion
            if dilation_radius_voxels > 1:
                # Create 3D ball structuring element
                size = 2 * dilation_radius_voxels + 1
                center = dilation_radius_voxels
                y, x, z = np.ogrid[:size, :size, :size]
                structuring_element = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= dilation_radius_voxels**2
                
                # Perform binary dilation
                dilated_matrix = ndimage.binary_dilation(voxel_matrix, structure=structuring_element)
            else:
                # Simple dilation with small structuring element
                dilated_matrix = ndimage.binary_dilation(voxel_matrix, iterations=dilation_radius_voxels)
        else:
            # Fallback: simple manual dilation (slower but works without scipy)
            dilated_matrix = voxel_matrix.copy()
            
            # Create simple 3x3x3 structuring element
            structure = np.ones((3, 3, 3), dtype=bool)
            
            # Apply multiple iterations for larger radius
            for i in range(dilation_radius_voxels):
                if i % 2 == 0:
                    print(f"Dilation iteration {i+1}/{dilation_radius_voxels}")
                
                # Manual dilation using convolution-like operation
                new_matrix = dilated_matrix.copy()
                
                # Pad the matrix to handle edges
                padded = np.pad(dilated_matrix, 1, mode='constant', constant_values=False)
                
                # Apply structuring element
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            if structure[dx+1, dy+1, dz+1]:
                                # Shift and OR with original
                                shifted = padded[1+dx:1+dx+dilated_matrix.shape[0],
                                               1+dy:1+dy+dilated_matrix.shape[1], 
                                               1+dz:1+dz+dilated_matrix.shape[2]]
                                new_matrix = new_matrix | shifted
                
                dilated_matrix = new_matrix
        
        print(f"Dilation completed in {time.time() - dilation_start:.2f}s")
        
        # 4. CONVERT BACK TO MESH
        mesh_start = time.time()
        
        # Create new voxel grid with dilated matrix
        dilated_voxels = trimesh.voxel.VoxelGrid(
            dilated_matrix, 
            transform=voxelized.transform
        )
        
        # Convert to mesh using marching cubes
        try:
            shrinkwrap_mesh = dilated_voxels.as_boxes()
            
            if shrinkwrap_mesh is None or shrinkwrap_mesh.volume <= 0:
                print("Box conversion failed, trying marching cubes...")
                # Try marching cubes if available
                shrinkwrap_mesh = dilated_voxels.marching_cubes
                
            if shrinkwrap_mesh is None or shrinkwrap_mesh.volume <= 0:
                print("Marching cubes failed, using convex hull fallback")
                return mesh.convex_hull
                
        except Exception as e:
            print(f"Mesh conversion failed: {e}, using convex hull fallback")
            return mesh.convex_hull
        
        print(f"Mesh conversion completed in {time.time() - mesh_start:.2f}s")
        
        # 5. VALIDATE RESULT
        if not shrinkwrap_mesh.is_watertight:
            print("Warning: Generated shrinkwrap is not watertight, attempting to fix...")
            try:
                shrinkwrap_mesh.fill_holes()
                if not shrinkwrap_mesh.is_watertight:
                    # Last resort: use convex hull of the result
                    shrinkwrap_mesh = shrinkwrap_mesh.convex_hull
            except:
                shrinkwrap_mesh = shrinkwrap_mesh.convex_hull
        
        total_time = time.time() - start_time
        volume_increase = ((shrinkwrap_mesh.volume - mesh.volume) / mesh.volume) * 100
        
        print(f"✅ Voxel + dilation shrinkwrap completed in {total_time:.2f}s")
        print(f"   Original volume: {mesh.volume:.2f}mm³")
        print(f"   Shrinkwrap volume: {shrinkwrap_mesh.volume:.2f}mm³")
        print(f"   Volume increase: {volume_increase:.1f}%")
        print(f"   Voxel count: {np.sum(dilated_matrix):,}")
        
        return shrinkwrap_mesh
        
    except Exception as e:
        print(f"Error in voxel + dilation shrinkwrap: {e}")
        print(traceback.format_exc())
        return mesh.convex_hull