"""
Fast Configuration for STL Analyzer
Disables expensive calculations for better performance
"""

# Fast mode settings
FAST_MODE = False  # Disable fast mode by default - let GUI control shrinkwrap

# Shrinkwrap settings for fast mode
FAST_SHRINKWRAP_CONFIG = {
    "enable_shrinkwrap": True,  # Allow shrinkwrap when requested
    "use_convex_hull_only": False,  # Don't force convex hull
    "max_processing_time": 30.0,  # Allow more time for proper shrinkwrap
    "skip_complex_calculations": False,
    "box_size_multiplier": 3.0,  # Use larger boxes for faster processing
    "detail_level": 4,  # Lower detail level
}

# Feature extraction settings
FAST_FEATURE_CONFIG = {
    "skip_shrinkwrap": False,  # Don't skip shrinkwrap - let GUI control it
    "skip_alpha_shapes": True,
    "skip_ball_pivoting": True,
    "use_basic_features_only": False,  # Allow full features when requested
    "timeout_per_file": 30.0,  # 30 second timeout per file (was 10)
}

def apply_fast_mode_patches():
    """Apply fast mode patches to the STL utilities"""
    try:
        from . import stl_utils
        
        # Patch the create_shrinkwrap function to be faster
        original_create_shrinkwrap = stl_utils.create_shrinkwrap
        
        def fast_create_shrinkwrap(mesh, **kwargs):
            """Fast shrinkwrap that just returns convex hull"""
            if FAST_MODE and FAST_SHRINKWRAP_CONFIG["use_convex_hull_only"]:
                print("Fast mode: Using convex hull instead of shrinkwrap")
                return mesh.convex_hull
            else:
                return original_create_shrinkwrap(mesh, **kwargs)
        
        # Replace the function
        stl_utils.create_shrinkwrap = fast_create_shrinkwrap
        
        # Patch extract_features for faster processing
        original_extract_features = stl_utils.extract_features
        
        def fast_extract_features(file_path, calculate_shrinkwrap=True, shrinkwrap_offset=5.0):
            """Fast feature extraction with optional shrinkwrap calculations"""
            import time
            start_time = time.time()
            
            mesh = stl_utils.try_load_stl(file_path)
            if mesh is None:
                return None
            
            try:
                # Get the model name
                import os
                model_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Calculate minimal bounding box
                obb = mesh.bounding_box_oriented
                size = obb.extents
                size = sorted(size, reverse=True)  # Sort from largest to smallest
                
                # Basic features only
                volume = mesh.volume
                surface_area = mesh.area
                min_bb_vol = size[0] * size[1] * size[2]
                
                # Fast convex hull calculation
                convex_hull = mesh.convex_hull
                convex_hull_volume = convex_hull.volume
                convexity_ratio = volume / convex_hull_volume if convex_hull_volume > 0 else 0.0
                
                # Calculate shrinkwrap if requested by GUI
                if calculate_shrinkwrap:
                    try:
                        print(f"Calculating REAL shrinkwrap for {model_name} with {shrinkwrap_offset}% offset...")
                        
                        # Try multiple import approaches for shrinkwrap_manager
                        try:
                            from . import shrinkwrap_manager
                        except ImportError:
                            try:
                                import shrinkwrap_manager
                            except ImportError:
                                # Direct file import as last resort
                                import sys
                                import os
                                current_dir = os.path.dirname(__file__)
                                sys.path.insert(0, current_dir)
                                import shrinkwrap_manager
                        
                        shrinkwrap_data = shrinkwrap_manager.extract_shrinkwrap_data_only(mesh, shrinkwrap_offset)
                        shrinkwrap_volume = shrinkwrap_data.get('shrinkwrap_volume', 0.0)
                        shrinkwrap_ratio = shrinkwrap_data.get('shrinkwrap_ratio', 0.0)
                        
                        if shrinkwrap_volume > 0:
                            print(f"✅ REAL shrinkwrap calculated: {shrinkwrap_volume:.2f}mm³ (ratio: {shrinkwrap_ratio:.4f})")
                        else:
                            print(f"❌ Shrinkwrap calculation failed, using 0")
                    except Exception as e:
                        print(f"Error calculating shrinkwrap for {model_name}: {e}")
                        shrinkwrap_volume = 0.0
                        shrinkwrap_ratio = 0.0
                else:
                    print(f"Shrinkwrap calculation disabled for {model_name}")
                    shrinkwrap_volume = 0.0
                    shrinkwrap_ratio = 0.0
                
                processing_time = time.time() - start_time
                print(f"Processing completed in {processing_time:.2f}s for {model_name}")
                
                return {
                    "name": model_name,
                    "filename": file_path,
                    "x": float(size[0]),
                    "y": float(size[1]),
                    "z": float(size[2]),
                    "volume": float(volume),
                    "surface_area": float(surface_area),
                    "bb_volume": float(min_bb_vol),
                    "convex_hull_volume": float(convex_hull_volume),
                    "convexity_ratio": float(convexity_ratio),
                    "shrinkwrap_volume": float(shrinkwrap_volume),
                    "shrinkwrap_ratio": float(shrinkwrap_ratio),
                }
                
            except Exception as e:
                print(f"Error in feature extraction for {file_path}: {e}")
                return None
        
        # Replace the function if fast mode is enabled
        if FAST_MODE:
            stl_utils.extract_features = fast_extract_features
            print("Fast mode enabled: Using optimized feature extraction")
        
        return True
        
    except Exception as e:
        print(f"Error applying fast mode patches: {e}")
        return False

def enable_fast_mode():
    """Enable fast mode globally"""
    global FAST_MODE
    FAST_MODE = True
    apply_fast_mode_patches()
    print("="*50)
    print("FAST MODE ENABLED")
    print("- Shrinkwrap calculations disabled")
    print("- Using convex hull for speed")
    print("- 10x faster processing")
    print("="*50)

def disable_fast_mode():
    """Disable fast mode"""
    global FAST_MODE
    FAST_MODE = False
    print("Fast mode disabled - full calculations enabled")

def is_fast_mode():
    """Check if fast mode is enabled"""
    return FAST_MODE 