"""
Processing thread for extracting features from STL files
"""

import os
import time
import datetime
from PyQt5.QtCore import QThread, pyqtSignal

class ProcessThread(QThread):
    """Worker thread for processing STL files"""
    progress = pyqtSignal(int)
    processed_file = pyqtSignal(dict)
    process_complete = pyqtSignal(int)
    
    def __init__(self, file_list, generate_shrinkwrap=False, shrinkwrap_offset=5.0, calculate_shrinkwrap=True, generate_files=False, output_mode="dedicated", custom_dir=""):
        super().__init__()
        self.file_list = file_list
        self.stop_flag = False
        self.generate_shrinkwrap = generate_shrinkwrap  # Legacy parameter - deprecated
        self.shrinkwrap_offset = shrinkwrap_offset
        
        # New separated controls
        self.calculate_shrinkwrap = calculate_shrinkwrap  # Whether to calculate shrinkwrap data
        self.generate_files = generate_files  # Whether to generate STL files
        self.output_mode = output_mode  # "dedicated", "same", "custom"
        self.custom_dir = custom_dir  # Custom directory for shrinkwrap files
        
    def run(self):
        """Process STL files and extract features"""
        processed_count = 0
        
        total_files = len(self.file_list)
        for i, file_path in enumerate(self.file_list):
            if self.stop_flag:
                break
                
            try:
                print(f"🔄 Processing: {os.path.basename(file_path)}")
                
                # Import the extract_features function
                try:
                    from ...stl_utils import extract_features, is_valid_entry
                except ImportError:
                    # Try direct import if relative import fails
                    try:
                        from stl_utils import extract_features, is_valid_entry
                    except ImportError:
                        print(f"❌ Could not import feature extraction functions for {file_path}")
                        continue
                
                # Extract features (shrinkwrap is already calculated here!)
                features = extract_features(file_path)
                
                if features is None:
                    print(f"❌ Failed to extract features from: {os.path.basename(file_path)}")
                    continue
                    
                print(f"✅ Features extracted for {features.get('name', 'unknown')}")
                
                # Check if shrinkwrap data was calculated in extract_features
                shrinkwrap_vol = features.get('shrinkwrap_volume', 0)
                shrinkwrap_ratio = features.get('shrinkwrap_ratio', 0)
                
                if shrinkwrap_vol > 0:
                    print(f"✅ Shrinkwrap data found: {shrinkwrap_vol:.2f}mm³ (ratio: {shrinkwrap_ratio:.4f})")
                    
                    # Skip generating shrinkwrap STL files - volume calculation is sufficient
                    if self.generate_files:
                        print(f"📊 Shrinkwrap volume already calculated: {shrinkwrap_vol:.2f}mm³ (no file generation needed)")
                    else:
                        print(f"ℹ️  STL file generation disabled")
                else:
                    print(f"ℹ️  No shrinkwrap data calculated (or disabled)")
                    # Ensure we have default values
                    if 'shrinkwrap_volume' not in features:
                        features['shrinkwrap_volume'] = 0.0
                    if 'shrinkwrap_ratio' not in features:
                        features['shrinkwrap_ratio'] = 0.0
                
                if is_valid_entry(features):
                    self.processed_file.emit(features)
                    processed_count += 1
                    print(f"✅ Valid features emitted for: {os.path.basename(file_path)}")
                else:
                    print(f"⚠️  Invalid features extracted from: {os.path.basename(file_path)}")
                        
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {e}")
                import traceback
                traceback.print_exc()
            
            # Update progress
            self.progress.emit(int((i + 1) / total_files * 100))
            
            # Small delay to prevent UI freezing and allow for stopping
            time.sleep(0.01)
            
        self.process_complete.emit(processed_count)
        print(f"🏁 Processing complete: {processed_count}/{total_files} files processed successfully")
    
    def _get_shrinkwrap_output_path(self, original_file_path):
        """Get the output path for shrinkwrap file based on settings"""
        import os
        
        base_dir = os.path.dirname(original_file_path)
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        filename = f"{base_name}_shrinkwrap_{self.shrinkwrap_offset:.1f}mm.stl"
        
        if self.output_mode == "same":
            # Same directory as original
            output_dir = base_dir
        elif self.output_mode == "custom" and self.custom_dir:
            # Custom directory
            output_dir = self.custom_dir
        else:
            # Default: dedicated subfolder
            output_dir = os.path.join(base_dir, "shrinkwrap")
        
        # Ensure output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create output directory {output_dir}: {e}")
            return None
        
        return os.path.join(output_dir, filename)
        
    def stop(self):
        """Stop the processing"""
        self.stop_flag = True 