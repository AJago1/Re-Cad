"""
Scanning thread for finding STL files in directories
"""

import os
import time
from PyQt5.QtCore import QThread, pyqtSignal

class ScanThread(QThread):
    """Worker thread for scanning directories"""
    progress = pyqtSignal(int)
    found_file = pyqtSignal(str)
    scan_complete = pyqtSignal(int)
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.stop_flag = False
        
    def run(self):
        """Scan the directory for STL files"""
        found_files = []
        processed_count = 0
        
        # Find all STL files
        for root, _, files in os.walk(self.directory):
            if self.stop_flag:
                break
                
            for file in files:
                if self.stop_flag:
                    break
                    
                if file.lower().endswith('.stl'):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    self.found_file.emit(full_path)
        
        # Process STL files and emit progress
        total_files = len(found_files)
        for i, file_path in enumerate(found_files):
            if self.stop_flag:
                break
                
            self.found_file.emit(file_path)
            self.progress.emit(int((i + 1) / total_files * 100))
            processed_count += 1
            
            # Small delay to prevent UI freezing and allow for stopping
            time.sleep(0.01)
            
        self.scan_complete.emit(processed_count)
    
    def stop(self):
        """Stop the scanning process"""
        self.stop_flag = True 