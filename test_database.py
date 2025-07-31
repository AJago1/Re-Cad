#!/usr/bin/env python
"""
Test script for database functionality
"""

import os
import sys
import pandas as pd

# Add the stl_analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stl_analyzer'))

from database import STLDatabase
from stl_utils import extract_features, is_valid_entry

def test_database():
    """Test basic database operations"""
    print("Testing STL Database functionality...")
    
    # Create a test database
    db = STLDatabase()
    
    # Check if database is empty
    df = db.get_all_entries()
    print(f"Initial database size: {len(df)} entries")
    
    # Test with a sample STL file if available
    test_files = []
    
    # Look for STL files in common locations
    search_dirs = [
        ".",
        "..",
        "test_files",
        "../test_files",
        "../../test_files"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.lower().endswith('.stl'):
                    test_files.append(os.path.join(search_dir, file))
                    if len(test_files) >= 3:  # Only test with first 3 files
                        break
        if test_files:
            break
    
    if not test_files:
        print("No STL files found for testing")
        return
    
    print(f"Found {len(test_files)} STL files for testing")
    
    # Test feature extraction and database operations
    valid_count = 0
    invalid_count = 0
    
    for file_path in test_files:
        print(f"\nTesting: {os.path.basename(file_path)}")
        
        # Extract features
        features = extract_features(file_path)
        
        if features is None:
            print(f"  ❌ Failed to extract features")
            invalid_count += 1
            continue
            
        # Validate features
        if not is_valid_entry(features):
            print(f"  ❌ Invalid features detected")
            print(f"     Volume: {features.get('volume', 'N/A')}")
            print(f"     Surface Area: {features.get('surface_area', 'N/A')}")
            invalid_count += 1
            continue
            
        print(f"  ✅ Valid features extracted")
        print(f"     Volume: {features['volume']:.2f} mm³")
        print(f"     Surface Area: {features['surface_area']:.2f} mm²")
        print(f"     Dimensions: {features['x']:.1f} × {features['y']:.1f} × {features['z']:.1f} mm")
        
        # Add to database
        db.add_entry(features)
        valid_count += 1
    
    print(f"\n📊 Results:")
    print(f"   Valid entries: {valid_count}")
    print(f"   Invalid entries: {invalid_count}")
    
    # Test database operations
    df_after = db.get_all_entries()
    print(f"   Database size after adding: {len(df_after)} entries")
    
    # Test saving and loading
    print("\n💾 Testing save/load...")
    db.save_database()
    
    # Create new database instance and load
    db2 = STLDatabase()
    df_loaded = db2.get_all_entries()
    print(f"   Loaded database size: {len(df_loaded)} entries")
    
    if len(df_loaded) == len(df_after):
        print("   ✅ Save/load test passed")
    else:
        print("   ❌ Save/load test failed")
    
    # Show sample data
    if not df_loaded.empty:
        print("\n📋 Sample database entries:")
        print(df_loaded[['name', 'volume', 'surface_area', 'x', 'y', 'z']].head())

if __name__ == "__main__":
    test_database() 