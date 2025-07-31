#!/usr/bin/env python
"""
Toggle Fast Mode for STL Analyzer
Allows switching between fast and full processing modes
"""

import os
import sys

def toggle_fast_mode():
    """Toggle between fast and full processing modes"""
    try:
        # Import the fast config module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stl_analyzer'))
        from fast_config import is_fast_mode, enable_fast_mode, disable_fast_mode
        
        current_mode = is_fast_mode()
        
        print("STL Analyzer Performance Mode Toggle")
        print("="*40)
        print(f"Current mode: {'FAST' if current_mode else 'FULL'}")
        print()
        
        if current_mode:
            print("Fast mode is currently enabled:")
            print("✓ Shrinkwrap calculations disabled")
            print("✓ Using convex hull for speed")
            print("✓ ~10x faster processing")
            print("✗ Less accurate volume calculations")
            print()
            choice = input("Switch to FULL mode? (y/N): ").lower().strip()
            if choice in ['y', 'yes']:
                disable_fast_mode()
                print("\n✓ Switched to FULL mode")
                print("- All calculations enabled")
                print("- More accurate results")
                print("- Slower processing")
            else:
                print("\nStaying in FAST mode")
        else:
            print("Full mode is currently enabled:")
            print("✓ All calculations enabled")
            print("✓ Most accurate results")
            print("✗ Slower processing")
            print("✗ May take 30+ seconds per file")
            print()
            choice = input("Switch to FAST mode? (Y/n): ").lower().strip()
            if choice not in ['n', 'no']:
                enable_fast_mode()
                print("\n✓ Switched to FAST mode")
                print("- 10x faster processing")
                print("- Uses convex hull approximation")
                print("- Recommended for large batches")
            else:
                print("\nStaying in FULL mode")
        
        print("\nRestart the application for changes to take effect.")
        
    except Exception as e:
        print(f"Error toggling fast mode: {e}")
        return False
    
    return True

if __name__ == "__main__":
    toggle_fast_mode()
    input("\nPress Enter to exit...") 