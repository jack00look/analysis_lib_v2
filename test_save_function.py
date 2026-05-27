#!/usr/bin/env python3
"""
Unit test for save_density_error_profile function.

Tests the function in isolation without needing full waterfall data.
"""

import numpy as np
import sys
import os

# Add waterfall_v2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waterfall_v2.waterfall_lib import save_density_error_profile

def test_save_density_profile():
    """Test the save_density_error_profile function directly."""
    
    print("="*80)
    print("Unit Test: save_density_error_profile()")
    print("="*80)
    
    # Test Case 1: Normal case with valid data
    print("\nTest 1: Normal case with valid density data")
    section_centers_x = np.array([2.6, 3.8, 5.0, 6.2, 7.4])
    densities = np.array([0.100105, 0.100449, 0.100325, 0.099981, 0.099337])
    params = {}
    
    print(f"  Input: {len(section_centers_x)} x-points, {len(densities)} densities")
    print(f"  X positions: {section_centers_x}")
    print(f"  Densities: {densities}")
    
    try:
        save_density_error_profile(section_centers_x, densities, params)
        
        # Check if file was created
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        density_file = os.path.join(script_dir, 'waterfall_v2', 'density_error_profile.txt')
        
        if os.path.exists(density_file):
            file_size = os.path.getsize(density_file)
            print(f"  ✓ File created: {file_size} bytes")
            with open(density_file, 'r') as f:
                lines = f.readlines()[:3]
                for line in lines:
                    print(f"    {line.rstrip()}")
            return True
        else:
            print(f"  ✗ File NOT created at {density_file}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_save_density_profile()
    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED")
        sys.exit(0)
    else:
        print("✗ TEST FAILED")
        sys.exit(1)
