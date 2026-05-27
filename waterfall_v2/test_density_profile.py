#!/usr/bin/env python3
"""
Simple test to verify density error profile saving works.

This test creates mock data and verifies:
1. fit_params_all is accessible in plot_main_waterfall()
2. save_density_error_profile() can be called
3. The density_error_profile.txt file is created
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the waterfall_v2 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waterfall_v2.waterfall_lib import plot_main_waterfall, _sectioned_sigmoid_analysis

def test_density_profile_save():
    """Test that density profiles are saved correctly."""
    
    print("="*80)
    print("Testing density profile saving functionality")
    print("="*80)
    
    # Create synthetic data
    n_shots = 10
    n_pixels = 100
    n_sections = 5
    
    # Mock magnetization data
    M = np.random.randn(n_shots, n_pixels) * 0.1 + 0.5
    M = np.clip(M, 0, 1)  # Keep between 0 and 1
    
    # Mock density data - vary with position and time
    D = np.random.randn(n_shots, n_pixels) * 0.01 + 0.1
    D = np.clip(D, 0.05, 0.15)  # Keep positive and reasonable
    
    # Scan variable (y-axis)
    y_unique = np.linspace(0, 10, n_shots)
    dy = y_unique[1] - y_unique[0]
    y_axis_label = "Time (ms)"
    
    # Space axis
    um_per_px = 0.1
    
    # Minimal params
    params = {
        'NUM_SECTIONS': n_sections,
        'X_MIN_INTEGRATION': 2.0,
        'X_MAX_INTEGRATION': 8.0,
        'WATERFALL_MAG_CLIM': None,
        'WATERFALL_DENSITY_CLIM': None,
        'SIGMOID_FIT_X_MIN': None,
        'SIGMOID_FIT_X_MAX': None,
    }
    
    # Plot flags with sectioned_sigmoid enabled
    plot_flags = {
        'sectioned_sigmoid': True,  # Enable sigmoid and density saving
    }
    
    print("\n1. Testing _sectioned_sigmoid_analysis()...")
    print(f"   - M shape: {M.shape}")
    print(f"   - D shape: {D.shape}")
    print(f"   - Number of sections: {n_sections}")
    
    x_plot_um = np.arange(M.shape[1]) * um_per_px
    xmin_ind = np.argmin(np.abs(x_plot_um - params['X_MIN_INTEGRATION']))
    xmax_ind = np.argmin(np.abs(x_plot_um - params['X_MAX_INTEGRATION']))
    
    sx, sy, se, avg_dens, fit_params_all, fit_params_filtered = _sectioned_sigmoid_analysis(
        M, y_unique, x_plot_um, xmin_ind, xmax_ind, n_sections, D=D
    )
    
    print(f"   ✓ Got {len(fit_params_all)} fit_params entries")
    
    if len(fit_params_all) > 0:
        # Check density values
        densities = [p.get('density', np.nan) for p in fit_params_all]
        print(f"   ✓ Density values: {[f'{d:.6f}' if not np.isnan(d) else 'NaN' for d in densities]}")
        
        # Check x positions
        x_positions = [p['x'] for p in fit_params_all]
        print(f"   ✓ X positions: {[f'{x:.2f}' for x in x_positions]}")
    else:
        print(f"   ✗ ERROR: fit_params_all is empty!")
        return False
    
    print("\n2. Testing plot_main_waterfall() with density saving...")
    print("   - Calling plot_main_waterfall with sectioned_sigmoid=True")
    
    # Mock other parameters
    title = "Test Waterfall"
    scan = "bubbles"
    
    try:
        # This should trigger the density profile save
        plot_main_waterfall(
            M=M,
            D=D,
            y_unique=y_unique,
            dy=dy,
            y_axis_label=y_axis_label,
            title=title,
            scan=scan,
            params=params,
            plot_flags=plot_flags,
            um_per_px=um_per_px,
            average=False,
        )
        print("   ✓ plot_main_waterfall() completed without errors")
    except Exception as e:
        print(f"   ✗ ERROR in plot_main_waterfall(): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Checking if density_error_profile.txt was created...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    density_file = os.path.join(script_dir, 'density_error_profile.txt')
    
    if os.path.exists(density_file):
        file_size = os.path.getsize(density_file)
        print(f"   ✓ SUCCESS! density_error_profile.txt created ({file_size} bytes)")
        
        # Show content preview
        with open(density_file, 'r') as f:
            lines = f.readlines()[:5]
            print(f"   First few lines:")
            for line in lines:
                print(f"     {line.rstrip()}")
        
        return True
    else:
        print(f"   ✗ ERROR: density_error_profile.txt NOT found at {density_file}")
        # List files in the directory
        print(f"   Files in {script_dir}:")
        for f in os.listdir(script_dir):
            if f.endswith('.txt'):
                print(f"     - {f}")
        return False

if __name__ == '__main__':
    plt.ioff()  # Turn off interactive mode
    success = test_density_profile_save()
    plt.close('all')  # Close all figures
    
    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED: Density profile saving works correctly!")
        sys.exit(0)
    else:
        print("✗ TEST FAILED: Issues with density profile saving")
        sys.exit(1)
