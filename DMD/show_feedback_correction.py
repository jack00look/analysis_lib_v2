#!/usr/bin/env python3
"""
Show feedback correction profiles for a specific iteration.

Usage:
    python show_feedback_correction.py --it 14
    python show_feedback_correction.py --it 14 --show-components
    python show_feedback_correction.py --it 14 --x-min 900 --x-max 1200
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback_automation_config import (
    MAGNETIZATION_FEEDBACK_FOLDER,
    NEW_PROFILE_KP_SIGMOID,
    NEW_PROFILE_SMOOTHING_SIGMA_SIGMOID,
    NEW_PROFILE_KP_DENSITY,
    NEW_PROFILE_SMOOTHING_SIGMA_DENSITY,
)

# DMD folder and profile names
DMD_FOLDER = os.path.dirname(os.path.abspath(__file__))
DMD_PROFILE_TXT_NAME = 'dmd_profile.txt'


def load_profile_from_txt(filepath):
    """Load x and y values from txt file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None
    
    try:
        data = np.loadtxt(filepath, skiprows=3)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def extend_profile(x_target, x_source, y_source):
    """Extend profile from source x-range to target x-range via interpolation."""
    if len(x_source) < 2:
        print(f"Warning: Not enough points to interpolate ({len(x_source)})")
        return np.zeros_like(x_target)
    
    kind = 'cubic' if len(x_source) >= 4 else 'linear'
    try:
        interp_func = interp1d(x_source, y_source, kind=kind, fill_value='extrapolate')
        return interp_func(x_target)
    except Exception as e:
        print(f"Error interpolating: {e}")
        return np.zeros_like(x_target)


def main():
    parser = argparse.ArgumentParser(
        description='Show feedback correction profiles for a specific iteration'
    )
    parser.add_argument('--it', type=int, required=True, 
                       help='Iteration number (e.g., 14)')
    parser.add_argument('--show-components', action='store_true',
                       help='Show each component separately')
    parser.add_argument('--x-min', type=float, default=None,
                       help='Minimum x range (um)')
    parser.add_argument('--x-max', type=float, default=None,
                       help='Maximum x range (um)')
    
    args = parser.parse_args()
    iteration = args.it
    
    print(f"\n{'='*80}")
    print(f"Feedback Correction Visualization - Iteration {iteration}")
    print(f"{'='*80}\n")
    
    # Load base DMD profile
    dmd_profile_path = os.path.join(DMD_FOLDER, DMD_PROFILE_TXT_NAME)
    x_dmd, y_dmd = load_profile_from_txt(dmd_profile_path)
    
    if x_dmd is None:
        print(f"Error: Could not load base DMD profile from {dmd_profile_path}")
        return False
    
    print(f"Loaded base DMD profile: {len(x_dmd)} points")
    
    # Load sigmoid profile
    sigmoid_file = f"sigmoid_center_interpolation_update_{iteration}.txt"
    sigmoid_path = os.path.join(MAGNETIZATION_FEEDBACK_FOLDER, sigmoid_file)
    x_sig, y_sig = load_profile_from_txt(sigmoid_path)
    
    if x_sig is None:
        print(f"Warning: Sigmoid profile not found: {sigmoid_file}")
        x_sig_ext = None
    else:
        print(f"Loaded sigmoid profile: {len(x_sig)} points")
        # Extend to DMD range
        x_sig_ext = extend_profile(x_dmd, x_sig, y_sig)
        # Apply smoothing and scaling
        y_sig_smoothed = gaussian_filter(x_sig_ext, sigma=DEFAULT_SMOOTHING_SIGMA_SIGMOID)
        y_sig_scaled = DEFAULT_KP_SIGMOID * y_sig_smoothed
    
    # Load density profile
    density_file = f"density_error_profile_update_{iteration}.txt"
    density_path = os.path.join(MAGNETIZATION_FEEDBACK_FOLDER, density_file)
    x_dens, y_dens = load_profile_from_txt(density_path)
    
    if x_dens is None:
        print(f"Warning: Density profile not found: {density_file}")
        x_dens_ext = None
    else:
        print(f"Loaded density profile: {len(x_dens)} points")
        # Extend to DMD range
        x_dens_ext = extend_profile(x_dmd, x_dens, y_dens)
        # Apply smoothing and scaling
        y_dens_smoothed = gaussian_filter(x_dens_ext, sigma=DEFAULT_SMOOTHING_SIGMA_DENSITY)
        y_dens_scaled = DEFAULT_KP_DENSITY * y_dens_smoothed
    
    # Compute total correction
    total_correction = np.zeros_like(x_dmd)
    if x_sig_ext is not None:
        total_correction += y_sig_scaled
    if x_dens_ext is not None:
        total_correction += y_dens_scaled
    
    # Set x-range for plotting
    if args.x_min is not None and args.x_max is not None:
        x_min_idx = np.argmin(np.abs(x_dmd - args.x_min))
        x_max_idx = np.argmin(np.abs(x_dmd - args.x_max))
        x_plot = x_dmd[x_min_idx:x_max_idx]
        y_dmd_plot = y_dmd[x_min_idx:x_max_idx]
        if x_sig_ext is not None:
            y_sig_scaled_plot = y_sig_scaled[x_min_idx:x_max_idx]
        if x_dens_ext is not None:
            y_dens_scaled_plot = y_dens_scaled[x_min_idx:x_max_idx]
        total_correction_plot = total_correction[x_min_idx:x_max_idx]
    else:
        x_plot = x_dmd
        y_dmd_plot = y_dmd
        if x_sig_ext is not None:
            y_sig_scaled_plot = y_sig_scaled
        if x_dens_ext is not None:
            y_dens_scaled_plot = y_dens_scaled
        total_correction_plot = total_correction
    
    # Create plots
    if args.show_components:
        # Show individual components
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), tight_layout=True)
        
        # Sigmoid
        if x_sig_ext is not None:
            axes[0].plot(x_plot, y_sig_scaled_plot, 'b-', linewidth=2, label='Sigmoid Error')
            axes[0].set_ylabel('Sigmoid Correction (a.u.)')
            axes[0].set_title(f'Iteration {iteration}: Sigmoid Error Profile (kp={DEFAULT_KP_SIGMOID}, σ={DEFAULT_SMOOTHING_SIGMA_SIGMOID})')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(0, color='k', linewidth=0.5, alpha=0.5)
            axes[0].legend()
        
        # Density
        if x_dens_ext is not None:
            axes[1].plot(x_plot, y_dens_scaled_plot, 'r-', linewidth=2, label='Density Error')
            axes[1].set_ylabel('Density Correction (a.u.)')
            axes[1].set_title(f'Iteration {iteration}: Density Error Profile (kp={DEFAULT_KP_DENSITY}, σ={DEFAULT_SMOOTHING_SIGMA_DENSITY})')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.5)
            axes[1].legend()
        
        # Total
        axes[2].plot(x_plot, total_correction_plot, 'g-', linewidth=2.5, label='Total Correction')
        axes[2].set_xlabel('Position (μm)')
        axes[2].set_ylabel('Total Correction (a.u.)')
        axes[2].set_title(f'Iteration {iteration}: Total Correction (Sigmoid + Density)')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='k', linewidth=0.5, alpha=0.5)
        axes[2].legend()
        
    else:
        # Show all on same plot
        fig, ax = plt.subplots(figsize=(14, 6), tight_layout=True)
        
        if x_sig_ext is not None:
            ax.plot(x_plot, y_sig_scaled_plot, 'b-', linewidth=2, alpha=0.7, 
                   label=f'Sigmoid (kp={DEFAULT_KP_SIGMOID}, σ={DEFAULT_SMOOTHING_SIGMA_SIGMOID})')
        
        if x_dens_ext is not None:
            ax.plot(x_plot, y_dens_scaled_plot, 'r-', linewidth=2, alpha=0.7, 
                   label=f'Density (kp={DEFAULT_KP_DENSITY}, σ={DEFAULT_SMOOTHING_SIGMA_DENSITY})')
        
        ax.plot(x_plot, total_correction_plot, 'g-', linewidth=3, alpha=0.9, 
               label='Total Correction')
        
        ax.set_xlabel('Position (μm)', fontsize=12)
        ax.set_ylabel('Correction (a.u.)', fontsize=12)
        ax.set_title(f'Iteration {iteration}: Feedback Corrections', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=11)
    
    plt.show()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
