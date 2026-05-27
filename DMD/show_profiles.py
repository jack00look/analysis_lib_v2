#!/usr/bin/env python3
"""
Show sigmoid and density profiles with their kp values applied.
Reads parameters from feedback_sigmoids_list.py for the specified iteration.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import importlib.util

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback_automation_config import MAGNETIZATION_FEEDBACK_FOLDER

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


def load_sigmoid_list():
    """Load the feedback sigmoid list from feedback_sigmoids_list.py."""
    sigmoid_list_path = os.path.join(os.path.dirname(__file__), 'feedback_sigmoids_list.py')
    spec = importlib.util.spec_from_file_location("feedback_sigmoids_list", sigmoid_list_path)
    sigmoid_list_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sigmoid_list_mod)
    return sigmoid_list_mod.SIGMOID_PROFILES


def find_profile_by_iteration(sigmoid_list, iteration):
    """Find profile entry for the given iteration number."""
    for profile in sigmoid_list:
        # Check if it's new format with sigmoid_filename
        if 'sigmoid_filename' in profile:
            # Extract iteration from filename
            sig_file = profile['sigmoid_filename']
            if f'_update_{iteration}.' in sig_file:
                return profile
        # Check if it's old format with filename
        elif 'filename' in profile:
            file = profile['filename']
            if f'_update_{iteration}.' in file:
                return profile
    return None


def main():
    parser = argparse.ArgumentParser(description='Show sigmoid and density profiles')
    parser.add_argument('--it', type=int, required=True, help='Iteration number')
    args = parser.parse_args()
    
    iteration = args.it
    
    # Load sigmoid list and find profile for this iteration
    try:
        sigmoid_list = load_sigmoid_list()
        profile = find_profile_by_iteration(sigmoid_list, iteration)
        
        if profile is None:
            print(f"Error: Iteration {iteration} not found in sigmoid_list")
            return False
        
        print(f"Found iteration {iteration} in sigmoid_list")
        
        # Get filenames
        if 'sigmoid_filename' in profile:
            sigmoid_filename = profile['sigmoid_filename']
            density_filename = profile.get('density_filename')
            kp_sig = profile.get('kp_sigmoid', 0.5)
            sigma_sig = profile.get('smoothing_sigma_sigmoid', 3.0)
            kp_dens = profile.get('kp_density', 0.3)
            sigma_dens = profile.get('smoothing_sigma_density', 3.0)
        else:
            # Old format (sigmoid only)
            sigmoid_filename = profile['filename']
            density_filename = None
            kp_sig = profile.get('kp', 0.5)
            sigma_sig = profile.get('smoothing_sigma', 2.0)
            kp_dens = 0.3
            sigma_dens = 3.0
            
    except Exception as e:
        print(f"Error loading sigmoid list: {e}")
        return False
    
    # Load sigmoid
    sigmoid_path = os.path.join(MAGNETIZATION_FEEDBACK_FOLDER, sigmoid_filename)
    x_sig, y_sig = load_profile_from_txt(sigmoid_path)
    
    if x_sig is None:
        print(f"Could not load sigmoid profile")
        return False
    
    print(f"Sigmoid: {len(x_sig)} points, range x=[{x_sig[0]:.1f}, {x_sig[-1]:.1f}]")
    print(f"Sigmoid: y range=[{y_sig.min():.4f}, {y_sig.max():.4f}]")
    
    # Load density if available
    x_dens = None
    y_dens = None
    if density_filename:
        density_path = os.path.join(MAGNETIZATION_FEEDBACK_FOLDER, density_filename)
        x_dens, y_dens = load_profile_from_txt(density_path)
        
        if x_dens is not None:
            print(f"Density: {len(x_dens)} points, range x=[{x_dens[0]:.1f}, {x_dens[-1]:.1f}]")
            print(f"Density: y range=[{y_dens.min():.4f}, {y_dens.max():.4f}]")
        else:
            print(f"Could not load density profile (continuing with sigmoid only)")
    
    # Apply smoothing
    y_sig_smoothed = gaussian_filter(y_sig, sigma=sigma_sig)
    
    # Subtract means before scaling
    y_sig_error = y_sig_smoothed - np.mean(y_sig_smoothed)
    
    # Apply kp scaling
    y_sig_scaled = kp_sig * y_sig_error
    
    y_dens_scaled = None
    if x_dens is not None:
        y_dens_smoothed = gaussian_filter(y_dens, sigma=sigma_dens)
        y_dens_error = y_dens_smoothed - np.mean(y_dens_smoothed)
        y_dens_scaled = kp_dens * y_dens_error
    
    # Combine
    if y_dens_scaled is not None:
        y_total = y_sig_scaled + y_dens_scaled
    else:
        y_total = y_sig_scaled
    
    print(f"\nParameters from sigmoid_list:")
    print(f"  Sigmoid: kp={kp_sig}, σ={sigma_sig}")
    if y_dens_scaled is not None:
        print(f"  Density: kp={kp_dens}, σ={sigma_dens}")
    
    print(f"\nAfter kp scaling and smoothing:")
    print(f"Sigmoid: y range=[{y_sig_scaled.min():.6f}, {y_sig_scaled.max():.6f}]")
    if y_dens_scaled is not None:
        print(f"Density: y range=[{y_dens_scaled.min():.6f}, {y_dens_scaled.max():.6f}]")
    print(f"Total: y range=[{y_total.min():.6f}, {y_total.max():.6f}]")
    
    # Plot
    if y_dens_scaled is not None:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Sigmoid
        axes[0].plot(x_sig, y_sig_scaled, 'b-', linewidth=2)
        axes[0].set_ylabel('Sigmoid × kp')
        axes[0].set_title(f'Iteration {iteration}: Sigmoid Profile (kp={kp_sig}, σ={sigma_sig})')
        axes[0].grid(True, alpha=0.3)
        
        # Density
        axes[1].plot(x_dens, y_dens_scaled, 'r-', linewidth=2)
        axes[1].set_ylabel('Density × kp')
        axes[1].set_title(f'Iteration {iteration}: Density Profile (kp={kp_dens}, σ={sigma_dens})')
        axes[1].grid(True, alpha=0.3)
        
        # Total
        axes[2].plot(x_sig, y_total, 'g-', linewidth=2)
        axes[2].set_ylabel('Sigmoid + Density')
        axes[2].set_title(f'Iteration {iteration}: Total Correction (Sigmoid + Density)')
        axes[2].grid(True, alpha=0.3)
        
        # Overlaid
        axes[3].plot(x_sig, y_sig_scaled, 'b-', linewidth=2, alpha=0.7, label='Sigmoid')
        axes[3].plot(x_dens, y_dens_scaled, 'r-', linewidth=2, alpha=0.7, label='Density')
        axes[3].plot(x_sig, y_total, 'g--', linewidth=2, alpha=0.7, label='Sum')
        axes[3].set_xlabel('Field (µm)')
        axes[3].set_ylabel('Correction (a.u.)')
        axes[3].set_title(f'Iteration {iteration}: All Profiles Overlaid')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Sigmoid
        axes[0].plot(x_sig, y_sig_scaled, 'b-', linewidth=2)
        axes[0].set_ylabel('Sigmoid × kp')
        axes[0].set_title(f'Iteration {iteration}: Sigmoid Profile (kp={kp_sig}, σ={sigma_sig})')
        axes[0].grid(True, alpha=0.3)
        
        # Overlaid (just sigmoid)
        axes[1].plot(x_sig, y_sig_scaled, 'b-', linewidth=2, label='Sigmoid')
        axes[1].set_xlabel('Field (µm)')
        axes[1].set_ylabel('Correction (a.u.)')
        axes[1].set_title(f'Iteration {iteration}: Sigmoid Profile')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'feedback_profiles_it{iteration}.png', dpi=150)
    print(f"\nSaved plot to feedback_profiles_it{iteration}.png")
    plt.show()
    
    return True


if __name__ == '__main__':
    main()
