"""
Validation plot for domain extraction.

Shows raw magnetization profiles for all shots with extracted domains overlaid.
This allows visual verification that domain detection is working correctly.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from waterfall.domain_extraction_lib import create_domain_info_per_shot, get_magnetization_matrix


def plot_domains_validation(df, domain_info_dict, params, mode_cfg, 
                           output_dir='./results/window_analysis', 
                           figname='domains_validation.png'):
    """
    Plot raw magnetization profiles for each shot (line plot).
    
    Shows individual magnetization profiles stacked vertically to see
    raw data and spatial variation.
    
    Parameters:
        df: DataFrame with magnetization data
        domain_info_dict: dict from domain extraction (not used for plotting)
        params: parameters dict with UM_PER_PX and column info
        mode_cfg: mode configuration dict
        output_dir: directory to save plot
        figname: filename for plot
    """
    
    # Build the same normalized magnetization used by the main waterfall plot.
    data_origin = mode_cfg.get('data_origin', 'show_ODs_v2') if isinstance(mode_cfg, dict) else 'show_ODs_v2'
    M_full, um_per_px = get_magnetization_matrix(df, mode_cfg, params, data_origin=data_origin)
    if M_full is None:
        print("Error: normalized magnetization matrix could not be built.")
        return None
    
    print(f"\nPlotting raw magnetization for {len(df)} shots...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), tight_layout=True)
    
    x_axis = None
    
    # Plot each shot
    for shot_idx in range(len(df)):
        mag_profile = np.asarray(M_full[shot_idx], dtype=float)
        
        # Generate x-axis (same for all shots)
        if x_axis is None:
            x_axis = np.arange(len(mag_profile)) * um_per_px
        
        # Plot with vertical offset
        y_offset = shot_idx
        ax.plot(x_axis, mag_profile + y_offset, color='black', alpha=0.6, linewidth=0.8)
    
    ax.set_xlabel('Position (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shot Index (with offset for clarity)', fontsize=12, fontweight='bold')
    ax.set_title('Raw Magnetization Profiles for All Shots', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / figname
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Raw magnetization plot saved to {filepath}\n")
    
    plt.show()
    
    return fig


if __name__ == '__main__':
    print("Domain Extraction Validation Plot")
    print("="*80)
    
    # This would be called from waterfall_v2_example.py
