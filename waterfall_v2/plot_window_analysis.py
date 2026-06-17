"""
Plotting script for sliding window analysis results.

Plots:
1. Defect density (defects/window_size) vs window_center
2. Field at zero magnetization vs window_center (on twin axis)
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from waterfall.domain_extraction_lib import find_field_column, get_magnetization_matrix
from waterfall.window_analysis_lib import results_to_dataframe


def load_window_analysis_results(json_path):
    """Load window analysis results from JSON file."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def compute_defect_density(results, window_size_um):
    """
    Get defect density from results using new interpolated values.
    
    The window_analysis now stores domain_density_at_zero which is 
    the interpolated defect density at the zero-crossing field.
    """
    defect_densities = []
    defect_errors = []
    
    for result in results:
        # Use the new interpolated values
        density = result.get('domain_density_at_zero', np.nan)
        error = result.get('domain_density_error_at_zero', np.nan)
        
        defect_densities.append(density)
        defect_errors.append(error)
    
    return np.array(defect_densities), np.array(defect_errors)


def plot_window_analysis(results, window_size_um, output_dir='./results/window_analysis', figname='window_analysis_twin_axis.png'):
    """
    Create plot with defect density and field_at_zero_mag vs window_center.
    
    Uses interpolated defect density at the zero-crossing field from linear fit.
    
    Parameters:
        results: list of window analysis result dicts
        window_size_um: size of each window in μm (for reference)
        output_dir: directory to save plot
        figname: filename for plot
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Get interpolated defect densities and errors
    defect_densities, defect_errors = compute_defect_density(results, window_size_um)
    df['defect_density'] = defect_densities
    df['defect_density_error'] = defect_errors
    
    # Create figure with twin axes
    fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
    
    # Plot 1: Defect density on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Window Center (μm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Defect Density (defects/μm)', fontsize=12, fontweight='bold', color=color1)
    
    # Plot with error bars
    ax1.errorbar(df['window_center_um'], df['defect_density'], 
                 yerr=df['defect_density_error'],
                 fmt='o-', color=color1, linewidth=2.0, markersize=6, 
                 label='Defect Density', alpha=0.8, capsize=5, capthick=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Field at zero magnetization on right y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Field at Zero Magnetization (T)', fontsize=12, fontweight='bold', color=color2)
    
    line2 = ax2.plot(df['window_center_um'], df['field_at_zero'], 
                     's-', color=color2, linewidth=2.0, markersize=6, 
                     label='Field at Zero Mag', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    fig.suptitle('Sliding Window Analysis: Defect Density and Field Balance vs Position\n'
                 '(using linear fit to zero magnetization)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Combine legends
    lines1 = ax1.get_lines()
    lines2 = line2
    lines = lines1 + lines2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    # Tight layout
    fig.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / figname
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {filepath}")
    
    # Show plot
    plt.show()
    
    return fig, (df, defect_densities, defect_errors)


def plot_window_analysis_with_domains(results, window_size_um, output_dir='./results/window_analysis', figname='window_analysis_detailed.png'):
    """
    Create a more detailed plot including domain count information.
    
    Creates a figure with 3 subplots:
    1. Defect density vs window_center (with error bars)
    2. Field at zero magnetization vs window_center
    3. Domain count at zero magnetization vs window_center
    
    Uses interpolated defect density at the zero-crossing field.
    """
    df = results_to_dataframe(results)
    defect_densities, defect_errors = compute_defect_density(results, window_size_um)
    df['defect_density'] = defect_densities
    df['defect_density_error'] = defect_errors
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True, sharex=True)
    
    # Plot 1: Defect Density with Error Bars
    ax = axes[0]
    ax.errorbar(df['window_center_um'], df['defect_density'],
                yerr=df['defect_density_error'], 
                fmt='o-', color='tab:blue', linewidth=2.0, markersize=6, 
                label='Defect Density', alpha=0.8, capsize=5, capthick=2)
    ax.set_ylabel('Defect Density\n(defects/μm)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Plot 2: Field at Zero Magnetization
    ax = axes[1]
    ax.plot(df['window_center_um'], df['field_at_zero'], 's-', 
            color='tab:red', linewidth=2.0, markersize=6, label='Field at Zero Mag', alpha=0.8)
    ax.set_ylabel('Field at Zero Mag\n(T)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Plot 3: Domain Count with Error Bars
    ax = axes[2]
    ax.errorbar(df['window_center_um'], df['domain_mean_at_zero'], 
                yerr=df['domain_std_at_zero'], fmt='D-', 
                color='tab:green', linewidth=2.0, markersize=6, 
                capsize=5, capthick=1.5, label='Domain Count', alpha=0.8)
    ax.set_ylabel('Avg Domain Count\nat Zero Mag', fontsize=11, fontweight='bold')
    ax.set_xlabel('Window Center (μm)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Title
    fig.suptitle('Sliding Window Analysis: Defects, Field Balance, and Domain Structure\n'
                 '(using linear fit to zero magnetization, errors shown for interpolated density)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / figname
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed plot saved to {filepath}")
    
    plt.show()
    
    return fig, df


def plot_raw_magnetization_by_field(df, field_summary, domain_info_dict, params, output_dir='./results/window_analysis', figname='raw_magnetization_by_field.png'):
    """
    Plot raw magnetization profiles grouped by field value with domain overlays.
    
    Creates a waterfall-style plot showing all shots' magnetization profiles,
    ordered by field value and colored by field. Overlays the extracted domains
    to validate domain detection.
    
    Parameters:
        df: DataFrame with magnetization data
        field_summary: dict with field_value -> statistics
        domain_info_dict: dict from domain extraction with domain info per shot
        params: parameters dict with UM_PER_PX and column info
        output_dir: directory to save plot
        figname: filename for plot
    """
    data_origin = params.get('data_origin', 'show_ODs_v2') if isinstance(params, dict) else 'show_ODs_v2'
    M_full, um_per_px = get_magnetization_matrix(df, {}, params, data_origin=data_origin)
    if M_full is None:
        print("Warning: normalized magnetization matrix could not be built")
        return None
    
    # Get field column
    field_col = find_field_column(df)
    if field_col is None:
        print("Warning: Field column not found")
        return None
    
    # Get x-position column
    x_col = None
    for col in df.columns:
        if isinstance(col, tuple) and 'x_centers_um' in str(col):
            x_col = col
            break
    
    # Sort by field value
    df_with_domain_key = df.copy()
    df_with_domain_key['_domain_info_key'] = np.arange(len(df_with_domain_key), dtype=int)
    df_sorted = df_with_domain_key.sort_values(field_col).reset_index(drop=True)
    
    # Get unique field values
    field_values = sorted(field_summary.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), tight_layout=True)
    
    # Extract magnetization profiles
    shot_count = 0
    cmap = plt.cm.RdYlBu_r
    field_norm = plt.Normalize(vmin=min(field_values), vmax=max(field_values))
    
    # Colors for domains
    domain_color_pos = np.array([0.2, 0.8, 0.2, 0.3])  # Green with transparency
    domain_color_neg = np.array([0.8, 0.2, 0.2, 0.3])  # Red with transparency
    
    # Plot each shot
    for sorted_idx in range(len(df_sorted)):
        shot_data = df_sorted.iloc[sorted_idx]
        
        # Get original row-position key used by domain_info_dict
        original_shot_idx = int(shot_data['_domain_info_key'])
        
        # Get magnetization profile
        mag_profile = np.asarray(M_full[original_shot_idx], dtype=float)
        
        # Get field value
        field_val = float(shot_data[field_col])
        
        # Generate x-axis if not available
        if x_col is not None:
            x_axis = np.asarray(shot_data[x_col])
        else:
            x_axis = np.arange(len(mag_profile)) * um_per_px
        
        # Color by field value
        color = cmap(field_norm(field_val))
        
        # Plot magnetization
        y_offset = shot_count
        ax.plot(x_axis, mag_profile + y_offset, color=color, alpha=0.8, linewidth=1.2, label=f'{field_val:.3f}T' if shot_count < 3 else '')
        ax.text(-60, y_offset, f'{field_val:.3f}T', fontsize=7, ha='right', va='center', fontweight='bold')
        
        # Overlay domains from domain_info_dict if available
        if domain_info_dict and original_shot_idx in domain_info_dict:
            shot_domains = domain_info_dict[original_shot_idx]
            domains = shot_domains.get('domains', [])
            
            if domains:
                for domain in domains:
                    x_start = domain['x_start_um']
                    x_end = domain['x_end_um']
                    sign = domain['sign']
                    
                    # Determine domain color
                    if sign > 0:
                        domain_color = domain_color_pos
                    else:
                        domain_color = domain_color_neg
                    
                    # Draw vertical band for domain
                    ax.axvspan(x_start, x_end, ymin=(y_offset-0.3)/(len(df_sorted)+2), 
                              ymax=(y_offset+0.3)/(len(df_sorted)+2),
                              color=domain_color, zorder=1, alpha=0.5)
        
        shot_count += 1
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=field_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.15)
    cbar.set_label('Field (T)', fontsize=11, fontweight='bold')
    
    # Add legend for domains
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2, 0.5], label='Domain (+)'),
        Patch(facecolor=[0.8, 0.2, 0.2, 0.5], label='Domain (-)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Position (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shot Index (ordered by field)', fontsize=12, fontweight='bold')
    ax.set_title('Raw Magnetization Profiles with Extracted Domains (grouped by field)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / figname
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Raw magnetization plot with domains saved to {filepath}")
    
    plt.show()
    
    return fig


def main():
    """Main function to load results and create plots."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*80}")
    logger.info("PLOTTING WINDOW ANALYSIS RESULTS")
    logger.info(f"{'='*80}\n")
    
    # Path to results
    results_dir = Path('./results/window_analysis')
    json_path = results_dir / 'window_analysis.json'
    
    if not json_path.exists():
        logger.error(f"Results file not found: {json_path}")
        logger.error("Please run sliding_window_analysis.py first.")
        return
    
    # Load results
    logger.info(f"Loading results from {json_path}")
    results = load_window_analysis_results(json_path)
    
    if not results:
        logger.error("No results loaded.")
        return
    
    logger.info(f"✓ Loaded {len(results)} windows\n")
    
    # Get window size (from first result or default)
    window_size_um = results[0]['window_end_um'] - results[0]['window_start_um']
    logger.info(f"Window size: {window_size_um:.1f} μm\n")
    
    # Create plots
    logger.info("Creating plots...")
    
    # Simple plot
    fig1, data1 = plot_window_analysis(results, window_size_um, 
                                       output_dir=str(results_dir),
                                       figname='window_analysis_twin_axis.png')
    
    # Detailed plot with 3 subplots
    fig2, data2 = plot_window_analysis_with_domains(results, window_size_um,
                                                     output_dir=str(results_dir),
                                                     figname='window_analysis_detailed.png')
    
    logger.info(f"\n{'='*80}")
    logger.info("Plotting complete!")
    logger.info(f"{'='*80}\n")
    
    return results, data1, data2


if __name__ == '__main__':
    results, data1, data2 = main()
