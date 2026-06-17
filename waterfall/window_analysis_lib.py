"""
Windowed analysis for KZ_det_scan_window mode.

Performs sliding window analysis to compute:
- Average magnetization in each window
- Defect counts in each window
- Field value where average magnetization is closest to zero
- Average number of domains at that field value
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from waterfall.domain_extraction_lib import find_field_column, get_magnetization_matrix


def analyze_window(df, window_start_um, window_size_um, mode_cfg, params, domain_info_dict=None, M_full=None, um_per_px=None):
    """
    Analyze a single window across all shots.
    
    For each shot:
    - Extract magnetization profile in window
    - Compute average magnetization
    - Count defects in window
    
    Returns organized by field value.
    
    Parameters:
        df: DataFrame with magnetization data
        window_start_um: left edge of window in μm
        window_size_um: window width in μm
        mode_cfg: mode configuration dict
        params: analysis parameters dict
        domain_info_dict: optional pre-computed domain info
    
    Returns:
        window_data: {
            field_value: {
                'mags': [mag1, mag2, ...],  # per-shot averages in window
                'defect_counts': [count1, count2, ...],
                'domain_counts': [n_dom1, n_dom2, ...],
                'shot_indices': [idx1, idx2, ...]
            },
            ...
        }
    """
    window_end_um = window_start_um + window_size_um
    
    data_origin = mode_cfg.get('data_origin', 'show_ODs_v2') if isinstance(mode_cfg, dict) else 'show_ODs_v2'
    if M_full is None or um_per_px is None:
        M_full, um_per_px = get_magnetization_matrix(df, mode_cfg, params, data_origin=data_origin)
    if M_full is None:
        raise ValueError("Could not build normalized magnetization matrix")
    
    # Find field column
    field_col = find_field_column(df)
    
    if field_col is None:
        raise ValueError("Field column (ARPKZ_final_set_field) not found in DataFrame")
    
    # Organize data by field value
    window_data = {}
    
    for shot_idx in range(len(df)):
        try:
            # Get field value
            field_val = float(df[field_col].iloc[shot_idx])
            
            # Get normalized magnetization profile
            m_profile = np.asarray(M_full[shot_idx], dtype=float)
            
            # Create x-axis
            x_centers_um = np.arange(m_profile.size) * um_per_px
            
            # Find pixels in window
            window_mask = (x_centers_um >= window_start_um) & (x_centers_um < window_end_um)
            
            if not np.any(window_mask):
                continue  # No pixels in this window for this shot
            
            # Compute average magnetization in window
            avg_mag = float(np.mean(m_profile[window_mask]))
            
            # Count defects in window
            defect_count = 0
            if domain_info_dict and shot_idx in domain_info_dict:
                # Use pre-computed domain info
                domains = domain_info_dict[shot_idx].get('domains', [])
                for domain in domains:
                    x_start = domain['x_start_um']
                    x_end = domain['x_end_um']
                    # A defect (domain wall) is where two domains meet
                    # Count defects that fall within window
                    if window_start_um <= x_start < window_end_um or \
                       window_start_um <= x_end < window_end_um:
                        defect_count += 1
                n_domains = len(domains)
            else:
                # No domain info available, use 0
                n_domains = 0
            
            # Initialize field entry if needed
            if field_val not in window_data:
                window_data[field_val] = {
                    'mags': [],
                    'defect_counts': [],
                    'domain_counts': [],
                    'shot_indices': []
                }
            
            # Add to field group
            window_data[field_val]['mags'].append(avg_mag)
            window_data[field_val]['defect_counts'].append(defect_count)
            window_data[field_val]['domain_counts'].append(n_domains)
            window_data[field_val]['shot_indices'].append(shot_idx)
            
        except Exception as e:
            print(f"Warning: Error processing shot {shot_idx} in window [{window_start_um:.1f}, {window_end_um:.1f}]: {e}")
            continue
    
    return window_data


def summarize_window_by_field(window_data, window_size_um=None):
    """
    Summarize window data by field value.
    
    Computes statistics for each field:
    - mean and std of magnetization
    - mean and std of defect counts
    - mean and std of domain counts
    - defect density (defects per μm) and error
    
    Returns:
        field_summary: {
            field_value: {
                'mag_mean': float,
                'mag_std': float,
                'mag_sem': float,
                'defect_mean': float,
                'defect_std': float,
                'defect_density': float,  # defects / window_size_um
                'defect_density_error': float,  # SEM of defect count / window_size_um
                'domain_mean': float,
                'domain_std': float,
                'n_shots': int
            },
            ...
        }
    """
    field_summary = {}
    
    for field_val, data in window_data.items():
        mags = np.asarray(data['mags'], dtype=float)
        defects = np.asarray(data['defect_counts'], dtype=float)
        domains = np.asarray(data['domain_counts'], dtype=float)
        n_shots = len(data['shot_indices'])
        
        defect_mean = float(np.mean(defects)) if len(defects) > 0 else np.nan
        defect_std = float(np.std(defects, ddof=1)) if len(defects) > 1 else np.nan
        defect_sem = float(defect_std / np.sqrt(len(defects))) if len(defects) > 1 else np.nan
        
        # Compute defect density if window_size_um is provided
        if window_size_um is not None and window_size_um > 0:
            defect_density = defect_mean / window_size_um
            defect_density_error = defect_sem / window_size_um
        else:
            defect_density = np.nan
            defect_density_error = np.nan
        
        field_summary[field_val] = {
            'mag_mean': float(np.mean(mags)) if len(mags) > 0 else np.nan,
            'mag_std': float(np.std(mags, ddof=1)) if len(mags) > 1 else np.nan,
            'mag_sem': float(np.std(mags, ddof=1) / np.sqrt(len(mags))) if len(mags) > 1 else np.nan,
            'defect_mean': defect_mean,
            'defect_std': defect_std,
            'defect_sem': defect_sem,
            'defect_density': defect_density,
            'defect_density_error': defect_density_error,
            'domain_mean': float(np.mean(domains)) if len(domains) > 0 else np.nan,
            'domain_std': float(np.std(domains, ddof=1)) if len(domains) > 1 else np.nan,
            'n_shots': int(n_shots)
        }
    
    return field_summary


def find_zero_crossing_field(field_summary, domain_balance_fit_window=0.4):
    """
    Find field value where average magnetization crosses zero using linear fit.
    
    Algorithm:
    1. Select field points where |mag_mean| <= domain_balance_fit_window
    2. Linear fit: mag_mean = a * field + b
    3. Find zero crossing: field_at_zero = -b/a
    4. Interpolate defect density at zero crossing using bracketing points
    
    Parameters:
        field_summary: dict of field_value -> {mag_mean, mag_std, domain_mean, ...}
        domain_balance_fit_window: max |mag_mean| for points to include in fit
    
    Returns:
        (field_at_zero_fit, result_info) where result_info contains:
            - field_at_zero: field where fit crosses zero magnetization
            - mag_mean_at_zero: estimated magnetization at zero (should be ~0)
            - domain_density_at_zero: interpolated defect density
            - domain_density_error_at_zero: interpolated error
            - field_below, field_above: bracketing field points
            - fit_slope, fit_intercept: linear fit parameters
        or (None, None) if fit fails
    """
    from scipy.stats import linregress
    
    # Collect points within the fit window
    fit_fields = []
    fit_mags = []
    fit_densities = []
    fit_errors = []
    
    for field_val, summary in field_summary.items():
        mag_mean = summary['mag_mean']
        if np.isfinite(mag_mean) and np.abs(mag_mean) <= domain_balance_fit_window:
            fit_fields.append(field_val)
            fit_mags.append(mag_mean)
            # Store defect density and error for later interpolation
            if 'defect_density' in summary:
                fit_densities.append(summary['defect_density'])
            if 'defect_density_error' in summary:
                fit_errors.append(summary['defect_density_error'])
            else:
                fit_errors.append(0.0)
    
    if len(fit_fields) < 2:
        print(f"    Warning: Not enough points for linear fit (found {len(fit_fields)})")
        return None, None
    
    # Perform linear fit: mag = a * field + b
    fit_fields_arr = np.asarray(fit_fields)
    fit_mags_arr = np.asarray(fit_mags)
    
    slope, intercept, r_value, p_value, std_err = linregress(fit_fields_arr, fit_mags_arr)
    
    if np.abs(slope) < 1e-10:
        print(f"    Warning: Fit slope too close to zero ({slope:.2e}), cannot determine zero crossing")
        return None, None
    
    # Find zero crossing: solve mag = 0 = a * field + b => field = -b/a
    field_at_zero_fit = -intercept / slope
    mag_at_zero_fit = slope * field_at_zero_fit + intercept  # Should be ~0
    
    print(f"    Linear fit: mag = {slope:.6f} * field + {intercept:.6f}")
    print(f"    Zero crossing at field = {field_at_zero_fit:.6f}, mag = {mag_at_zero_fit:.6g}")
    print(f"    Fit R² = {r_value**2:.4f}, using {len(fit_fields)} points in window |mag| <= {domain_balance_fit_window}")
    
    # Find bracketing field points (just above and just below the zero crossing)
    field_below = None
    field_above = None
    density_below = None
    density_above = None
    error_below = None
    error_above = None
    
    sorted_fields = sorted(field_summary.keys())
    for i, field_val in enumerate(sorted_fields):
        if field_val <= field_at_zero_fit:
            field_below = field_val
            if 'defect_density' in field_summary[field_val]:
                density_below = field_summary[field_val]['defect_density']
                error_below = field_summary[field_val].get('defect_density_error', 0.0)
        if field_val >= field_at_zero_fit and field_above is None:
            field_above = field_val
            if 'defect_density' in field_summary[field_val]:
                density_above = field_summary[field_val]['defect_density']
                error_above = field_summary[field_val].get('defect_density_error', 0.0)
    
    if field_below is not None and field_above is not None and field_above != field_below:
        weight_above = (field_at_zero_fit - field_below) / (field_above - field_below)
        weight_below = 1.0 - weight_above
    else:
        weight_above = 0.5
        weight_below = 0.5

    # Interpolate defect density at zero crossing
    if field_below is not None and field_above is not None and density_below is not None and density_above is not None:
        # Linear interpolation between the two bracketing points
        density_at_zero = weight_below * density_below + weight_above * density_above
        error_at_zero = weight_below * error_below + weight_above * error_above
        
        print(f"    Interpolated density at zero: {density_at_zero:.6f} "
              f"(between field {field_below:.6f}→{density_below:.6f}±{error_below:.6f} "
              f"and {field_above:.6f}→{density_above:.6f}±{error_above:.6f})")
    else:
        # Fallback: use simple average if defect densities not available
        density_at_zero = np.mean([d for d in fit_densities if d is not None]) if fit_densities else 0.0
        error_at_zero = np.mean([e for e in fit_errors if e is not None]) if fit_errors else 0.0
        print(f"    Fallback: using average density from fit window: {density_at_zero:.6f}±{error_at_zero:.6f}")
    
    def _interp_summary_value(key, default=np.nan):
        if field_below is None and field_above is None:
            return default
        if field_below is None:
            return field_summary[field_above].get(key, default)
        if field_above is None:
            return field_summary[field_below].get(key, default)

        below_val = field_summary[field_below].get(key, default)
        above_val = field_summary[field_above].get(key, default)
        if not (np.isfinite(below_val) and np.isfinite(above_val)):
            finite_vals = [v for v in (below_val, above_val) if np.isfinite(v)]
            return float(finite_vals[0]) if finite_vals else default
        if field_above == field_below:
            return float(0.5 * (below_val + above_val))
        return float(weight_below * below_val + weight_above * above_val)

    domain_mean_at_zero = _interp_summary_value('domain_mean')
    domain_std_at_zero = _interp_summary_value('domain_std')
    mag_sem_at_zero = _interp_summary_value('mag_sem')
    n_shots_at_zero = int(round(_interp_summary_value('n_shots', default=0)))

    result_info = {
        'field_at_zero': field_at_zero_fit,
        'mag_mean_at_zero': float(mag_at_zero_fit),
        'mag_sem_at_zero': float(mag_sem_at_zero) if np.isfinite(mag_sem_at_zero) else np.nan,
        'domain_density_at_zero': float(density_at_zero),
        'domain_density_error_at_zero': float(error_at_zero),
        'domain_mean': float(domain_mean_at_zero) if np.isfinite(domain_mean_at_zero) else np.nan,
        'domain_std': float(domain_std_at_zero) if np.isfinite(domain_std_at_zero) else np.nan,
        'n_shots': int(n_shots_at_zero),
        'field_below': float(field_below) if field_below is not None else None,
        'field_above': float(field_above) if field_above is not None else None,
        'fit_slope': float(slope),
        'fit_intercept': float(intercept),
        'fit_r_squared': float(r_value**2),
        'n_fit_points': len(fit_fields),
    }
    
    return field_at_zero_fit, result_info


def sliding_window_analysis(df, x_min_um, x_max_um, window_size_um, window_step_um, 
                            mode_cfg, params, domain_info_dict=None):
    """
    Perform complete sliding window analysis across x range.
    
    Parameters:
        df: DataFrame with magnetization and field data
        x_min_um: leftmost x position
        x_max_um: rightmost x position
        window_size_um: window width
        window_step_um: step size between windows
        mode_cfg: mode configuration
        params: analysis parameters
        domain_info_dict: optional pre-computed domain info
    
    Returns:
        results: list of dicts, one per window:
        [
            {
                'window_center_um': center,
                'window_start_um': start,
                'window_end_um': end,
                'field_at_zero': field_value,
                'mag_mean_at_zero': mag_mean,
                'mag_std_at_zero': mag_std,
                'mag_sem_at_zero': mag_sem,
                'domain_mean_at_zero': domain_mean,
                'domain_std_at_zero': domain_std,
                'n_shots_at_zero': n_shots,
                'field_summary': {...}  # full summary for this window
            },
            ...
        ]
    """
    results = []
    window_start = x_min_um
    window_num = 0
    data_origin = mode_cfg.get('data_origin', 'show_ODs_v2') if isinstance(mode_cfg, dict) else 'show_ODs_v2'
    M_full, um_per_px = get_magnetization_matrix(df, mode_cfg, params, data_origin=data_origin)
    if M_full is None:
        raise ValueError("Could not build normalized magnetization matrix for sliding-window analysis")
    
    while window_start + window_size_um <= x_max_um:
        window_end = window_start + window_size_um
        window_center = (window_start + window_end) / 2.0
        
        print(f"Analyzing window {window_num}: [{window_start:.1f}, {window_end:.1f}] μm (center: {window_center:.1f})")
        
        try:
            # Analyze this window
            window_data = analyze_window(
                df,
                window_start,
                window_size_um,
                mode_cfg,
                params,
                domain_info_dict,
                M_full=M_full,
                um_per_px=um_per_px,
            )
            
            if not window_data:
                print(f"  No data in this window, skipping.")
                window_start += window_step_um
                window_num += 1
                continue
            
            # Summarize by field
            field_summary = summarize_window_by_field(window_data, window_size_um=window_size_um)
            
            # Get domain_balance_fit_window from config
            domain_balance_fit_window = mode_cfg.get('domain_balance_fit_window', 0.4)
            
            # Find zero crossing using linear fit
            field_at_zero, zero_info = find_zero_crossing_field(field_summary, domain_balance_fit_window)
            
            if field_at_zero is None:
                print(f"  No valid zero crossing found, skipping.")
                window_start += window_step_um
                window_num += 1
                continue
            
            # Store result with interpolated defect density at zero crossing
            result = {
                'window_num': window_num,
                'window_center_um': float(window_center),
                'window_start_um': float(window_start),
                'window_end_um': float(window_end),
                'field_at_zero': float(zero_info['field_at_zero']),
                'mag_mean_at_zero': float(zero_info['mag_mean_at_zero']),
                'mag_sem_at_zero': float(zero_info['mag_sem_at_zero']),
                'domain_density_at_zero': float(zero_info['domain_density_at_zero']),
                'domain_density_error_at_zero': float(zero_info['domain_density_error_at_zero']),
                'domain_mean_at_zero': float(zero_info['domain_mean']),
                'domain_std_at_zero': float(zero_info['domain_std']),
                'n_shots_at_zero': int(zero_info['n_shots']),
                'field_below': zero_info.get('field_below'),
                'field_above': zero_info.get('field_above'),
                'fit_slope': float(zero_info['fit_slope']),
                'fit_intercept': float(zero_info['fit_intercept']),
                'fit_r_squared': float(zero_info['fit_r_squared']),
                'n_fit_points': int(zero_info['n_fit_points']),
                'field_summary': field_summary
            }
            
            results.append(result)
            
            print(f"  ✓ Window {window_num}: field_at_zero={field_at_zero:.6g}, "
                  f"domain_mean={zero_info['domain_mean']:.2f}±{zero_info['domain_std']:.2f}")
            
        except Exception as e:
            print(f"  Error analyzing window: {e}")
            window_start += window_step_um
            window_num += 1
            continue
        
        # Move to next window
        window_start += window_step_um
        window_num += 1
    
    return results


def save_window_analysis_results(results, output_dir='./results', filename='window_analysis.json'):
    """
    Save sliding window analysis results to JSON.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")
    return filepath


def results_to_dataframe(results):
    """
    Convert results to a concise DataFrame for easy plotting and analysis.
    
    Returns DataFrame with columns:
    - window_num
    - window_center_um
    - window_start_um
    - window_end_um
    - field_at_zero
    - mag_mean_at_zero
    - mag_sem_at_zero
    - domain_mean_at_zero
    - domain_std_at_zero
    - n_shots_at_zero
    """
    rows = []
    for result in results:
        row = {
            'window_num': result['window_num'],
            'window_center_um': result['window_center_um'],
            'window_start_um': result['window_start_um'],
            'window_end_um': result['window_end_um'],
            'field_at_zero': result['field_at_zero'],
            'mag_mean_at_zero': result['mag_mean_at_zero'],
            'mag_sem_at_zero': result['mag_sem_at_zero'],
            'domain_mean_at_zero': result['domain_mean_at_zero'],
            'domain_std_at_zero': result['domain_std_at_zero'],
            'n_shots_at_zero': result['n_shots_at_zero']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_window_analysis_csv(results, output_dir='./results', filename='window_analysis.csv'):
    """
    Save sliding window analysis results to CSV.
    """
    df = results_to_dataframe(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Results CSV saved to {filepath}")
    return filepath
