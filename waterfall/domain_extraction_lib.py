"""
Domain extraction and saving utilities for KZ_det_scan_window and similar modes.

Provides functions to:
1. Extract domain boundaries and signs from magnetization profiles
2. Organize domain info with field values
3. Save domain data to files for further analysis
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter1d


def extract_domain_boundaries_and_signs(m_profile, x_axis_um, x_min_um, x_max_um, 
                                        gaussian_sigma=0.3, pos_threshold=0.08, 
                                        neg_threshold=-0.08):
    """
    Extract domain boundaries and normalized signs from a magnetization profile.
    
    Parameters:
        m_profile: 1D magnetization profile
        x_axis_um: x positions in μm
        x_min_um, x_max_um: integration region bounds
        gaussian_sigma: smoothing sigma
        pos_threshold: positive derivative threshold
        neg_threshold: negative derivative threshold
    
    Returns:
        domain_boundaries: list of (x_start, x_end) tuples in μm
        domain_signs: list of +1/-1 signs (normalized to most magnetized)
        peaks_pos: list of positive peak positions
        peaks_neg: list of negative peak positions
    """
    m_roi = np.asarray(m_profile, dtype=float)
    x_roi_um = np.asarray(x_axis_um, dtype=float)
    
    if m_roi.size < 3:
        return [], [], [], []
    
    # Smooth magnetization
    if gaussian_sigma > 0:
        m_smooth = gaussian_filter1d(m_roi, sigma=gaussian_sigma)
    else:
        m_smooth = m_roi.copy()
    
    # Compute derivative
    d_roi = np.gradient(m_smooth)
    
    # Find threshold crossings (peaks in derivative)
    peaks_pos = []
    peaks_neg = []
    
    for i in range(1, len(d_roi) - 1):
        if d_roi[i] > pos_threshold and d_roi[i] > d_roi[i-1] and d_roi[i] > d_roi[i+1]:
            peaks_pos.append(float(x_roi_um[i]))
        elif d_roi[i] < neg_threshold and d_roi[i] < d_roi[i-1] and d_roi[i] < d_roi[i+1]:
            peaks_neg.append(float(x_roi_um[i]))
    
    all_peaks = sorted(peaks_pos + peaks_neg)
    
    # Assign raw domain signs based on peak analysis
    domain_boundaries = []
    domain_signs_raw = []
    
    if len(all_peaks) > 0:
        # Create peak sign dict
        peak_signs = {}
        for px in peaks_pos:
            peak_signs[px] = +1
        for px in peaks_neg:
            peak_signs[px] = -1
        
        # First domain: x_min to first peak
        first_peak_sign = peak_signs[all_peaks[0]]
        first_sign = -first_peak_sign
        domain_boundaries.append((x_min_um, all_peaks[0]))
        domain_signs_raw.append(first_sign)
        
        # Intermediate domains: between peaks
        for i in range(len(all_peaks) - 1):
            peak_sign_i = peak_signs[all_peaks[i]]
            peak_sign_next = peak_signs[all_peaks[i+1]]
            
            # Only add new domain if sign changes
            if peak_sign_i != peak_sign_next:
                domain_boundaries.append((all_peaks[i], all_peaks[i+1]))
                domain_signs_raw.append(peak_sign_i)
        
        # Last domain: last peak to x_max
        last_peak_sign = peak_signs[all_peaks[-1]]
        domain_boundaries.append((all_peaks[-1], x_max_um))
        domain_signs_raw.append(last_peak_sign)
    else:
        # No peaks: single domain
        domain_boundaries = [(x_min_um, x_max_um)]
        m_avg = np.mean(m_roi)
        domain_signs_raw = [1 if m_avg > 0 else -1]
    
    # Normalize signs: +1 to group with higher average magnetization
    domain_signs = normalize_domain_signs(domain_boundaries, domain_signs_raw, m_profile, x_axis_um)
    
    return domain_boundaries, domain_signs, peaks_pos, peaks_neg


def normalize_domain_signs(domain_boundaries, domain_signs_raw, m_profile, x_axis_um):
    """
    Normalize domain signs so that +1 corresponds to the domain sign group 
    with higher average magnetization.
    
    Returns normalized domain_signs list.
    """
    m_roi = np.asarray(m_profile, dtype=float)
    x_roi_um = np.asarray(x_axis_um, dtype=float)
    
    if len(domain_boundaries) == 0:
        return []
    
    # Compute average magnetization for each raw domain sign
    pos_domains_m = []
    neg_domains_m = []
    
    for (x_start, x_end), raw_sign in zip(domain_boundaries, domain_signs_raw):
        mask = (x_roi_um >= x_start) & (x_roi_um <= x_end)
        if np.any(mask):
            domain_m_avg = np.mean(m_roi[mask])
            if raw_sign > 0:
                pos_domains_m.append(domain_m_avg)
            else:
                neg_domains_m.append(domain_m_avg)
    
    # Compute group averages
    pos_group_avg = np.mean(pos_domains_m) if len(pos_domains_m) > 0 else -np.inf
    neg_group_avg = np.mean(neg_domains_m) if len(neg_domains_m) > 0 else -np.inf
    
    # Assign +1 to group with larger magnetization
    if pos_group_avg > neg_group_avg:
        sign_correction = 1
    else:
        sign_correction = -1
    
    domain_signs = [sign * sign_correction for sign in domain_signs_raw]
    return domain_signs


def create_domain_info_per_shot(df, seqs, scan, data_origin, mode_cfg, params):
    """
    Extract domain information for all shots in the analysis.
    
    Returns a dictionary with shot indices as keys and domain info as values:
    {
        shot_index: {
            'field': field_value,
            'domains': [
                {'x_start': x1, 'x_end': x2, 'sign': +1},
                ...
            ],
            'n_domains': number_of_domains,
            'defect_count': number_of_defects
        },
        ...
    }
    """
    domain_info_dict = {}
    
    if scan not in ('KZ_det_scan', 'KZ_det_scan_window'):
        return domain_info_dict
    
    # Extract defect analysis config
    defect_cfg = mode_cfg.get('defect_analysis', {}) if isinstance(mode_cfg, dict) else {}
    gauss_sigma = float(defect_cfg.get('gaussian_sigma', 0.3))
    pos_thr = float(defect_cfg.get('derivative_threshold_pos', 0.08))
    neg_thr = float(defect_cfg.get('derivative_threshold_neg', -0.08))
    
    # Get integration region
    x_min_um = float(params.get('DOMAIN_WALL_X_MIN', 0))
    x_max_um = float(params.get('DOMAIN_WALL_X_MAX', 100))
    um_per_px = float(params.get('UM_PER_PX', 1.019))
    
    # Find field column
    field_col = None
    for col in df.columns:
        if isinstance(col, tuple) and 'ARPKZ_final_set_field' in str(col[0]):
            field_col = col
            break
    
    if field_col is None:
        print(f"[Domain Extraction] Warning: Could not find ARPKZ_final_set_field column")
        return domain_info_dict
    
    # Find magnetization column - same approach as KZ_det_scan
    # Try to load PTAI_m1 magnetization profile
    mag_col = None
    candidates = [
        (data_origin, 'PTAI_m1_n1D_x'),
        (data_origin, 'PTAI_m1_SVD_n1D_x'),
        (data_origin, 'PTAI_m1_1d'),
    ]
    
    for cand in candidates:
        if cand in df.columns:
            mag_col = cand
            break
    
    if mag_col is None:
        print(f"[Domain Extraction] Warning: Could not find magnetization column")
        print(f"[Domain Extraction] Tried: {candidates}")
        return domain_info_dict
    
    print(f"[Domain Extraction] Found columns: mag={mag_col}, field={field_col}")
    
    # Iterate through shots
    for shot_idx in range(len(df)):
        try:
            field_value = float(df[field_col].iloc[shot_idx])
            
            m_profile = df[mag_col].iloc[shot_idx]
            if not isinstance(m_profile, np.ndarray):
                m_profile = np.asarray(m_profile, dtype=float)
            
            # Create x-axis
            x_centers_um = np.arange(m_profile.size) * um_per_px
            
            # Extract domain boundaries and signs
            domains_bd, domains_signs, peaks_pos, peaks_neg = extract_domain_boundaries_and_signs(
                m_profile, x_centers_um, x_min_um, x_max_um,
                gaussian_sigma=gauss_sigma,
                pos_threshold=pos_thr,
                neg_threshold=neg_thr
            )
            
            # Create domain info list
            domain_list = []
            for (x_start, x_end), sign in zip(domains_bd, domains_signs):
                domain_list.append({
                    'x_start_um': float(x_start),
                    'x_end_um': float(x_end),
                    'sign': int(sign),
                    'color': 'blue' if sign > 0 else 'red'
                })
            
            # Store in dict
            domain_info_dict[shot_idx] = {
                'field': float(field_value),
                'domains': domain_list,
                'n_domains': len(domain_list),
                'defect_count': len(peaks_pos) + len(peaks_neg),
                'n_pos_defects': len(peaks_pos),
                'n_neg_defects': len(peaks_neg),
            }
            
        except Exception as e:
            print(f"Warning: Could not extract domain info for shot {shot_idx}: {e}")
            continue
    
    return domain_info_dict


def save_domain_info_to_csv(domain_info_dict, output_dir='./results', filename='domain_info.csv'):
    """
    Save domain information to a CSV file.
    
    One row per shot with columns:
    shot_index, field, n_domains, defect_count, domain_json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    rows = []
    for shot_idx, info in domain_info_dict.items():
        row = {
            'shot_index': shot_idx,
            'field': info['field'],
            'n_domains': info['n_domains'],
            'defect_count': info['defect_count'],
            'n_pos_defects': info['n_pos_defects'],
            'n_neg_defects': info['n_neg_defects'],
            'domain_json': json.dumps(info['domains'])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    filepath = Path(output_dir) / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Domain info saved to {filepath}")
    return filepath


def save_domain_info_to_json(domain_info_dict, output_dir='./results', filename='domain_info.json'):
    """
    Save complete domain information to a JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = Path(output_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(domain_info_dict, f, indent=2)
    print(f"✓ Domain info saved to {filepath}")
    return filepath


def load_domain_info_from_json(filepath):
    """
    Load domain information from a JSON file.
    """
    with open(filepath, 'r') as f:
        domain_info = json.load(f)
    return domain_info
