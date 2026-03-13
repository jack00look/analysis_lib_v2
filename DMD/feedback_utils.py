"""
Utility functions for DMD feedback scripts.
"""

import numpy as np
import zerorpc
import os
import h5py
from datetime import datetime
from scipy.interpolate import interp1d


def compute_magnetization_profile(images, roi_integration=None, y_roi=None):
    """
    Computes magnetization profile from two-component images.
    
    Parameters:
    -----------
    images : dict
        Dictionary containing 'PTAI_m1' and 'PTAI_m2' keys with 2D image arrays
    roi_integration : slice object, tuple of slices, or None
        ROI for integration (used directly as array index)
        If None, uses full image
    y_roi : tuple or None
        Y-axis ROI for averaging (y_start, y_end)
        If None, uses full y-axis
        
    Returns:
    --------
    z1d : np.ndarray
        1D magnetization profile along x-axis
    Z : np.ndarray
        2D magnetization map
    """
    img_m1 = images['PTAI_m1']
    img_m2 = images['PTAI_m2']
    
    # Apply ROI if specified (use directly as index like in feedback_atoms.py)
    if roi_integration is not None:
        img_m1 = img_m1[roi_integration]
        img_m2 = img_m2[roi_integration]
    
    # Compute total density and magnetization
    D_tot = img_m1 + img_m2
    
    # Compute magnetization Z = (m2 - m1) / (m1 + m2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (img_m2 - img_m1) / D_tot
    
    # Replace NaN/Inf with 0
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip to physical range [-1, 1]
    Z = np.clip(Z, -1, 1)
    
    # Integrate along y to get 1D profile
    if y_roi is not None:
        y_start, y_end = y_roi
        z_roi_y = Z[y_start:y_end, :]
    else:
        z_roi_y = Z
    
    # Average along y-axis
    z1d = np.nanmean(z_roi_y, axis=0)
    
    return z1d, Z


def get_dmd_profile(client, h5file, results_group):
    """
    Gets the DMD profile either from saved data (if reanalysis) or from server.
    
    Parameters:
    -----------
    client : zerorpc.Client
        ZeroRPC client connected to DMD server
    h5file : h5py.File
        HDF5 file handle
    results_group : str
        HDF5 group path where results are stored
        
    Returns:
    --------
    last_profile_x : np.ndarray
        X-axis coordinates of the profile (in microns)
    last_profile_y : np.ndarray
        Y-axis values (intensity) of the profile
    already_used : bool
        Whether this shot was already used for feedback
    """
    flag_dataset = results_group + '/dmd_profile_updated'
    
    # Check if this shot already updated the DMD profile
    already_used = flag_dataset in h5file and bool(h5file[flag_dataset][()])
    
    if already_used:
        # Load the previously saved profiles from h5
        print('This shot has already been used for DMD feedback.')
        print('Loading saved DMD profile from H5 file.')
        last_profile_x = np.array(h5file[results_group + '/last_profile_x_um'][...])
        last_profile_y = np.array(h5file[results_group + '/last_profile_y'][...])
    else:
        # First run: get current state from DMD server
        print('Getting current DMD profile from server.')
        last_profile = client.get_last_atoms_profile()
        # Use server x-axis if provided, otherwise will need to be set by caller
        last_profile_x = np.array(last_profile['x']) if 'x' in last_profile else None
        last_profile_y = np.array(last_profile['y'])
    
    return last_profile_x, last_profile_y, already_used


def update_dmd_profile(client, profile_x, profile_y, already_used):
    """
    Updates the DMD profile on the server if not already used.
    
    Parameters:
    -----------
    client : zerorpc.Client
        ZeroRPC client connected to DMD server
    profile_x : np.ndarray
        X-axis coordinates of the profile (in microns)
    profile_y : np.ndarray
        Y-axis values (intensity) of the profile
    already_used : bool
        Whether this shot was already used for feedback
        
    Returns:
    --------
    status : str
        Status message from server or indication if skipped
    updated_now : bool
        Whether the profile was actually updated in this call
    """
    if already_used:
        status = 'skipped_already_used_for_feedback'
        updated_now = False
        print('DMD profile update: NO (already used for feedback)')
    else:
        print('Sending profile to DMD server...')
        status = client.load_1d_profile(profile_x.tolist(), profile_y.tolist())
        updated_now = True
        print('status: ', status)
        print('DMD profile update: YES')
    
    return status, updated_now


def parse_h5_filename_for_timestamp(filename):
    """
    Extract timestamp from h5 filename.
    Expected format: YYYY-MM-DD_HHMMSS_<program_name>_<shot_number>.h5
    
    Returns:
    --------
    timestamp : float or None
        Unix timestamp in seconds, or None if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) < 2:
            return None
        
        date_str = parts[0]  # YYYY-MM-DD
        time_str = parts[1]  # HHMMSS
        
        # Parse to datetime
        dt_str = f"{date_str} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        
        return dt.timestamp()
    except Exception as e:
        print(f'Warning: Could not parse timestamp from filename {filename}: {e}')
        return None


def get_h5_run_time(h5_path):
    """
    Read 'run time' attribute from h5 file.
    
    Returns:
    --------
    timestamp : float or None
        Unix timestamp in seconds
    """
    try:
        with h5py.File(h5_path, 'r') as h5:
            run_time = h5.attrs.get('run time', None)
            if run_time is None:
                return None
            
            # If it's bytes, decode
            if isinstance(run_time, bytes):
                run_time = run_time.decode('utf-8')
            
            # If it's a string, try parsing
            if isinstance(run_time, str):
                try:
                    return float(run_time)
                except:
                    # Try parsing as ISO format
                    dt = datetime.fromisoformat(run_time.replace('Z', '+00:00'))
                    return dt.timestamp()
            
            # Otherwise assume it's numeric
            return float(run_time)
    except Exception as e:
        print(f'Warning: Could not read run time from {h5_path}: {e}')
        return None


def find_eligible_previous_shots(current_h5_path, program_name, n_shots_needed, 
                                  max_lookback_time, results_group, strict_mode=True):
    """
    Find previous shots eligible for multi-shot averaging.
    
    Parameters:
    -----------
    current_h5_path : str
        Path to current shot's h5 file
    program_name : str
        Name of the sequence program to match
    n_shots_needed : int
        Number of previous shots needed (N-1 for N-shot average)
    max_lookback_time : float
        Maximum time to look back (seconds)
    results_group : str
        HDF5 group where feedback results are stored
    strict_mode : bool
        If True, exclude shots that already triggered feedback
        
    Returns:
    --------
    eligible_shots : list
        List of h5 file paths (most recent first, excluding current shot)
    """
    # Get current shot timestamp
    current_timestamp = get_h5_run_time(current_h5_path)
    if current_timestamp is None:
        current_timestamp = parse_h5_filename_for_timestamp(current_h5_path)
    
    if current_timestamp is None:
        print('Warning: Could not determine current shot timestamp')
        return []
    
    # Get directory containing shots
    shot_dir = os.path.dirname(current_h5_path)
    
    # Scan for h5 files
    eligible_shots = []
    
    try:
        all_files = sorted(os.listdir(shot_dir), reverse=True)  # Most recent first
    except Exception as e:
        print(f'Warning: Could not list directory {shot_dir}: {e}')
        return []
    
    for filename in all_files:
        if not filename.endswith('.h5'):
            continue
        
        filepath = os.path.join(shot_dir, filename)
        
        # Skip current shot
        if os.path.abspath(filepath) == os.path.abspath(current_h5_path):
            continue
        
        # Check timestamp
        shot_timestamp = get_h5_run_time(filepath)
        if shot_timestamp is None:
            shot_timestamp = parse_h5_filename_for_timestamp(filepath)
        
        if shot_timestamp is None:
            continue
        
        # Check if within time window
        time_diff = current_timestamp - shot_timestamp
        if time_diff < 0 or time_diff > max_lookback_time:
            continue
        
        # Check if shot is eligible
        try:
            with h5py.File(filepath, 'r') as h5:
                # Check program name
                program_dataset = results_group + '/program_name'
                if program_dataset in h5:
                    stored_program = h5[program_dataset][()]
                    if isinstance(stored_program, bytes):
                        stored_program = stored_program.decode('utf-8')
                    if stored_program != program_name:
                        continue
                else:
                    continue
                
                # Check eligibility
                eligible_dataset = results_group + '/feedback_eligible'
                if eligible_dataset in h5:
                    is_eligible = bool(h5[eligible_dataset][()])
                    if not is_eligible:
                        continue
                else:
                    # If eligibility not stored, skip (old shot format)
                    continue
                
                # In strict mode, skip shots that already triggered feedback
                if strict_mode:
                    trigger_dataset = results_group + '/multishot_trigger'
                    if trigger_dataset in h5:
                        is_trigger = bool(h5[trigger_dataset][()])
                        if is_trigger:
                            continue
                
                # Check that error profile exists
                error_x_dataset = results_group + '/error_profile_x'
                error_dataset = results_group + '/error_profile'
                if error_x_dataset not in h5 or error_dataset not in h5:
                    continue
                
                # This shot is eligible
                eligible_shots.append(filepath)
                
                if len(eligible_shots) >= n_shots_needed:
                    break
                    
        except Exception as e:
            print(f'Warning: Could not check eligibility of {filepath}: {e}')
            continue
    
    return eligible_shots


def retrieve_error_profiles(h5_paths, results_group):
    """
    Load error profiles from a list of h5 files.
    
    Parameters:
    -----------
    h5_paths : list
        List of h5 file paths
    results_group : str
        HDF5 group where feedback results are stored
        
    Returns:
    --------
    error_profiles : list
        List of (x_vals, error_profile) tuples
    """
    profiles = []
    
    for path in h5_paths:
        try:
            with h5py.File(path, 'r') as h5:
                error_x = np.array(h5[results_group + '/error_profile_x'][...])
                error_y = np.array(h5[results_group + '/error_profile'][...])
                profiles.append((error_x, error_y))
        except Exception as e:
            print(f'Warning: Could not load error profile from {path}: {e}')
            continue
    
    return profiles


def average_error_profiles(error_profiles_list):
    """
    Average multiple error profiles, handling different x-axes via interpolation.
    
    Parameters:
    -----------
    error_profiles_list : list
        List of (x_vals, error_profile) tuples
        
    Returns:
    --------
    x_common : np.ndarray
        Common x-axis for averaged profile
    error_averaged : np.ndarray
        Averaged error profile
    """
    if len(error_profiles_list) == 0:
        raise ValueError('No error profiles provided for averaging')
    
    if len(error_profiles_list) == 1:
        return error_profiles_list[0]
    
    # Use the first (current) shot's x-axis as reference
    x_common = error_profiles_list[0][0]
    
    # Interpolate all profiles to common x-axis
    interpolated_profiles = []
    
    for x_vals, error_vals in error_profiles_list:
        if np.array_equal(x_vals, x_common):
            # Already on common grid
            interpolated_profiles.append(error_vals)
        else:
            # Interpolate to common grid
            try:
                interp_func = interp1d(x_vals, error_vals, kind='linear', 
                                       bounds_error=False, fill_value=0.0)
                error_interp = interp_func(x_common)
                interpolated_profiles.append(error_interp)
            except Exception as e:
                print(f'Warning: Could not interpolate profile: {e}')
                continue
    
    # Average all profiles
    error_averaged = np.mean(interpolated_profiles, axis=0)
    
    return x_common, error_averaged


def mark_shots_as_contributors(shot_paths, results_group, iteration_id):
    """
    Mark shots as having contributed to a feedback iteration.
    
    Parameters:
    -----------
    shot_paths : list
        List of h5 file paths that contributed
    results_group : str
        HDF5 group where feedback results are stored
    iteration_id : str
        Unique identifier for this feedback iteration
    """
    for path in shot_paths:
        try:
            with h5py.File(path, 'r+') as h5:
                # Mark as contributor
                contrib_dataset = results_group + '/multishot_contributor'
                if contrib_dataset in h5:
                    del h5[contrib_dataset]
                h5.create_dataset(contrib_dataset, data=np.bool_(True))
                
                # Add to list of iterations this shot participated in
                iterations_dataset = results_group + '/contributing_to_iterations'
                if iterations_dataset in h5:
                    existing = [item.decode('utf-8') if isinstance(item, bytes) else str(item) 
                               for item in h5[iterations_dataset][:]]
                    if iteration_id not in existing:
                        existing.append(iteration_id)
                        del h5[iterations_dataset]
                        h5.create_dataset(iterations_dataset, 
                                        data=np.array([np.bytes_(s) for s in existing]))
                else:
                    h5.create_dataset(iterations_dataset, 
                                    data=np.array([np.bytes_(iteration_id)]))
                    
        except Exception as e:
            print(f'Warning: Could not mark {path} as contributor: {e}')
            continue
