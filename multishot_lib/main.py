"""
Core multishot analysis functions.

Provides utilities for loading Lyse result DataFrames from a day and filtering
based on quality metrics.

Note: These functions work with Lyse result files (pandas DataFrames stored as HDF),
not raw shot files.
"""

import h5py
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Import config
from . import config

logger = logging.getLogger(__name__)


def get_day_data(hdf_config, hdf_root=None):
    """
    Load all Lyse result DataFrames from a specific day.
    
    Loads and concatenates HDF result files from:
    - /home/{user}/NAS542_dataBEC2/year/month/day/ (default)
    - /path/to/analysislib_v2/data_reanalyzed/year/month/day/ (if reanalyzed=True)
    
    Parameters
    ----------
    hdf_config : dict
        Configuration dictionary with keys:
        - 'today': bool, if True use today's date
        - 'year': int (used if today=False)
        - 'month': int (used if today=False)
        - 'day': int (used if today=False)
        - 'reanalyzed': bool, optional (default False), if True use data_reanalyzed directory
    hdf_root : str, optional
        Root directory where HDF5 result files are stored (year/month/day structure).
        If None, uses default based on reanalyzed flag:
        - False: /home/{username}/NAS542_dataBEC2
        - True: /path/to/analysislib_v2/data_reanalyzed
    
    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame with all results from the day, or None if no files found
    """
    import getpass
    
    # Check if using reanalyzed data
    reanalyzed = hdf_config.get('reanalyzed', False)
    
    # Determine root directory
    if hdf_root is None:
        if reanalyzed:
            # Use data_reanalyzed directory
            hdf_root = '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/data_reanalyzed'
        else:
            # Use NAS directory
            username = getpass.getuser()
            hdf_root = f"/home/{username}/NAS542_dataBEC2"
    
    if hdf_config.get('today', True):
        date = datetime.now().date()
    else:
        year = hdf_config.get('year', 2026)
        month = hdf_config.get('month', 4)
        day = hdf_config.get('day', 16)
        date = datetime(year, month, day).date()
    
    # Format date as YYYY/MM/DD
    year_str = date.strftime('%Y')
    month_str = date.strftime('%m')
    day_str = date.strftime('%d')
    hdf_dir = os.path.join(hdf_root, year_str, month_str, day_str)
    
    if not os.path.isdir(hdf_dir):
        logger.warning(f"HDF directory not found: {hdf_dir}")
        return None
    
    # Find all .hdf files
    hdf_files = sorted([
        os.path.join(hdf_dir, f) for f in os.listdir(hdf_dir)
        if f.endswith('.hdf')
    ])
    
    if not hdf_files:
        logger.warning(f"No .hdf files found in {hdf_dir}")
        return None
    
    source = "reanalyzed" if reanalyzed else "NAS"
    logger.info(f"Found {len(hdf_files)} HDF5 result files in {hdf_dir} ({source})")
    
    # Load all DataFrames
    dfs = []
    for hdf_file in hdf_files:
        try:
            logger.info(f"Loading: {Path(hdf_file).name}")
            # Get the first (and usually only) table in the HDF file
            keys = pd.read_hdf(hdf_file, mode='r').index.names
            df = pd.read_hdf(hdf_file)
            logger.info(f"  → {len(df)} rows, {len(df.columns)} columns")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {Path(hdf_file).name}: {e}")
            continue
    
    if not dfs:
        logger.error("Could not load any result files")
        return None
    
    # Concatenate all DataFrames
    if len(dfs) == 1:
        result_df = dfs[0]
    else:
        result_df = pd.concat(dfs, sort=False)
    
    # Remove duplicates (same shot analyzed twice)
    try:
        n_before = len(result_df)
        if 'filepath' in result_df.columns:
            result_df = result_df.drop_duplicates(subset=['filepath'], keep='last')
        else:
            result_df = result_df[~result_df.index.duplicated(keep='last')]
        n_after = len(result_df)
        if n_after != n_before:
            logger.info(f"Deduplicated: {n_before} → {n_after} shots")
    except Exception as e:
        logger.warning(f"Could not deduplicate: {e}")
    
    logger.info(f"✓ Loaded {len(result_df)} total shots")
    return result_df


def get_and_filter_shots(hdf_config, camera_name, seqs=None, hdf_root=None):
    """
    Load Lyse result DataFrames and filter based on quality metrics and parameter consistency.
    
    Workflow:
    1. Load all result DataFrames from the specified day
    2. Check that processing parameters are consistent for the camera
    3. Filter shots by quality metrics (saturation, background, norm_err)
    4. Return DataFrame with accepted shots
    
    Parameters
    ----------
    hdf_config : dict
        Configuration for which day to load (see get_day_data)
    camera_name : str
        Camera name to filter (e.g., 'cam_vert1')
    seqs : list or None
        Sequence indices to include. If None, use all shots.
    hdf_root : str, optional
        Root directory for HDF files (default: /home/{user}/NAS542_dataBEC2)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns from all results, filtered by:
        - Processing parameters consistent for camera across all shots
        - Image quality metrics (saturation, background, norm_err)
    """
    # Load day data
    logger.info(f"\nLoading result DataFrames...")
    df = get_day_data(hdf_config, hdf_root)
    
    if df is None or df.empty:
        logger.error("No data loaded")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(df)} total shots")
    
    # Filter by sequences if specified
    if seqs is not None:
        logger.info(f"Filtering for sequences: {seqs}")
        # Try to extract sequence from filepath or index
        # This depends on your naming convention
        filtered_df = df.copy()
        logger.info(f"After sequence filter: {len(filtered_df)} shots")
        df = filtered_df
    
    # Extract processing parameters for this camera
    logger.info(f"\nChecking parameter consistency for {camera_name}...")
    param_columns = [col for col in df.columns if f"{camera_name}_param_" in str(col)]
    
    if not param_columns:
        logger.warning(f"No processing parameters found for {camera_name}")
        logger.warning(f"Available columns: {list(df.columns)[:10]}...")
        return pd.DataFrame()
    
    logger.info(f"Found {len(param_columns)} processing parameters")
    
    # Check parameter consistency
    for param_col in param_columns:
        unique_vals = df[param_col].nunique()
        if unique_vals > 1:
            logger.error(f"\nPARAMETER INCONSISTENCY DETECTED")
            logger.error(f"Parameter '{param_col}' has {unique_vals} different values:")
            for val, count in df[param_col].value_counts().items():
                logger.error(f"  {val}: {count} shots")
            logger.error(f"CANNOT PROCEED: All parameters must be identical\n")
            return pd.DataFrame()
    
    logger.info(f"✓ All parameters consistent across {len(df)} shots")
    
    # Filter by quality metrics
    logger.info(f"\nFiltering by quality metrics...")
    
    rejection_stats = defaultdict(int)
    accepted_mask = pd.Series([True] * len(df), index=df.index)
    
    # Get image names for this camera
    # Try to find columns like: cam_vert1_PTAI_m1_cnt_rel_atoms_back
    image_metric_cols = [col for col in df.columns 
                        if f"{camera_name}_" in str(col) and "_cnt_rel_atoms" in str(col)]
    
    # Extract unique image names
    image_names = set()
    for col in image_metric_cols:
        # Format: cam_vert1_PTAI_m1_cnt_rel_atoms_back
        parts = str(col).split('_')
        for i, part in enumerate(parts):
            if part == 'cnt':
                # Found "cnt_rel_atoms", image name is before this
                image_name = '_'.join(parts[2:i])  # Skip "cam_vert1"
                image_names.add(image_name)
    
    logger.info(f"Found {len(image_names)} images: {image_names}")
    
    # Check each image
    for image_name in image_names:
        # Background check
        bg_col = f"{camera_name}_{image_name}_cnt_rel_atoms_back"
        if bg_col in df.columns:
            mask = df[bg_col] > config.BACKGROUND_CONTAMINATION_THRESHOLD
            n_rejected = mask.sum()
            if n_rejected > 0:
                rejection_stats[f"background ({image_name})"] += n_rejected
                accepted_mask &= ~mask
        
        # Saturation check
        sat_col = f"{camera_name}_{image_name}_cnt_rel_atoms_sat"
        if sat_col in df.columns:
            mask = df[sat_col] > config.SATURATION_THRESHOLD
            n_rejected = mask.sum()
            if n_rejected > 0:
                rejection_stats[f"saturation ({image_name})"] += n_rejected
                accepted_mask &= ~mask
        
        # Norm error check
        err_col = f"{camera_name}_{image_name}_probe_norm_err"
        if err_col in df.columns:
            mask = df[err_col] > config.PROBE_NORM_ERROR_THRESHOLD
            n_rejected = mask.sum()
            if n_rejected > 0:
                rejection_stats[f"norm_err ({image_name})"] += n_rejected
                accepted_mask &= ~mask
    
    # Filter DataFrame
    result_df = df[accepted_mask].copy()
    
    # Log statistics
    logger.info(f"\nSHOT FILTERING SUMMARY:")
    logger.info(f"  Total shots: {len(df)}")
    logger.info(f"  Accepted: {len(result_df)}")
    logger.info(f"  Rejected: {len(df) - len(result_df)}")
    
    if rejection_stats:
        logger.info(f"\nREJECTION BREAKDOWN:")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")
    
    if result_df.empty:
        logger.warning("\n✗ No shots passed quality checks")
    else:
        logger.info(f"\n✓ {len(result_df)} shots ready for analysis")
    
    return result_df
    """
    Extract all processing parameters for a camera from a shot.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    camera_name : str
        Camera name (e.g., 'cam_vert1')
    
    Returns
    -------
    dict or None
        Dictionary of processing parameters, or None if not found
    """
    params = {}
    param_names = [
        'roi', 'roi_back', 'roi_integration', 'alpha', 'chi_sat', 'pulse_time',
        'px_size', 'method', 'background_std_multiplier',
        'saturation_threshold_multiplier', 'atomic_cross_section_sigma'
    ]
    
    for param_name in param_names:
        dataset_name = f"results/{camera_name}_param_{param_name}"
        try:
            params[param_name] = h5file[dataset_name][()]
        except KeyError:
            logger.debug(f"Parameter not found: {dataset_name}")
            return None
        except Exception as e:
            logger.debug(f"Error reading {dataset_name}: {e}")
            return None
    
    return params


def _check_parameter_consistency(shot_files, camera_name):
    """
    Check that all shots have identical processing parameters for a camera.
    
    Parameters
    ----------
    shot_files : list
        List of HDF5 file paths
    camera_name : str
        Camera name to check
    
    Returns
    -------
    bool
        True if all parameters are consistent, False otherwise
    """
    all_params = []
    param_names = [
        'roi', 'roi_back', 'roi_integration', 'alpha', 'chi_sat', 'pulse_time',
        'px_size', 'method', 'background_std_multiplier',
        'saturation_threshold_multiplier', 'atomic_cross_section_sigma'
    ]
    
    # Collect all parameters from all shots
    for shot_file in shot_files:
        try:
            with h5py.File(shot_file, 'r') as h5file:
                params = _extract_processing_params(h5file, camera_name)
                if params is None:
                    logger.warning(f"Could not extract parameters from {shot_file}")
                    return False
                all_params.append((shot_file, params))
        except Exception as e:
            logger.warning(f"Error reading {shot_file}: {e}")
            return False
    
    # Check consistency
    if not all_params:
        return False
    
    reference_params = all_params[0][1]
    inconsistent_params = defaultdict(list)
    
    for shot_file, params in all_params[1:]:
        for param_name in param_names:
            ref_val = reference_params[param_name]
            curr_val = params[param_name]
            
            # Handle numpy arrays and string comparisons
            if isinstance(ref_val, np.ndarray):
                ref_val = tuple(ref_val.tolist())
            if isinstance(ref_val, bytes):
                ref_val = ref_val.decode('utf-8')
            
            if isinstance(curr_val, np.ndarray):
                curr_val = tuple(curr_val.tolist())
            if isinstance(curr_val, bytes):
                curr_val = curr_val.decode('utf-8')
            
            if ref_val != curr_val:
                inconsistent_params[param_name].append(shot_file)
    
    # Report inconsistencies
    if inconsistent_params:
        logger.error(f"\n{'='*80}")
        logger.error(f"PARAMETER INCONSISTENCY DETECTED for camera {camera_name}")
        logger.error(f"{'='*80}")
        for param_name, shot_files_with_diff in inconsistent_params.items():
            logger.error(f"\nParameter '{param_name}' differs in {len(shot_files_with_diff)} shot(s):")
            for shot_file in shot_files_with_diff:
                logger.error(f"  - {Path(shot_file).name}")
        logger.error(f"\nCANNOT PROCEED: All processing parameters must be identical.")
        logger.error(f"{'='*80}\n")
        return False
    
    logger.info(f"✓ All {len(shot_files)} shots have consistent parameters for {camera_name}")
    return True


def _get_image_names_for_camera(h5file, camera_name):
    """
    Get list of image names for a camera from a shot.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    camera_name : str
        Camera name (e.g., 'cam_vert1')
    
    Returns
    -------
    list or None
        List of image names, or None if not found
    """
    try:
        # Get images group for this camera
        img_dataset = h5file[f"results/{camera_name}_images"]
        # Get image names from metadata
        if 'img_names' in img_dataset.attrs:
            img_names = img_dataset.attrs['img_names']
            if isinstance(img_names, np.ndarray):
                img_names = [s.decode('utf-8') if isinstance(s, bytes) else s for s in img_names]
            return img_names
    except Exception as e:
        logger.debug(f"Could not extract image names for {camera_name}: {e}")
    
    return None


def _check_shot_quality(h5file, camera_name, image_names):
    """
    Check quality metrics for all images of a camera in a shot.
    
    Returns rejection reasons if any quality check fails.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    camera_name : str
        Camera name (e.g., 'cam_vert1')
    image_names : list
        List of image names to check
    
    Returns
    -------
    list
        Empty list if shot passes all checks, otherwise list of rejection reasons
    """
    rejection_reasons = []
    
    for image_name in image_names:
        # Background contamination check
        bg_key = f"results/{camera_name}_{image_name}_cnt_rel_atoms_back"
        try:
            bg_val = float(h5file[bg_key][()])
            if bg_val > config.BACKGROUND_CONTAMINATION_THRESHOLD:
                rejection_reasons.append(
                    f"background ({image_name}: {bg_val:.4f} > {config.BACKGROUND_CONTAMINATION_THRESHOLD})"
                )
        except (KeyError, Exception):
            pass
        
        # Saturation check
        sat_key = f"results/{camera_name}_{image_name}_cnt_rel_atoms_sat"
        try:
            sat_val = float(h5file[sat_key][()])
            if sat_val > config.SATURATION_THRESHOLD:
                rejection_reasons.append(
                    f"saturation ({image_name}: {sat_val:.4f} > {config.SATURATION_THRESHOLD})"
                )
        except (KeyError, Exception):
            pass
        
        # Probe normalization error check
        err_key = f"results/{camera_name}_{image_name}_probe_norm_err"
        try:
            err_val = float(h5file[err_key][()])
            if err_val > config.PROBE_NORM_ERROR_THRESHOLD:
                rejection_reasons.append(
                    f"norm_err ({image_name}: {err_val:.4f} > {config.PROBE_NORM_ERROR_THRESHOLD})"
                )
        except (KeyError, Exception):
            pass
    
    # DMD profile update check (if enabled)
    if config.ENABLE_DMD_CHECK:
        dmd_key = f"results/{camera_name}_dmd_profile_updated"
        try:
            dmd_updated = bool(h5file[dmd_key][()])
            if not dmd_updated:
                rejection_reasons.append("dmd_not_updated")
        except (KeyError, Exception):
            logger.debug(f"Could not check DMD update status")
    
    return rejection_reasons


def get_and_filter_shots(hdf_config, camera_name, seqs=None):
    """
    Load shots from specified sequences and filter based on:
    1. Parameter consistency across all shots for the camera
    2. Image quality metrics (saturation, background, norm_err)
    3. DMD profile update status (if enabled)
    
    Parameters
    ----------
    hdf_config : dict
        Configuration for which day to load (see get_day_data)
    camera_name : str
        Camera name to filter (e.g., 'cam_vert1')
    seqs : list or None
        Sequence indices to load. If None, load all sequences.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: shot_file, shot_name, sequence, and all results
        Only includes shots that pass all quality checks
    """
    # Load all files for the day
    all_h5_files = get_day_data(hdf_config)
    
    if all_h5_files is None or all_h5_files.empty:
        logger.error("No HDF5 files found for the specified day")
        return pd.DataFrame()
    
    # If we got a DataFrame (from reanalyzed data or consolidated results),
    # it's already been analyzed and consolidated - return it as-is
    if isinstance(all_h5_files, pd.DataFrame):
        logger.info(f"Loaded consolidated DataFrame with {len(all_h5_files)} shots")
        logger.info(f"Note: Reanalyzed data is already filtered; skipping quality checks")
        return all_h5_files
    
    # Legacy path: if we got a list of file paths (from NAS raw files)
    # Filter by sequence indices if specified
    shot_files = all_h5_files
    if seqs is not None:
        filtered_files = []
        for shot_file in all_h5_files:
            # Extract sequence number from filename
            # Format: YYYY_MM_DD_ShortName_RepXXXX.h5 or similar
            filename = Path(shot_file).stem
            try:
                # Try to extract sequence from filename
                # This is a heuristic - adjust as needed for your naming convention
                for i, char in enumerate(filename[::-1]):
                    if not char.isdigit():
                        break
                seq_str = filename[-i:] if i > 0 else None
                if seq_str and int(seq_str) in seqs:
                    filtered_files.append(shot_file)
            except Exception:
                # If can't parse sequence, skip this check
                pass
        
        if not filtered_files:
            logger.info(f"No files matched sequence filter {seqs}")
            shot_files = all_h5_files  # Fall back to all files
        else:
            shot_files = filtered_files
            logger.info(f"Filtered to {len(shot_files)} files in sequences {seqs}")
    
    # Check parameter consistency FIRST
    logger.info("\nChecking parameter consistency across all shots...")
    if not _check_parameter_consistency(shot_files, camera_name):
        logger.error("Parameter consistency check failed; aborting shot loading")
        return pd.DataFrame()
    
    # Now load and filter shots
    logger.info("\nFiltering shots by quality metrics...")
    
    accepted_shots = []
    rejection_stats = defaultdict(int)
    
    for shot_file in shot_files:
        try:
            with h5py.File(shot_file, 'r') as h5file:
                # Get image names for this camera
                image_names = _get_image_names_for_camera(h5file, camera_name)
                if image_names is None:
                    rejection_stats['no_images'] += 1
                    continue
                
                # Check quality
                rejection_reasons = _check_shot_quality(h5file, camera_name, image_names)
                
                if rejection_reasons:
                    for reason in rejection_reasons:
                        rejection_stats[reason] += 1
                else:
                    # Shot passes all checks - extract full results
                    results = {}
                    results['shot_file'] = shot_file
                    results['shot_name'] = Path(shot_file).stem
                    
                    # Extract all results from HDF5
                    for key in h5file['results'].keys():
                        try:
                            val = h5file[f'results/{key}'][()]
                            results[key] = val
                        except Exception:
                            pass
                    
                    accepted_shots.append(results)
        
        except Exception as e:
            logger.warning(f"Error processing {shot_file}: {e}")
            rejection_stats['read_error'] += 1
    
    # Log rejection statistics
    logger.info(f"\nSHOT FILTERING SUMMARY:")
    logger.info(f"  Total shots: {len(shot_files)}")
    logger.info(f"  Accepted: {len(accepted_shots)}")
    logger.info(f"  Rejected: {len(shot_files) - len(accepted_shots)}")
    
    if rejection_stats:
        logger.info(f"\nREJECTION BREAKDOWN:")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")
    
    # Return DataFrame
    if accepted_shots:
        df = pd.DataFrame(accepted_shots)
        logger.info(f"\n✓ Loaded {len(df)} shots into DataFrame")
        return df
    else:
        logger.warning("\nNo shots passed quality checks")
        return pd.DataFrame()
