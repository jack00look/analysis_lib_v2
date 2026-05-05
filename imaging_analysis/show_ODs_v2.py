"""
Refactored OD display and analysis script using imaging_analysis_lib_v2

This script:
1. Loads OD images from HDF5 files
2. Displays them with 1D projections
3. Saves detailed processing parameters for reproducibility
4. Removes DMD-related functionality
5. Improves code organization and logging
"""

import h5py
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time
import logging
import sys
import os
from datetime import datetime

# Add paths for imports
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load modules using v2 library
spec = importlib.util.spec_from_file_location(
    "imaging_analysis_lib.camera_settings",
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py"
)
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location(
    "imaging_analysis_lib_v2.main",
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib_v2/main.py"
)
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location(
    "general_lib.main",
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py"
)
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

cameras = camera_settings.cameras

# DMD validation parameters
LOAD_TIME_WINDOW_S = 20.0


def _to_unix_timestamp(value):
    """Best-effort conversion of server/h5 timestamps to unix seconds."""
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')

    if isinstance(value, str):
        value = value.strip()
        if value == '':
            return None

        # Numeric string (epoch seconds)
        try:
            return float(value)
        except Exception:
            pass

        # ISO-like timestamp
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            pass

        # Common explicit formats
        for fmt in ('%Y%m%dT%H%M%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except Exception:
                continue

    return None


def _get_shot_run_time_timestamp(h5file):
    """Read shot 'run time' global from h5 and return unix seconds."""
    try:
        # Direct access from h5file.attrs (not globals.attrs)
        run_time_value = h5file.attrs['run time']
        ts = _to_unix_timestamp(run_time_value)
        if ts is not None:
            logger.info(f"DEBUG: 'run time' parsed successfully: {repr(run_time_value)} -> {ts}")
            return ts
        else:
            logger.warning(f"DEBUG: Could not parse 'run time' value: {repr(run_time_value)}")
            return None
    except KeyError:
        logger.warning(f"DEBUG: 'run time' key not found in h5file.attrs")
        logger.warning(f"DEBUG: available keys in h5file.attrs: {list(h5file.attrs.keys())}")
        return None
    except Exception as e:
        logger.warning(f'Error reading run time from h5file.attrs: {e}')
        return None


def _validate_dmd_state_h5(h5file, run_time_ts):
    """
    Validate DMD state using h5 file /data/dmd group.
    
    Returns tuple: (is_valid, validation_message)
    """
    try:
        if '/data/dmd/is_loading' not in h5file:
            return True, 'DMD validation: /data/dmd/is_loading not found (skipping validation)'
        
        is_loading = bool(h5file['/data/dmd/is_loading'][()])
        if is_loading:
            return False, 'DMD validation: DMD was loading when shot was taken -> INVALID'
        
        # DMD was not loading, check if last load finished in time
        if '/data/dmd/last_load_finished' not in h5file:
            return True, 'DMD validation: /data/dmd/last_load_finished not found (skipping timing check)'
        
        if run_time_ts is None:
            return True, 'DMD validation: run_time_ts is None, cannot perform timing check'
        
        last_load_finished_ts = float(h5file['/data/dmd/last_load_finished'][()])
        dt = run_time_ts - last_load_finished_ts
        
        if dt <= LOAD_TIME_WINDOW_S:
            return False, (
                f'DMD validation: Load finished too recently '
                f'(only {dt:.1f} s ago, need > {LOAD_TIME_WINDOW_S:.0f} s) -> INVALID'
            )
        else:
            return True, (
                f'DMD validation: Load finished long enough ago '
                f'({dt:.1f} s ago > {LOAD_TIME_WINDOW_S:.0f} s) -> VALID'
            )
    
    except Exception as e:
        return True, f'DMD validation: Error reading DMD validation data from h5: {e}'


def show_ODs(h5file, show=True, show_SVD=True):
    """
    Show OD images from h5 file with detailed processing information.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file with image data
    show : bool
        Whether to display plots (default: True)
    show_SVD : bool
        Whether to show SVD-reconstructed probe images (default: True)
    
    Returns
    -------
    dict
        Analysis results including:
        - n1D projections (n1D_x, n1D_y)
        - Center of mass (com_x, com_y)
        - Processing parameters (alpha, chi_sat, pulse_time, roi, etc.)
        - Image metadata (OD sums, normalization errors, etc.)
    """
    infos_analysis = {}
    
    # =========================================================================
    # DMD Validation Check (h5-based)
    # =========================================================================
    run_time_ts = _get_shot_run_time_timestamp(h5file)
    dmd_is_valid, dmd_validation_msg = _validate_dmd_state_h5(h5file, run_time_ts)
    logger.info(dmd_validation_msg)
    infos_analysis['dmd_validation_valid'] = dmd_is_valid
    infos_analysis['dmd_validation_message'] = dmd_validation_msg
    
    try:
        for cam_entry in cameras:
            cam = cameras[cam_entry]
            logger.info(f"Processing camera: {cam['name']}")
            
            # Process standard (non-SVD) images
            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
            if dict_images is None:
                logger.info(f"No images found for {cam['name']}")
                continue
            
            # Process SVD images if requested
            dict_images_SVD = None
            if show_SVD:
                dict_images_SVD = imaging_analysis_lib_mod.get_images_camera_waxes_SVD(h5file, cam)
                if dict_images_SVD is not None:
                    logger.info(f"Found SVD images: {list(dict_images_SVD.images.keys())}")
                else:
                    logger.info(f"No SVD images found for camera {cam['name']}")
            
            # Process standard images
            images = dict_images.images
            infos_images = dict_images.infos
            process_params = dict_images.process_params
            
            keys = list(images.keys())
            if len(keys) == 0:
                continue
            
            normal_keys = [key for key in keys if 'SVD' not in key]
            L = len(normal_keys)
            
            # Create figure for normal ODs
            fig, ax = plt.subplots(
                L * 2, 2,
                figsize=(L * 6, 6),
                constrained_layout=True,
                gridspec_kw={'width_ratios': [1, 1]}
            )
            
            if L == 1:
                ax = np.array([ax])  # Ensure 2D array (L*2, 2) for consistency
                # Now reshape to ensure it's 2D with correct dimensions
                ax = ax.reshape(L * 2, 2)
            
            logger.info(f"Creating OD figure with {L} images, {L*2} rows of subplots")
            
            for i in range(L):
                key = normal_keys[i]
                logger.info(f"Plotting: {key}")
                j = 0
                
                # Save all image metadata
                for key_infos in infos_images[key]:
                    infos_analysis[f"{key}_{key_infos}"] = infos_images[key][key_infos]
                
                # Plot the OD
                n1D_x, n1D_y = imaging_analysis_lib_mod.plot_OD(dict_images, key, cam, fig, ax, i, j)
                
                # Save 1D projections
                infos_analysis[f"{key}_n1D_x"] = n1D_x
                infos_analysis[f"{key}_n1D_y"] = n1D_y
                
                # Calculate center of mass
                x_vals = np.arange(len(n1D_x))
                y_vals = np.arange(len(n1D_y))
                com_x = np.sum(n1D_x * x_vals) / np.sum(n1D_x) if np.sum(n1D_x) > 0 else 0
                com_y = np.sum(n1D_y * y_vals) / np.sum(n1D_y) if np.sum(n1D_y) > 0 else 0
                
                infos_analysis[f"{key}_com_x"] = com_x
                infos_analysis[f"{key}_com_y"] = com_y
                
                # Maintain backwards compatibility with old naming
                infos_analysis[f"{key[5:]}_1d"] = n1D_x
                
                # Share axes for visual consistency
                if i > 0:
                    ax[2 * i, 1].sharex(ax[0, 1])
                    ax[2 * i, 1].sharey(ax[0, 1])
            
            # Set title
            title = general_lib_mod.get_title(h5file, cam)
            fig.suptitle(title, fontsize=12)
            
            # Save processing parameters used
            if process_params:
                for param_name, param_value in process_params.items():
                    infos_analysis[f"{cam['name']}_param_{param_name}"] = param_value
            
            # Process SVD images if available
            if dict_images_SVD is not None:
                images_svd = dict_images_SVD.images
                infos_images_svd = dict_images_SVD.infos
                process_params_svd = dict_images_SVD.process_params
                svd_keys = list(images_svd.keys())
                L_svd = len(svd_keys)
                
                if L_svd > 0:
                    fig_svd, ax_svd = plt.subplots(
                        L_svd * 2, 2,
                        figsize=(L_svd * 6, 6),
                        constrained_layout=True,
                        gridspec_kw={'width_ratios': [1, 1]}
                    )
                    
                    if L_svd == 1:
                        ax_svd = np.array([ax_svd])
                    
                    logger.info(f"Creating SVD OD figure with {L_svd} images, {L_svd*2} rows")
                    
                    for i in range(L_svd):
                        key = svd_keys[i]
                        logger.info(f"Plotting SVD: {key}")
                        j = 0
                        
                        # Save SVD metadata
                        for key_infos in infos_images_svd[key]:
                            infos_analysis[f"{key}_{key_infos}"] = infos_images_svd[key][key_infos]
                        
                        # Plot the SVD OD
                        n1D_x, n1D_y = imaging_analysis_lib_mod.plot_OD(
                            dict_images_SVD, key, cam, fig_svd, ax_svd, i, j
                        )
                        
                        # Save 1D projections
                        infos_analysis[f"{key}_n1D_x"] = n1D_x
                        infos_analysis[f"{key}_n1D_y"] = n1D_y
                        
                        # Calculate center of mass
                        x_vals = np.arange(len(n1D_x))
                        y_vals = np.arange(len(n1D_y))
                        com_x = np.sum(n1D_x * x_vals) / np.sum(n1D_x) if np.sum(n1D_x) > 0 else 0
                        com_y = np.sum(n1D_y * y_vals) / np.sum(n1D_y) if np.sum(n1D_y) > 0 else 0
                        
                        infos_analysis[f"{key}_com_x"] = com_x
                        infos_analysis[f"{key}_com_y"] = com_y
                        
                        # Share axes
                        if i > 0:
                            ax_svd[2 * i, 1].sharex(ax_svd[0, 1])
                            ax_svd[2 * i, 1].sharey(ax_svd[0, 1])
                    
                    title_svd = general_lib_mod.get_title(h5file, cam) + "\n[SVD Reconstructed Probes]"
                    fig_svd.suptitle(title_svd, fontsize=12)
                    
                    # Save SVD processing parameters
                    if process_params_svd:
                        for param_name, param_value in process_params_svd.items():
                            infos_analysis[f"{cam['name']}_svd_param_{param_name}"] = param_value
        
        logger.info(f"Analysis complete. Saved {len(infos_analysis)} results.")
        return infos_analysis
    
    except Exception as e:
        logger.error(f"Error in show_ODs: {e}", exc_info=True)
        return infos_analysis

if __name__ == '__main__':
    time_0 = time.time()
    
    # Check if running from lyse or from command line
    h5_file_path = None
    if len(sys.argv) > 1:
        h5_file_path = sys.argv[1]
    
    try:
        if h5_file_path:
            # Command line mode: process the specified file
            logger.info(f"Processing file: {h5_file_path}")
            with h5py.File(h5_file_path, 'r') as h5file:
                dict_results = show_ODs(h5file)
            
            if dict_results is not None:
                logger.info(f"Results obtained: {len(dict_results)} items")
                for key in dict_results:
                    logger.debug(f"  {key}: {type(dict_results[key])}")
        else:
            # Lyse mode: use lyse.path
            import lyse
            
            run = lyse.Run(lyse.path)
            dict_results = None
            
            with h5py.File(lyse.path, 'r+') as h5file:
                dict_results = show_ODs(h5file)
            
            # Save all results to lyse database
            if dict_results is not None:
                for key in dict_results:
                    try:
                        run.save_result(key, dict_results[key])
                    except Exception as e:
                        logger.warning(f"Could not save result {key}: {e}")
        
        elapsed = time.time() - time_0
        logger.info(f'Analysis completed in {elapsed:.2f} s')
        print(f'Elapsed time: {elapsed:.2f} s')
    
    except Exception as e:
        elapsed = time.time() - time_0
        logger.error(f'Error: {e}', exc_info=True)
        print(f'Elapsed time: {elapsed:.2f} s')
