"""
Simple PHC image display and analysis script using imaging_analysis_lib_v2

This script:
1. Loads PHC images from HDF5 files (cam_vert2_PHC only)
2. Displays the image with 1D projections
3. Saves detailed processing parameters for reproducibility
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
from scipy.ndimage import gaussian_filter1d

# Add paths for imports
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

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


def show_PHC_image(h5file, show=True):
    """
    Show phase contrast image from cam_vert2_PHC with 1D projections.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file with image data
    show : bool
        Whether to display plots (default: True)
    
    Returns
    -------
    dict
        Analysis results including 1D projections and image metadata
    """
    infos_analysis = {}
    cam_name = 'cam_vert2_PHC'
    
    try:
        logger.info(f"Processing PHC image from {cam_name}...")
        
        # Get camera configuration
        if cam_name not in cameras:
            logger.error(f"Camera {cam_name} not found in settings")
            return infos_analysis
        
        cam = cameras[cam_name]
        logger.info(f"Camera settings: {cam['name']}, method: {cam['method']}")
        
        # Load images using the standard library function
        dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
        
        if dict_images is None:
            logger.error(f"Failed to load images for {cam_name}")
            return infos_analysis
        
        # Extract image data
        images = dict_images.images
        infos_images = dict_images.infos
        process_params = dict_images.process_params
        
        logger.info(f"Available image keys: {list(images.keys())}")
        
        if len(images) == 0:
            logger.error("No images found in data")
            return infos_analysis
        
        # Get first non-SVD image
        normal_keys = [key for key in images.keys() if 'SVD' not in key]
        if not normal_keys:
            logger.error("No normal (non-SVD) images found")
            return infos_analysis
        
        image_key = normal_keys[0]
        logger.info(f"Using image key: {image_key}")
        
        # Create figure with 3 plots in a column (2D image, 1D projection, 1D derivative)
        fig, ax = plt.subplots(
            3, 1,
            figsize=(8, 12),
            constrained_layout=True,
            gridspec_kw={'height_ratios': [2, 1, 1]}
        )
        
        # Get image data
        n2D = dict_images.images[image_key]
        ODlog = dict_images.images_ODlog[image_key]
        roi_integration = cam['roi_integration']
        print(f"ROI for integration: {roi_integration}")
        ROI_y_min = roi_integration[0].start
        ROI_y_max = roi_integration[0].stop


        
        # Calculate 1D projections
        from imaging_analysis_lib_v2.main import get_n1Ds, get_axes
        n1D_x, n1D_y = get_n1Ds(n2D[roi_integration], cam)
        x_vals, y_vals, X_name, Y_name, inverted = get_axes(dict_images.axes)
        
        # Get coordinate arrays
        X, Y = np.meshgrid(x_vals, y_vals)
        X_name_labeled = X_name + ' [m]'
        
        # Setup Z data based on method
        method = cam['method']
        if method == 'phase_contrast':
            Z = ODlog - 1.0
            vmin, vmax = -1.0, 1.0
            cmap = 'RdBu'
        else:
            Z = ODlog
            vmin, vmax = 0.0, 1.4e14
            cmap = 'gist_stern'
        
        # Apply inversion if needed
        if inverted:
            Z = Z.T
            n1D_x, n1D_y = n1D_y, n1D_x
        
        # Plot 2D image (top panel)
        logger.info(f"Plotting 2D image for {image_key}...")
        im = ax[0].pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax[0])
        cbar.set_label('OD [m$^{-2}$]', fontsize=11)
        ax[0].set_title(f"{cam['name']} - {image_key}", fontsize=14)
        ax[0].set_ylabel('Y [m]', fontsize=11)
        ax[0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax[0].axhline(ROI_y_min*cam['px_size'], color='k', linestyle='--', linewidth=1, label='ROI Integration')
        ax[0].axhline(ROI_y_max*cam['px_size'], color='k', linestyle='--', linewidth=1)
        ax[0].legend()
        
        # Plot 1D x-projection (bottom panel)
        logger.info(f"Plotting 1D projection for {image_key}...")
        x_vals_1D = x_vals
        ax[1].plot(x_vals_1D, n1D_x, linewidth=2)
        ax[1].set_xlabel(X_name_labeled, fontsize=11)
        ax[1].set_ylabel('n1D [m$^{-1}$]', fontsize=11)
        ax[1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax[1].grid(True, alpha=0.3)
        
        # Share x-axis between plots
        ax[1].sharex(ax[0])
        ax[2].sharex(ax[0])
        
        # Calculate derivative of n1D_x with Gaussian filtering
        # Filter the signal with a Gaussian filter
        sigma = 2.0  # Gaussian filter sigma in pixels
        n1D_x_filtered = gaussian_filter1d(n1D_x, sigma=sigma)
        ax[1].plot(x_vals_1D, n1D_x_filtered, linewidth=2, color='orange', label=f'Gaussian Filtered (σ={sigma}px)')
        ax[1].legend()
        
        # Calculate spatial derivative
        # np.gradient uses the pixel spacing (1 pixel)
        n1D_x_derivative = np.gradient(n1D_x_filtered)
        
        # Plot derivative (third panel)
        logger.info(f"Plotting derivative of 1D projection...")
        ax[2].plot(x_vals_1D, n1D_x_derivative, linewidth=2)
        ax[2].set_xlabel(X_name_labeled, fontsize=11)
        ax[2].set_ylabel('dn1D/dx [m$^{-2}$]', fontsize=11)
        ax[2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax[2].grid(True, alpha=0.3)
        ax[2].set_ylim(-1e-6, 1e-6)
        ax[2].axhline(0.4e-6, color='k', linestyle='--', linewidth=1)
        ax[2].axhline(-0.4e-6, color='k', linestyle='--', linewidth=1)
        
        # Set axis limits to [950 to 1200 px] in meters
        px_min = 932 * 1.019e-6  # Convert from pixels to meters\
        px_max = 1320 * 1.019e-6
        py_min = 32*1.019e-6
        py_max = 14*1.019e-6
        ax[0].set_xlim(px_min, px_max)
        ax[0].set_ylim(py_min, py_max)
        ax[1].set_xlim(px_min, px_max)
        ax[1].set_ylim(np.min(n1D_x), 1.2e-5)
        ax[2].set_xlim(px_min, px_max)
        logger.info(f"Set x-axis limits to {px_min:.4e} to {px_max:.4e} m ({950} to {1200} px)")
        
        # Save results
        infos_analysis[f'{image_key}_n1D_x'] = n1D_x
        infos_analysis[f'{image_key}_n1D_y'] = n1D_y
        infos_analysis[f'{image_key}_n1D_x_filtered'] = n1D_x_filtered
        infos_analysis[f'{image_key}_n1D_x_derivative'] = n1D_x_derivative
        
        # Calculate center of mass
        x_vals = np.arange(len(n1D_x))
        y_vals = np.arange(len(n1D_y))
        com_x = np.sum(n1D_x * x_vals) / np.sum(n1D_x) if np.sum(n1D_x) > 0 else 0
        com_y = np.sum(n1D_y * y_vals) / np.sum(n1D_y) if np.sum(n1D_y) > 0 else 0
        
        infos_analysis[f'{image_key}_com_x'] = com_x
        infos_analysis[f'{image_key}_com_y'] = com_y
        
        logger.info(f"Center of mass: x={com_x:.1f}, y={com_y:.1f}")
        
        # Save image metadata
        for key_info in infos_images.get(image_key, {}):
            infos_analysis[f"{image_key}_{key_info}"] = infos_images[image_key][key_info]
            logger.info(f"  {key_info}: {infos_images[image_key][key_info]}")
        
        # Save processing parameters
        if process_params:
            for param_name, param_value in process_params.items():
                infos_analysis[f"{cam_name}_param_{param_name}"] = param_value
                logger.info(f"  param_{param_name}: {param_value}")
        
        # Set title and save figure
        title = general_lib_mod.get_title(h5file, cam)
        fig.suptitle(title, fontsize=12)
        
        output_path = f"PHC_{image_key}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to: {output_path}")
        
        logger.info(f"Analysis complete. Saved {len(infos_analysis)} results.")
        
        # Try to display (works better when not in lyse)
        if show and 'lyse' not in sys.modules:
            try:
                plt.show()
            except Exception as e:
                logger.warning(f"Could not display plot: {e}")
        
        return infos_analysis
    
    except Exception as e:
        logger.error(f"Error in show_PHC_image: {e}", exc_info=True)
        return infos_analysis


if __name__ == '__main__':
    time_0 = time.time()
    
    h5_file_path = None
    if len(sys.argv) > 1:
        h5_file_path = sys.argv[1]
    
    try:
        if h5_file_path:
            logger.info(f"Processing file: {h5_file_path}")
            with h5py.File(h5_file_path, 'r') as h5file:
                dict_results = show_PHC_image(h5file, show=True)
            
            if dict_results is not None:
                logger.info(f"Results obtained: {len(dict_results)} items")
        else:
            import lyse
            
            run = lyse.Run(lyse.path)
            
            with h5py.File(lyse.path, 'r+') as h5file:
                dict_results = show_PHC_image(h5file, show=False)
            
            # Save results to lyse database
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
