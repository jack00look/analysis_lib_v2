"""
Magnetization (Z) analysis script using imaging_analysis_lib_v2

This script:
1. Loads OD images (absorption and phase contrast) from HDF5 files
2. Displays n2D and n1D projections for m1, m2 states
3. Computes and displays magnetization from absorption data
4. Compares magnetization with phase contrast data
5. Applies coordinate transformation to phase contrast data (x' = a + bx)

The final output includes:
- Plot 1: n2D and n1D for absorption (cam_vert_1) and phase contrast (cam_vert_2)
- Plot 2: Magnetization (from absorption) and phase contrast n1D on same axis
"""

import h5py
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Coordinate transformation parameters for phase contrast
PC_TRANSFORM_A = 0  # x' = a + bx
PC_TRANSFORM_B = 1


def show_Z(h5file, show=True):
    """
    Show magnetization analysis with n2D and n1D plots.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file with image data
    show : bool
        Whether to display plots (default: True)
    
    Returns
    -------
    dict
        Analysis results including:
        - n1D and n2D data for absorption and phase contrast
        - Magnetization computed from absorption
        - Processed phase contrast data with coordinate transformation
    """
    infos_analysis = {}
    
    try:
        # Find absorption and phase contrast cameras
        cam_abs = None  # absorption camera (cam_vert_1)
        cam_pc = None   # phase contrast camera (cam_vert_2)
        dict_images_abs = None
        dict_images_pc = None
        
        # Iterate through all cameras and identify absorption vs phase contrast
        for cam_entry in cameras:
            cam = cameras[cam_entry]
            logger.info(f"Processing camera: {cam['name']}")
            
            # Try to get images for this camera
            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
            if dict_images is None:
                logger.info(f"No images found for {cam['name']}")
                continue
            
            logger.info(f"Found images for {cam['name']}: {list(dict_images.images.keys())}")
            
            # Classify camera as absorption or phase contrast
            # Look for 'vert1' or similar for absorption, 'vert2' or 'PHC' for phase contrast
            if 'vert1' in cam['name'].lower() or 'vert_1' in cam['name'].lower():
                cam_abs = cam
                dict_images_abs = dict_images
                logger.info(f"Identified {cam['name']} as absorption camera")
            elif 'vert2' in cam['name'].lower() or 'phc' in cam['name'].lower():
                cam_pc = cam
                dict_images_pc = dict_images
                logger.info(f"Identified {cam['name']} as phase contrast camera")
        
        # Prepare figure layout
        has_absorption = dict_images_abs is not None
        has_phase_contrast = dict_images_pc is not None
        
        if not has_absorption and not has_phase_contrast:
            logger.warning("No absorption or phase contrast data found")
            logger.warning(f"Available cameras in config: {list(cameras.keys())}")
            for cam_entry in cameras:
                cam = cameras[cam_entry]
                logger.warning(f"  Camera: {cam['name']}")
            return infos_analysis
        
        if has_absorption:
            logger.info(f"Found absorption images from {cam_abs['name']}")
        
        if has_phase_contrast:
            logger.info(f"Found phase contrast images from {cam_pc['name']}")
        
        # Create main figure with 4 rows x 2 columns
        # (0,0): n2D m2,  (0,1): n1D m2
        # (1,0): n2D m1,  (1,1): n1D m1
        # (2,0): n2D PC,  (2,1): n1D PC
        # (3,0): Magnetization + PC n1D overlay
        fig1, axes1 = plt.subplots(
            4, 2,
            figsize=(12, 16),
            constrained_layout=True
        )
        
        logger.info(f"Created figure with 4 rows x 2 columns for visualization")
        
        row_idx = 0
        
        # =====================================================================
        # Plot Absorption Data (m2 and m1) - Using plot_OD with legend hidden
        # =====================================================================
        if has_absorption:
            logger.info("Processing absorption images")
            images_abs = dict_images_abs.images
            infos_images_abs = dict_images_abs.infos
            process_params_abs = dict_images_abs.process_params
            
            # Get absorption image keys
            abs_keys = sorted([k for k in images_abs.keys() if 'SVD' not in k])
            logger.info(f"Absorption image keys: {abs_keys}")
            
            # Extract m2 and m1 images
            m2_key = None
            m1_key = None
            for key in abs_keys:
                if 'm2' in key:
                    m2_key = key
                elif 'm1' in key:
                    m1_key = key
            
            # Get axes info for ROI conversion (pixel to physical coordinates)
            x_vals, y_vals = None, None
            if hasattr(dict_images_abs, 'axes'):
                try:
                    x_vals, y_vals, X_name, Y_name, inverted = imaging_analysis_lib_mod.get_axes(dict_images_abs.axes)
                    logger.info(f"Got axes: X={X_name}, Y={Y_name}, inverted={inverted}")
                except Exception as e:
                    logger.warning(f"Could not get axes: {e}")
            
            # Get ROI edges
            roi_back_edges = None
            roi_int_edges = None
            if x_vals is not None and y_vals is not None:
                try:
                    roi_back_edges = imaging_analysis_lib_mod.get_roiback_edges(cam_abs, x_vals, y_vals)
                    roi_int_edges = imaging_analysis_lib_mod.get_roi_integration_edges(cam_abs, x_vals, y_vals)
                    logger.info(f"Got ROI edges for absorption")
                except Exception as e:
                    logger.warning(f"Could not get ROI edges: {e}")
            
            # Process m2: n2D at (0,0), n1D_y at (0,1)
            if m2_key is not None:
                logger.info(f"Plotting m2 absorption: {m2_key}")
                image_2d_m2 = images_abs[m2_key]
                
                # Plot n2D at (0,0) with absorption color scaling and colormap
                ax_2d = axes1[0, 0]
                
                # Use extent to display physical coordinates
                extent = None
                if x_vals is not None and y_vals is not None:
                    extent = [x_vals[0], x_vals[-1], y_vals[-1], y_vals[0]]  # [left, right, bottom, top]
                    logger.info(f"Using extent: {extent}")
                    im = ax_2d.imshow(image_2d_m2, cmap='gist_stern', vmin=0.0, vmax=1.4e14, 
                                     extent=extent, origin='upper', aspect='auto', interpolation='none')
                else:
                    logger.warning(f"No axes values available for m2, using pixel coordinates")
                    im = ax_2d.pcolormesh(image_2d_m2, cmap='gist_stern', shading='auto', vmin=0.0, vmax=1.4e14)
                
                ax_2d.set_title(f'{m2_key} n2D', fontsize=10)
                ax_2d.set_xlabel('x [um]' if x_vals is not None else 'x pixel')
                ax_2d.set_ylabel('y [um]' if y_vals is not None else 'y pixel')
                plt.colorbar(im, ax=ax_2d, label='n2D [m$^{-2}$]')
                
                # Overlay ROI rectangles (in physical coordinates)
                if roi_back_edges is not None:
                    rect_back = patches.Rectangle(
                        (roi_back_edges.xmin, roi_back_edges.ymin),
                        roi_back_edges.xmax - roi_back_edges.xmin,
                        roi_back_edges.ymax - roi_back_edges.ymin,
                        linewidth=2, edgecolor='g', facecolor='none', linestyle='--', label='roi_back'
                    )
                    ax_2d.add_patch(rect_back)
                    logger.info(f"Added roi_back rectangle")
                
                if roi_int_edges is not None:
                    rect_int = patches.Rectangle(
                        (roi_int_edges.xmin, roi_int_edges.ymin),
                        roi_int_edges.xmax - roi_int_edges.xmin,
                        roi_int_edges.ymax - roi_int_edges.ymin,
                        linewidth=2, edgecolor='b', facecolor='none', linestyle='--', label='roi_integration'
                    )
                    ax_2d.add_patch(rect_int)
                    logger.info(f"Added roi_integration rectangle")
                
                # Compute 1D projections
                n1D_y_m2 = np.sum(image_2d_m2, axis=0)  # sum over rows (integrate along y)
                n1D_x_m2 = np.sum(image_2d_m2, axis=1)  # sum over columns (integrate along x)
                
                infos_analysis[f"{m2_key}_n1D_x"] = n1D_x_m2
                infos_analysis[f"{m2_key}_n1D_y"] = n1D_y_m2
                
                # Plot n1D_y at (0,1) - y-integrated profile
                axes1[0, 1].plot(n1D_y_m2, linewidth=2, color='blue')
                axes1[0, 1].set_title(f'{m2_key} n1D (y-integrated)', fontsize=10)
                axes1[0, 1].set_xlabel('y pixel')
                axes1[0, 1].set_ylabel('Intensity')
                axes1[0, 1].grid(True, alpha=0.3)
                
                # Save metadata
                for key_info in infos_images_abs[m2_key]:
                    infos_analysis[f"{m2_key}_{key_info}"] = infos_images_abs[m2_key][key_info]
            
            # Process m1: n2D at (1,0), n1D_y at (1,1)
            if m1_key is not None:
                logger.info(f"Plotting m1 absorption: {m1_key}")
                image_2d_m1 = images_abs[m1_key]
                
                # Plot n2D at (1,0) with absorption color scaling and colormap
                ax_2d = axes1[1, 0]
                
                # Use extent to display physical coordinates
                extent = None
                if x_vals is not None and y_vals is not None:
                    extent = [x_vals[0], x_vals[-1], y_vals[-1], y_vals[0]]  # [left, right, bottom, top]
                    logger.info(f"Using extent: {extent}")
                    im = ax_2d.imshow(image_2d_m1, cmap='gist_stern', vmin=0.0, vmax=1.4e14, 
                                     extent=extent, origin='upper', aspect='auto', interpolation='none')
                else:
                    logger.warning(f"No axes values available for m1, using pixel coordinates")
                    im = ax_2d.pcolormesh(image_2d_m1, cmap='gist_stern', shading='auto', vmin=0.0, vmax=1.4e14)
                
                ax_2d.set_title(f'{m1_key} n2D', fontsize=10)
                ax_2d.set_xlabel('x [um]' if x_vals is not None else 'x pixel')
                ax_2d.set_ylabel('y [um]' if y_vals is not None else 'y pixel')
                plt.colorbar(im, ax=ax_2d, label='n2D [m$^{-2}$]')
                
                # Overlay ROI rectangles (in physical coordinates)
                if roi_back_edges is not None:
                    rect_back = patches.Rectangle(
                        (roi_back_edges.xmin, roi_back_edges.ymin),
                        roi_back_edges.xmax - roi_back_edges.xmin,
                        roi_back_edges.ymax - roi_back_edges.ymin,
                        linewidth=2, edgecolor='g', facecolor='none', linestyle='--', label='roi_back'
                    )
                    ax_2d.add_patch(rect_back)
                    logger.info(f"Added roi_back rectangle")
                
                if roi_int_edges is not None:
                    rect_int = patches.Rectangle(
                        (roi_int_edges.xmin, roi_int_edges.ymin),
                        roi_int_edges.xmax - roi_int_edges.xmin,
                        roi_int_edges.ymax - roi_int_edges.ymin,
                        linewidth=2, edgecolor='b', facecolor='none', linestyle='--', label='roi_integration'
                    )
                    ax_2d.add_patch(rect_int)
                    logger.info(f"Added roi_integration rectangle")
                
                # Compute 1D projections
                n1D_y_m1 = np.sum(image_2d_m1, axis=0)  # sum over rows (integrate along y)
                n1D_x_m1 = np.sum(image_2d_m1, axis=1)  # sum over columns (integrate along x)
                
                infos_analysis[f"{m1_key}_n1D_x"] = n1D_x_m1
                infos_analysis[f"{m1_key}_n1D_y"] = n1D_y_m1
                
                # Plot n1D_y at (1,1) - y-integrated profile
                axes1[1, 1].plot(n1D_y_m1, linewidth=2, color='blue')
                axes1[1, 1].set_title(f'{m1_key} n1D (y-integrated)', fontsize=10)
                axes1[1, 1].set_xlabel('y pixel')
                axes1[1, 1].set_ylabel('Intensity')
                axes1[1, 1].grid(True, alpha=0.3)
                
                # Save metadata
                for key_info in infos_images_abs[m1_key]:
                    infos_analysis[f"{m1_key}_{key_info}"] = infos_images_abs[m1_key][key_info]
            
            # Save processing parameters
            if process_params_abs:
                for param_name, param_value in process_params_abs.items():
                    infos_analysis[f"{cam_abs['name']}_param_{param_name}"] = param_value
        
        # =====================================================================
        # Plot Phase Contrast Data at (2,0) and (2,1)
        # =====================================================================
        if has_phase_contrast:
            logger.info("Processing phase contrast images")
            images_pc = dict_images_pc.images
            infos_images_pc = dict_images_pc.infos
            process_params_pc = dict_images_pc.process_params
            
            # Get phase contrast image keys
            pc_keys = sorted([k for k in images_pc.keys() if 'SVD' not in k])
            logger.info(f"Phase contrast image keys: {pc_keys}")
            
            # Process first available phase contrast image
            if len(pc_keys) > 0:
                pc_key = pc_keys[0]
                logger.info(f"Plotting phase contrast: {pc_key}")
                
                # Get the processed ODlog data (not raw images)
                if hasattr(dict_images_pc, 'images_ODlog') and pc_key in dict_images_pc.images_ODlog:
                    image_2d = dict_images_pc.images_ODlog[pc_key] - 1.0
                else:
                    # Fallback to raw images if ODlog not available
                    image_2d = images_pc[pc_key]
                
                # Get axes info for phase contrast camera
                x_vals_pc, y_vals_pc = None, None
                if hasattr(dict_images_pc, 'axes'):
                    try:
                        x_vals_pc, y_vals_pc, _, _, _ = imaging_analysis_lib_mod.get_axes(dict_images_pc.axes)
                        logger.info(f"Got PC axes")
                    except Exception as e:
                        logger.warning(f"Could not get PC axes: {e}")
                
                # Plot n2D at (2,0) with phase contrast color scaling and colormap
                ax_2d = axes1[2, 0]
                
                # Use extent to display physical coordinates
                extent = None
                if x_vals_pc is not None and y_vals_pc is not None:
                    extent = [x_vals_pc[0], x_vals_pc[-1], y_vals_pc[-1], y_vals_pc[0]]
                    logger.info(f"Using PC extent: {extent}")
                    im = ax_2d.imshow(image_2d, cmap='RdBu', vmin=-1.0, vmax=1.0, 
                                     extent=extent, origin='upper', aspect='auto', interpolation='none')
                else:
                    logger.warning(f"No axes values available for PC, using pixel coordinates")
                    im = ax_2d.pcolormesh(image_2d, cmap='RdBu', shading='auto', vmin=-1.0, vmax=1.0)
                
                ax_2d.set_title(f'{pc_key} n2D', fontsize=10)
                ax_2d.set_xlabel('x [um]' if x_vals_pc is not None else 'x pixel')
                ax_2d.set_ylabel('y [um]' if y_vals_pc is not None else 'y pixel')
                plt.colorbar(im, ax=ax_2d, label='Phase Contrast')
                
                # Compute 1D projections (y-integrated and x-integrated)
                n1D_y_pc = np.sum(image_2d, axis=0)  # sum over rows (integrate along y)
                n1D_x_pc = np.sum(image_2d, axis=1)  # sum over columns (integrate along x)
                
                infos_analysis[f"{pc_key}_n1D_x"] = n1D_x_pc
                infos_analysis[f"{pc_key}_n1D_y"] = n1D_y_pc
                
                # Plot n1D_y at (2,1) - y-integrated profile
                axes1[2, 1].plot(n1D_y_pc, linewidth=2, color='red')
                axes1[2, 1].set_title(f'{pc_key} n1D (y-integrated)', fontsize=10)
                axes1[2, 1].set_xlabel('y pixel')
                axes1[2, 1].set_ylabel('Intensity')
                axes1[2, 1].grid(True, alpha=0.3)
                
                # Save metadata
                for key_info in infos_images_pc[pc_key]:
                    infos_analysis[f"{pc_key}_{key_info}"] = infos_images_pc[pc_key][key_info]
            
            # Save processing parameters
            if process_params_pc:
                for param_name, param_value in process_params_pc.items():
                    infos_analysis[f"{cam_pc['name']}_param_{param_name}"] = param_value
        
        # Set title for main figure
        title = general_lib_mod.get_title(h5file, cam_abs if cam_abs else cam_pc)
        fig1.suptitle(title + "\n[n2D and n1D Projections + Magnetization]", fontsize=12)
        
        # =====================================================================
        # Plot Magnetization vs Phase Contrast at (3,0)
        # =====================================================================
        if has_absorption and ('m1_key' in locals() and m1_key is not None) and ('m2_key' in locals() and m2_key is not None):
            logger.info("Creating magnetization plot at (3,0)")
            
            # Get absorption n1D data (x-integrated, which are the n1D_x values)
            n1D_x_m1 = infos_analysis.get(f"{m1_key}_n1D_x", np.array([]))
            n1D_x_m2 = infos_analysis.get(f"{m2_key}_n1D_x", np.array([]))
            
            if len(n1D_x_m1) > 0 and len(n1D_x_m2) > 0:
                # Magnetization: M = (N_m1 - N_m2) / (N_m1 + N_m2)
                # Handle potential division issues
                denominator = n1D_x_m1 + n1D_x_m2
                magnetization = np.zeros_like(denominator)
                valid_indices = denominator != 0
                magnetization[valid_indices] = (n1D_x_m1[valid_indices] - n1D_x_m2[valid_indices]) / denominator[valid_indices]
                
                x_pixels = np.arange(len(magnetization))
                ax_mag = axes1[3, 0]
                ax_mag.plot(x_pixels, magnetization, 'b-', linewidth=2, label='Magnetization Z (absorption)')
                infos_analysis['magnetization'] = magnetization
                
                logger.info(f"Computed magnetization with {len(magnetization)} pixels")
                
                # Add phase contrast data on same plot if available
                if has_phase_contrast and 'pc_key' in locals() and 'n1D_y_pc' in locals():
                    # Transform phase contrast coordinates: x' = a + bx
                    x_pixels_pc = np.arange(len(n1D_y_pc))
                    x_pixels_pc_transformed = PC_TRANSFORM_A + PC_TRANSFORM_B * x_pixels_pc
                    
                    # Normalize phase contrast to compare with magnetization
                    n1D_pc_normalized = n1D_y_pc / np.max(np.abs(n1D_y_pc)) if np.max(np.abs(n1D_y_pc)) > 0 else n1D_y_pc
                    
                    ax_mag_right = ax_mag.twinx()
                    ax_mag_right.plot(x_pixels_pc_transformed, n1D_pc_normalized, 'r-', linewidth=2, 
                                     label=f'Phase contrast n1D (a={PC_TRANSFORM_A}, b={PC_TRANSFORM_B})')
                    ax_mag_right.set_ylabel('Phase contrast (normalized)', fontsize=11, color='r')
                    ax_mag_right.tick_params(axis='y', labelcolor='r')
                    
                    infos_analysis['n1D_pc_transformed'] = n1D_pc_normalized
                    infos_analysis['x_pc_transformed'] = x_pixels_pc_transformed
                    
                    logger.info(f"Plotted phase contrast with coordinate transformation")
                
                ax_mag.set_xlabel('x pixel', fontsize=11)
                ax_mag.set_ylabel('Magnetization Z', fontsize=11, color='b')
                ax_mag.tick_params(axis='y', labelcolor='b')
                ax_mag.set_title('Magnetization vs Phase Contrast', fontsize=11)
                ax_mag.grid(True, alpha=0.3)
        
        logger.info(f"Analysis complete. Saved {len(infos_analysis)} results.")
        return infos_analysis
    
    except Exception as e:
        logger.error(f"Error in show_Z: {e}", exc_info=True)
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
                dict_results = show_Z(h5file)
            
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
                dict_results = show_Z(h5file)
            
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
