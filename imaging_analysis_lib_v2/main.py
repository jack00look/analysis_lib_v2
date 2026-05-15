"""
Refactored imaging analysis library - v2

Handles OD image processing from absorption and phase contrast imaging.
Cleaner separation of concerns, data classes, type hints, and logging.
"""

import numpy as np
import h5py
import logging
import importlib.util
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Optional, Any
from scipy.ndimage import rotate
from scipy.stats import f
import traceback
import matplotlib.patches as patches

# Load general_lib
spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
if spec and spec.loader:
    general_lib_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(general_lib_mod)
else:
    raise ImportError("Could not load general_lib")

from fitting.models import J_Flat2DModel
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel

# ============================================================================
# CONSTANTS - Physical parameters and thresholds
# ============================================================================

# Atomic cross-section for Rb-87 (meters^2)
ATOMIC_CROSS_SECTION_SIGMA = 1.656425e-13

# Background noise detection: threshold for flagging low background
# Used as: back_max = mean(background) + BACKGROUND_STD_MULTIPLIER * std(background)
# Pixels below back_max are counted as potential background issues
BACKGROUND_STD_MULTIPLIER_DEFAULT = 3.0

# Saturation detection: threshold as fraction of dynamic_range
# Used as: sat_min = dynamic_range * SATURATION_THRESHOLD_MULTIPLIER
# Pixels above sat_min are counted as saturated
SATURATION_THRESHOLD_MULTIPLIER_DEFAULT = 0.95

# 2D to 1D fitting
F_TEST_THRESHOLD = 150.0  # Threshold for F-test in 2D fitting (flat vs model)

# Plotting
PLOT_FONTSIZE = 12

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES - Type-safe data structures
# ============================================================================

@dataclass
class ProcessedImageMetadata:
    """Metadata computed during OD calculation for one image."""
    OD_log_bare_sum: float
    OD_lin_sum: float
    OD_log_sum: float
    probe_norm_err: float
    cnt_rel_probe_back: float
    cnt_rel_atoms_back: float
    cnt_rel_probe_sat: float
    cnt_rel_atoms_sat: float
    N_atoms: float  # Total atom number computed from full n_2D


@dataclass
class ProcessedImages:
    """Container for all processed image data from a single camera."""
    images: Dict  # n2D arrays indexed by image key
    axes: Dict    # Coordinate arrays (x, y, z) in lab frame
    infos: Dict   # Metadata for each image
    images_ODlog: Dict  # OD_log for plotting
    process_params: Optional[Dict] = None  # Parameters used in processing


@dataclass
class ROIEdges:
    """Physical coordinate boundaries of a region of interest."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class AxisInfo:
    """Information about image axis orientation in lab frame."""
    name_1: str  # 'x', 'y', or 'z'
    name_2: str  # 'x', 'y', or 'z'
    invert_1: bool  # Whether first axis is inverted
    invert_2: bool  # Whether second axis is inverted


# ============================================================================
# CONSTANTS & CONFIGURATION MANAGEMENT
# ============================================================================

def get_processing_params(cam):
    """Extract and document all parameters used for processing.
    
    This is saved with results to enable reproducibility.
    """
    return {
        'roi': str(cam['roi']),
        'roi_back': str(cam['roi_back']),
        'roi_integration': str(cam['roi_integration']),
        'alpha': cam['alpha'],
        'chi_sat': cam['chi_sat'],
        'pulse_time': cam['pulse_time'],
        'px_size': cam['px_size'],
        'method': cam['method'],
        'background_std_multiplier': BACKGROUND_STD_MULTIPLIER_DEFAULT,
        'saturation_threshold_multiplier': SATURATION_THRESHOLD_MULTIPLIER_DEFAULT,
        'atomic_cross_section_sigma': ATOMIC_CROSS_SECTION_SIGMA,
    }


# ============================================================================
# CORE IMAGE PROCESSING
# ============================================================================

def calculate_OD(atoms, probe, cam, back=None, roi_background=None, alpha=None, 
                 chi_sat=None, pulse_time=None, detuning=0, method='absorption'):
    """Calculate 2D column density (n2D) from absorption images.
    
    Parameters
    ----------
    atoms : np.ndarray
        Raw atom image
    probe : np.ndarray
        Reference probe image
    cam : Dict
        Camera configuration dictionary
    back : np.ndarray
        Background image
    detuning : float
        Probe detuning in units of linewidth (default: 0)
    
    Returns
    -------
    n_2D : np.ndarray
        2D column density (atoms/m²)
    metadata : ProcessedImageMetadata
        Metadata and quality indicators
    OD_log_bare : np.ndarray
        Raw logarithmic OD (before alpha correction)
    """
    method = cam['method']
    
    # Clip to ROI
    roi = cam['roi']
    atoms = atoms[roi]
    probe = probe[roi]
    back = back[roi]
    
    # Quality metrics
    px_number = np.size(atoms.flatten())
    back_max = np.mean(back) + BACKGROUND_STD_MULTIPLIER_DEFAULT * np.std(back)
    sat_min = cam['dynamic_range'] * SATURATION_THRESHOLD_MULTIPLIER_DEFAULT
    
    cnt_probe_back = np.sum(probe < back_max) / px_number
    cnt_atoms_back = np.sum(atoms < back_max) / px_number
    cnt_probe_sat = np.sum(probe > sat_min) / px_number
    cnt_atoms_sat = np.sum(atoms > sat_min) / px_number
    
    # Background subtraction
    if probe.dtype in (np.float64, np.float32):
        atoms_diff = np.subtract(atoms.astype(np.float64), back.astype(np.float64))
        probe_diff = np.subtract(probe, back.astype(np.float64))
    else:
        atoms_diff = np.subtract(atoms, back, dtype=np.int64)
        probe_diff = np.subtract(probe, back, dtype=np.int64)
    
    atoms = np.maximum(atoms_diff, 0)
    probe = np.maximum(probe_diff, 0)
    
    # Normalize probe to atoms using background region
    roi_background = cam['roi_back']
    probe = probe * (atoms[roi_background].mean() / probe[roi_background].mean())
    probe = probe.clip(1)
    atoms = atoms.clip(1)
    
    # Normalization error metric
    roi_background_size = np.sum(np.ones_like(atoms)[roi_background])
    error_norm = np.sqrt(
        np.sum((atoms[roi_background] - probe[roi_background]) ** 2 / (np.sqrt(2) * probe[roi_background]))
    ) / roi_background_size
    
    if method == 'absorption':
        alpha = cam['alpha']
        chi_sat = cam['chi_sat']
        pulse_time = cam['pulse_time']
        
        # Compute optical densities
        OD_log_bare = np.log(np.array(probe).clip(1) / np.array(atoms).clip(1))
        OD_log = OD_log_bare * alpha * (1 + (2 * detuning) ** 2)
        OD_lin = (probe - atoms) / (chi_sat * pulse_time)
        n_2D = (OD_log + OD_lin) / ATOMIC_CROSS_SECTION_SIGMA
        
        # Clip negative values to zero (noise from background subtraction)
        n_2D = np.maximum(n_2D, 0)
        
        # Compute total atom number from full n_2D (not cut by roi_integration)
        pixel_size = cam['px_size']
        N_atoms = float(np.sum(n_2D) * pixel_size ** 2)
        
        metadata = ProcessedImageMetadata(
            OD_log_bare_sum=float(np.sum(OD_log_bare)),
            OD_lin_sum=float(np.sum(OD_lin)),
            OD_log_sum=float(np.sum(OD_log)),
            probe_norm_err=float(error_norm),
            cnt_rel_probe_back=float(cnt_probe_back),
            cnt_rel_atoms_back=float(cnt_atoms_back),
            cnt_rel_probe_sat=float(cnt_probe_sat),
            cnt_rel_atoms_sat=float(cnt_atoms_sat),
            N_atoms=N_atoms,
        )
        
        return n_2D, metadata, OD_log_bare
    
    elif method == 'phase_contrast':
        logger.info('Calculating phase contrast OD')
        OD_log_bare = np.array(atoms).clip(1) / np.array(probe).clip(1)
        n_2D = OD_log_bare
        
        # Clip negative values to zero (noise from background subtraction)
        n_2D = np.maximum(n_2D, 0)
        
        # Compute total atom number from full n_2D (not cut by roi_integration)
        pixel_size = cam['px_size']
        N_atoms = float(np.sum(n_2D) * pixel_size ** 2)
        
        metadata = ProcessedImageMetadata(
            OD_log_bare_sum=0.0,
            OD_lin_sum=0.0,
            OD_log_sum=0.0,
            probe_norm_err=float(error_norm),
            cnt_rel_probe_back=float(cnt_probe_back),
            cnt_rel_atoms_back=float(cnt_atoms_back),
            cnt_rel_probe_sat=float(cnt_probe_sat),
            cnt_rel_atoms_sat=float(cnt_atoms_sat),
            N_atoms=N_atoms,
        )
        
        return n_2D, metadata, OD_log_bare
    
    else:
        raise ValueError(f"Unknown imaging method: {method}")


# ============================================================================
# 1D PROJECTIONS
# ============================================================================

def get_n1Ds(n_2D, cam):
    """Project 2D column density to 1D profiles.
    
    Parameters
    ----------
    n_2D : np.ndarray
        2D column density array
    cam : Dict
        Camera configuration (used for px_size)
    
    Returns
    -------
    n1D_x : np.ndarray
        1D projection along x-axis
    n1D_y : np.ndarray
        1D projection along y-axis
    """
    px_size = cam['px_size']
    n1D_x = np.sum(n_2D, axis=0) * px_size
    n1D_y = np.sum(n_2D, axis=1) * px_size
    return n1D_x, n1D_y


# ============================================================================
# COORDINATE & AXIS MANAGEMENT
# ============================================================================

def get_axes(axes):
    """Parse and order coordinate axes.
    
    Returns
    -------
    x_vals, y_vals : np.ndarray
        Coordinate arrays
    X_name, Y_name : str
        Axis labels ('x', 'y', 'z')
    inverted : bool
        Whether axes are inverted from standard order
    """
    axes_names = list(axes.keys())
    order = np.array(['x', 'y', 'z'])
    axes_names_ordered = sorted(axes_names, key=lambda name: np.where(order == name)[0][0])
    inverted = (axes_names[0] != axes_names_ordered[0])
    
    X_name = axes_names_ordered[0]
    Y_name = axes_names_ordered[1]
    x_vals = axes[X_name]
    y_vals = axes[Y_name]
    
    return x_vals, y_vals, X_name, Y_name, inverted


def get_axes_infos(cam):
    """Determine axis orientation in lab frame from axis_matrix.
    
    The axis_matrix transforms camera pixel coords to lab frame.
    """
    axis_matrix = cam['axis_matrix'].copy()
    axis_names = ['x', 'y', 'z']
    
    x_ax_cam_array = np.array([1, 0])
    y_ax_cam_array = np.array([0, 1])
    
    img_ax_1 = np.dot(axis_matrix, x_ax_cam_array)
    img_ax_2 = np.dot(axis_matrix, y_ax_cam_array)
    
    ind_1 = np.where(np.abs(img_ax_1) != 0)[0][0]
    ind_2 = np.where(np.abs(img_ax_2) != 0)[0][0]
    
    name_1 = axis_names[ind_1]
    name_2 = axis_names[ind_2]
    invert_1 = img_ax_1[ind_1] < 0
    invert_2 = img_ax_2[ind_2] < 0
    
    return AxisInfo(name_1=name_1, name_2=name_2, invert_1=invert_1, invert_2=invert_2)


def _get_roi_edges(roi_slice, x_vals, y_vals, axis_info):
    """Compute physical coordinate boundaries for a region of interest.
    
    Parameters
    ----------
    roi_slice : tuple
        NumPy slice notation for ROI (e.g., np.s_[100:200, 50:150])
    x_vals, y_vals : np.ndarray
        Coordinate arrays in physical units
    axis_info : AxisInfo
        Axis orientation information
    
    Returns
    -------
    ROIEdges
        Physical boundaries of the ROI
    """
    X, Y = np.meshgrid(x_vals, y_vals)
    
    if axis_info.invert_1:
        X_slice = X[:, ::-1][roi_slice]
    else:
        X_slice = X[roi_slice]
    
    if axis_info.invert_2:
        Y_slice = Y[::-1][roi_slice]
    else:
        Y_slice = Y[roi_slice]
    
    return ROIEdges(
        xmin=float(X_slice.min()),
        xmax=float(X_slice.max()),
        ymin=float(Y_slice.min()),
        ymax=float(Y_slice.max()),
    )


def get_roiback_edges(cam, x_vals, y_vals):
    """Get physical edges of background ROI."""
    axis_info = get_axes_infos(cam)
    return _get_roi_edges(cam['roi_back'], x_vals, y_vals, axis_info)


def get_roi_integration_edges(cam, x_vals, y_vals):
    """Get physical edges of integration ROI."""
    axis_info = get_axes_infos(cam)
    return _get_roi_edges(cam['roi_integration'], x_vals, y_vals, axis_info)


# ============================================================================
# DATA I/O & PIPELINE
# ============================================================================

def get_raws_camera(h5file, cam, show_errs=False):
    """Load raw images from HDF5 file for a camera.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    cam : Dict
        Camera configuration with 'h5group' key
    show_errs : bool
        Whether to log missing camera groups
    
    Returns
    -------
    Dict or None
        Dictionary mapping image names to raw image arrays
    """
    try:
        if cam['h5group'] not in h5file:
            if show_errs:
                logger.warning(f"{h5file.filename} has no {cam['name']} images")
            return None
        
        imgs = h5file[cam['h5group']]
        img_names = [s.decode('utf-8') for s in imgs.attrs['img_names']]
        images = dict(zip(img_names, imgs))
        return images
    
    except Exception as e:
        logger.error(f"Error loading raws from {cam['name']}: {e}", exc_info=True)
        return None


def _process_camera_images(h5file, cam, images_raws, detuning, probe_source='standard'):
    """Internal function: Process raw images to n2D and assemble result.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    cam : Dict
        Camera configuration
    images_raws : Dict
        Raw images from HDF5
    detuning : float
        Probe detuning
    probe_source : str
        'standard' or 'svd' - which probe to use
    
    Returns
    -------
    ProcessedImages or None
    """
    atoms_keys = cam['atoms_images'].copy()
    probe_key = cam['probe_image']
    background_key = cam['background_image']
    
    images = {}
    images_ODlog = {}
    infos = {}
    
    for key in atoms_keys:
        if key not in images_raws:
            logger.debug(f"Skipping missing key: {key}")
            continue
        
        # Get probe based on source
        if probe_source == 'standard':
            probe = images_raws[probe_key]
        elif probe_source == 'svd':
            svd_probe_key = f"results/raw/{cam['name']}/{key}_SVDprobe"
            if svd_probe_key not in h5file:
                logger.debug(f"No SVD probe for {key}")
                continue
            probe = np.array(h5file[svd_probe_key])
        else:
            raise ValueError(f"Unknown probe_source: {probe_source}")
        
        try:
            n_2D, metadata, OD_log_bare = calculate_OD(
                images_raws[key],
                probe,
                cam,
                images_raws[background_key],
                detuning=detuning
            )
        except Exception as e:
            logger.error(f"Error calculating OD for {key} in {cam['name']}: {e}", exc_info=True)
            continue
        
        if probe_source == 'svd':
            images[f"{key}_SVD"] = n_2D
            infos[f"{key}_SVD"] = asdict(metadata)
            images_ODlog[f"{key}_SVD"] = n_2D
        else:
            images[key] = n_2D
            infos[key] = asdict(metadata)
            images_ODlog[key] = n_2D
    
    if len(images) == 0:
        logger.warning(f"No images processed for {cam['name']}")
        return None
    
    # Build coordinate axes
    images_keys = list(images.keys())
    px_size = cam['px_size']
    x_ax_cam = np.arange(images[images_keys[0]].shape[1]) * px_size
    y_ax_cam = np.arange(images[images_keys[0]].shape[0]) * px_size
    
    axis_info = get_axes_infos(cam)
    axis_image = {
        axis_info.name_1: x_ax_cam,
        axis_info.name_2: y_ax_cam,
    }
    
    # Apply axis inversions
    for key in images:
        if axis_info.invert_1:
            images[key] = images[key][:, ::-1]
            images_ODlog[key] = images_ODlog[key][:, ::-1]
        if axis_info.invert_2:
            images[key] = images[key][::-1, :]
            images_ODlog[key] = images_ODlog[key][::-1, :]
    
    process_params = get_processing_params(cam)
    
    return ProcessedImages(
        images=images,
        axes=axis_image,
        infos=infos,
        images_ODlog=images_ODlog,
        process_params=process_params,
    )


def get_images_camera_waxes(h5file, cam, show_errs=False):
    """Load and process camera images (standard probe).
    
    This is the main public API for image processing.
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    cam : Dict
        Camera configuration dictionary
    show_errs : bool
        Whether to log missing data
    
    Returns
    -------
    ProcessedImages or None
        Processed images with metadata, axes, and parameters used
    """
    try:
        images_raws = get_raws_camera(h5file, cam, show_errs=show_errs)
        if images_raws is None:
            return None
        
        # Get detuning from globals
        try:
            probe_det_0 = float(h5file['globals'].attrs['probe_detuning_0'])
        except KeyError:
            probe_det_0 = 0.0
            logger.info('No probe_detuning_0 in globals, using 0')
        
        detuning = float((h5file['globals'].attrs['probe_detuning_debug'] - probe_det_0) / 9.7946)
        
        return _process_camera_images(h5file, cam, images_raws, detuning, probe_source='standard')
    
    except Exception as e:
        logger.error(f"Error processing images for {cam['name']}: {e}", exc_info=True)
        return None


def get_images_camera_waxes_SVD(h5file, cam, show_errs=False):
    """Load and process camera images (SVD-reconstructed probe).
    
    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file
    cam : Dict
        Camera configuration dictionary
    show_errs : bool
        Whether to log missing data
    
    Returns
    -------
    ProcessedImages or None
        Processed images with SVD probes
    """
    try:
        # Check if SVD results exist
        cam_raws_h5name = f"results/raw/{cam['name']}"
        if cam_raws_h5name not in h5file:
            if show_errs:
                logger.info(f"No SVD results found for {cam['name']}")
            return None
        
        images_raws = get_raws_camera(h5file, cam, show_errs=show_errs)
        if images_raws is None:
            return None
        
        # Get detuning from globals
        try:
            probe_det_0 = float(h5file['globals'].attrs['probe_detuning_0'])
        except KeyError:
            probe_det_0 = 0.0
            logger.info('No probe_detuning_0 in globals, using 0')
        
        detuning = float((h5file['globals'].attrs['probe_detuning_debug'] - probe_det_0) / 9.7946)
        
        return _process_camera_images(h5file, cam, images_raws, detuning, probe_source='svd')
    
    except Exception as e:
        logger.error(f"Error processing SVD images for {cam['name']}: {e}", exc_info=True)
        return None


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_OD(dict_images, key, cam, fig, ax, i, j):
    """Plot OD image with 1D projections and ROI overlays.
    
    Creates a 2×2 subplot arrangement:
    - [i, j]: 1D y-projection
    - [i, j+1]: 2D image with colormaps and ROI boxes
    - [i+1, j]: (off)
    - [i+1, j+1]: 1D x-projection
    
    Parameters
    ----------
    dict_images : ProcessedImages
        Processed image data
    key : str
        Image key to plot
    cam : Dict
        Camera configuration
    fig, ax : matplotlib objects
        Figure and axes
    i, j : int
        Subplot indices
    
    Returns
    -------
    n1D_x, n1D_y : np.ndarray
        1D projections for further analysis
    """
    # Extract data
    n2D = dict_images.images[key]
    ODlog = dict_images.images_ODlog[key]
    roi_integration = cam['roi_integration']
    n1D_x, n1D_y = get_n1Ds(n2D[roi_integration], cam)
    
    # Get axes
    x_vals, y_vals, X_name, Y_name, inverted = get_axes(dict_images.axes)
    
    # Build coordinate arrays for roi_integration region
    X_1D, Y_1D = np.meshgrid(x_vals, y_vals)
    X_1D = X_1D[roi_integration]
    Y_1D = Y_1D[roi_integration]
    x_vals_1D = X_1D[0, :]
    y_vals_1D = Y_1D[:, 0]
    
    # Get ROI boundaries
    roi_back_edges = get_roiback_edges(cam, x_vals, y_vals)
    roi_int_edges = get_roi_integration_edges(cam, x_vals, y_vals)
    
    # Setup for plotting
    X, Y = np.meshgrid(x_vals, y_vals)
    X_name_labeled = X_name + ' [m]'
    Y_name_labeled = Y_name + ' [m]'
    
    method = cam['method']
    if method == 'absorption':
        Z = ODlog
        vmin, vmax = 0.0, 1.4e14
        cmap = 'gist_stern'
    elif method == 'phase_contrast':
        Z = ODlog - 1.0
        vmin, vmax = -1.0, 1.0
        cmap = 'RdBu'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply inversion if needed
    if inverted:
        Z = Z.T
        n1D_x, n1D_y = n1D_y, n1D_x
    
    # Plot 2D image
    fig_temp = ax[2 * i, j + 1].pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(fig_temp, ax=ax[2 * i, j + 1])
    cbar.set_label('n2D [m$^{-2}$]', fontsize=PLOT_FONTSIZE)
    
    # Overlay ROI rectangles
    rect_back = patches.Rectangle(
        (roi_back_edges.xmin, roi_back_edges.ymin),
        roi_back_edges.xmax - roi_back_edges.xmin,
        roi_back_edges.ymax - roi_back_edges.ymin,
        linewidth=3, edgecolor='g', facecolor='none', linestyle='--', label='roi_back'
    )
    rect_int = patches.Rectangle(
        (roi_int_edges.xmin, roi_int_edges.ymin),
        roi_int_edges.xmax - roi_int_edges.xmin,
        roi_int_edges.ymax - roi_int_edges.ymin,
        linewidth=3, edgecolor='b', facecolor='none', linestyle='--', label='roi_integration'
    )
    ax[2 * i, j + 1].add_patch(rect_back)
    ax[2 * i, j + 1].add_patch(rect_int)
    ax[2 * i, j + 1].set_title(key)
    ax[2 * i, j + 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[2 * i, j + 1].legend(loc='upper right', fontsize=PLOT_FONTSIZE - 2)
    
    # Plot x-projection (bottom right)
    ax[2 * i + 1, j + 1].plot(x_vals_1D, n1D_x)
    ax[2 * i + 1, j + 1].set_xlabel(X_name_labeled, fontsize=PLOT_FONTSIZE)
    ax[2 * i + 1, j + 1].set_ylabel('n1D [m$^{-1}$]', fontsize=PLOT_FONTSIZE)
    ax[2 * i + 1, j + 1].sharex(ax[2 * i, j + 1])
    ax[2 * i + 1, j + 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    
    # Plot y-projection (left)
    ax[2 * i, j].plot(n1D_y, y_vals_1D)
    ax[2 * i, j].set_ylabel(Y_name_labeled, fontsize=PLOT_FONTSIZE)
    ax[2 * i, j].set_xlabel('n1D [m$^{-1}$]', fontsize=PLOT_FONTSIZE)
    ax[2 * i, j].sharey(ax[2 * i, j + 1])
    ax[2 * i, j].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    
    # Turn off unused subplot
    ax[2 * i + 1, j].axis('off')
    
    return n1D_x, n1D_y


# ============================================================================
# 1D FITTING (kept for compatibility, can be extended later)
# ============================================================================

def do_fit_1D(y_vals, x_vals, MODEL, prefix, ax, invert=False):
    """Fit 1D data to a model.
    
    Parameters
    ----------
    y_vals : np.ndarray
        Data to fit
    x_vals : np.ndarray
        Coordinate values
    MODEL : lmfit Model class
        Model to use (e.g., ThomasFermi1DModel)
    prefix : str
        Parameter prefix for lmfit
    ax : matplotlib axis
        Axis for plotting
    invert : bool
        Whether to invert axes on plot
    
    Returns
    -------
    out : lmfit result
        Fit result object
    """
    model = MODEL(prefix=prefix)
    p0 = model.guess(y_vals, X=x_vals)
    
    if not invert:
        ax.plot(x_vals, model.eval(p0, X=x_vals), 'k--', label='initial guess')
    else:
        ax.plot(model.eval(p0, X=x_vals), x_vals, 'k--', label='initial guess')
    
    out = model.fit(y_vals, p0, X=x_vals)
    model.plot_best(out.params, x_vals, ax, invert)
    ax.legend()
    
    return out


def do_fit_2D(Z_vals, X_vals, Y_vals, MODEL, prefix, ax, i, j):
    """Fit 2D data to a model with flat background comparison.
    
    Uses F-test to determine if model is significantly better than flat fit.
    
    Parameters
    ----------
    Z_vals : np.ndarray
        2D data to fit
    X_vals, Y_vals : np.ndarray
        2D coordinate arrays
    MODEL : lmfit Model class
        2D model to fit
    prefix : str
        Parameter prefix for lmfit
    ax : matplotlib axes array
        [i,j]: surface, [i+1,j]: x-projection, [i,j-1]: y-projection
    i, j : int
        Subplot indices
    
    Returns
    -------
    fit_results : lmfit result
    is_flat : bool
        Whether F-test indicates data is consistent with flat background
    """
    # Fit to flat background
    flat_model = J_Flat2DModel(prefix=prefix)
    p0_flat = flat_model.guess(Z_vals, X=X_vals, Y=Y_vals)
    out_flat = flat_model.fit(Z_vals, p0_flat, X=X_vals, Y=Y_vals)
    chisqr_flat = out_flat.chisqr
    nfree_flat = out_flat.nfree
    
    # Fit to model
    ax_surface = ax[i, j]
    ax_x = ax[i + 1, j]
    ax_y = ax[i, j - 1]
    
    x_vals = X_vals[0, :]
    y_vals = Y_vals[:, 0]
    Z_vals_x = np.sum(Z_vals, axis=0)
    Z_vals_y = np.sum(Z_vals, axis=1)
    
    model = MODEL(prefix=prefix)
    p0 = model.guess(Z_vals, X=X_vals, Y=Y_vals)
    guess_fit = model.eval(p0, X=X_vals, Y=Y_vals)
    guess_fit_x = np.sum(guess_fit, axis=0)
    guess_fit_y = np.sum(guess_fit, axis=1)
    
    fit_results = model.fit(Z_vals, p0, X=X_vals, Y=Y_vals)
    chisqr = fit_results.chisqr
    nfree = fit_results.nfree
    
    # F-test
    F_test = (chisqr_flat - chisqr) / (nfree_flat - nfree) / (chisqr / nfree)
    p_val = 1 - f.cdf(F_test, nfree_flat - nfree, nfree)
    is_flat = F_test < F_TEST_THRESHOLD
    
    logger.info(f"F-test: {F_test:.2f}, p-value: {p_val:.4f}, is_flat: {is_flat}")
    
    # Plot
    best_fit = model.eval(fit_results.params, X=X_vals, Y=Y_vals)
    best_fit_x = np.sum(best_fit, axis=0)
    best_fit_y = np.sum(best_fit, axis=1)
    
    ax_surface.pcolormesh(X_vals, Y_vals, Z_vals - best_fit)
    
    ax_x.plot(x_vals, Z_vals_x, label='data')
    ax_x.plot(x_vals, guess_fit_x, 'k--', label='initial guess')
    ax_x.plot(x_vals, best_fit_x, 'r-', label='best fit')
    ax_x.legend()
    
    ax_y.plot(Z_vals_y, y_vals, label='data')
    ax_y.plot(guess_fit_y, y_vals, 'k--', label='initial guess')
    ax_y.plot(best_fit_y, y_vals, 'r-', label='best fit')
    ax_y.legend()
    
    ax[i + 1, j - 1].axis('off')
    
    return fit_results, is_flat
