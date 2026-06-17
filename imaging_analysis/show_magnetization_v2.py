import h5py
import importlib
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time
from scipy.ndimage import gaussian_filter1d
import lmfit as lm

# =============================================================================
# CONSTANTS
# =============================================================================
AC_REGION_START = 250
AC_REGION_END = 1900

# Pre-autocorrelation filter sigma
G2_PRE_FILTER_SIGMA = 2.0

# Dynamic Y-ROI parameters
Y_ROI_HALF_WIDTH = 60
Z1D_SMOOTH_SIGMA = 10
DOMAIN_THRESHOLD = 0.04
ROI_LINE_WIDTH = 2.0

# Signed-derivative peak detection thresholds (waterfall-style)
DERIVATIVE_THRESHOLD_POS = 0.09
DERIVATIVE_THRESHOLD_NEG = -0.09
THRESHOLD_SCAN_POS_MIN = 0.04
THRESHOLD_SCAN_POS_MAX = 0.1
THRESHOLD_SCAN_NEG_MIN = -0.10
THRESHOLD_SCAN_NEG_MAX = -0.04
THRESHOLD_SCAN_POINTS = 50
PLATEAU_DELTA_DEFECTS = 0.05
PLATEAU_MIN_POINTS = 3
# =============================================================================
# LIBRARY IMPORTS
# =============================================================================
try:
    spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
    camera_settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_settings)

    spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
    imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imaging_analysis_lib_mod)

    spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
    general_lib_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(general_lib_mod)
    
    spec = importlib.util.spec_from_file_location("defect_finding_lib", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/defect_finding_lib.py")
    defect_finding_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defect_finding_lib)
    
    CAMERAS = camera_settings.cameras

except ImportError as e:
    print(f"FAILED TO IMPORT LIBRARIES: {e}")
    # Define generic fallbacks or exit if these are strictly required, 
    # but allowing script to define functions even if imports fail slightly helps debugging.
    CAMERAS = {}
    defect_finding_lib = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def autocorr(x):
    """
    Computes the autocorrelation of x.
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:] / result[result.size//2]


def get_camera_images(h5file, camera_name='cam_vert1'):
    """
    Retrieves images for the specified camera from the H5 file.
    """
    if camera_name not in CAMERAS:
        print(f"DEBUG: Camera {camera_name} not found in settings.")
        return None

    cam = CAMERAS[camera_name]
    print(f"DEBUG: Requesting images for camera: {camera_name}")
    
    # Call the library function
    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
    
    if dict_images is None:
        print("DEBUG: imaging_analysis_lib_mod.get_images_camera_waxes returned None")
        return None
        
    images = dict_images.get('images')
    images_ODlog = dict_images.get('images_ODlog')
    
    if not images or len(images) == 0:
        print("DEBUG: No images found in dict_images['images']")
        return None
        
    print(f"DEBUG: Retrieved images. Keys: {list(images.keys())}")
    return dict_images


def load_affine_coefficients():
    """
    Loads affine correction coefficients from waterfall_config.py.
    Returns dict with keys: a1, b1, c1, a2, b2, c2
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "waterfall_config",
            "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_config.py"
        )
        waterfall_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(waterfall_config)
        
        affine = waterfall_config.MAGNETIZATION_MODALITIES['vert1_two_component']['affine_correction']
        print(f"DEBUG: Loaded affine coefficients: {affine}")
        return affine
    except Exception as e:
        print(f"DEBUG: Failed to load affine coefficients: {e}")
        # Return identity transform if loading fails
        return {'a1': 1.0, 'b1': 0.0, 'c1': 0.0, 'a2': 1.0, 'b2': 0.0, 'c2': 0.0}


def load_defect_analysis_params():
    """
    Loads defect analysis parameters from waterfall_config.py.
    Returns dict with derivative threshold parameters.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "waterfall_config",
            "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_config.py"
        )
        waterfall_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(waterfall_config)
        
        params = waterfall_config.DEFECT_ANALYSIS_PARAMS
        print(f"DEBUG: Loaded defect analysis params. Thresholds: pos={params.get('derivative_threshold_pos')}, neg={params.get('derivative_threshold_neg')}")
        return params
    except Exception as e:
        print(f"DEBUG: Failed to load defect analysis params: {e}")
        # Return defaults if loading fails
        return {
            'derivative_threshold_pos': 0.1,
            'derivative_threshold_neg': -0.1,
            'peak_step_filter_window_px': 4,
            'peak_step_filter_min_abs_delta_m': 0.3,
        }


def run_full_defect_analysis(m_profile, x_axis_um, x_min_um, x_max_um, Z=None, y_roi=None, title="Defect Analysis"):
    """
    Runs the complete waterfall-style defect-finding algorithm with step-by-step visualization.
    
    Parameters:
    -----------
    m_profile : ndarray
        1D magnetization profile
    x_axis_um : ndarray
        Position axis in micrometers
    x_min_um, x_max_um : float
        ROI boundaries in micrometers
    Z : ndarray, optional
        2D magnetization array for visualization
    y_roi : tuple, optional
        Y ROI bounds (y_start, y_end) for overlaying on 2D image
    title : str
        Title for the visualization figure
    
    Returns:
    --------
    results : dict
        Dictionary containing all analysis results including:
        - count: total defect count
        - threshold_x_pos, threshold_x_neg: positions of threshold-detected peaks
        - corrected_x: positions of zero-crossing corrected peaks
        - rejected_x: positions of rejected peaks
        - plus intermediate data for visualization
    """
    if defect_finding_lib is None:
        print("ERROR: defect_finding_lib not loaded!")
        return None
    
    # Load config parameters
    defect_params = load_defect_analysis_params()
    
    # Call the main defect finding algorithm
    results = defect_finding_lib.find_defects_in_profile(
        m_profile,
        x_axis_um,
        defect_cfg=defect_params,
        x_min_um=x_min_um,
        x_max_um=x_max_um
    )
    
    print(f"\n=== FULL DEFECT ANALYSIS RESULTS ===")
    print(f"Total defect count: {results['count']}")
    print(f"  Positive threshold peaks: {len(results['threshold_x_pos'])}")
    print(f"  Negative threshold peaks: {len(results['threshold_x_neg'])}")
    print(f"  Zero-crossing corrections: {len(results['corrected_x'])}")
    print(f"  Rejected by peak-step filter: {len(results['rejected_x'])}")
    
    # Create comprehensive visualization
    m_roi = results['m_roi']
    d_roi = results['d_roi']
    x_roi_um = results['x_roi_um']
    threshold_x_pos = results['threshold_x_pos']
    threshold_x_neg = results['threshold_x_neg']
    corrected_x = results['corrected_x']
    rejected_x = results['rejected_x']
    
    # Compute domain regions from algorithm-detected peaks (consistent across all steps)
    all_peaks = np.concatenate([threshold_x_pos, threshold_x_neg, corrected_x]) if len(corrected_x) > 0 else np.concatenate([threshold_x_pos, threshold_x_neg]) if len(threshold_x_pos) > 0 or len(threshold_x_neg) > 0 else np.array([])
    all_peaks = np.sort(all_peaks)
    
    # Build peak list with signs: +1 for positive peaks, -1 for negative peaks
    peak_signs_dict = {}
    for px in threshold_x_pos:
        peak_signs_dict[px] = +1  # positive derivative peak
    for px in threshold_x_neg:
        peak_signs_dict[px] = -1  # negative derivative peak
    for px in corrected_x:
        # Corrected peaks are "opposite" peaks - infer sign from derivative direction
        # Actually, we need to look at the derivative to determine if this is a + or - correction
        d_idx = np.argmin(np.abs(x_roi_um - px))
        if 0 < d_idx < len(d_roi) - 1:
            # Check if this is a local min or max in derivative
            is_local_min = (d_roi[d_idx] <= d_roi[d_idx-1]) and (d_roi[d_idx] < d_roi[d_idx+1])
            is_local_max = (d_roi[d_idx] >= d_roi[d_idx-1]) and (d_roi[d_idx] > d_roi[d_idx+1])
            if is_local_min:
                peak_signs_dict[px] = -1
            elif is_local_max:
                peak_signs_dict[px] = +1
            else:
                peak_signs_dict[px] = +1 if d_roi[d_idx] > 0 else -1
        else:
            peak_signs_dict[px] = +1 if d_roi[d_idx] > 0 else -1
    
    # Compute domain boundaries and their signs
    # Strategy: Create domain boundaries only where peak sign changes
    domain_boundaries = []
    domain_signs_raw = []  # Track raw signs before normalization
    
    if len(all_peaks) > 0:
        # First domain sign is opposite of the first peak (domain before a peak has opposite sign)
        first_peak_sign = peak_signs_dict[all_peaks[0]]
        first_sign = -first_peak_sign
        
        domain_boundaries.append((x_min_um, all_peaks[0]))
        domain_signs_raw.append(first_sign)
        
        # Add boundaries only where peak sign changes
        # After each peak, the domain has the same sign as the peak (peak direction determines domain sign)
        for i in range(len(all_peaks) - 1):
            peak_sign_i = peak_signs_dict[all_peaks[i]]
            peak_sign_next = peak_signs_dict[all_peaks[i+1]]
            
            # Domain after peak has the sign of the peak transition
            current_sign = peak_sign_i  # After a +peak, we're in positive; after -peak, we're in negative
            
            # Only add new domain if sign actually changes
            if peak_sign_i != peak_sign_next:
                domain_boundaries.append((all_peaks[i], all_peaks[i+1]))
                domain_signs_raw.append(current_sign)
        
        # Last domain: from last peak to x_max
        last_peak_sign = peak_signs_dict[all_peaks[-1]]
        last_sign = last_peak_sign  # Domain after last peak has sign of that peak
        domain_boundaries.append((all_peaks[-1], x_max_um))
        domain_signs_raw.append(last_sign)
    else:
        # No peaks found - single domain
        m_avg = np.mean(m_roi)
        domain_boundaries.append((x_min_um, x_max_um))
        domain_signs_raw = [1 if m_avg > 0 else -1]
    
    # Normalize domain signs: assign +1 to the sign group with larger average magnetization
    if len(domain_boundaries) > 0:
        # Compute average magnetization for each raw domain sign
        pos_domains_m = []
        neg_domains_m = []
        
        for (x_start, x_end), raw_sign in zip(domain_boundaries, domain_signs_raw):
            # Find magnetization in this domain
            start_idx = np.argmin(np.abs(x_roi_um - x_start))
            end_idx = np.argmin(np.abs(x_roi_um - x_end))
            domain_m_avg = np.mean(m_roi[start_idx:end_idx+1])
            
            if raw_sign > 0:
                pos_domains_m.append(domain_m_avg)
            else:
                neg_domains_m.append(domain_m_avg)
        
        # Compute group averages
        pos_group_avg = np.mean(pos_domains_m) if len(pos_domains_m) > 0 else -np.inf
        neg_group_avg = np.mean(neg_domains_m) if len(neg_domains_m) > 0 else -np.inf
        
        # Assign +1 to the group with LARGER average magnetization (more positive = +1)
        # This ensures +1 always means "more magnetized in positive direction"
        if pos_group_avg > neg_group_avg:
            # pos_domains_raw already have +1, keep them as is
            sign_correction = 1
        else:
            # neg_domains_raw should become +1 instead, flip everything
            sign_correction = -1
        
        domain_signs = [sign * sign_correction for sign in domain_signs_raw]
    else:
        domain_signs = []
    
    # Create figure with all algorithm steps
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    
    # Helper function to shade domains consistently
    def shade_domains(ax, x_roi_um, boundaries, signs, y_min, y_max):
        """Shade domains with consistent colors across all plots"""
        for (x_start, x_end), sign in zip(boundaries, signs):
            color = 'blue' if sign > 0 else 'red'
            # Create mask for this domain region
            domain_mask = (x_roi_um >= x_start) & (x_roi_um <= x_end)
            ax.fill_between(x_roi_um, y_min, y_max, where=domain_mask, alpha=0.1, color=color)
    
    # Step 1: 2D Magnetization image
    ax = axes[0]
    if Z is not None:
        px_size_um = 1.019
        roi_start_um = x_min_um
        roi_end_um = x_max_um
        
        # Convert ROI boundaries to pixel indices
        roi_start_idx = int(np.argmin(np.abs(x_axis_um - roi_start_um)))
        roi_end_idx = int(np.argmin(np.abs(x_axis_um - roi_end_um)))
        
        # Crop 2D image to ROI horizontally
        Z_roi = Z[:, roi_start_idx:roi_end_idx+1]
        x_roi_um_array = x_axis_um[roi_start_idx:roi_end_idx+1]
        
        # Set extent with y-axis in micrometers
        y_max_um = Z.shape[0] * px_size_um
        im = ax.imshow(Z_roi, vmin=-1, vmax=1, cmap='RdBu', aspect='auto', 
                      extent=[x_roi_um_array[0], x_roi_um_array[-1], y_max_um, 0])
        
        # Draw domain boundaries as thin black vertical lines
        for (x_start, x_end), sign in zip(domain_boundaries, domain_signs):
            if x_start > x_roi_um_array[0] and x_start < x_roi_um_array[-1]:
                ax.axvline(x_start, color='k', linestyle='-', linewidth=0.8, alpha=0.6)
            if x_end > x_roi_um_array[0] and x_end < x_roi_um_array[-1]:
                ax.axvline(x_end, color='k', linestyle='-', linewidth=0.8, alpha=0.6)
        
        ax.axvline(roi_start_um, color='g', linestyle=':', linewidth=2, alpha=0.7, label='ROI boundaries')
        ax.axvline(roi_end_um, color='g', linestyle=':', linewidth=2, alpha=0.7)
        if y_roi is not None:
            y_start, y_end = y_roi
            # Convert pixel coordinates to micrometers
            y_start_um = y_start * px_size_um
            y_end_um = y_end * px_size_um
            ax.axhline(y_start_um, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y_end_um, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Lock x-axis to match other plots
        ax.set_xlim(x_roi_um_array[0], x_roi_um_array[-1])
        
        ax.set_ylabel('Y (µm)', fontsize=9, fontweight='bold')
        ax.set_title('Step 1: 2D Magnetization', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No 2D data provided', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Step 1: 2D Magnetization (no data)', fontsize=10, fontweight='bold')
    
    # Step 2: Raw and smoothed rescaled z1d
    ax = axes[1]
    m_smooth = defect_finding_lib.gaussian_filter1d(m_roi, sigma=defect_params.get('gaussian_sigma', 0.3))
    shade_domains(ax, x_roi_um, domain_boundaries, domain_signs, m_roi.min() - 0.1, m_roi.max() + 0.1)
    ax.plot(x_roi_um, m_roi, 'b-', linewidth=1.5, alpha=0.6, label='Raw rescaled z1d')
    ax.plot(x_roi_um, m_smooth, 'r-', linewidth=2.5, label='Gaussian smoothed (σ={:.2f})'.format(defect_params.get('gaussian_sigma', 0.3)))
    ax.axvline(x_min_um, color='g', linestyle=':', linewidth=2, alpha=0.7, label='ROI boundaries')
    ax.axvline(x_max_um, color='g', linestyle=':', linewidth=2, alpha=0.7)
    # Add peak lines for reference
    if len(all_peaks) > 0:
        for peak in all_peaks:
            ax.axvline(peak, color='gray', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.set_ylabel('Magnetization', fontsize=9, fontweight='bold')
    ax.set_title('Step 2: Raw and Smoothed Rescaled z1d Signal', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # Step 3: Derivative peaks (renumbered to axes[2])
    ax = axes[2]
    ax.plot(x_roi_um, d_roi, 'k-', linewidth=1.8, label='Derivative')
    
    # Threshold lines
    thr_pos = defect_params.get('derivative_threshold_pos', 0.1)
    thr_neg = defect_params.get('derivative_threshold_neg', -0.1)
    ax.axhline(thr_pos, color='tab:blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Pos threshold: {thr_pos:.4f}')
    ax.axhline(thr_neg, color='tab:red', linestyle='--', linewidth=2, alpha=0.7, label=f'Neg threshold: {thr_neg:.4f}')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Fill above/below threshold regions
    ax.fill_between(x_roi_um, thr_pos, d_roi.max() + 0.1, where=(d_roi >= thr_pos), 
                    alpha=0.15, color='blue', label='Above pos threshold')
    ax.fill_between(x_roi_um, d_roi.min() - 0.1, thr_neg, where=(d_roi <= thr_neg), 
                    alpha=0.15, color='red', label='Below neg threshold')
    
    # Mark threshold peaks
    if len(threshold_x_pos) > 0:
        d_pos_vals = np.interp(threshold_x_pos, x_roi_um, d_roi)
        ax.scatter(threshold_x_pos, d_pos_vals, color='tab:blue', s=120, marker='^', 
                  linewidths=1.5, edgecolors='darkblue', zorder=5, label=f'Pos peaks ({len(threshold_x_pos)})')
    if len(threshold_x_neg) > 0:
        d_neg_vals = np.interp(threshold_x_neg, x_roi_um, d_roi)
        ax.scatter(threshold_x_neg, d_neg_vals, color='tab:red', s=150, marker='v', 
                  linewidths=2, edgecolors='darkred', zorder=5, label=f'Neg peaks ({len(threshold_x_neg)})')
    
    # Mark rejected peaks
    if len(rejected_x) > 0:
        d_rej_vals = np.interp(rejected_x, x_roi_um, d_roi)
        ax.scatter(rejected_x, d_rej_vals, color='orange', s=120, marker='x', 
                  linewidths=3, alpha=0.7, zorder=4, label=f'Rejected peaks ({len(rejected_x)})')
    
    ax.set_ylabel('d(m)/dx', fontsize=9, fontweight='bold')
    ax.set_title('Step 3: Derivative Peaks (Threshold detection + Filtering)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.set_ylim(d_roi.min() - 0.05, d_roi.max() + 0.05)
    
    # Step 4: Final result with all peaks
    ax = axes[3]
    shade_domains(ax, x_roi_um, domain_boundaries, domain_signs, -1.2, 1.2)
    ax.plot(x_roi_um, m_smooth, 'b-', linewidth=2.5, label='Smoothed magnetization')
    ax.axvline(x_min_um, color='g', linestyle=':', linewidth=2, alpha=0.7, label='ROI boundaries')
    ax.axvline(x_max_um, color='g', linestyle=':', linewidth=2, alpha=0.7)
    
    # All threshold peaks
    all_threshold_x = np.concatenate([threshold_x_pos, threshold_x_neg]) if len(threshold_x_pos) > 0 or len(threshold_x_neg) > 0 else np.array([])
    all_threshold_m = np.interp(all_threshold_x, x_roi_um, m_smooth) if len(all_threshold_x) > 0 else np.array([])
    if len(all_threshold_x) > 0:
        ax.scatter(all_threshold_x, all_threshold_m, color='purple', s=100, marker='D', 
                  linewidths=1.5, edgecolors='indigo', zorder=5, label=f'Threshold peaks ({len(all_threshold_x)})')
    
    # Zero-crossing corrected peaks
    if len(corrected_x) > 0:
        corrected_m = np.interp(corrected_x, x_roi_um, m_smooth)
        ax.scatter(corrected_x, corrected_m, color='green', s=150, marker='*', 
                  linewidths=1.5, edgecolors='darkgreen', zorder=6, label=f'Zero-crossing corrections ({len(corrected_x)})')
    
    # Draw vertical lines for all final peaks
    if len(all_peaks) > 0:
        for peak in all_peaks:
            ax.axvline(peak, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax.set_ylabel('Magnetization', fontsize=9, fontweight='bold')
    ax.set_xlabel('Position (μm)', fontsize=9, fontweight='bold')
    ax.set_title(f'Step 4: Final Result - Total {results["count"]} Defects / {len(domain_boundaries)} Domains', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    ax.tick_params(labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    return results, fig


def compute_magnetization_and_density(img_m1, img_m2):
    """
    Computes total density (D) and magnetization (Z).
    """
    D_tot = img_m1 + img_m2
    
    # Avoid division by zero
    # We can use np.divide with out argument or just standard checks.
    # Given the context, we'll do standard division and handle potential NaNs if needed, 
    # but the original code just did raw division.
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (img_m2 - img_m1) / D_tot
        
    # Z might have NaNs where D_tot is 0
    return D_tot, Z


def find_y_roi(images_ODlog):
    """
    Finds the center of the cloud by integrating OD along x and smoothing.
    Returns (start, end) indices for the Y-ROI.
    """
    # Sum OD across all x
    od_tot = images_ODlog['PTAI_m1'] + images_ODlog['PTAI_m2']
    profile_y = np.nansum(od_tot, axis=1)
    
    # Smooth to find robust peak
    profile_smooth = gaussian_filter1d(profile_y, sigma=5)
    center = np.argmax(profile_smooth)
    
    start = max(0, center - Y_ROI_HALF_WIDTH)
    end = min(len(profile_y), center + Y_ROI_HALF_WIDTH)
    
    print(f"DEBUG: Found cloud center at Y={center}. ROI: [{start}:{end}]")
    return start, end


def identify_domains(fluctuations, threshold):
    """
    Identifies domains with state persistence (hysteresis).
    Returns (state_mask, pos_count, neg_count).
    State: +1 (pos), -1 (neg), 0 (undefined/initial)
    """
    n = len(fluctuations)
    states = np.zeros(n)
    
    # Initial state determination: find the first threshold crossing
    current_state = 0
    first_crossing_idx = -1
    for i in range(n):
        if fluctuations[i] > threshold:
            current_state = 1
            first_crossing_idx = i
            break
        elif fluctuations[i] < -threshold:
            current_state = -1
            first_crossing_idx = i
            break
            
    if current_state == 0:
        return states, 0, 0
        
    # Back-fill initial 0s with the first found state
    states[:first_crossing_idx] = current_state
    
    # Forward pass with hysteresis
    for i in range(first_crossing_idx, n):
        if fluctuations[i] > threshold:
            current_state = 1
        elif fluctuations[i] < -threshold:
            current_state = -1
        states[i] = current_state
        
    # Count domains (transitions)
    # A transition is only counted if it changes sign
    transitions = np.diff(states)
    pos_count = np.sum(transitions == 2) # -1 to 1
    neg_count = np.sum(transitions == -2) # 1 to -1
    
    # We also need to count the initial domain
    if states[0] == 1:
        pos_count += 1
    else:
        neg_count += 1
        
    return states, pos_count, neg_count


def sweep_smoothing_sigmas(roi_fluctuations, max_sigma = 300, steps=50):
    """
    Sweeps the pre-filter sigma and returns the resulting fitted correlation length.
    """
    sigmas_filter = np.geomspace(0.5, max_sigma, steps)
    sigmas_fit = []
    
    for s in sigmas_filter:
        # 1. Smooth
        smoothed = gaussian_filter1d(roi_fluctuations, sigma=s)
        # 2. Autocorrelate
        g2 = autocorr(smoothed)
        # 3. Fit
        _, exp_fit_res = fit_g2(g2)
        if exp_fit_res and exp_fit_res.success:
            sigmas_fit.append(exp_fit_res.params['sigma'].value)
        else:
            sigmas_fit.append(np.nan)
            
    return sigmas_filter, np.array(sigmas_fit)


def sweep_thresholds(fluctuations, max_val=0.2, steps=50):
    """
    Sweeps thresholds from 0 to max_val and returns domain counts.
    """
    thresholds = np.linspace(0, max_val, steps)
    pos_counts = []
    neg_counts = []
    
    for t in thresholds:
        _, p, n = identify_domains(fluctuations, t)
        pos_counts.append(p)
        neg_counts.append(n)
        
    return thresholds, np.array(pos_counts), np.array(neg_counts)


def find_optimal_threshold(t_vals, counts):
    """
    Finds the optimal threshold by identifying the first plateau in the count vs threshold curve.
    Uses the midpoint of the FULL first stable region after the initial noise peak.
    """
    if len(counts) < 3 or np.max(counts) == 0:
        return 0.05 # Default fallback
    
    # Smooth counts slightly to ignore tiny fluctuations for plateau detection
    counts_smooth = np.round(gaussian_filter1d(counts.astype(float), sigma=1)).astype(int)
    
    # Identify all plateaus (contiguous regions of constant smoothed count)
    is_constant = (np.diff(counts_smooth) == 0).astype(int)
    plateaus = []
    curr_start = -1
    
    for i in range(len(is_constant)):
        if is_constant[i] == 1:
            if curr_start == -1:
                curr_start = i
        else:
            if curr_start != -1:
                # Plateau from curr_start to i (inclusive in counts_smooth)
                plateaus.append((curr_start, i))
                curr_start = -1
    if curr_start != -1:
        plateaus.append((curr_start, len(is_constant)))

    # Filter plateaus:
    # 1. Past the initial noise divergence (first 5% instead of 10% to be more inclusive)
    # 2. Not count = 0
    # 3. Minimum length
    min_plateau_len = 3
    
    for start, end in plateaus:
        length = end - start + 1
        mid_idx = (start + end) // 2
        # Ensure it's not the 'vacuum' (zero count) and past noise
        if length >= min_plateau_len and counts_smooth[mid_idx] > 0 and start > len(t_vals)*0.05:
            return t_vals[mid_idx]
            
    # Fallback to the longest plateau if no 'first' meets criteria
    if plateaus:
        lengths = [e - s for s, e in plateaus]
        best_p = plateaus[np.argmax(lengths)]
        return t_vals[(best_p[0] + best_p[1]) // 2]
                
    return t_vals[len(t_vals)//4] # Final fallback


def find_first_zero_crossing(x_data, y_data):
    """
    Finds the first x-value where y_data crosses zero using linear interpolation.
    """
    if len(y_data) < 2:
        return None
        
    # Find indices where sign changes
    # We only care about the first one
    sign_changes = np.where(np.diff(np.sign(y_data)))[0]
    
    if len(sign_changes) == 0:
        return None
        
    idx = sign_changes[0]
    
    # Linear interpolation for better precision
    x1, x2 = x_data[idx], x_data[idx+1]
    y1, y2 = y_data[idx], y_data[idx+1]
    
    # y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
    # 0 = y1 + (y2 - y1) / (x2 - x1) * (x_zero - x1)
    # x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
    
    if y2 == y1:
        return x1
        
    x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
    return x_zero

def find_zero_crossing_with_optimal_threshold_and_derivative(x_data, y_data):
    """
    Finds the boundaries of the domain using the optimization and derivative.
    """
    #1. Smooth the data
    y_data_smooth = gaussian_filter1d(y_data, sigma=Z1D_SMOOTH_SIGMA)
    
    #2. Find the optimal threshold
    optimal_threshold = find_optimal_threshold(x_data, y_data_smooth)
    
    #3. Find the zero crossing
    zero_crossing = find_first_zero_crossing(x_data, y_data_smooth - optimal_threshold)
    
    return zero_crossing    
    
    

def calculate_g2(Z, images_ODlog=None):
    """
    Calculates the g2 (autocorrelation) function of the Z fluctuations.
    1. Detects Y-ROI dynamically if images_ODlog is provided.
    2. Integrates Z along y (axis 0) within the ROI.
    3. Subtracts smoothed background to get local fluctuations.
    4. Computes autocorrelation in the ROI.
    """
    # 1. Determine Y-ROI
    if images_ODlog is not None:
        y_start, y_end = find_y_roi(images_ODlog)
    else:
        # Fallback to full range if no OD provided
        y_start, y_end = 0, Z.shape[0]

    # Integrate along y (mean) within the ROI
    z_roi_y = Z[y_start:y_end, :]
    z1d = np.nanmean(z_roi_y, axis=0) 
    
    # Check bounds
    if AC_REGION_START >= len(z1d) or AC_REGION_END > len(z1d):
        print(f"DEBUG: ROI [{AC_REGION_START}:{AC_REGION_END}] out of bounds for image width {len(z1d)}")
        return None, None, None, None, None

    # CLIP Z1D
    z1d = np.clip(z1d, -1, 1)
        
    # 2. Subtract background (Gaussian smooth)
    # Use the constant for sigma
    z1d_smooth = gaussian_filter1d(z1d, sigma=Z1D_SMOOTH_SIGMA)
    fluctuations = z1d - z1d_smooth
    
    # 3. Slice ROI for autocorrelation
    roi_fluctuations = fluctuations[AC_REGION_START:AC_REGION_END]
    
    # 3b. Apply pre-filter before autocorrelating (User Request)
    roi_fluctuations_filt = gaussian_filter1d(roi_fluctuations, sigma=G2_PRE_FILTER_SIGMA)
    
    # 4. Autocorrelate
    g2 = autocorr(roi_fluctuations_filt)
    
    # 5. Identify domains in the ROI
    domain_states, pos_domains, neg_domains = identify_domains(roi_fluctuations, DOMAIN_THRESHOLD)
    
    # 6. Sweep thresholds (uses raw ROI fluctuations for counting)
    sweep_data = sweep_thresholds(roi_fluctuations)
    
    # 7. Fit g2 and |g2|
    fit_res, exp_fit_res = fit_g2(g2)
    
    # 8. Smoothing Sweep (Analyze dependency of sigma_fit on sigma_filter)
    smooth_sweep_data = sweep_smoothing_sigmas(roi_fluctuations)
    
    return z1d, fluctuations, g2, (y_start, y_end), (domain_states, pos_domains, neg_domains), sweep_data, fit_res, exp_fit_res, smooth_sweep_data


def gaussian_func(x, A, sigma, offset):
    return A * np.exp(-x**2 / (2*sigma**2)) + offset


def gaussian_decay(x, A, sigma, C):
    return A * np.exp(-x**2 / (2 * sigma**2)) + C


def fit_g2(g2_data):
    """
    Fits the g2 data with a Gaussian model and |g2| with an exponential model.
    """
    lags = np.arange(len(g2_data))
    fit_res = None
    exp_fit_res = None
    try:
        # 1. Gaussian fit for g2 (original)
        g_model = lm.models.GaussianModel()
        g_params = g_model.guess(g2_data, x=lags)
        fit_res = g_model.fit(g2_data, g_params, x=lags)
        
        # 2. Gaussian decay fit for |g2| (requested)
        # Model: y = A * exp(-x^2 / 2sigma^2) + C
        abs_g2 = np.abs(g2_data)
        g_decay_mod = lm.Model(gaussian_decay)
        
        # Robust Guess:
        # A = initial value, C = tail value, sigma = where it drops to ~0.6 A
        A_guess = abs_g2[0] - np.mean(abs_g2[-10:])
        C_guess = np.mean(abs_g2[-10:])
        # Simple sigma guess: where it first drops below ~0.6 of amplitude (e^-0.5)
        sigma_idx = np.where(abs_g2 < (A_guess * 0.6 + C_guess))[0]
        sigma_guess = sigma_idx[0] if len(sigma_idx) > 0 else 20.0
        
        params = g_decay_mod.make_params(A=max(0.1, A_guess), sigma=max(1.0, sigma_guess), C=C_guess)
        # Enforce saturation to 1 at x=0: A + C = 1  => A = 1 - C
        params['A'].set(expr='1 - C')
        params['sigma'].set(min=0.5, max=len(lags)*5) # Sensible bounds
        params['C'].set(min=-0.2, max=0.8) # Keep offset reasonable
        
        exp_fit_res = g_decay_mod.fit(abs_g2, params, x=lags)
    except Exception as e:
        print(f"DEBUG: G2 fitting failed: {e}")

    return fit_res, exp_fit_res


def plot_g2_analysis(Z, g2, z1d, fluctuations, fit_result, exp_fit_result, y_roi, domain_info, sweep_data, title):
    """
    Plots the g2 analysis window.
    """
    y_start, y_end = y_roi
    domain_states, pos_domains, neg_domains = domain_info
    t_vals, p_counts, n_counts = sweep_data
    
    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(4, 2)
    
    # Plot 1: Full Z1D and Background
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(z1d, label='Z 1D (y-averaged)')
    # Re-calculate smooth for visualization
    z1d_smooth = z1d - fluctuations
    ax1.plot(z1d_smooth, 'r--', label='Smoothed Background')
    # Mark ROI
    ax1.axvline(AC_REGION_START, color='g', linestyle=':', label='ROI Start')
    ax1.axvline(AC_REGION_END, color='g', linestyle=':', label='ROI End')
    ax1.set_title("Z 1D Profile (ROI only)")
    ax1.set_xlim(AC_REGION_START, AC_REGION_END)
    ax1.legend()
    

    # Plot 2: 2D Magnetization (Z) - Now SECOND
    ax0 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax0.imshow(Z, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
    ax0.set_title("2D Magnetization (ROI)")
    # Add horizontal lines for Y integration ROI
    ax0.axhline(y_start, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)
    ax0.axhline(y_end, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)

    # Draw domain boundaries on 2D plot using optimal threshold
    roi_x = np.arange(AC_REGION_START, AC_REGION_END)
    t_vals, p_counts, n_counts = sweep_data
    total_counts = p_counts + n_counts
    t_opt = find_optimal_threshold(t_vals, total_counts)
    # Recalculate domains for optimal threshold
    roi_data = fluctuations[AC_REGION_START:AC_REGION_END]
    states_opt, pos_opt, neg_opt = identify_domains(roi_data, t_opt)
    bounds_opt = np.where(np.diff(states_opt) != 0)[0]
    for idx in bounds_opt:
        x_val = roi_x[idx]
        ax0.axvline(x_val, color='magenta', linestyle=':', linewidth=1.5, alpha=0.8)
    
    # Plot 3: Fluctuations with OPTIMAL threshold (now 3rd plot)
    ax_opt = fig.add_subplot(gs[2, :], sharex=ax1)
    roi_data = fluctuations[AC_REGION_START:AC_REGION_END]
    total_counts = p_counts + n_counts
    t_opt = find_optimal_threshold(t_vals, total_counts)
    # Recalculate domains for optimal threshold
    states_opt, pos_opt, neg_opt = identify_domains(roi_data, t_opt)
    ax_opt.plot(roi_x, roi_data, 'k', linewidth=1)
    ax_opt.fill_between(roi_x, -1, 1, where=(states_opt == 1), 
                        color='blue', alpha=0.2, label='Pos Domain')
    ax_opt.fill_between(roi_x, -1, 1, where=(states_opt == -1), 
                        color='red', alpha=0.2, label='Neg Domain')
    ax_opt.set_ylim(min(roi_data)*1.2, max(roi_data)*1.2)
    # Mark boundaries for optimal
    bounds_opt = np.where(np.diff(states_opt) != 0)[0]
    for idx in bounds_opt:
        ax_opt.axvline(roi_x[idx], color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax_opt.axhline(t_opt, color='magenta', linestyle='--', linewidth=1, alpha=0.5, label=f'Optimal Thr={t_opt:.3f}')
    ax_opt.axhline(-t_opt, color='magenta', linestyle='--', linewidth=1, alpha=0.5)
    domain_text_opt = f"Domains: +{pos_opt}, -{neg_opt}"
    ax_opt.set_title(f"Local Z Fluctuations (Optimal Thr={t_opt:.3f}, {domain_text_opt})")
    ax_opt.legend()

    # Plot 4: Threshold Sweep (Bottom Span)
    ax4 = fig.add_subplot(gs[3, :])
    ax4.plot(t_vals, p_counts, 'b-', label='Pos')
    ax4.plot(t_vals, n_counts, 'r-', label='Neg')
    total_counts = p_counts + n_counts
    ax4.plot(t_vals, total_counts, 'k--', label='Total', alpha=0.3)
    # Smooth total counts for visualization
    total_smooth = gaussian_filter1d(total_counts.astype(float), sigma=1)
    ax4.plot(t_vals, total_smooth, 'k-', linewidth=2, label='Total (smooth)')
    # Find and plot optimal threshold
    t_opt = find_optimal_threshold(t_vals, total_counts)
    ax4.axvline(t_opt, color='magenta', linestyle='--', linewidth=2, label=f'Optimal: {t_opt:.3f}')
    ax4.axvline(DOMAIN_THRESHOLD, color='green', linestyle='--', label=f'Cur: {DOMAIN_THRESHOLD}')
    ax4.set_title("Domain Number vs. Threshold")
    ax4.set_xlabel("Threshold")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize='small')
    fig.suptitle(title + "\n Domain Counting Analysis", fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_smoothing_analysis(g2, fit_result, exp_fit_result, smooth_sweep_data, title, pre_filter_sigma):
    """
    Plots the smoothing sweep analysis and the G2 autocorrelation fit in a separate window.
    """
    sig_filter, sig_fit = smooth_sweep_data
    
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1)
    
    # 1. G2 Autocorrelation Plot (Moved here)
    ax3 = fig.add_subplot(gs[0, 0])
    ax3.plot(g2, 'b-', alpha=0.5, label='g2')
    ax3.plot(np.abs(g2), 'r-', label='|g2|')
    
    lags = np.arange(len(g2))
    x_zero = find_first_zero_crossing(lags, g2)
    if x_zero is not None:
        ax3.axvline(x_zero, color='k', linestyle='--', alpha=0.6, label=f'Zero: {x_zero:.1f}')
        
    if fit_result:
        ax3.plot(fit_result.best_fit, 'b--', alpha=0.5, label=f'Gauss $\sigma$={fit_result.params["sigma"].value:.2f}')
        
    if exp_fit_result:
        sigma_corr = exp_fit_result.params['sigma'].value
        ax3.plot(lags, exp_fit_result.best_fit, 'g-', linewidth=2.5, label=f'Gauss Fit ($\sigma$={sigma_corr:.1f})')
        ax3.text(0.05, 0.95, f'$\sigma_{{corr}} = {sigma_corr:.1f}$ px', transform=ax3.transAxes, 
                color='green', fontweight='bold', verticalalignment='top')
        
    ax3.set_title("g2 Autocorrelation & Fit")
    ax3.set_xlabel("Lag (pixels)")
    ax3.legend(fontsize='small')
    ax3.set_ylim(-1.1, 1.1)

    # 2. Smoothing Sweep Plot
    ax = fig.add_subplot(gs[1, 0])
    # Main sweep data
    ax.plot(sig_filter, sig_fit, 'o-', color='darkgreen', markersize=8, label='Fitted Correlation Length ($\sigma_{fit}$)')
    
    # Operating point
    ax.axvline(pre_filter_sigma, color='gray', linestyle='--', label=f'Operating Sigma: {pre_filter_sigma}')
    
    # Reference lines
    lims = [min(sig_filter), max(sig_filter)]
    ax.plot(lims, lims, 'k:', alpha=0.3, label='Resolution Limit ($y=x$)')
    
    # Quadrature Reference
    valid_sig = sig_fit[~np.isnan(sig_fit)]
    if len(valid_sig) > 0:
        sig_phys = valid_sig[0] 
        sig_theory = np.sqrt(sig_phys**2 + sig_filter**2)
        ax.plot(sig_filter, sig_theory, 'r--', alpha=0.5, label=f'Quadrature ($\sigma_{{phys}}={sig_phys:.1f}$)')

    ax.set_title("Smoothing Sweep: Fit Result vs. Filter Sigma")
    ax.set_xlabel("Gaussian Filter Sigma (px)")
    ax.set_ylabel("Gaussian Fit Sigma (px)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize='small')
    
    plt.tight_layout()
    return fig


def plot_magnetization(images_ODlog, img_m1, img_m2, Z, D, y_roi, title, show=True):
    """
    Plots the m1, m2, Magnetization (Z), and Total Density (m1+m2).
    """
    y_start, y_end = y_roi
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Plot m1
    # Note: aspect='auto' fixes the "locking" issue when zooming with shared axes
    ax[0].imshow(images_ODlog['PTAI_m1'], vmin=0, vmax=1.5e14, cmap='gist_stern', aspect='auto')
    ax[0].set_title('m1')
    
    # Plot m2
    ax[1].imshow(images_ODlog['PTAI_m2'], vmin=0, vmax=1.5e14, cmap='gist_stern', aspect='auto')
    ax[1].set_title('m2')
    
    # Plot Z
    ax[2].imshow(Z, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
    ax[2].set_title('Z = (m2-m1)/(m1+m2)')

    
    # Plot D (m1 + m2)
    # Using ODlog sum for visualization as in original code
    d_visual = images_ODlog['PTAI_m1'] + images_ODlog['PTAI_m2']
    ax[3].imshow(d_visual, vmin=0, vmax=1.5e14, cmap='gist_stern', aspect='auto')
    ax[3].set_title('m1 + m2 (OD)')
    
    # Add vertical lines for ROI limits to all plots
    for axis in ax:
        axis.axvline(AC_REGION_START, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)
        axis.axvline(AC_REGION_END, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)
        axis.axhline(y_start, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.5)
        axis.axhline(y_end, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.5)
    
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    
    if show:
        # standard matplotlib show usage, but in lyse it might be different. 
        # Usually lyse handles figure display automatically if plotted.
        pass 
        
    return fig


def get_plot_title(h5file, camera_name):
    """
    Generates a descriptive title from the H5 file metadata.
    """
    try:
        title = 'CAM ' + CAMERAS[camera_name]['name']
        h5file_infos = general_lib_mod.get_h5file_infos(h5file.filename)
        title += '\n' + f"{h5file_infos['year']}/{h5file_infos['month']}/{h5file_infos['day']}"
        title += '\n' + f"seq:{h5file_infos['seq']}   it:{h5file_infos['it']}   rep:{h5file_infos['rep']}"
        title += '\n' + h5file_infos['program_name']
        
        # Add Kibble-Zurek parameters if found in globals
        if 'globals' in h5file:
            ga = h5file['globals'].attrs
            # Check for both singular and plural versions of ramp time
            ramp = ga.get('ARPKZ_omega_ramp_time', ga.get('ARPKZ_omegas_ramp_time', None))
            wait = ga.get('ARPKZ_waiting_time', None)
            
            kz_info = []
            if ramp is not None:
                kz_info.append(f"Ramp: {ramp} s")
            if wait is not None:
                kz_info.append(f"Wait: {wait} s")
            
            if kz_info:
                title += '\n' + "   ".join(kz_info)
                
        return title
    except Exception:
        return f"CAM {camera_name} (Metadata Error)"


def find_peaks_above_threshold(x, y, threshold):
    """
    Find local maxima in y that are above the given threshold.
    Returns x-coordinates and values of peaks.
    """
    peaks = []
    peak_x = []
    
    # Find local maxima (compare with neighbors)
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= threshold:
            peaks.append(y[i])
            peak_x.append(x[i])  # Use the actual x-coordinate, not the index
    
    return np.array(peak_x), np.array(peaks)


def _count_peak_indices_pos(d_signal, threshold):
    if d_signal.size < 3:
        return np.array([], dtype=int)
    local_max = (d_signal[1:-1] >= d_signal[:-2]) & (d_signal[1:-1] > d_signal[2:])
    above = d_signal[1:-1] >= threshold
    return np.where(local_max & above)[0] + 1


def _count_peak_indices_neg(d_signal, threshold):
    if d_signal.size < 3:
        return np.array([], dtype=int)
    local_min = (d_signal[1:-1] <= d_signal[:-2]) & (d_signal[1:-1] < d_signal[2:])
    below = d_signal[1:-1] <= threshold
    return np.where(local_min & below)[0] + 1


def _suggest_threshold_from_plateau(thr_values, avg_curve, current_thr, delta_tol, min_points):
    """Choose threshold from widest plateau in count vs threshold curve."""
    if len(thr_values) == 0 or len(avg_curve) == 0:
        return float(current_thr), None, 'fallback-empty'
    if len(thr_values) == 1:
        v = float(thr_values[0])
        return v, (v, v), 'single-point'

    min_points = max(2, int(min_points))
    delta_tol = max(0.0, float(delta_tol))

    diffs = np.abs(np.diff(avg_curve))
    flat_edges = diffs <= delta_tol

    candidates = []
    i = 0
    while i < len(flat_edges):
        if not flat_edges[i]:
            i += 1
            continue
        j = i
        while (j + 1) < len(flat_edges) and flat_edges[j + 1]:
            j += 1
        start_idx = i
        end_idx = j + 1
        width_pts = end_idx - start_idx + 1
        if width_pts >= min_points:
            mid_idx = (start_idx + end_idx) // 2
            dist_to_current = abs(float(thr_values[mid_idx]) - float(current_thr))
            candidates.append((width_pts, -dist_to_current, start_idx, end_idx, mid_idx))
        i = j + 1

    if len(candidates) > 0:
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        _, _, start_idx, end_idx, mid_idx = candidates[0]
        return (
            float(thr_values[mid_idx]),
            (float(thr_values[start_idx]), float(thr_values[end_idx])),
            'plateau',
        )

    min_diff_idx = int(np.argmin(diffs))
    mid_idx = min_diff_idx + 1
    start_idx = min_diff_idx
    end_idx = min_diff_idx + 1
    return (
        float(thr_values[mid_idx]),
        (float(thr_values[start_idx]), float(thr_values[end_idx])),
        'fallback-min-diff',
    )


def plot_z1d_derivative(
    Z,
    img_m1,
    img_m2,
    y_roi,
    title,
    derivative_threshold_pos=None,
    derivative_threshold_neg=None,
    optimize_thresholds=True,
):
    """
    Plots rescaled m1 and m2 magnetization and its derivative in a separate window.
    Marks signed derivative peaks using the same logic as waterfall:
    - positive peaks: local maxima above `derivative_threshold_pos`
    - negative peaks: local minima below `derivative_threshold_neg`
    If thresholds are None, they are loaded from waterfall_config.
    """
    # Load thresholds from waterfall_config if not provided
    if derivative_threshold_pos is None or derivative_threshold_neg is None:
        defect_params = load_defect_analysis_params()
        if derivative_threshold_pos is None:
            derivative_threshold_pos = defect_params.get('derivative_threshold_pos', 0.1)
        if derivative_threshold_neg is None:
            derivative_threshold_neg = defect_params.get('derivative_threshold_neg', -0.1)
    
    # Load affine coefficients
    affine = load_affine_coefficients()
    a1, b1, c1 = affine['a1'], affine['b1'], affine['c1']
    a2, b2, c2 = affine['a2'], affine['b2'], affine['c2']
    
    y_start, y_end = y_roi
    
    # Compute line densities (1D profiles) integrated along y-axis
    # Note: We integrate with pixel size to get proper units (atoms/m)
    # The camera pixel size should be obtained from camera settings
    px_size = 1.019e-6  # Approximate pixel size for cam_vert1 in meters (1.019 um/pixel)
    
    n1d_m1 = np.sum(img_m1, axis=0) * px_size  # atoms/m
    n1d_m2 = np.sum(img_m2, axis=0) * px_size  # atoms/m
    
    # Apply affine correction
    n1d_m1_rescaled = a1 * n1d_m1 + b1 * n1d_m2 + c1  # atoms/m
    n1d_m2_rescaled = a2 * n1d_m2 + b2 * n1d_m1 + c2  # atoms/m
    
    # Compute rescaled magnetization from rescaled densities
    n1d_tot = n1d_m1_rescaled + n1d_m2_rescaled
    with np.errstate(divide='ignore', invalid='ignore'):
        m_rescaled = (n1d_m2_rescaled - n1d_m1_rescaled) / n1d_tot
    m_rescaled = np.clip(m_rescaled, -1, 1)
    
    # Create x-axis in micrometers for all plots (this is the ONLY definition)
    px_size_um = 1.019  # pixel size in micrometers
    x_axis_um = np.arange(len(m_rescaled)) * px_size_um  # convert to micrometers
    roi_start_um = AC_REGION_START * px_size_um
    roi_end_um = AC_REGION_END * px_size_um
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 2D Magnetization (ROI) - keep in pixels for imshow
    ax[0].imshow(Z, vmin=-1, vmax=1, cmap='RdBu', aspect='auto', extent=[x_axis_um[0], x_axis_um[-1], Z.shape[0], 0])
    ax[0].set_title('2D Magnetization')
    ax[0].grid(True, alpha=0.3)
    # Mark ROI in um coordinates
    ax[0].axvline(roi_start_um, color='g', linestyle=':', label='ROI Start')
    ax[0].axvline(roi_end_um, color='g', linestyle=':', label='ROI End')
    ax[0].axhline(y_start, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)
    ax[0].axhline(y_end, color='white', linestyle='--', linewidth=ROI_LINE_WIDTH, alpha=0.7)
    ax[0].set_xlabel('Position (μm)')
    ax[0].legend()
    
    # Plot rescaled magnetization
    m_rescaled_avg = np.mean(m_rescaled[AC_REGION_START:AC_REGION_END])
    m_rescaled_std = np.std(m_rescaled[AC_REGION_START:AC_REGION_END])
    # Smooth rescaled magnetization to remove high-frequency noise
    m_rescaled_smooth = gaussian_filter1d(m_rescaled, sigma=1.)
    ax[1].plot(x_axis_um, m_rescaled, 'b-', label='m_rescaled')
    ax[1].plot(x_axis_um, m_rescaled_smooth, 'r--', label='m_rescaled_smooth')
    ax[1].set_title('Rescaled Magnetization Signal (from m1 and m2 line densities)')
    ax[1].grid(True, alpha=0.3)
    ax[1].axhline(m_rescaled_avg, color='r', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label=f'Mean: {m_rescaled_avg:.3f} and Std: {m_rescaled_std:.3f}')
    # Mark ROI
    ax[1].axvline(roi_start_um, color='g', linestyle=':')
    ax[1].axvline(roi_end_um, color='g', linestyle=':')
    
    # Calculate derivative
    dm = np.gradient(m_rescaled)
    dm_smooth = np.gradient(m_rescaled_smooth)

    # Signed peak detection (waterfall-style) in ROI
    d_roi = dm_smooth[AC_REGION_START:AC_REGION_END]
    thr_pos_used = float(derivative_threshold_pos)
    thr_neg_used = float(derivative_threshold_neg)
    plateau_pos = None
    plateau_neg = None
    mode_pos = 'fixed'
    mode_neg = 'fixed'

    if optimize_thresholds and len(d_roi) >= 3:
        thr_vals_pos = np.linspace(THRESHOLD_SCAN_POS_MIN, THRESHOLD_SCAN_POS_MAX, max(2, int(THRESHOLD_SCAN_POINTS)))
        thr_vals_neg = np.linspace(THRESHOLD_SCAN_NEG_MIN, THRESHOLD_SCAN_NEG_MAX, max(2, int(THRESHOLD_SCAN_POINTS)))

        counts_pos = np.asarray([len(_count_peak_indices_pos(d_roi, thr)) for thr in thr_vals_pos], dtype=float)
        counts_neg = np.asarray([len(_count_peak_indices_neg(d_roi, thr)) for thr in thr_vals_neg], dtype=float)

        thr_pos_used, plateau_pos, mode_pos = _suggest_threshold_from_plateau(
            thr_vals_pos,
            counts_pos,
            derivative_threshold_pos,
            PLATEAU_DELTA_DEFECTS,
            PLATEAU_MIN_POINTS,
        )
        thr_neg_used, plateau_neg, mode_neg = _suggest_threshold_from_plateau(
            thr_vals_neg,
            counts_neg,
            derivative_threshold_neg,
            PLATEAU_DELTA_DEFECTS,
            PLATEAU_MIN_POINTS,
        )

    if len(d_roi) >= 3:
        local_max = (d_roi[1:-1] >= d_roi[:-2]) & (d_roi[1:-1] > d_roi[2:])
        above = d_roi[1:-1] >= thr_pos_used
        peak_pos_roi = np.where(local_max & above)[0] + 1

        local_min = (d_roi[1:-1] <= d_roi[:-2]) & (d_roi[1:-1] < d_roi[2:])
        below = d_roi[1:-1] <= thr_neg_used
        peak_neg_roi = np.where(local_min & below)[0] + 1
    else:
        peak_pos_roi = np.array([], dtype=int)
        peak_neg_roi = np.array([], dtype=int)

    peak_pos_indices = peak_pos_roi + AC_REGION_START
    peak_neg_indices = peak_neg_roi + AC_REGION_START
    peak_indices = np.sort(np.concatenate([peak_pos_indices, peak_neg_indices]).astype(int))
    peak_values = dm_smooth[peak_indices] if len(peak_indices) > 0 else np.array([])
    
    # Build domain states between peaks based on average magnetization
    roi_x_um = x_axis_um[AC_REGION_START:AC_REGION_END]
    domain_states = np.zeros(len(roi_x_um))
    if len(peak_indices) > 0:
        sorted_peaks = np.sort(peak_indices)
        boundaries = [AC_REGION_START] + list(sorted_peaks) + [AC_REGION_END]
        for i in range(len(boundaries) - 1):
            start_idx = int(boundaries[i])
            end_idx = int(boundaries[i + 1])
            avg_m = np.mean(m_rescaled_smooth[start_idx:end_idx])
            state = 1 if avg_m > 0 else -1
            roi_start = start_idx - AC_REGION_START
            roi_end = end_idx - AC_REGION_START
            domain_states[roi_start:roi_end] = state
    else:
        avg_m = np.mean(m_rescaled_smooth[AC_REGION_START:AC_REGION_END])
        domain_states[:] = 1 if avg_m > 0 else -1
    
    # Plot derivative
    ax[2].plot(x_axis_um, dm, 'b-', alpha=0.5, label='d(m_rescaled)/dx')
    ax[2].plot(x_axis_um, dm_smooth, 'k-', linewidth=1.2, label='d(m_rescaled_smooth)/dx')

    # Mark threshold lines
    ax[2].axhline(thr_pos_used, color='tab:blue', linestyle='--', linewidth=1.5,
                  label=f'+ threshold: {thr_pos_used:.4f}')
    ax[2].axhline(thr_neg_used, color='tab:red', linestyle='--', linewidth=1.5,
                  label=f'- threshold: {thr_neg_used:.4f}')

    # Mark signed peaks (convert indices to um for plotting)
    peak_pos_indices_um = peak_pos_indices * px_size_um if len(peak_pos_indices) > 0 else []
    peak_neg_indices_um = peak_neg_indices * px_size_um if len(peak_neg_indices) > 0 else []
    
    if len(peak_pos_indices) > 0:
        ax[2].scatter(
            peak_pos_indices_um,
            dm_smooth[peak_pos_indices],
            color='tab:blue',
            s=80,
            marker='x',
            linewidths=2,
            label=f'Positive peaks (n={len(peak_pos_indices)})',
            zorder=5,
        )
    if len(peak_neg_indices) > 0:
        ax[2].scatter(
            peak_neg_indices_um,
            dm_smooth[peak_neg_indices],
            color='tab:red',
            s=80,
            marker='x',
            linewidths=2,
            label=f'Negative peaks (n={len(peak_neg_indices)})',
            zorder=5,
        )

    for idx, idx_um in zip(peak_indices, peak_indices * px_size_um):
        ax[2].axvline(idx_um, color='0.4', linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax[2].set_title(
        f'Smoothed Derivative of Rescaled Magnetization - {len(peak_indices)} Peaks Detected '
        f'(+{len(peak_pos_indices)}, -{len(peak_neg_indices)})'
    )
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    ax[2].axvline(roi_start_um, color='g', linestyle=':', label='ROI Start')
    ax[2].axvline(roi_end_um, color='g', linestyle=':', label='ROI End')
    ax[2].set_ylim(-0.15, 0.15)
    ax[2].set_xlim(roi_start_um - 10*px_size_um, roi_end_um + 10*px_size_um)
    ax[2].set_xlabel('Position (μm)')

    m_rescaled_domain_avg = np.mean(domain_states)

    ax[1].plot(roi_x_um, domain_states, 'k-', alpha=0.5, label='Domain State')
    ax[1].axhline(np.mean(domain_states), color='k', linestyle='--', alpha=0.5, label=f'Domain Avg: {np.mean(domain_states):.2f}')
    ax[1].legend()
    ax[1].set_xlabel('Position (μm)')

    # Color domains on derivative plot based on magnetization sign
    y_min, y_max = ax[2].get_ylim()
    ax[2].fill_between(roi_x_um, y_min, y_max, where=(domain_states == 1),
                       color='blue', alpha=0.15, label='Pos Magnetization')
    ax[2].fill_between(roi_x_um, y_min, y_max, where=(domain_states == -1),
                       color='red', alpha=0.15, label='Neg Magnetization')

    
    # Print statistics
    print("DEBUG: Signed Derivative Peak Analysis (Rescaled Magnetization)")
    print(
        f"  Threshold optimization: {'on' if optimize_thresholds else 'off'} | "
        f"+thr={thr_pos_used:.4f} ({mode_pos}) -thr={thr_neg_used:.4f} ({mode_neg})"
    )
    print(
        f"  Total peaks found: {len(peak_indices)} "
        f"(+{len(peak_pos_indices)} above {thr_pos_used:.4f}, "
        f"-{len(peak_neg_indices)} below {thr_neg_used:.4f})"
    )
    if len(peak_indices) > 0:
        print(f"  Peak positions (indices): {peak_indices}")
        print(f"  Peak values: {peak_values}")
    
    # Create a separate figure for original and rescaled m1 and m2 profiles
    fig_rescaled = plt.figure(figsize=(14, 10))
    gs_rescaled = fig_rescaled.add_gridspec(2, 2)
    
    # Plot original m1
    ax_m1_orig = fig_rescaled.add_subplot(gs_rescaled[0, 0])
    ax_m1_orig.plot(x_axis_um, n1d_m1, 'b-', linewidth=2)
    ax_m1_orig.axvline(roi_start_um, color='g', linestyle=':', alpha=0.7)
    ax_m1_orig.axvline(roi_end_um, color='g', linestyle=':', alpha=0.7)
    ax_m1_orig.set_title('Original m1 Line Density')
    ax_m1_orig.set_ylabel('Line Density (atoms/m)')
    ax_m1_orig.grid(True, alpha=0.3)
    
    # Plot rescaled m1
    ax_m1_resc = fig_rescaled.add_subplot(gs_rescaled[0, 1], sharex=ax_m1_orig)
    ax_m1_resc.plot(x_axis_um, n1d_m1_rescaled, 'b-', linewidth=2)
    ax_m1_resc.axvline(roi_start_um, color='g', linestyle=':', alpha=0.7)
    ax_m1_resc.axvline(roi_end_um, color='g', linestyle=':', alpha=0.7)
    ax_m1_resc.set_title('Rescaled m1 Line Density')
    ax_m1_resc.set_ylabel('Line Density (atoms/m)')
    ax_m1_resc.grid(True, alpha=0.3)
    
    # Plot original m2
    ax_m2_orig = fig_rescaled.add_subplot(gs_rescaled[1, 0], sharex=ax_m1_orig)
    ax_m2_orig.plot(x_axis_um, n1d_m2, 'r-', linewidth=2)
    ax_m2_orig.axvline(roi_start_um, color='g', linestyle=':', alpha=0.7)
    ax_m2_orig.axvline(roi_end_um, color='g', linestyle=':', alpha=0.7)
    ax_m2_orig.set_title('Original m2 Line Density')
    ax_m2_orig.set_xlabel('Position (μm)')
    ax_m2_orig.set_ylabel('Line Density (atoms/m)')
    ax_m2_orig.grid(True, alpha=0.3)
    
    # Plot rescaled m2
    ax_m2_resc = fig_rescaled.add_subplot(gs_rescaled[1, 1], sharex=ax_m1_orig)
    ax_m2_resc.plot(x_axis_um, n1d_m2_rescaled, 'r-', linewidth=2)
    ax_m2_resc.axvline(roi_start_um, color='g', linestyle=':', alpha=0.7)
    ax_m2_resc.axvline(roi_end_um, color='g', linestyle=':', alpha=0.7)
    ax_m2_resc.set_title('Rescaled m2 Line Density')
    ax_m2_resc.set_xlabel('Position (μm)')
    ax_m2_resc.set_ylabel('Line Density (atoms/m)')
    ax_m2_resc.grid(True, alpha=0.3)
    
    # Add text box with rescaling parameters
    params_text = (
        f"Affine Correction Coefficients Applied:\n"
        f"m1_rescaled = {a1:.4f}·m1 + {b1:.4f}·m2 + {c1:.2e}\n"
        f"m2_rescaled = {a2:.4f}·m2 + {b2:.4f}·m1 + {c2:.2e}\n\n"
        f"Pixel size: 1.019 μm/px\n"
        f"Units: atoms/m"
    )
    fig_rescaled.text(
        0.5, 0.02, params_text, ha='center', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, family='monospace'
    )
    
    fig_rescaled.suptitle(title + "\n Rescaled m1 and m2 Profiles (with affine correction)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # ===== RUN FULL DEFECT ANALYSIS WITH STEP-BY-STEP VISUALIZATION =====
    print("\nDEBUG: Running full defect-finding algorithm (KZ_det_scan style)...")
    try:
        defect_result = run_full_defect_analysis(
            m_rescaled,
            x_axis_um,
            x_min_um=roi_start_um,
            x_max_um=roi_end_um,
            Z=Z,
            y_roi=y_roi,
            title=title
        )
        if defect_result is not None:
            defect_results, fig_defect_analysis = defect_result
            print(f"Full defect analysis complete: {defect_results['count']} total defects found")
    except Exception as e:
        print(f"DEBUG: Error running full defect analysis: {e}")
        import traceback
        traceback.print_exc()
    
    return fig, len(peak_indices), peak_indices, m_rescaled_avg, m_rescaled_std, m_rescaled_domain_avg
    


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def show_magnetization_v2(h5file, show=True):
    """
    Main analysis function to show magnetization.
    Returns dictionary of analysis results (currently empty in original, but extensible).
    """
    infos_analysis = {}

    try:
        print("DEBUG: Entering show_magnetization_v2")
        camera_name = 'cam_vert1'
        
        # 1. Get Images
        print("DEBUG: Calling get_camera_images...")
        dict_images = get_camera_images(h5file, camera_name)
        if dict_images is None:
            print("DEBUG: get_camera_images returned None. Aborting.")
            return None
            
        images = dict_images['images']
        images_ODlog = dict_images['images_ODlog']
        
        # Check if we have valid images
        normal_keys = [key for key in images.keys() if 'SVD' not in key]
        if not normal_keys:
            print("DEBUG: No normal keys found in images dictionary.")
            return None
            
        img_m1 = images['PTAI_m1']
        img_m2 = images['PTAI_m2']
        
        # 2. Compute Physics
        print("DEBUG: Computing magnetization and density...")
        D, Z = compute_magnetization_and_density(img_m1, img_m2)
        print("DEBUG: Compute finished.")
        
        # 3. Plot
        print("DEBUG: Starting plot...")
        title = get_plot_title(h5file, camera_name)
        
        # 4. Compute g2 (Now includes dynamic Y-ROI and domain counting)
        print("DEBUG: Calculating g2...")
        g2_tuple = calculate_g2(Z, images_ODlog)
        if g2_tuple[0] is None:
            print("DEBUG: g2 calculation returned None, skipping advanced analysis but saving basic results.")

             
        z1d, fluctuations, g2, y_roi, domain_info, sweep_data, fit_res, exp_fit_res, smooth_sweep_data = g2_tuple
        
        # Plot main magnetization with ROI
        plot_magnetization(images_ODlog, img_m1, img_m2, Z, D, y_roi, title, show=show)
        print("DEBUG: Main plot finished.")

        #if g2 is not None:
        if True:
            print("DEBUG: Plotting Domain and Correlation analysis...")
            # Removed per user request:
            #   - Z 1D Profile (ROI only)
            #   - 2D Magnetization (ROI)
            #   - Local Z Fluctuations
            #   - Domain Number vs. Threshold
            #   - Autocorrelation figure (plot_smoothing_analysis) - deleted
            # Plot rescaled magnetization and derivative in separate window
            fig_deriv, nb_zero_crossings, zero_crossing_indices, z1d_avg, z1d_std, z1d_domain_avg = plot_z1d_derivative(Z, img_m1, img_m2, y_roi, title, optimize_thresholds=False)

            # Store zero crossing count in results
            infos_analysis['nb_zero_crossings'] = int(nb_zero_crossings)
            infos_analysis['z1d_avg'] = float(z1d_avg)
            infos_analysis['z1d_std'] = float(z1d_std)
            infos_analysis['z1d_domain_avg'] = float(z1d_domain_avg)

            print("DEBUG: plotting finished.")
            
            # 5. Extract results for saving to lyse
            t_vals, p_counts, n_counts = sweep_data
            t_opt = find_optimal_threshold(t_vals, p_counts + n_counts)
            
            # Find the closest index to t_opt in t_vals
            opt_idx = np.argmin(np.abs(t_vals - t_opt))
            
            infos_analysis['nb_domains_opt_pos'] = int(p_counts[opt_idx])
            infos_analysis['nb_domains_opt_neg'] = int(n_counts[opt_idx])
            infos_analysis['nb_domains_opt_tot'] = int(nb_zero_crossings)

            
            # Zero crossing of g2
            lags = np.arange(len(g2))
            x_zero = find_first_zero_crossing(lags, g2)
            if x_zero is not None:
                infos_analysis['g2_zero_crossing'] = float(x_zero)
                
            # Correlation length from Gaussian decay fit
            if exp_fit_res and 'sigma' in exp_fit_res.params:
                infos_analysis['g2_correlation_length'] = float(exp_fit_res.params['sigma'].value)

        else:
            print("DEBUG: g2 calculation returned None.")
        
        # 5. Integrate (Logic preserved from original but commented out there -- 
        #    we can calculate it here if needed for return values)
        n1d_m1 = np.sum(img_m1, axis=0)
        n1d_m2 = np.sum(img_m2, axis=0)
        n1d_tot = n1d_m1 + n1d_m2
        
        # We can store these in infos_analysis if we want to save them to lyse
        # infos_analysis['n1d_tot'] = n1d_tot
        
        return infos_analysis
        
    except Exception as e:
        print(traceback.format_exc())
        return None


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    time_0 = time.time()
    try:
        import lyse
        
        # Use existing lyse path
        try:
            path = lyse.path
        except AttributeError:
            # Fallback for testing outside lyse if needed, or error
            path = ""
            
        if path:
            run = lyse.Run(path)
            res_dict = None
            
            with h5py.File(path, 'r+') as h5file:
                res_dict = show_magnetization_v2(h5file)
                
            if res_dict is not None:
                for key in res_dict:
                    run.save_result(key, res_dict[key])
                print("DEBUG: Result saved to lyse run.")
            else:
                print("DEBUG: No results returned from show_magnetization_v2.")
                    
            print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
    except Exception as e:
        print('DEBUG: Top level exception in __main__')
        print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
        print(traceback.format_exc())
