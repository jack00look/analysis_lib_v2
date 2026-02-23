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
AC_REGION_START = 820
AC_REGION_END = 1180

# Pre-autocorrelation filter sigma
G2_PRE_FILTER_SIGMA = 2.0

# Dynamic Y-ROI parameters
Y_ROI_HALF_WIDTH = 30
Z1D_SMOOTH_SIGMA = 10
DOMAIN_THRESHOLD = 0.04
ROI_LINE_WIDTH = 2.0
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
    
    CAMERAS = camera_settings.cameras

except ImportError as e:
    print(f"FAILED TO IMPORT LIBRARIES: {e}")
    # Define generic fallbacks or exit if these are strictly required, 
    # but allowing script to define functions even if imports fail slightly helps debugging.
    CAMERAS = {}


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
    ax[0].imshow(images_ODlog['PTAI_m1'], vmin=0, vmax=1.5, cmap='gist_stern', aspect='auto')
    ax[0].set_title('m1')
    
    # Plot m2
    ax[1].imshow(images_ODlog['PTAI_m2'], vmin=0, vmax=1.5, cmap='gist_stern', aspect='auto')
    ax[1].set_title('m2')
    
    # Plot Z
    ax[2].imshow(Z, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
    ax[2].set_title('Z = (m2-m1)/(m1+m2)')

    
    # Plot D (m1 + m2)
    # Using ODlog sum for visualization as in original code
    d_visual = images_ODlog['PTAI_m1'] + images_ODlog['PTAI_m2']
    ax[3].imshow(d_visual, vmin=0, vmax=1.5, cmap='gist_stern', aspect='auto')
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


def plot_z1d_derivative(Z, z1d, y_roi, title, derivative_threshold=0.1):
    """
    Plots z1d and its derivative in a separate window.
    Marks peaks in the absolute derivative above a threshold (signature of zero crossings).
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    # Plot 2D Magnetization (ROI)
    ax[0].imshow(Z, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
    ax[0].set_title('2D Magnetization')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # Mark ROI
    ax[0].axvline(AC_REGION_START, color='g', linestyle=':', label='ROI Start')
    ax[0].axvline(AC_REGION_END, color='g', linestyle=':', label='ROI End')
    #ax[0].xlim(AC_REGION_START - 10, AC_REGION_END + 10)
    
    # Plot z1d
    # Smooth z1d to remove high-frequency noise
    z1d_smooth = gaussian_filter1d(z1d, sigma=1.)
    ax[1].plot(z1d, 'b-', label='z1d')
    ax[1].plot(z1d_smooth, 'r--', label='z1d_smooth')
    ax[1].set_title('z1d Signal')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    # Mark ROI
    ax[1].axvline(AC_REGION_START, color='g', linestyle=':', label='ROI Start')
    ax[1].axvline(AC_REGION_END, color='g', linestyle=':', label='ROI End')
    #ax[1].xlim(AC_REGION_START - 10, AC_REGION_END + 10)
    
    # Calculate derivative
    dz = np.gradient(z1d)
    dz_smooth = np.gradient(z1d_smooth)
    abs_dz_smooth = np.abs(dz_smooth)
    
    # Find peaks in the absolute derivative above threshold
    x_indices = np.arange(len(abs_dz_smooth))
    peak_indices, peak_values = find_peaks_above_threshold(
        x_indices[AC_REGION_START:AC_REGION_END],
        abs_dz_smooth[AC_REGION_START:AC_REGION_END],
        derivative_threshold
    )
    
    # Build domain states between peaks based on average magnetization
    roi_x = np.arange(AC_REGION_START, AC_REGION_END)
    domain_states = np.zeros(len(roi_x))
    if len(peak_indices) > 0:
        sorted_peaks = np.sort(peak_indices)
        boundaries = [AC_REGION_START] + list(sorted_peaks) + [AC_REGION_END]
        for i in range(len(boundaries) - 1):
            start_idx = int(boundaries[i])
            end_idx = int(boundaries[i + 1])
            avg_z = np.mean(z1d_smooth[start_idx:end_idx])
            state = 1 if avg_z > 0 else -1
            roi_start = start_idx - AC_REGION_START
            roi_end = end_idx - AC_REGION_START
            domain_states[roi_start:roi_end] = state
    else:
        avg_z = np.mean(z1d_smooth[AC_REGION_START:AC_REGION_END])
        domain_states[:] = 1 if avg_z > 0 else -1
    
    # Plot derivative
    ax[2].plot(dz, 'b-', label='d(z1d)/dx')
    ax[2].plot(abs_dz_smooth, 'r--', label='d(z1d_smooth)/dx (abs)')
    
    # Mark the threshold line
    ax[2].axhline(derivative_threshold, color='orange', linestyle='--', linewidth=1.5, 
                  label=f'Threshold: {derivative_threshold}')
    
    # Mark peaks above threshold
    if len(peak_indices) > 0:
        ax[2].scatter(peak_indices, peak_values, color='red', s=100, marker='x', 
                     linewidths=2, label=f'Peaks (n={len(peak_indices)})', zorder=5)
        # Draw vertical lines for each peak
        for idx in peak_indices:
            ax[2].axvline(idx, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax[2].set_title(f'Derivative of z1d - {len(peak_indices)} Zero Crossings Detected')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    ax[2].axvline(AC_REGION_START, color='g', linestyle=':', label='ROI Start')
    ax[2].axvline(AC_REGION_END, color='g', linestyle=':', label='ROI End')
    ax[2].set_ylim(-0.05, 0.15)
    ax[2].set_xlim(AC_REGION_START - 10, AC_REGION_END + 10)
    
    # Color domains on derivative plot based on magnetization sign
    y_min, y_max = ax[2].get_ylim()
    ax[2].fill_between(roi_x, y_min, y_max, where=(domain_states == 1),
                       color='blue', alpha=0.15, label='Pos Magnetization')
    ax[2].fill_between(roi_x, y_min, y_max, where=(domain_states == -1),
                       color='red', alpha=0.15, label='Neg Magnetization')

    
    # Print statistics
    print(f"DEBUG: Zero Crossing Analysis")
    print(f"  Total peaks found above threshold ({derivative_threshold}): {len(peak_indices)}")
    if len(peak_indices) > 0:
        print(f"  Peak positions (indices): {peak_indices}")
        print(f"  Peak values: {peak_values}")
    
    return fig, len(peak_indices), peak_indices
    


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
             return None
             
        z1d, fluctuations, g2, y_roi, domain_info, sweep_data, fit_res, exp_fit_res, smooth_sweep_data = g2_tuple
        
        # Plot main magnetization with ROI
        plot_magnetization(images_ODlog, img_m1, img_m2, Z, D, y_roi, title, show=show)
        print("DEBUG: Main plot finished.")

        if g2 is not None:
            print("DEBUG: Plotting Domain and Correlation analysis...")
            plot_g2_analysis(Z, g2, z1d, fluctuations, fit_res, exp_fit_res, y_roi, domain_info, sweep_data, title)
            plot_smoothing_analysis(g2, fit_res, exp_fit_res, smooth_sweep_data, title, G2_PRE_FILTER_SIGMA)
            # Plot z1d derivative in separate window
            fig_deriv, nb_zero_crossings, zero_crossing_indices = plot_z1d_derivative(Z, z1d, y_roi, title)
            
            # Store zero crossing count in results
            infos_analysis['nb_zero_crossings'] = int(nb_zero_crossings)

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
