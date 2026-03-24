"""
Manual DMD Profile Update Script

This script allows manual updating of DMD profiles by:
1. Loading a previous shot's DMD profile as starting point (specified by day, sequence, it, rep)
2. Reading one or more error profiles from txt files (x_um, y columns)
3. Extending each error profile to the full DMD x-region (zero outside txt file region)
4. Applying Gaussian blur to each error profile
5. Adding each blurred error with its own gain factor (KP) to the old profile
6. Sending the updated profile to the DMD server
"""

import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Import modules
spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.main", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("feedback_config", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_config.py")
feedback_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_config)

spec = importlib.util.spec_from_file_location("general_lib.settings", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings_mod)


# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

# Previous shot specification (starting DMD profile source)
YEAR = 2026
MONTH = 3
DAY = 20
SEQUENCE = 2               # Sequence number (integer)
ITERATION = 0                # Iteration number
REPETITION = 35            # Repetition number

# Error profiles (in DMD feedback folder)
# Each item must define:
#   - filename: txt file with two columns (x_um, error_value)
#   - kp: proportional gain applied to that profile
# Optional per-profile override:
#   - smoothing_sigma: Gaussian blur sigma for that profile (in pixels)
ERROR_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation1.txt',
        'kp': 0.5,
        'smoothing_sigma': 2.0,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation2.txt',
        'kp': 0.5,
        'smoothing_sigma': 2.0,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation3.txt',
        'kp': 0.5,
        'smoothing_sigma': 2.0,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation4.txt',
        'kp': 0.5,
        'smoothing_sigma': 2.0,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation5.txt',
        'kp': 0.9,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation6.txt',
        'kp': 0.5,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation7.txt',
        'kp': 0.8,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation8.txt',
        'kp': 0.8,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation9.txt',
        'kp': 0.8,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation10.txt',
        'kp': 0.5,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation11.txt',
        'kp': 0.8,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
    {
        'filename': 'sigmoid_center_interpolation12.txt',
        'kp': 0.5,
        'smoothing_sigma': 5.,  # Optional override for default smoothing sigma
    },
]

# Default Gaussian blur sigma for error profiles (in pixels)
SMOOTHING_SIGMA = 2.0

# DMD server connection
DMD_SERVER_IP = feedback_config.DMD_SERVER_IP
DMD_SERVER_PORT = feedback_config.DMD_SERVER_PORT

# Wall configuration (optional - set to None to disable walls)
WALL_TYPE = feedback_config.WALL_TYPE  # 'hard', 'soft', or None
X_CENTER = feedback_config.X_CENTER
FEEDBACK_WIDTH = feedback_config.FEEDBACK_WIDTH
SOFT_WALL_WIDTH = feedback_config.SOFT_WALL_WIDTH

# =============================================================================


def load_h5_file(year, month, day, sequence, iteration, repetition, bec2_path):
    """
    Find the h5 file using general_lib_mod.get_h5filename().
    
    Returns the full path to the h5 file, or None if not found.
    """
    h5_path = general_lib_mod.get_h5filename(
        year=year,
        month=month,
        day=day,
        sequence=sequence,
        it=iteration,
        rep=repetition,
        bec2_path=bec2_path,
        show_errs=True,
    )
    
    return h5_path


def load_dmd_profile_from_h5(h5_path):
    """
    Load the DMD profile from a previous shot's h5 file.
    Follows the same logic as get_dmd_profile() in feedback_utils.py
    
    This loads the "last_profile" which is what was on the atoms BEFORE feedback,
    i.e., the DMD state that the atoms experienced during that shot.
    
    Returns (x_um, profile_y) or (None, None) if not found.
    """
    try:
        with h5py.File(h5_path, 'r') as h5file:
            results_group = 'results/feedback_atoms'
            
            # Check if this shot has feedback results
            if results_group + '/last_profile_x_um' in h5file:
                # Load the "last profile" - this is what was loaded on DMD before feedback
                x_um = np.array(h5file[results_group + '/last_profile_x_um'][...])
                profile_y = np.array(h5file[results_group + '/last_profile_y'][...])
                print(f"Loaded DMD profile (last_profile) from feedback results: {len(x_um)} points")
                print(f"  This is the profile that was on the atoms during the specified shot.")
                return x_um, profile_y
            
            # Alternative: load the profile that was SENT after feedback 
            if results_group + '/profile_x_um' in h5file:
                x_um = np.array(h5file[results_group + '/profile_x_um'][...])
                profile_y = np.array(h5file[results_group + '/profile_y_sent'][...])
                print(f"Loaded DMD profile (sent after feedback) from feedback results: {len(x_um)} points")
                print(f"  Note: This is the profile sent AFTER feedback from the specified shot.")
                return x_um, profile_y
            
            # Fallback: try to reconstruct from camera data
            cam_vert1 = camera_settings.cameras['cam_vert1']
            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam_vert1)
            
            if dict_images is not None:
                axes = dict_images['axes']
                x_vals, y_vals, X_name, Y_name, inverted = imaging_analysis_lib_mod.get_axes(axes)
                x_um = x_vals * 1e6  # Convert to um
                
                # Create a default flat profile
                profile_y = np.ones_like(x_um)
                print(f"Warning: No saved DMD profile found in feedback results.")
                print(f"Using flat profile with {len(x_um)} points from camera axis.")
                return x_um, profile_y
                
    except Exception as e:
        print(f"Error loading DMD profile from h5 file: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None


def load_error_profile_from_txt(txt_path):
    """
    Load error profile from text file.
    
    Expected format: two columns (x_um, error_value), tab or space delimited.
    Lines starting with # are treated as comments.
    
    Returns (x_um, error_values) or (None, None) if file not found/invalid.
    """
    try:
        if not os.path.exists(txt_path):
            print(f"Error: File not found: {txt_path}")
            return None, None
        
        data = np.loadtxt(txt_path, comments='#')
        
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Error: Expected 2 columns in {txt_path}, got shape {data.shape}")
            return None, None
        
        x_um = data[:, 0]

        error_values = data[:, 1]

        error_values = (error_values - np.mean(error_values))  # Center error around zero
        
        
        print(f"Loaded error profile from {txt_path}: {len(x_um)} points")
        print(f"  x range: {x_um.min():.2f} to {x_um.max():.2f} um")
        print(f"  error range: {error_values.min():.4f} to {error_values.max():.4f}")
        
        return x_um, error_values
        
    except Exception as e:
        print(f"Error loading error profile from txt file: {e}")
        return None, None


def get_error_profile_specs():
    """
    Return normalized error-profile specifications.

    Supports the new ERROR_PROFILES list and keeps backward compatibility
    with the older ERROR_PROFILE_FILENAME/KP configuration if present.
    """
    if 'ERROR_PROFILES' in globals() and ERROR_PROFILES is not None:
        specs = ERROR_PROFILES
    else:
        specs = [{
            'filename': globals()['ERROR_PROFILE_FILENAME'],
            'kp': globals()['KP'],
        }]

    normalized_specs = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f'ERROR_PROFILES[{i}] must be a dict.')
        if 'filename' not in spec:
            raise ValueError(f'ERROR_PROFILES[{i}] is missing "filename".')
        if 'kp' not in spec:
            raise ValueError(f'ERROR_PROFILES[{i}] is missing "kp".')

        normalized_specs.append({
            'filename': spec['filename'],
            'kp': float(spec['kp']),
            'smoothing_sigma': float(spec.get('smoothing_sigma', SMOOTHING_SIGMA)),
        })

    if len(normalized_specs) == 0:
        raise ValueError('ERROR_PROFILES must contain at least one profile.')

    return normalized_specs


def extend_error_profile(x_dmd, x_error, error_values):
    """
    Extend the error profile to match the full DMD x-axis.
    Uses interpolation within the error profile region, zeros outside.
    
    Args:
        x_dmd: Full DMD x-axis (um)
        x_error: x-axis from error profile txt file (um)
        error_values: Error values from txt file
    
    Returns:
        error_extended: Error profile extended to full DMD x-axis
    """
    error_extended = np.zeros_like(x_dmd)
    
    # Find indices where DMD x-axis is within error profile range
    x_min_error = x_error.min()
    x_max_error = x_error.max()
    
    mask_inside = (x_dmd >= x_min_error) & (x_dmd <= x_max_error)
    
    if not np.any(mask_inside):
        print("Warning: No overlap between DMD x-axis and error profile x range.")
        return error_extended
    
    # Interpolate error values onto DMD x-axis (within error profile range)
    interp_func = interp1d(x_error, error_values, kind='linear', 
                          bounds_error=False, fill_value=0.0)
    error_extended[mask_inside] = interp_func(x_dmd[mask_inside])
    
    print(f"Extended error profile: {np.sum(mask_inside)} points inside error region, "
          f"{np.sum(~mask_inside)} points set to zero outside")
    
    return error_extended


def apply_walls(profile, x_um):
    """
    Apply wall constraints to the profile based on configuration.
    
    Args:
        profile: 1D array of profile values
        x_um: 1D array of x positions in um
    
    Returns:
        profile: Modified profile with walls applied
    """
    if WALL_TYPE is None:
        return profile
    
    ind_outside = (x_um < X_CENTER - FEEDBACK_WIDTH) | (x_um > X_CENTER + FEEDBACK_WIDTH)
    ind_inside = ~ind_outside
    
    if WALL_TYPE == 'hard':
        # Hard walls: immediately set to 1.0 outside feedback region
        profile[ind_outside] = 1.0
        
    elif WALL_TYPE == 'soft':
        # Soft walls: cos^2 transition from edge value to 1.0
        left_edge = X_CENTER - FEEDBACK_WIDTH
        right_edge = X_CENTER + FEEDBACK_WIDTH
        
        # Left wall transition
        left_wall_start = left_edge - SOFT_WALL_WIDTH
        ind_left_wall = (x_um >= left_wall_start) & (x_um < left_edge)
        if np.any(ind_left_wall):
            edge_value = profile[x_um >= left_edge][0] if np.any(x_um >= left_edge) else 0.5
            dist = left_edge - x_um[ind_left_wall]
            transition = np.cos(np.pi/2 * (1 - dist/SOFT_WALL_WIDTH))**2
            profile[ind_left_wall] = edge_value + (1.0 - edge_value) * transition
        
        # Right wall transition  
        right_wall_end = right_edge + SOFT_WALL_WIDTH
        ind_right_wall = (x_um > right_edge) & (x_um <= right_wall_end)
        if np.any(ind_right_wall):
            edge_value = profile[x_um <= right_edge][-1] if np.any(x_um <= right_edge) else 0.5
            dist = x_um[ind_right_wall] - right_edge
            transition = np.cos(np.pi/2 * (1 - dist/SOFT_WALL_WIDTH))**2
            profile[ind_right_wall] = edge_value + (1.0 - edge_value) * transition
        
        # Set to 1.0 beyond the soft wall regions
        profile[x_um < left_wall_start] = 1.0
        profile[x_um > right_wall_end] = 1.0
    
    return profile


def main():
    """Main execution function."""
    
    print("="*70)
    print("Manual DMD Profile Update Script")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1) Find and load the previous shot's h5 file
    # -------------------------------------------------------------------------
    print(f"\nSearching for shot: year={YEAR}, month={MONTH}, day={DAY}, sequence={SEQUENCE}, it={ITERATION}, rep={REPETITION}")
    
    h5_path = load_h5_file(YEAR, MONTH, DAY, SEQUENCE, ITERATION, REPETITION, settings_mod.bec2path)
    
    if h5_path is None:
        print(f"ERROR: Could not find h5 file for specified shot.")
        return
    
    print(f"Found h5 file: {h5_path}")
    
    # Load DMD profile from h5 file
    x_dmd, old_profile = load_dmd_profile_from_h5(h5_path)
    
    if x_dmd is None:
        print("ERROR: Could not load DMD profile from h5 file.")
        return
    
    error_profile_specs = get_error_profile_specs()

    # -------------------------------------------------------------------------
    # 2) Load error profile(s) from txt file(s)
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profile_results = []
    total_scaled_error = np.zeros_like(old_profile, dtype=float)

    for i, spec in enumerate(error_profile_specs, start=1):
        error_file_path = os.path.join(script_dir, spec['filename'])

        print(f"\nLoading error profile {i}/{len(error_profile_specs)} from: {error_file_path}")

        x_error, error_values = load_error_profile_from_txt(error_file_path)
        if x_error is None:
            print(f"ERROR: Could not load error profile {spec['filename']}.")
            return

        print(f"\nExtending error profile {i} to full DMD x-axis...")
        error_extended = extend_error_profile(x_dmd, x_error, error_values)

        sigma = spec['smoothing_sigma']
        print(f"\nApplying Gaussian blur to profile {i} (sigma={sigma})...")
        error_blurred = gaussian_filter(error_extended, sigma=sigma)

        scaled_error = spec['kp'] * error_blurred
        total_scaled_error += scaled_error

        profile_results.append({
            'filename': spec['filename'],
            'kp': spec['kp'],
            'smoothing_sigma': sigma,
            'x_error': x_error,
            'error_values': error_values,
            'error_extended': error_extended,
            'error_blurred': error_blurred,
            'scaled_error': scaled_error,
        })

    # -------------------------------------------------------------------------
    # 3) Generate new profile: old + sum_i(KP_i * blurred_error_i)
    # -------------------------------------------------------------------------
    print(f"\nGenerating new profile from {len(profile_results)} error profile(s)...")
    for result in profile_results:
        print(
            f"  {result['filename']}: KP={result['kp']:+.4f}, "
            f"sigma={result['smoothing_sigma']:.3f}"
        )
    
    new_profile = old_profile + total_scaled_error

    # Ensure non-negativity
    min_val = np.min(new_profile)
    if min_val < 0:
        print(f"Shifting profile up by {-min_val:.4f} to ensure non-negativity")
        new_profile -= min_val
    new_profile[new_profile < 0.0] = 0.0
    
    # -------------------------------------------------------------------------
    # 4) Apply walls if configured
    # -------------------------------------------------------------------------
    if WALL_TYPE is not None:
        print(f"\nApplying {WALL_TYPE} walls...")
        new_profile = apply_walls(new_profile, x_dmd)
    
    # -------------------------------------------------------------------------
    # 5) Plot for visualization
    # -------------------------------------------------------------------------
    print("\nGenerating visualization plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)
    
    # Top plot: profiles
    ax1.plot(x_dmd, old_profile, 'b-', linewidth=2, label='Old DMD Profile', alpha=0.7)
    ax1.plot(x_dmd, new_profile, 'r-', linewidth=2, label='New DMD Profile', alpha=0.7)
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('DMD Intensity (a.u.)')
    ax1.set_title('DMD Profile Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark feedback region if walls are configured
    if WALL_TYPE is not None:
        ax1.axvline(X_CENTER - FEEDBACK_WIDTH, color='green', linestyle=':', 
                   alpha=0.5, label='Feedback Region')
        ax1.axvline(X_CENTER + FEEDBACK_WIDTH, color='green', linestyle=':', alpha=0.5)
        
        if WALL_TYPE == 'soft':
            ax1.axvline(X_CENTER - FEEDBACK_WIDTH - SOFT_WALL_WIDTH, 
                       color='orange', linestyle=':', alpha=0.5, label='Wall Boundary')
            ax1.axvline(X_CENTER + FEEDBACK_WIDTH + SOFT_WALL_WIDTH, 
                       color='orange', linestyle=':', alpha=0.5)
    
    # Bottom plot: error profiles and total correction
    cmap = plt.get_cmap('tab10')
    component_colors = cmap(np.linspace(0, 1, max(len(profile_results), 1)))
    for idx, result in enumerate(profile_results):
        color = component_colors[idx % len(component_colors)]
        ax2.plot(
            result['x_error'],
            result['error_values'],
            linestyle='-',
            marker='o',
            linewidth=1.0,
            markersize=3,
            color=color,
            alpha=0.45,
            label=f"Raw: {result['filename']}"
        )
        ax2.plot(
            x_dmd,
            result['scaled_error'],
            linewidth=1.8,
            color=color,
            alpha=0.9,
            label=(
                f"Scaled: {result['filename']} "
                f"(KP={result['kp']:+.3f}, σ={result['smoothing_sigma']:.2f})"
            )
        )

    ax2.plot(
        x_dmd,
        total_scaled_error,
        'k--',
        linewidth=2.5,
        alpha=0.9,
        label='Total applied correction'
    )
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('Error (a.u.)')
    ax2.set_title('Error Profile Processing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.show()
    
    # -------------------------------------------------------------------------
    # 6) Send to DMD server
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    try:
        print(f"Connecting to DMD server at {DMD_SERVER_IP}:{DMD_SERVER_PORT}...")
        client = zerorpc.Client()
        client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")
        
        print("Testing connection...")
        hello_response = client.hello()
        print(f"Server response: {hello_response}")
        
        print("Sending new profile to DMD...")
        status = client.load_1d_profile(x_dmd.tolist(), new_profile.tolist())
        print(f"DMD update status: {status}")
        
        client.close()
        print("\nDMD profile successfully updated!")
        
    except Exception as e:
        print(f"\nERROR: Failed to update DMD profile: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Script completed.")
    print("="*70)


if __name__ == '__main__':
    main()
