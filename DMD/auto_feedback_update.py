"""
Automated DMD Feedback Update Script

Complete streamlined feedback workflow with profile accumulation:
1. Load base DMD profile (from h5 or txt)
2. Load new sigmoid profile from waterfall
3. Save sigmoid to DMD folder with auto-increment naming
4. Add sigmoid to persistent feedback_sigmoids_list.py file
5. Load ALL sigmoids from the list
6. Apply feedback corrections (with configurable kp and sigma for each)
7. Visualize old vs new profile with error decomposition
8. Update plot_sigmoid_txt.py configuration
9. Load profile to the DMD server (default, can skip with --no-server)

Workflow features:
- Persistent profile list accumulates profile pairs (sigmoid + density) across multiple runs
- Each profile pair has independent kp and smoothing_sigma for sigmoid and density
- Backward compatible: works with sigmoid-only if density profile is missing
- Can manually edit feedback_sigmoids_list.py to adjust/add/remove profiles
- Automatic visualization and DMD loading
- Can delete last correction with --delete-last and reload without it
- Wall edges can be adjusted at runtime with command-line options

Usage:
    python auto_feedback_update.py [OPTIONS]

Options:
    --kp KP_VALUE              Override KP gain for sigmoid profile
    --sigma SIGMA_VALUE        Override smoothing sigma for sigmoid profile
    --kp-d KP_DENSITY          Override KP gain for density profile
    --sigma-d SIGMA_DENSITY    Override smoothing sigma for density profile
    --x-center X_CENTER        Override X_CENTER (center position of feedback region in µm)
    --feedback-width WIDTH     Override FEEDBACK_WIDTH (half-width of feedback region in µm)
    --soft-wall-width WIDTH    Override SOFT_WALL_WIDTH (width of soft wall transition in µm)
    --no-server                Do not load to DMD server
    --no-plot-update           Do not update plot config
    --reset                    Reset all profiles and clear list (WARNING: cannot be undone)
    --delete-last              Delete last correction and reload without it (WARNING: deletes file)

Examples:
    python auto_feedback_update.py                                              # Full workflow with defaults
    python auto_feedback_update.py --kp 1.5 --sigma 5.0 --kp-d 0.5 --sigma-d 3.0  # Override both profiles
    python auto_feedback_update.py --feedback-width 200 --soft-wall-width 100   # Adjust wall positions
    python auto_feedback_update.py --x-center 1000 --feedback-width 150         # Custom feedback region
    python auto_feedback_update.py --no-server                                  # Just save, don't load to DMD
    python auto_feedback_update.py --delete-last                                # Remove last correction and reload
    python auto_feedback_update.py --reset                                      # Reset all profiles
"""

import os
import sys
import shutil
import argparse
import numpy as np
import importlib
import h5py
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Import modules - optional imports for h5 support
try:
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

    spec = importlib.util.spec_from_file_location("general_lib.settings", 
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
    settings_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_mod)
    
    h5_support = True
except Exception as e:
    log_error = str(e)
    h5_support = False

spec = importlib.util.spec_from_file_location("feedback_automation_config", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_automation_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

try:
    import zerorpc
except ImportError:
    zerorpc = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(msg, level=1):
    """Print log message if verbosity >= level."""
    if config.VERBOSITY >= level:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")


def load_dmd_profile_from_txt(txt_path):
    """Load DMD profile from text file (with optional header row)."""
    try:
        if not os.path.isabs(txt_path):
            dmd_folder = os.path.dirname(__file__)
            txt_path = os.path.join(dmd_folder, txt_path)
        
        if not os.path.exists(txt_path):
            log(f"Error: Profile file not found: {txt_path}")
            return None, None
        
        try:
            data = np.loadtxt(txt_path, comments='#')
        except ValueError as e:
            if "could not convert string" in str(e):
                log("Detected header row, skipping it...", level=2)
                data = np.loadtxt(txt_path, comments='#', skiprows=1)
            else:
                raise
        
        if data.ndim != 2 or data.shape[1] != 2:
            log(f"Error: Expected 2 columns in {txt_path}, got shape {data.shape}")
            return None, None
        
        x_um = data[:, 0]
        profile_y = data[:, 1]
        
        log(f"Loaded DMD profile from txt: {len(x_um)} points ({x_um.min():.1f} to {x_um.max():.1f} um)")
        return x_um, profile_y
        
    except Exception as e:
        log(f"Error loading DMD profile from txt: {e}")
        return None, None


def load_dmd_profile_from_h5(year, month, day, sequence, iteration, repetition):
    """Load DMD profile from h5 file."""
    if not h5_support:
        log(f"ERROR: H5 support not available: {log_error}")
        return None, None
    
    try:
        h5_path = general_lib_mod.get_h5filename(
            year=year, month=month, day=day,
            sequence=sequence, it=iteration, rep=repetition,
            bec2_path=settings_mod.bec2path, show_errs=True
        )
        
        if h5_path is None:
            log(f"Error: Could not find h5 file for shot {year}-{month:02d}-{day:02d} seq{sequence} it{iteration} rep{repetition}")
            return None, None
        
        log(f"Found h5 file: {h5_path}")
        
        with h5py.File(h5_path, 'r') as h5file:
            results_group = 'results/feedback_atoms'
            
            # Try to load the last profile
            if results_group + '/last_profile_x_um' in h5file:
                x_um = np.array(h5file[results_group + '/last_profile_x_um'][...])
                profile_y = np.array(h5file[results_group + '/last_profile_y'][...])
                log(f"Loaded DMD profile from h5 (last_profile): {len(x_um)} points")
                return x_um, profile_y
            
            # Try to load the sent profile
            if results_group + '/profile_x_um' in h5file:
                x_um = np.array(h5file[results_group + '/profile_x_um'][...])
                profile_y = np.array(h5file[results_group + '/profile_y_sent'][...])
                log(f"Loaded DMD profile from h5 (sent): {len(x_um)} points")
                return x_um, profile_y
            
            log("Warning: No saved DMD profile found in h5, creating flat profile from camera axis")
            cam_vert1 = camera_settings.cameras['cam_vert1']
            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam_vert1)
            
            if dict_images is not None:
                axes = dict_images['axes']
                x_vals, y_vals, X_name, Y_name, inverted = imaging_analysis_lib_mod.get_axes(axes)
                x_um = x_vals * 1e6
                profile_y = np.ones_like(x_um)
                return x_um, profile_y
    
    except Exception as e:
        log(f"Error loading DMD profile from h5: {e}")
    
    return None, None


def load_error_profile_from_txt(txt_path):
    """Load error profile from text file."""
    try:
        if not os.path.exists(txt_path):
            log(f"Error: File not found: {txt_path}")
            return None, None
        
        try:
            data = np.loadtxt(txt_path, comments='#')
        except ValueError as e:
            if "could not convert string" in str(e):
                log("Detected header row, skipping it...", level=2)
                data = np.loadtxt(txt_path, comments='#', skiprows=1)
            else:
                raise
        
        if data.ndim != 2 or data.shape[1] != 2:
            log(f"Error: Expected 2 columns, got {data.shape}")
            return None, None
        
        x_um = data[:, 0]
        error_values = data[:, 1]
        error_values = error_values - np.mean(error_values)
        
        log(f"Loaded error profile: {len(x_um)} points", level=2)
        return x_um, error_values
        
    except Exception as e:
        log(f"Error loading error profile: {e}")
        return None, None


def extend_error_profile(x_dmd, x_error, error_values):
    """Extend error profile to full DMD x-range (zero outside original range)."""
    # Create interpolation function for the error profile
    interp_func = interp1d(x_error, error_values, kind='linear', 
                          bounds_error=False, fill_value=0.0)
    
    # Extend to full DMD x-axis
    error_extended = interp_func(x_dmd)
    return error_extended


def apply_walls(profile, x_um, wall_type, x_center, feedback_width, soft_wall_width):
    """Apply wall configuration to profile."""
    if wall_type is None:
        return profile
    
    profile = profile.copy()

    ind_inside = (x_um >= x_center - feedback_width) & (x_um <= x_center + feedback_width)
    profile_min = np.min(profile[ind_inside])
    if (profile_min)<0.:
        profile = profile - (profile_min)
    
    if wall_type == 'hard':
        # Hard walls: set to 1.0 outside feedback region
        profile[x_um < x_center - feedback_width] = 1.0
        profile[x_um > x_center + feedback_width] = 1.0
    
    elif wall_type == 'soft':
        # Soft walls: cos^2 transition from edge value to 1.0
        left_edge = x_center - feedback_width
        right_edge = x_center + feedback_width
        
        # Left wall transition
        left_wall_start = left_edge - soft_wall_width
        ind_left_wall = (x_um >= left_wall_start) & (x_um < left_edge)
        if np.any(ind_left_wall):
            edge_value = profile[x_um >= left_edge][0] if np.any(x_um >= left_edge) else 0.5
            # Distance from left edge (0 at left_edge, soft_wall_width at left_wall_start)
            dist = left_edge - x_um[ind_left_wall]
            # cos^2 transition: 0 at wall_start, 1 at edge
            transition = np.cos(np.pi/2 * (1 - dist/soft_wall_width))**2
            profile[ind_left_wall] = edge_value + (1.0 - edge_value) * transition
        
        # Right wall transition  
        right_wall_end = right_edge + soft_wall_width
        ind_right_wall = (x_um > right_edge) & (x_um <= right_wall_end)
        if np.any(ind_right_wall):
            edge_value = profile[x_um <= right_edge][-1] if np.any(x_um <= right_edge) else 0.5
            # Distance from right edge (0 at right_edge, soft_wall_width at right_wall_end)
            dist = x_um[ind_right_wall] - right_edge
            # cos^2 transition: 0 at wall_end, 1 at edge
            transition = np.cos(np.pi/2 * (1 - dist/soft_wall_width))**2
            profile[ind_right_wall] = edge_value + (1.0 - edge_value) * transition
        
        # Set to 1.0 beyond the soft wall regions
        profile[x_um < left_wall_start] = 1.0
        profile[x_um > right_wall_end] = 1.0
    
    return profile


def load_sigmoid_list(dmd_folder):
    """Load the feedback sigmoid list from feedback_sigmoids_list.py."""
    try:
        list_file_path = os.path.join(dmd_folder, 'feedback_sigmoids_list.py')
        
        if not os.path.exists(list_file_path):
            log(f"Warning: Sigmoid list file not found: {list_file_path}")
            log("Creating new list file...")
            return []
        
        spec = importlib.util.spec_from_file_location("feedback_sigmoids_list", list_file_path)
        sigmoid_list_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sigmoid_list_mod)
        
        if hasattr(sigmoid_list_mod, 'SIGMOID_PROFILES'):
            profiles = sigmoid_list_mod.SIGMOID_PROFILES
            log(f"Loaded sigmoid list with {len(profiles)} profiles", level=2)
            return profiles
        else:
            log("Warning: No SIGMOID_PROFILES found in feedback_sigmoids_list.py")
            return []
    
    except Exception as e:
        log(f"Error loading sigmoid list: {e}")
        return []


def add_sigmoid_to_list(dmd_folder, sigmoid_filename, density_filename, kp_sigmoid, sigma_sigmoid, kp_density, sigma_density):
    """Add new sigmoid and density profiles to the feedback_sigmoids_list.py file."""
    try:
        list_file_path = os.path.join(dmd_folder, 'feedback_sigmoids_list.py')
        
        # Read current file
        with open(list_file_path, 'r') as f:
            content = f.read()
        
        # Find SIGMOID_PROFILES list and add new entry
        import re
        
        # Create new entry with both sigmoid and density profiles
        new_entry = f"""    {{
        'sigmoid_filename': '{sigmoid_filename}',
        'density_filename': '{density_filename}',
        'kp_sigmoid': {kp_sigmoid},
        'smoothing_sigma_sigmoid': {sigma_sigmoid},
        'kp_density': {kp_density},
        'smoothing_sigma_density': {sigma_density},
        'description': 'Auto-added from waterfall',
    }},"""
        
        # Find the closing bracket of SIGMOID_PROFILES list
        pattern = r"(SIGMOID_PROFILES\s*=\s*\[)(.*?)(\])"
        
        def replacer(match):
            start = match.group(1)
            middle = match.group(2)
            end = match.group(3)
            # Add new entry before closing bracket
            return start + middle + f"\n{new_entry}\n    " + end
        
        new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
        
        if new_content != content:
            with open(list_file_path, 'w') as f:
                f.write(new_content)
            log(f"Added new profile pair to list:")
            log(f"  Sigmoid: {sigmoid_filename} (kp={kp_sigmoid}, sigma={sigma_sigmoid})")
            log(f"  Density: {density_filename} (kp={kp_density}, sigma={sigma_density})")
            return True
        else:
            log("Warning: Could not update sigmoid list")
            return False
    
    except Exception as e:
        log(f"Error adding profiles to list: {e}")
        return False
    """Find the next available profile number."""
    max_num = -1
    for filename in os.listdir(dmd_folder):
        if filename.startswith('sigmoid_center_interpolation') and filename.endswith('.txt'):
            # Try to extract number from filename
            try:
                if '_update_' in filename:
                    # Format: sigmoid_center_interpolation_update_N.txt
                    num_str = filename.replace('sigmoid_center_interpolation_update_', '').replace('.txt', '')
                    num = int(num_str)
                    max_num = max(max_num, num)
                elif filename == 'sigmoid_center_interpolation.txt':
                    continue
                else:
                    # Format: sigmoid_center_interpolation.txt, sigmoid_center_interpolation0.txt, etc.
                    num_str = filename.replace('sigmoid_center_interpolation', '').replace('.txt', '')
                    if num_str and num_str[0] != '_':
                        try:
                            num = int(num_str)
                            max_num = max(max_num, num)
                        except ValueError:
                            pass
            except (ValueError, IndexError):
                pass
    
    return max_num + 1


def save_sigmoid_update(profile_data, dmd_folder, profile_number, new_sigmoid_path):
    """Save the updated sigmoid profile with proper naming."""
    output_filename = f'sigmoid_center_interpolation_update_{profile_number}.txt'
    output_path = os.path.join(dmd_folder, output_filename)
    
    x_dmd = profile_data['x']
    profile_y = profile_data['y']
    
    # Create header with information about the update
    header = (f"# Updated DMD profile\n"
              f"# Updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"# Source sigmoid: {os.path.basename(new_sigmoid_path)}\n"
              f"# Profile number: {profile_number}\n"
              f"# x (um) vs DMD intensity")
    
    data_to_save = np.column_stack((x_dmd, profile_y))
    np.savetxt(output_path, data_to_save, header=header, fmt='%.6f', 
               delimiter='\t', comments='')
    
    log(f"Saved profile to: {output_path}")
    return output_path


def save_density_update(profile_data, sigmoid_folder, profile_number, new_density_path):
    """Save the updated density profile with proper naming (for backward compatibility)."""
    try:
        x_dmd = profile_data['x']
        profile_y = profile_data['y']
        
        output_filename = f'density_error_profile_update_{profile_number}.txt'
        output_path = os.path.join(sigmoid_folder, output_filename)
        
        # Create header with information about the update
        header = (f"# Updated density error profile\n"
                  f"# Updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# Source density: {os.path.basename(new_density_path)}\n"
                  f"# Profile number: {profile_number}\n"
                  f"# x (um) vs density error (a.u.)")
        
        data_to_save = np.column_stack((x_dmd, profile_y))
        np.savetxt(output_path, data_to_save, header=header, fmt='%.6f', 
                   delimiter='\t', comments='')
        
        log(f"Saved density profile to: {output_path}")
        return output_path
    except Exception as e:
        log(f"Warning: Could not save density profile: {e}")
        return None


def update_plot_config(dmd_folder, profile_number, new_profile_label):
    """Update plot_sigmoid_txt.py configuration to include new profile."""
    plot_config_path = os.path.join(dmd_folder, 'plot_sigmoid_txt.py')
    
    if not os.path.exists(plot_config_path):
        log(f"Warning: plot_sigmoid_txt.py not found at {plot_config_path}")
        return False
    
    # Create backup
    if config.CREATE_BACKUPS:
        backup_path = plot_config_path + '.bak'
        shutil.copy2(plot_config_path, backup_path)
        log(f"Created backup: {backup_path}", level=2)
    
    # Read the file
    with open(plot_config_path, 'r') as f:
        content = f.read()
    
    # Find the PLOT_FILES list and add the new profile
    new_profile_entry = f"""    {{
        'filename': 'sigmoid_center_interpolation_update_{profile_number}.txt',
        'line_style': 'g-',
        'scatter_color': 'g',
        'label': '{new_profile_label}',
    }},"""
    
    # Look for the closing bracket of PLOT_FILES
    import re
    pattern = r"(\]\s*(?=\n\n|# ))"
    match = re.search(pattern, content)
    
    if match:
        insert_pos = match.start()
        new_content = content[:insert_pos] + f",\n{new_profile_entry}\n    " + content[insert_pos:]
        
        with open(plot_config_path, 'w') as f:
            f.write(new_content)
        
        log(f"Updated plot configuration: {plot_config_path}")
        return True
    else:
        log("Warning: Could not find PLOT_FILES list in plot_sigmoid_txt.py")
        return False


def load_to_dmd_server(x_dmd, profile_y, server_ip, server_port):
    """Load profile to DMD server via zerorpc."""
    if zerorpc is None:
        log("Warning: zerorpc not available, cannot load to DMD server")
        return False
    
    try:
        log("Connecting to DMD server...")
        client = zerorpc.Client()
        client.connect(f"tcp://{server_ip}:{server_port}")
        
        log("Testing connection...")
        hello_response = client.hello()
        log(f"Server response: {hello_response}", level=2)
        
        log("Sending profile to DMD server...")
        status = client.load_1d_profile(x_dmd.tolist(), profile_y.tolist())
        log(f"DMD update status: {status}")
        
        client.close()
        log("Successfully loaded profile to DMD server!")
        return True
        
    except Exception as e:
        log(f"Error loading profile to DMD: {e}")
        return False


def plot_profile_comparison(x_dmd, old_profile, new_profile, sigmoid_extended, sigmoid_smoothed, 
                            kp, sigma, wall_type, x_center, feedback_width, soft_wall_width):
    """Create comparison plots for single sigmoid."""
    import matplotlib.pyplot as plt
    
    log("\nGenerating visualization plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)
    
    # Top plot: Old vs New DMD Profile
    ax1.plot(x_dmd, old_profile, 'b-', linewidth=2, label='Old DMD Profile', alpha=0.7)
    ax1.plot(x_dmd, new_profile, 'r-', linewidth=2, label='New DMD Profile', alpha=0.7)
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('DMD Intensity (a.u.)')
    ax1.set_title(f'DMD Profile Comparison (kp={kp:.3f}, σ={sigma:.2f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark feedback region
    if wall_type is not None:
        ax1.axvline(x_center - feedback_width, color='green', linestyle=':', 
                   alpha=0.5, label='Feedback Region')
        ax1.axvline(x_center + feedback_width, color='green', linestyle=':', alpha=0.5)
    
    # Bottom plot: Error profile
    ax2.plot(x_dmd, sigmoid_extended, linewidth=1.0, color='purple', alpha=0.5, 
            label='Raw sigmoid')
    ax2.plot(x_dmd, sigmoid_smoothed, linewidth=2.0, color='purple', alpha=0.8, 
            label=f'Smoothed sigmoid (σ={sigma:.2f})')
    ax2.plot(x_dmd, kp * sigmoid_smoothed, 'k--', linewidth=2.5, alpha=0.9, 
            label=f'Applied correction (kp={kp:.3f})')
    
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('Correction (a.u.)')
    ax2.set_title('Error Profile Processing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.show()
    return fig


def plot_profile_comparison_multi(x_dmd, old_profile, new_profile, sigmoid_details,
                                  wall_type, x_center, feedback_width, soft_wall_width):
    """Create comparison plots for multiple sigmoids."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    log("\nGenerating visualization plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)
    
    # =========================================================================
    # Top plot: Old vs New DMD Profile
    # =========================================================================
    ax1.plot(x_dmd, old_profile, 'b-', linewidth=2, label='Old DMD Profile', alpha=0.7)
    ax1.plot(x_dmd, new_profile, 'r-', linewidth=2, label='New DMD Profile', alpha=0.7)
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('DMD Intensity (a.u.)')
    ax1.set_title(f'DMD Profile Comparison - {len(sigmoid_details)} sigmoids applied')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark feedback region
    if wall_type is not None:
        ax1.axvline(x_center - feedback_width, color='green', linestyle=':', 
                   alpha=0.5, label='Feedback Region')
        ax1.axvline(x_center + feedback_width, color='green', linestyle=':', alpha=0.5)
    
    # =========================================================================
    # Bottom plot: All sigmoid contributions
    # =========================================================================
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    
    total_correction = np.zeros_like(x_dmd)
    
    for i, details in enumerate(sigmoid_details):
        color = colors[i % len(colors)]
        filename = details['filename']
        kp = details['kp']
        sigma = details['sigma']
        extended = details['extended']
        smoothed = details['smoothed']
        
        # Plot raw and smoothed
        ax2.plot(x_dmd, extended, linewidth=0.8, color=color, alpha=0.3, linestyle='--')
        ax2.plot(x_dmd, smoothed, linewidth=1.5, color=color, alpha=0.7, 
                label=f'{filename} (σ={sigma:.1f})')
        
        # Add to total
        total_correction = total_correction + kp * smoothed
    
    # Plot total correction
    ax2.plot(x_dmd, total_correction, 'k-', linewidth=3, alpha=0.9, 
            label='Total applied correction')
    
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('Correction (a.u.)')
    ax2.set_title(f'Error Profiles - {len(sigmoid_details)} sigmoids')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    plt.show()
    return fig


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def reset_feedback_profiles(dmd_folder):
    """Reset feedback system: delete all profiles and clear the list."""
    log("="*70)
    log("RESET FEEDBACK PROFILES")
    log("="*70)
    
    sigmoid_folder = os.path.join(dmd_folder, config.MAGNETIZATION_FEEDBACK_FOLDER)
    
    # Confirm reset
    print("\nThis will:")
    print(f"  1. Delete all sigmoid profiles in: {sigmoid_folder}")
    print("  2. Clear feedback_sigmoids_list.py")
    print("\nThis cannot be undone!\n")
    
    confirm = input("Type 'yes' to confirm reset: ").strip().lower()
    
    if confirm != 'yes':
        log("Reset cancelled.")
        return False
    
    # Delete all sigmoid profiles from magnetization_feedback folder
    if os.path.exists(sigmoid_folder):
        deleted_count = 0
        for filename in os.listdir(sigmoid_folder):
            if filename.startswith('sigmoid_center_interpolation_') and filename.endswith('.txt'):
                filepath = os.path.join(sigmoid_folder, filename)
                try:
                    os.remove(filepath)
                    log(f"  Deleted: {filename}")
                    deleted_count += 1
                except Exception as e:
                    log(f"  ERROR deleting {filename}: {e}", level=1)
        
        log(f"Deleted {deleted_count} profile files")
    
    # Clear feedback_sigmoids_list.py
    list_file = os.path.join(dmd_folder, 'feedback_sigmoids_list.py')
    
    if os.path.exists(list_file):
        try:
            with open(list_file, 'r') as f:
                content = f.read()
            
            # Replace SIGMOID_PROFILES list with empty list
            import re
            pattern = r"(SIGMOID_PROFILES\s*=\s*)\[(.*?)\]"
            new_content = re.sub(pattern, r"\1[\n    ]", content, flags=re.DOTALL)
            
            with open(list_file, 'w') as f:
                f.write(new_content)
            
            log(f"Cleared SIGMOID_PROFILES list in: {os.path.basename(list_file)}")
        except Exception as e:
            log(f"ERROR clearing feedback list: {e}", level=1)
            return False
    
    log("="*70)
    log("Reset complete!")
    log("="*70 + "\n")
    
    return True


def delete_last_correction(dmd_folder):
    """Delete the last correction and reload the profile without it."""
    log("="*70)
    log("DELETE LAST CORRECTION")
    log("="*70)
    
    sigmoid_folder = os.path.join(dmd_folder, config.MAGNETIZATION_FEEDBACK_FOLDER)
    list_file = os.path.join(dmd_folder, 'feedback_sigmoids_list.py')
    
    # Load current list
    if not os.path.exists(list_file):
        log("ERROR: feedback_sigmoids_list.py not found")
        return False
    
    try:
        spec = importlib.util.spec_from_file_location("feedback_sigmoids_list", list_file)
        feedback_list_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feedback_list_mod)
        sigmoid_profiles = feedback_list_mod.SIGMOID_PROFILES
    except Exception as e:
        log(f"ERROR loading sigmoid list: {e}")
        return False
    
    if not sigmoid_profiles:
        log("ERROR: No sigmoids in the list to delete")
        return False
    
    # Show what will be deleted
    last_profile = sigmoid_profiles[-1]
    print("\nThis will:")
    print(f"  1. Delete: {last_profile['filename']}")
    print(f"     Description: {last_profile.get('description', 'N/A')}")
    print(f"  2. Remove it from feedback_sigmoids_list.py")
    print(f"  3. Reload the profile with {len(sigmoid_profiles) - 1} corrections")
    print(f"  4. Load the updated profile to DMD server\n")
    
    confirm = input("Type 'yes' to confirm deletion: ").strip().lower()
    
    if confirm != 'yes':
        log("Deletion cancelled.")
        return False
    
    # Delete the file
    profile_file = os.path.join(sigmoid_folder, last_profile['filename'])
    if os.path.exists(profile_file):
        try:
            os.remove(profile_file)
            log(f"  ✓ Deleted: {last_profile['filename']}")
        except Exception as e:
            log(f"  ERROR deleting file: {e}")
            return False
    
    # Remove from list
    try:
        with open(list_file, 'r') as f:
            content = f.read()
        
        # Parse and remove the last entry
        import re
        
        # Find all dictionaries in SIGMOID_PROFILES list
        pattern = r"(\{\s*'filename':\s*'[^']*'.*?\},?)"
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if matches:
            last_match = matches[-1]
            # Remove the last match and its trailing comma/whitespace
            start = last_match.start()
            end = last_match.end()
            
            # Include trailing comma and whitespace
            while end < len(content) and content[end] in ' \t\n,':
                end += 1
            
            new_content = content[:start] + content[end:]
            
            with open(list_file, 'w') as f:
                f.write(new_content)
            
            log(f"  ✓ Removed from: {os.path.basename(list_file)}")
        else:
            log("ERROR: Could not parse sigmoid list")
            return False
    except Exception as e:
        log(f"ERROR updating feedback list: {e}")
        return False
    
    # Now reload all remaining sigmoids and update the profile
    log("\n--- Reloading profile with remaining corrections ---\n")
    
    # Load base profile
    log("Loading base DMD profile...")
    if config.USE_H5_PROFILE:
        x_dmd, old_profile = load_dmd_profile_from_h5(
            config.H5_SHOT_YEAR, config.H5_SHOT_MONTH, config.H5_SHOT_DAY,
            config.H5_SHOT_SEQUENCE, config.H5_SHOT_ITERATION, config.H5_SHOT_REPETITION
        )
        # Fall back to txt if h5 loading fails
        if x_dmd is None:
            log("WARNING: H5 loading failed, falling back to txt file")
            x_dmd, old_profile = load_dmd_profile_from_txt(config.PROFILE_TXT_FILE)
    else:
        x_dmd, old_profile = load_dmd_profile_from_txt(config.PROFILE_TXT_FILE)
    
    if x_dmd is None:
        log("ERROR: Could not load base DMD profile from either h5 or txt")
        return False
    # Reload the (now shorter) sigmoid list
    try:
        spec = importlib.util.spec_from_file_location("feedback_sigmoids_list", list_file)
        feedback_list_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feedback_list_mod)
        sigmoid_profiles = feedback_list_mod.SIGMOID_PROFILES
        log(f"Loaded {len(sigmoid_profiles)} remaining corrections")
    except Exception as e:
        log(f"ERROR reloading sigmoid list: {e}")
        return False
    
    # Apply all remaining sigmoids
    new_profile = old_profile.copy()
    
    for idx, sigmoid_data in enumerate(sigmoid_profiles):
        filename = sigmoid_data['filename']
        kp = sigmoid_data.get('kp', config.NEW_PROFILE_KP)
        sigma = sigmoid_data.get('smoothing_sigma', config.DEFAULT_SMOOTHING_SIGMA)
        
        sigmoid_path = os.path.join(sigmoid_folder, filename)
        if not os.path.exists(sigmoid_path):
            log(f"WARNING: Sigmoid file not found: {filename}")
            continue
        
        # Load and apply sigmoid
        x_sigmoid, y_sigmoid = load_error_profile_from_txt(sigmoid_path)
        
        if x_sigmoid is None:
            log(f"WARNING: Could not load sigmoid from {filename}")
            continue
        
        # Extend to full DMD range
        y_sigmoid_extended = extend_error_profile(x_dmd, x_sigmoid, y_sigmoid)
        
        # Subtract mean and smooth
        y_error = y_sigmoid_extended - np.mean(y_sigmoid_extended)
        y_error_smooth = gaussian_filter(y_error, sigma=sigma)
        
        # Apply with kp and add to profile
        new_profile += kp * y_error_smooth
        
        log(f"  [{idx + 1}/{len(sigmoid_profiles)}] Applied: {filename} (kp={kp}, sigma={sigma})")
    
    # Ensure non-negativity
    min_val = np.min(new_profile)
    if min_val < 0:
        log(f"Shifting profile up by {-min_val:.4f} to ensure non-negativity")
        new_profile = new_profile - min_val
    new_profile[new_profile < 0.0] = 0.0
    
    # Clip to [0, 1] range before loading to server
    max_val = np.max(new_profile)
    if max_val > 1.0:
        log(f"Clipping profile from [0, {max_val:.4f}] to [0, 1.0]")
        new_profile = np.clip(new_profile, 0.0, 1.0)
    else:
        log(f"Profile range: [0, {max_val:.4f}] - within bounds")
    
    # Save new profile to base location
    log("\nSaving updated profile...")
    try:
        np.savetxt(config.PROFILE_TXT_FILE, np.column_stack([x_dmd, new_profile]), 
                   delimiter='\t', header='x_um\tfield_Hz', comments='')
        log(f"  ✓ Saved to: {config.PROFILE_TXT_FILE}")
    except Exception as e:
        log(f"ERROR saving profile: {e}")
        return False
    
    # Load to DMD server
    log("\nLoading profile to DMD server...")
    success = load_to_dmd_server(x_dmd, new_profile, config.DMD_SERVER_IP, config.DMD_SERVER_PORT)
    if success:
        log("  ✓ Profile loaded to server")
    else:
        log("  WARNING: Could not load to server")
    
    # Summary
    log("\n" + "="*70)
    log(f"✓ Deletion complete!")
    log(f"  Deleted: {last_profile['filename']}")
    log(f"  Remaining corrections: {len(sigmoid_profiles)}")
    log(f"  Profile updated and loaded to DMD server")
    log("="*70 + "\n")
    
    return True


def main():
    """Main automation workflow with profile accumulation."""
    
    print("\n" + "="*70)
    print("Automated DMD Feedback Update - Profile Accumulation Workflow")
    print("="*70 + "\n")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Automated DMD feedback update with accumulation')
    parser.add_argument('--kp', type=float, help='Override KP gain for sigmoid profile')
    parser.add_argument('--sigma', type=float, help='Override smoothing sigma for sigmoid profile')
    parser.add_argument('--kp-d', type=float, help='Override KP gain for density profile')
    parser.add_argument('--sigma-d', type=float, help='Override smoothing sigma for density profile')
    parser.add_argument('--no-server', action='store_true', help='Do not load to DMD server')
    parser.add_argument('--no-plot-update', action='store_true', help='Do not update plot config')
    parser.add_argument('--reset', action='store_true', help='Reset all profiles and clear list')
    parser.add_argument('--delete-last', action='store_true', help='Delete last correction and reload without it')
    parser.add_argument('--x-center', type=float, help='Override X_CENTER (center position of feedback region)')
    parser.add_argument('--feedback-width', type=float, help='Override FEEDBACK_WIDTH (half-width of feedback region)')
    parser.add_argument('--soft-wall-width', type=float, help='Override SOFT_WALL_WIDTH (width of soft wall transition)')
    args = parser.parse_args()
    
    # Determine DMD folder
    if config.DMD_PROFILES_FOLDER:
        dmd_folder = config.DMD_PROFILES_FOLDER
    else:
        dmd_folder = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isdir(dmd_folder):
        log(f"ERROR: DMD folder not found: {dmd_folder}")
        return False
    
    # Handle reset option
    if args.reset:
        return reset_feedback_profiles(dmd_folder)
    
    # Handle delete-last option
    if args.delete_last:
        return delete_last_correction(dmd_folder)
    
    # Get effective parameters for NEW sigmoid
    kp = args.kp if args.kp is not None else config.NEW_PROFILE_KP
    # Get effective parameters for NEW sigmoid profile
    kp = args.kp if args.kp is not None else config.NEW_PROFILE_KP_SIGMOID
    sigma = args.sigma if args.sigma is not None else config.NEW_PROFILE_SMOOTHING_SIGMA_SIGMOID
    
    # Get effective parameters for NEW density profile
    kp_d = args.kp_d if args.kp_d is not None else config.NEW_PROFILE_KP_DENSITY
    sigma_d = args.sigma_d if args.sigma_d is not None else config.NEW_PROFILE_SMOOTHING_SIGMA_DENSITY
    
    auto_update_plot = config.AUTO_UPDATE_PLOT_CONFIG and not args.no_plot_update
    load_to_server = not args.no_server
    
    # Get effective wall parameters
    x_center = args.x_center if args.x_center is not None else config.X_CENTER
    feedback_width = args.feedback_width if args.feedback_width is not None else config.FEEDBACK_WIDTH
    soft_wall_width = args.soft_wall_width if args.soft_wall_width is not None else config.SOFT_WALL_WIDTH
    
    log(f"Sigmoid profile - kp={kp}, sigma={sigma}")
    log(f"Density profile - kp_d={kp_d}, sigma_d={sigma_d}")
    log(f"Wall config - x_center={x_center}, feedback_width={feedback_width}, soft_wall_width={soft_wall_width}")
    log(f"Auto-update plot: {auto_update_plot}")
    log(f"Load to DMD server: {load_to_server}")
    
    # Create magnetization_feedback subfolder for all sigmoid files
    sigmoid_folder = os.path.join(dmd_folder, config.MAGNETIZATION_FEEDBACK_FOLDER)
    if not os.path.exists(sigmoid_folder):
        os.makedirs(sigmoid_folder)
        log(f"Created magnetization_feedback folder: {sigmoid_folder}")
    elif not os.path.isdir(sigmoid_folder):
        log(f"ERROR: {sigmoid_folder} exists but is not a directory")
        return False
    
    # =========================================================================
    # 1) Load base DMD profile
    # =========================================================================
    log("\n--- Step 1: Loading base DMD profile ---")
    
    if config.USE_H5_PROFILE:
        x_dmd, old_profile = load_dmd_profile_from_h5(
            config.H5_SHOT_YEAR, config.H5_SHOT_MONTH, config.H5_SHOT_DAY,
            config.H5_SHOT_SEQUENCE, config.H5_SHOT_ITERATION, config.H5_SHOT_REPETITION
        )
    else:
        x_dmd, old_profile = load_dmd_profile_from_txt(config.PROFILE_TXT_FILE)
    
    if x_dmd is None:
        log("ERROR: Could not load base DMD profile. Exiting.")
        return False
    
    # Apply smoothing to initial profile if configured
    if config.SMOOTH_INITIAL_PROFILE:
        log(f"Applying Gaussian smoothing to initial profile (sigma={config.INITIAL_PROFILE_SMOOTHING_SIGMA})...")
        old_profile = gaussian_filter(old_profile, sigma=config.INITIAL_PROFILE_SMOOTHING_SIGMA)
        log("Initial profile smoothed")
    
    # =========================================================================
    # 2) Load new sigmoid profile from waterfall
    # =========================================================================
    log("\n--- Step 2: Loading new sigmoid profile from waterfall ---")
    
    sigmoid_path = config.NEW_SIGMOID_PATH
    
    # If path doesn't exist, try alternatives
    if not os.path.exists(sigmoid_path):
        log(f"Path not found: {sigmoid_path}", level=2)
        
        # Try relative path in waterfall folder
        alt_path1 = os.path.join(os.path.dirname(dmd_folder), 'waterfall', 'sigmoid_center_interpolation.txt')
        
        if os.path.exists(alt_path1):
            sigmoid_path = alt_path1
            log(f"Using alternative path: {sigmoid_path}", level=2)
        else:
            log(f"ERROR: New sigmoid file not found at {sigmoid_path}")
            log(f"       Also tried: {alt_path1}")
            return False
    
    x_sigmoid, sigmoid_values = load_error_profile_from_txt(sigmoid_path)
    
    if x_sigmoid is None:
        log("ERROR: Could not load new sigmoid profile. Exiting.")
        return False
    
    # Try to load density profile (backward compatible - optional)
    density_path = sigmoid_path.replace('sigmoid_center_interpolation.txt', 'density_error_profile.txt')
    x_density, density_values = None, None
    has_density = False
    
    if os.path.exists(density_path):
        x_density, density_values = load_error_profile_from_txt(density_path)
        if x_density is not None:
            has_density = True
            log(f"Also loaded density error profile: {os.path.basename(density_path)}")
        else:
            log(f"Warning: Could not load density profile at {density_path}, continuing with sigmoid only")
    else:
        log(f"Note: No density profile found at {density_path}, using sigmoid only", level=2)
    
    # =========================================================================
    # 3) Save new sigmoid and density to magnetization_feedback folder
    # =========================================================================
    log("\n--- Step 3: Saving new profiles and updating profile list ---")
    
    # Find next profile update number
    max_num = -1
    for filename in os.listdir(sigmoid_folder):
        if filename.startswith('sigmoid_center_interpolation_update_') and filename.endswith('.txt'):
            try:
                num = int(filename.replace('sigmoid_center_interpolation_update_', '').replace('.txt', ''))
                max_num = max(max_num, num)
            except ValueError:
                pass
    
    new_profile_num = max_num + 1
    new_sigmoid_filename = f'sigmoid_center_interpolation_update_{new_profile_num}.txt'
    new_density_filename = f'density_error_profile_update_{new_profile_num}.txt' if has_density else None
    new_sigmoid_path = os.path.join(sigmoid_folder, new_sigmoid_filename)
    
    # Load the ORIGINAL sigmoid values from waterfall (NOT mean-subtracted)
    try:
        if not os.path.exists(sigmoid_path):
            log(f"Error: File not found: {sigmoid_path}")
            return False
        
        try:
            data = np.loadtxt(sigmoid_path, comments='#')
        except ValueError as e:
            if "could not convert string" in str(e):
                log("Detected header row, skipping it...", level=2)
                data = np.loadtxt(sigmoid_path, comments='#', skiprows=1)
            else:
                raise
        
        if data.ndim != 2 or data.shape[1] != 2:
            log(f"Error: Expected 2 columns, got {data.shape}")
            return False
        
        x_original = data[:, 0]
        y_original = data[:, 1]  # ORIGINAL values, NOT mean-subtracted
        
        # Save original sigmoid with full field values
        header = (f"# Sigmoid profile from waterfall analysis\n"
                  f"# Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# x (um) vs ARPB_final_set_field (full field values, NOT corrected)\n"
                  f"# This is the field profile showing where sample changes state")
        data_to_save = np.column_stack((x_original, y_original))
        np.savetxt(new_sigmoid_path, data_to_save, header=header, fmt='%.6f', delimiter='\t', comments='')
        log(f"Saved original sigmoid: {new_sigmoid_filename} (original field values, not corrected)")
        
    except Exception as e:
        log(f"Error saving sigmoid: {e}")
        return False
    
    # Save density profile if available
    if has_density:
        try:
            if not os.path.exists(density_path):
                log(f"Error: Density file not found: {density_path}")
                has_density = False
            else:
                try:
                    data = np.loadtxt(density_path, comments='#')
                except ValueError as e:
                    if "could not convert string" in str(e):
                        data = np.loadtxt(density_path, comments='#', skiprows=1)
                    else:
                        raise
                
                if data.ndim != 2 or data.shape[1] != 2:
                    log(f"Error: Expected 2 columns in density file, got {data.shape}")
                    has_density = False
                else:
                    x_dens = data[:, 0]
                    y_dens = data[:, 1]
                    
                    # Save original density error
                    header = (f"# Density error profile from waterfall analysis\n"
                              f"# Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                              f"# x (um) vs density error (a.u.)\n"
                              f"# Error is (density - mean density)")
                    data_to_save = np.column_stack((x_dens, y_dens))
                    density_full_path = os.path.join(sigmoid_folder, new_density_filename)
                    np.savetxt(density_full_path, data_to_save, header=header, fmt='%.6f', delimiter='\t', comments='')
                    log(f"Saved original density: {new_density_filename} (density error, not corrected)")
        except Exception as e:
            log(f"Warning: Error saving density profile: {e}, continuing without it")
            has_density = False
    
    # Add to profiles list (with or without density)
    if has_density:
        add_sigmoid_to_list(dmd_folder, new_sigmoid_filename, new_density_filename, 
                            kp, sigma, kp_d, sigma_d)
    else:
        # Backward compatibility: still add with a placeholder
        add_sigmoid_to_list(dmd_folder, new_sigmoid_filename, '', 
                            kp, sigma, kp_d, sigma_d)
    
    # =========================================================================
    # 4) Load ALL profiles from the list and apply corrections
    # =========================================================================
    log("\n--- Step 4: Loading all profiles from list and applying corrections ---")
    
    sigmoid_list = load_sigmoid_list(dmd_folder)
    
    if not sigmoid_list:
        log("ERROR: No profiles found in feedback_sigmoids_list.py")
    
    log(f"Processing {len(sigmoid_list)} profile groups:")
    
    # Start with original profile
    new_profile = old_profile.copy()
    total_error = np.zeros_like(old_profile, dtype=float)
    sigmoid_details = []
    
    # Process each profile pair in the list
    for i, profile_spec in enumerate(sigmoid_list):
        # Backward compatibility: handle both old 'filename' and new 'sigmoid_filename' keys
        sigmoid_filename = profile_spec.get('sigmoid_filename', profile_spec.get('filename'))
        density_filename = profile_spec.get('density_filename', '')
        kp_sig = profile_spec.get('kp_sigmoid', profile_spec.get('kp', 0.5))  # Backward compat
        sigma_sig = profile_spec.get('smoothing_sigma_sigmoid', profile_spec.get('smoothing_sigma', 2.0))
        kp_dens = profile_spec.get('kp_density', 0.3)
        sigma_dens = profile_spec.get('smoothing_sigma_density', 3.0)
        
        log(f"\n  [{i+1}/{len(sigmoid_list)}] Profile group {i+1}")
        
        # ===== Load and apply SIGMOID profile =====
        if sigmoid_filename.startswith('sigmoid_center_interpolation_update_'):
            sigmoid_file_path = os.path.join(sigmoid_folder, sigmoid_filename)
        else:
            sigmoid_file_path = os.path.join(dmd_folder, sigmoid_filename)
        
        if not os.path.exists(sigmoid_file_path):
            log(f"    WARNING: Sigmoid file not found: {sigmoid_file_path}")
            continue
        
        x_sig, sig_values = load_error_profile_from_txt(sigmoid_file_path)
        if x_sig is None:
            log(f"    WARNING: Could not load sigmoid {sigmoid_filename}")
            continue
        
        # Extend to full DMD x-range
        sig_extended = extend_error_profile(x_dmd, x_sig, sig_values)
        
        # Subtract mean
        sig_error = sig_extended - np.mean(sig_extended)
        
        # Apply smoothing
        sig_smoothed = gaussian_filter(sig_error, sigma=sigma_sig)
        
        # Scale
        scaled_sig = kp_sig * sig_smoothed
        new_profile = new_profile + scaled_sig
        total_error = total_error + scaled_sig
        
        # Track sigmoid details for visualization
        sigmoid_details.append({
            'filename': sigmoid_filename,
            'kp': kp_sig,
            'sigma': sigma_sig,
            'extended': sig_extended,
            'smoothed': sig_smoothed,
        })
        
        log(f"    Sigmoid: {sigmoid_filename} (kp={kp_sig}, sigma={sigma_sig})")
        
        # ===== Load and apply DENSITY profile if available =====
        scaled_dens = None
        if density_filename and density_filename.strip():  # Check if not empty
            if density_filename.startswith('density_error_profile_update_'):
                density_file_path = os.path.join(sigmoid_folder, density_filename)
            else:
                density_file_path = os.path.join(dmd_folder, density_filename)
            
            if os.path.exists(density_file_path):
                x_dens, dens_values = load_error_profile_from_txt(density_file_path)
                if x_dens is not None:
                    # Extend to full DMD x-range
                    dens_extended = extend_error_profile(x_dmd, x_dens, dens_values)
                    
                    # Apply smoothing
                    dens_smoothed = gaussian_filter(dens_extended, sigma=sigma_dens)
                    
                    # Scale
                    scaled_dens = kp_dens * dens_smoothed
                    new_profile = new_profile + scaled_dens
                    total_error = total_error + scaled_dens
                    
                    log(f"    Density:  {density_filename} (kp={kp_dens}, sigma={sigma_dens})")
                else:
                    log(f"    WARNING: Could not load density {density_filename}")
            else:
                log(f"    Note: No density file found (backward compatibility): {density_filename}", level=2)
        else:
            log(f"    (No density profile for this group - sigmoid only)")
    
    # Ensure non-negativity
    min_val = np.min(new_profile)
    if min_val < 0:
        log(f"\nShifting profile up by {-min_val:.4f} to ensure non-negativity")
        new_profile = new_profile - min_val
    new_profile[new_profile < 0.0] = 0.0
    
    # Apply walls if configured
    if config.WALL_TYPE is not None:
        log(f"Applying {config.WALL_TYPE} walls...")
        new_profile = apply_walls(new_profile, x_dmd, config.WALL_TYPE, 
                                  x_center, feedback_width, soft_wall_width)
    
    # Clip to [0, 1] range before loading to server
    max_val = np.max(new_profile)
    if max_val > 1.0:
        log(f"Clipping profile from [0, {max_val:.4f}] to [0, 1.0]")
        new_profile = np.clip(new_profile, 0.0, 1.0)
    else:
        log(f"Profile range: [0, {max_val:.4f}] - within bounds")
    
    # =========================================================================
    # 5) Load to DMD server (BEFORE plotting so it doesn't wait for plot window)
    # =========================================================================
    if load_to_server:
        log("\n--- Step 5: Loading profile to DMD server ---")
        load_to_dmd_server(x_dmd, new_profile, config.DMD_SERVER_IP, config.DMD_SERVER_PORT)
    else:
        log("\n--- Step 5: Skipping DMD server load (--no-server flag used) ---")
    
    # =========================================================================
    # 6) Visualize profiles with all sigmoids
    # =========================================================================
    log("\n--- Step 6: Visualizing profiles and corrections ---")
    plot_profile_comparison_multi(x_dmd, old_profile, new_profile, sigmoid_details,
                                 config.WALL_TYPE, x_center, 
                                 feedback_width, soft_wall_width)
    
    # =========================================================================
    # 7) Update plot configuration
    # =========================================================================
    if auto_update_plot:
        log("\n--- Step 7: Updating plot configuration ---")
        update_plot_config(dmd_folder, new_profile_num, config.NEW_PROFILE_PLOT_LABEL)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(f"✓ Feedback update complete!")
    print(f"  New sigmoid saved: {new_sigmoid_filename}")
    print(f"  Total sigmoids in list: {len(sigmoid_list)}")
    print(f"  Base profile: {config.PROFILE_TXT_FILE if not config.USE_H5_PROFILE else f'H5 shot'}")
    if load_to_server:
        print(f"  Profile loaded to DMD server")
    if auto_update_plot:
        print(f"  Plot config updated")
        print(f"  Profile loaded to DMD server")
    print(f"  Edit {dmd_folder}/feedback_sigmoids_list.py to adjust profiles")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
