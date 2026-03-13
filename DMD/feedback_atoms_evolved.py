"""
Evolved DMD feedback script with multi-shot averaging capability.

This script extends the original feedback_atoms.py with the ability to:
- Average error profiles across multiple shots
- Track shot eligibility and contribution to feedback iterations
- Provide detailed diagnostics for multi-shot feedback
"""

import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors
import traceback
import time
import subprocess
import sys
from datetime import datetime
from scipy.ndimage import gaussian_filter

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("feedback_config", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_config.py")
feedback_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_config)

spec = importlib.util.spec_from_file_location("feedback_utils", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_utils.py")
feedback_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_utils)

warning_script_name = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/open_warning_window.py"

client = zerorpc.Client()
client.connect(f"tcp://{feedback_config.DMD_SERVER_IP}:{feedback_config.DMD_SERVER_PORT}")

print('Server response:', client.hello())

cameras = camera_settings.cameras

# =============================================================================
# LOAD PARAMETERS FROM CONFIG
# =============================================================================
config = feedback_config.DENSITY_CONFIG

FEEDBACK_PROGRAM_NAME = config['program_name']
X_CENTER = feedback_config.X_CENTER
FEEDBACK_WIDTH = feedback_config.FEEDBACK_WIDTH
N_THRESHOLD = config['n_threshold']
KP = config['kp']
SMOOTHING_SIGMA = config['smoothing_sigma']

WALL_TYPE = feedback_config.WALL_TYPE
SOFT_WALL_WIDTH = feedback_config.SOFT_WALL_WIDTH

# Multi-shot averaging parameters
N_SHOTS_TO_AVERAGE = feedback_config.N_SHOTS_TO_AVERAGE
MAX_LOOKBACK_TIME = feedback_config.MAX_LOOKBACK_TIME
STRICT_MODE = feedback_config.STRICT_MODE
# =============================================================================

RESULTS_GROUP = 'results/feedback_atoms_evolved'
FLAG_DATASET = RESULTS_GROUP + '/dmd_profile_updated'
USED_SHOT_DATASET = RESULTS_GROUP + '/used_for_feedback'
PROGRAM_NAME_DATASET = RESULTS_GROUP + '/program_name'
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


def _extract_load_end_timestamp(load_status):
    """Extract load-completion timestamp from server get_load_status() output."""
    if load_status is None:
        return None

    if isinstance(load_status, dict):
        # If server explicitly reports loading in progress
        if bool(load_status.get('loading', False)) or bool(load_status.get('is_loading', False)):
            return None

        for key in (
            'completion_timestamp',
            'completed_timestamp',
            'last_load_completion_timestamp',
            'last_load_end_timestamp',
            'load_end_timestamp',
            'timestamp',
            'time',
        ):
            if key in load_status:
                ts = _to_unix_timestamp(load_status[key])
                if ts is not None:
                    return ts
        return None

    return _to_unix_timestamp(load_status)


def _get_shot_run_time_timestamp(h5file):
    """Read shot 'run time' global from h5 and return unix seconds."""
    try:
        # Direct access from h5file.attrs (not globals.attrs)
        run_time_value = h5file.attrs['run time']
        ts = _to_unix_timestamp(run_time_value)
        return ts
    except KeyError:
        print(f"DEBUG: 'run time' key not found in h5file.attrs")
        print(f"DEBUG: available keys in h5file.attrs: {list(h5file.attrs.keys())[:10]}")
    except Exception as e:
        print(f'Error reading run time from h5file.attrs: {e}')
        return None

    # Fallback: try alternatives
    for key in ('run_time', 'runtime'):
        try:
            run_time_value = h5file.attrs[key]
            ts = _to_unix_timestamp(run_time_value)
            print(f'DEBUG: found {key}, raw = {repr(run_time_value)}, parsed = {ts}')
            return ts
        except KeyError:
            continue

    return None


def main(client, cameras, h5file):
    # -------------------------------------------------------------------------
    # 1) Gate feedback by sequence/program name
    # -------------------------------------------------------------------------
    h5_infos = general_lib_mod.get_h5file_infos(h5file.filename)
    if h5_infos is None:
        print('Could not parse h5 filename; skipping DMD feedback.')
        print('DMD profile update: NO')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    program_name = h5_infos['program_name']
    if program_name != FEEDBACK_PROGRAM_NAME:
        print(f"Program '{program_name}' is not {FEEDBACK_PROGRAM_NAME}. Skipping DMD feedback.")
        general_lib_mod.save_data(h5file, PROGRAM_NAME_DATASET, np.bytes_(program_name))
        general_lib_mod.save_data(h5file, USED_SHOT_DATASET, np.bool_(False))
        general_lib_mod.save_data(h5file, FLAG_DATASET, np.bool_(False))
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/feedback_eligible', np.bool_(False))
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/eligibility_reason', np.bytes_('wrong_program'))
        print('DMD profile update: NO')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    # If this shot already updated the DMD profile, do not re-apply on reanalysis.
    # We still continue and show the plot for visual inspection.
    already_used_for_feedback = FLAG_DATASET in h5file and bool(h5file[FLAG_DATASET][()])
    if already_used_for_feedback:
        print('This shot already triggered feedback. Plot will be shown but DMD will not be re-updated.')

    # -------------------------------------------------------------------------
    # 2) Build atom-density error profile from the current shot
    # -------------------------------------------------------------------------
    cam_vert1 = cameras['cam_vert1']
    key = 'PTAI_m1'

    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam_vert1)
    if dict_images is None:
        print('ERROR: No images found for cam_vert1; skipping DMD feedback.')
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/feedback_eligible', np.bool_(False))
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/eligibility_reason', np.bytes_('no_images'))
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    images = dict_images['images']
    infos_images = dict_images['infos']
    n2D = dict_images['images'][key]
    roi_integration = cam_vert1['roi_integration']
    n1D_x, n1D_y = imaging_analysis_lib_mod.get_n1Ds(n2D[roi_integration], cam_vert1)
    axes = dict_images['axes']
    x_vals, y_vals, X_name, Y_name, inverted = imaging_analysis_lib_mod.get_axes(axes)

    # m to um conversion of both x_vals and n1D
    x_vals = x_vals * 1e6
    n1D_x = n1D_x * 1e-6

    # Define feedback region using parameters
    ind_outside = (x_vals < X_CENTER - FEEDBACK_WIDTH) | (x_vals > X_CENTER + FEEDBACK_WIDTH)
    ind_inside = ~ind_outside
    n1D_x_avg = np.mean(n1D_x[ind_inside])

    N_tot = np.sum(n1D_x[ind_inside])

    # -------------------------------------------------------------------------
    # 3) Read previous DMD/atoms state and validate timing
    # -------------------------------------------------------------------------
    can_apply_feedback = True
    skip_reason = ''

    if already_used_for_feedback:
        # Load the previously saved profiles from h5
        last_profile_x = np.array(h5file[RESULTS_GROUP + '/last_profile_x_um'][...])
        last_profile_y = np.array(h5file[RESULTS_GROUP + '/last_profile_y'][...])
    else:
        # First run: get current state from DMD server
        last_profile = client.get_last_atoms_profile()
        # Try to use server x-axis if provided, otherwise fallback to current shot axis
        last_profile_x = np.array(last_profile['x']) if 'x' in last_profile else x_vals
        last_profile_y = np.array(last_profile['y'])

        # Validate that this shot was taken in the 20 s before the last load completion
        load_status = client.get_load_status()
        load_end_ts = _extract_load_end_timestamp(load_status)
        run_time_ts = _get_shot_run_time_timestamp(h5file)

        if load_end_ts is None:
            can_apply_feedback = False
            skip_reason = 'skipped_load_in_progress_or_no_completion_timestamp'
            print('ERROR: DMD loading in progress or no completion timestamp available.')
        elif run_time_ts is None:
            can_apply_feedback = False
            skip_reason = 'skipped_missing_run_time_global'
            print("ERROR: Could not read 'run time' global from h5 file.")
        else:
            dt = load_end_ts - run_time_ts
            if dt > LOAD_TIME_WINDOW_S:
                can_apply_feedback = False
                skip_reason = 'skipped_run_time_outside_load_window'
                print(
                    f"ERROR: DMD load finished too late (load_end - run_time = {dt:.1f} s > {LOAD_TIME_WINDOW_S:.0f} s)."
                )

    # -------------------------------------------------------------------------
    # 4) Determine eligibility and save error profile
    # -------------------------------------------------------------------------
    eligibility_reason = ''
    is_eligible = True

    if N_tot < N_THRESHOLD:
        is_eligible = False
        eligibility_reason = 'insufficient_atoms'
        print(f'ERROR: Atom number ({N_tot:.2e}) below threshold ({N_THRESHOLD:.2e}).')
    elif not can_apply_feedback:
        is_eligible = False
        eligibility_reason = skip_reason
        print(f'ERROR: Shot not eligible: {eligibility_reason}')
    else:
        eligibility_reason = 'eligible'
        print(f'Shot is ELIGIBLE for feedback (N={N_tot:.2e})')

    # Compute current error profile
    n1D_x_err = n1D_x - n1D_x_avg
    n1D_x_err_smooth = np.zeros_like(n1D_x_err)
    n1D_x_err_smooth[ind_inside] = gaussian_filter(n1D_x_err[ind_inside], sigma=SMOOTHING_SIGMA)
    n1D_x_err_smooth[ind_outside] = 0.0

    # Save eligibility status and error profile immediately
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/feedback_eligible', np.bool_(is_eligible))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/eligibility_reason', np.bytes_(eligibility_reason))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/error_profile_x', x_vals)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/error_profile', n1D_x_err_smooth)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/total_atom_number', np.array(N_tot))

    if not is_eligible:
        print('Shot not eligible for feedback. Skipping.')
        return {'used_for_feedback': False, 'dmd_profile_updated': False, 'feedback_eligible': False}

    # -------------------------------------------------------------------------
    # 5) Multi-shot averaging (if enabled)
    # -------------------------------------------------------------------------
    contributing_paths = [h5file.filename]
    iteration_id = f"{datetime.now().timestamp()}"
    
    if N_SHOTS_TO_AVERAGE > 1 and not already_used_for_feedback:
        print(f'\n=== Multi-shot averaging enabled (N={N_SHOTS_TO_AVERAGE}) ===')
        
        # Find eligible previous shots
        prev_shots = feedback_utils.find_eligible_previous_shots(
            h5file.filename,
            program_name,
            N_SHOTS_TO_AVERAGE - 1,
            MAX_LOOKBACK_TIME,
            RESULTS_GROUP,
            strict_mode=STRICT_MODE
        )
        
        print(f'Found {len(prev_shots)} eligible previous shots for averaging')
        
        if len(prev_shots) > 0:
            # Retrieve error profiles from previous shots
            prev_error_profiles = feedback_utils.retrieve_error_profiles(prev_shots, RESULTS_GROUP)
            
            print(f'Successfully loaded {len(prev_error_profiles)} error profiles')
            
            # Combine current and previous profiles
            all_error_profiles = [(x_vals, n1D_x_err_smooth)] + prev_error_profiles
            contributing_paths = [h5file.filename] + prev_shots
            
            # Average error profiles
            x_avg, error_avg = feedback_utils.average_error_profiles(all_error_profiles)
            
            print(f'Averaging {len(all_error_profiles)} error profiles')
        else:
            print('No previous shots found, using only current shot')
            x_avg, error_avg = x_vals, n1D_x_err_smooth
    else:
        if N_SHOTS_TO_AVERAGE == 1:
            print('\n=== Multi-shot averaging disabled (N=1) ===')
        x_avg, error_avg = x_vals, n1D_x_err_smooth

    # -------------------------------------------------------------------------
    # 6) Apply feedback correction with averaged error
    # -------------------------------------------------------------------------
    new_profile = last_profile_y + KP * error_avg
    min_inside = np.min(new_profile[ind_inside])
    if min_inside < 0:
        new_profile -= min_inside  # Shift up to ensure non-negativity in feedback region
    new_profile[new_profile < 0.] = 0.

    # -------------------------------------------------------------------------
    # 7) Apply walls if configured
    # -------------------------------------------------------------------------
    if WALL_TYPE == 'hard':
        # Hard walls: immediately set to 1.0 outside feedback region
        new_profile[ind_outside] = 1.0

    elif WALL_TYPE == 'soft':
        # Soft walls: cos^2 transition from edge value to 1.0
        left_edge = X_CENTER - FEEDBACK_WIDTH
        right_edge = X_CENTER + FEEDBACK_WIDTH

        # Left wall transition
        left_wall_start = left_edge - SOFT_WALL_WIDTH
        ind_left_wall = (x_avg >= left_wall_start) & (x_avg < left_edge)
        if np.any(ind_left_wall):
            edge_value = new_profile[x_avg >= left_edge][0] if np.any(x_avg >= left_edge) else 0.5
            dist = left_edge - x_avg[ind_left_wall]
            transition = np.cos(np.pi / 2 * (1 - dist / SOFT_WALL_WIDTH)) ** 2
            new_profile[ind_left_wall] = edge_value + (1.0 - edge_value) * transition

        # Right wall transition
        right_wall_end = right_edge + SOFT_WALL_WIDTH
        ind_right_wall = (x_avg > right_edge) & (x_avg <= right_wall_end)
        if np.any(ind_right_wall):
            edge_value = new_profile[x_avg <= right_edge][-1] if np.any(x_avg <= right_edge) else 0.5
            dist = x_avg[ind_right_wall] - right_edge
            transition = np.cos(np.pi / 2 * (1 - dist / SOFT_WALL_WIDTH)) ** 2
            new_profile[ind_right_wall] = edge_value + (1.0 - edge_value) * transition

        # Set to 1.0 beyond the soft wall regions
        new_profile[x_avg < left_wall_start] = 1.0
        new_profile[x_avg > right_wall_end] = 1.0

    profile_x = x_avg
    profile_y = new_profile

    # -------------------------------------------------------------------------
    # 8) Plot for visualization
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Plot n1D_x and n1D_x_avg on left y-axis
    ax1.plot(x_vals, n1D_x, 'b-', label='n1D_x (current shot)', linewidth=1.5)
    ax1.axhline(n1D_x_avg, color='c', linestyle='--', label='n1D_x_avg', linewidth=1.5)
    ax1.set_xlabel('Position (µm)')
    ax1.set_ylabel('Atom Density (a.u.)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create second y-axis for old/new DMD-relevant profiles
    ax2 = ax1.twinx()
    ax2.plot(last_profile_x, last_profile_y, color='orange', linestyle='--', 
             label='Last Atoms Profile', linewidth=1.5)
    ax2.plot(profile_x, profile_y, 'r-', label='Generated Profile', linewidth=2)
    ax2.set_ylabel('DMD Intensity (a.u.)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Create third y-axis for error profiles
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot current shot error in light green
    ax3.plot(x_vals, n1D_x_err_smooth, 'g-', alpha=0.3, linewidth=1, 
             label='Current Error')
    
    # Plot averaged error (if different from current)
    if len(contributing_paths) > 1:
        ax3.plot(x_avg, error_avg, 'g-', linewidth=2, 
                label=f'Averaged Error (N={len(contributing_paths)})')
    
    ax3.set_ylabel('Error (a.u.)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    # Title with multi-shot info
    title = f'DMD Feedback Profile (Multi-shot: {len(contributing_paths)} shots)'
    if not is_eligible:
        title += f' ⚠️ NOT ELIGIBLE ({eligibility_reason})'
    elif already_used_for_feedback:
        title += ' [REANALYSIS - NO UPDATE]'
    
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper center')
    ax3.legend(loc='upper right')
    ax1.grid()

    # Mark feedback region boundaries
    ax1.axvline(X_CENTER - FEEDBACK_WIDTH, color='green', linestyle=':', alpha=0.5)
    ax1.axvline(X_CENTER + FEEDBACK_WIDTH, color='green', linestyle=':', alpha=0.5)

    # Mark soft wall regions if applicable
    if WALL_TYPE == 'soft':
        ax1.axvline(X_CENTER - FEEDBACK_WIDTH - SOFT_WALL_WIDTH, color='orange', 
                   linestyle=':', alpha=0.5)
        ax1.axvline(X_CENTER + FEEDBACK_WIDTH + SOFT_WALL_WIDTH, color='orange', 
                   linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 9) Update DMD and mark contributors
    # -------------------------------------------------------------------------
    if already_used_for_feedback:
        status = 'skipped_already_used_for_feedback'
        dmd_profile_updated_now = False
        is_trigger = False
        print('DMD profile update: NO (already used for feedback)')
    else:
        status = client.load_1d_profile(profile_x.tolist(), profile_y.tolist())
        dmd_profile_updated_now = True
        is_trigger = True
        print('DMD profile update: YES')
        
        # Mark all contributing shots (including current)
        if len(contributing_paths) > 1:
            print(f'Marking {len(contributing_paths)} shots as contributors...')
            feedback_utils.mark_shots_as_contributors(contributing_paths, RESULTS_GROUP, iteration_id)

    feedback_dict = {
        'program_name': program_name,
        'used_for_feedback': True,
        'dmd_profile_updated': True,
        'dmd_profile_updated_now': dmd_profile_updated_now,
        'dmd_profile_update_status': str(status),
        'feedback_eligible': is_eligible,
        'n_shots_in_average': len(contributing_paths),
    }

    # Save metadata and corresponding profiles/images used for feedback
    general_lib_mod.save_data(h5file, PROGRAM_NAME_DATASET, np.bytes_(feedback_dict['program_name']))
    general_lib_mod.save_data(h5file, USED_SHOT_DATASET, np.bool_(feedback_dict['used_for_feedback']))
    general_lib_mod.save_data(h5file, FLAG_DATASET, np.bool_(feedback_dict['dmd_profile_updated']))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/dmd_profile_updated_now', 
                            np.bool_(feedback_dict['dmd_profile_updated_now']))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/dmd_profile_update_status', 
                            np.bytes_(feedback_dict['dmd_profile_update_status']))

    # Mark if this shot triggered the feedback
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/multishot_trigger', np.bool_(is_trigger))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/multishot_contributor', np.bool_(is_trigger))
    
    if is_trigger:
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/feedback_iteration_id', np.bytes_(iteration_id))
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/n_shots_in_average', 
                                np.int32(len(contributing_paths)))
        
        # Store paths of all contributing shots
        contributing_paths_bytes = [np.bytes_(p) for p in contributing_paths]
        general_lib_mod.save_data(h5file, RESULTS_GROUP + '/contributing_shot_paths', 
                                np.array(contributing_paths_bytes))

    # Save the previous profile that was on the atoms (loaded before this feedback)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/last_profile_x_um', last_profile_x)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/last_profile_y', last_profile_y)

    # Save the new profile that was sent to DMD
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/profile_x_um', profile_x)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/profile_y_sent', profile_y)

    # Save derived data and images
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/atoms_density_profile_x', n1D_x)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/atoms_density_profile_x_avg', np.array(n1D_x_avg))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/atoms_density_profile_x_err_smooth', n1D_x_err_smooth)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/averaged_error_profile', error_avg)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/atoms_image_2d', n2D)

    return feedback_dict


if __name__ == '__main__':
    time_0 = time.time()
    try:
        import lyse
        run = lyse.Run(lyse.path)
        with h5py.File(lyse.path, 'r+') as h5file:
            feedback_dict = main(client, cameras, h5file)
            if feedback_dict is not None:
                for key in feedback_dict:
                    run.save_result(key, feedback_dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
    except Exception as e:
        print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
        print(traceback.format_exc())
    finally:
        client.close()
