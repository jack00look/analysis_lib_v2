import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import traceback
import time
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

spec = importlib.util.spec_from_file_location("feedback_utils", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_utils.py")
feedback_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_utils)

spec = importlib.util.spec_from_file_location("feedback_config", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_config.py")
feedback_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_config)

warning_script_name = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/open_warning_window.py"

client = zerorpc.Client()
client.connect(f"tcp://{feedback_config.DMD_SERVER_IP}:{feedback_config.DMD_SERVER_PORT}")

print('Server response:', client.hello())

cameras = camera_settings.cameras

# =============================================================================
# LOAD PARAMETERS FROM CONFIG
# =============================================================================
config = feedback_config.MAGNETIZATION_CONFIG

FEEDBACK_PROGRAM_NAME = config['program_name']
X_CENTER = feedback_config.X_CENTER
FEEDBACK_WIDTH = feedback_config.FEEDBACK_WIDTH
N_THRESHOLD = config['n_threshold']
KP = config['kp']
SMOOTHING_SIGMA = config['smoothing_sigma']

WALL_TYPE = feedback_config.WALL_TYPE
SOFT_WALL_WIDTH = feedback_config.SOFT_WALL_WIDTH
# =============================================================================

RESULTS_GROUP = 'results/feedback_magnetization'
FLAG_DATASET = RESULTS_GROUP + '/dmd_profile_updated'
USED_SHOT_DATASET = RESULTS_GROUP + '/used_for_feedback'
PROGRAM_NAME_DATASET = RESULTS_GROUP + '/program_name'
LOAD_TIME_WINDOW_S = 20.0


def _to_unix_timestamp(value):
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
        try:
            return float(value)
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            pass
        for fmt in ('%Y%m%dT%H%M%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except Exception:
                continue

    return None


def _extract_load_end_timestamp(load_status):
    if load_status is None:
        return None

    if isinstance(load_status, dict):
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
    keys = ('run time', 'run_time', 'runtime')

def _get_shot_run_time_timestamp(h5file):
    """Read shot 'run time' global from h5 and return unix seconds."""
    try:
        # Direct access from h5file.attrs (not globals.attrs)
        run_time_value = h5file.attrs['run time']
        ts = _to_unix_timestamp(run_time_value)
        print(f'DEBUG: raw run time value = {repr(run_time_value)}, parsed = {ts}')
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
        print('ERROR: Could not parse h5 filename.')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    program_name = h5_infos['program_name']
    if program_name != FEEDBACK_PROGRAM_NAME:
        general_lib_mod.save_data(h5file, PROGRAM_NAME_DATASET, np.bytes_(program_name))
        general_lib_mod.save_data(h5file, USED_SHOT_DATASET, np.bool_(False))
        general_lib_mod.save_data(h5file, FLAG_DATASET, np.bool_(False))
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    # -------------------------------------------------------------------------
    # 2) Build magnetization error profile from the current shot
    # -------------------------------------------------------------------------
    cam_vert1 = cameras['cam_vert1']

    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam_vert1)
    if dict_images is None:
        print('ERROR: No images found for cam_vert1.')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    images = dict_images['images']
    infos_images = dict_images['infos']
    axes = dict_images['axes']
    
    # Check if we have both m1 and m2 images
    if 'PTAI_m1' not in images or 'PTAI_m2' not in images:
        print('ERROR: Missing m1 or m2 images.')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    # Get axes
    x_vals, y_vals, X_name, Y_name, inverted = imaging_analysis_lib_mod.get_axes(axes)
    
    # Convert m to um for x_vals
    x_vals = x_vals * 1e6

    # Compute magnetization using utility function
    roi_integration = cam_vert1['roi_integration']
    z1d, Z = feedback_utils.compute_magnetization_profile(
        images, 
        roi_integration=roi_integration
    )
    
    # Get 1D density profiles for both components (with proper unit conversion)
    n2D_m1 = images['PTAI_m1']
    n2D_m2 = images['PTAI_m2']
    n1D_m1_x, n1D_m1_y = imaging_analysis_lib_mod.get_n1Ds(n2D_m1[roi_integration], cam_vert1)
    n1D_m2_x, n1D_m2_y = imaging_analysis_lib_mod.get_n1Ds(n2D_m2[roi_integration], cam_vert1)
    
    # Convert m to um for density profiles (n1D is in 1/m, convert to 1/um)
    n1D_m1 = n1D_m1_x * 1e-6
    n1D_m2 = n1D_m2_x * 1e-6

    # Define integration region
    x_center = X_CENTER
    w = FEEDBACK_WIDTH

    ind_outside = (x_vals < x_center - w) | (x_vals > x_center + w)
    ind_inside = ~ind_outside
    
    # Target magnetization: average in the region
    z1d_avg = np.mean(z1d[ind_inside])
    
    # Compute total atom number from density profiles
    N_tot = np.sum(n1D_m1[ind_inside]) + np.sum(n1D_m2[ind_inside])

    # Check if total atom number is sufficient for feedback
    if N_tot < N_THRESHOLD:
        print(f'ERROR: Atom number ({N_tot:.2e}) below threshold ({N_THRESHOLD:.2e}). Skipping feedback.')
        return {'used_for_feedback': False, 'dmd_profile_updated': False}

    # Compute error: deviation from average
    z1d_err = z1d - z1d_avg

    # Smooth the error
    z1d_err_smooth = gaussian_filter(z1d_err, sigma=SMOOTHING_SIGMA)
    z1d_err_smooth[ind_outside] = 0.0

    # Proportional gain (adjust based on your system response)
    kp = KP

    # -------------------------------------------------------------------------
    # 3) Read previous DMD/magnetization state and validate timing
    # -------------------------------------------------------------------------
    last_profile_x, last_profile_y, already_used = feedback_utils.get_dmd_profile(
        client, h5file, RESULTS_GROUP
    )

    can_apply_feedback = True
    skip_reason = ''

    # If no x-axis from server, use current shot's x-axis
    if last_profile_x is None:
        last_profile_x = x_vals

    if already_used:
        print('Plot will still be shown for inspection, but DMD profile will not be re-updated.')
    else:
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

    # Compute new profile
    new_profile = last_profile_y + kp * z1d_err_smooth
    #new_profile = gaussian_filter(new_profile, sigma=SMOOTHING_SIGMA/2)  # Optional: smooth the new profile to avoid abrupt changes
    new_profile[new_profile < 0.] = 0.

    profile_x = x_vals
    profile_y = new_profile
    
    # -------------------------------------------------------------------------
    # 4) Plot for visualization
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    
    # Plot z1d and z1d_avg on left y-axis
    ax1.plot(x_vals, z1d, 'b-', label='z1d (magnetization)')
    ax1.axhline(z1d_avg, color='c', linestyle='--', label='z1d_avg')
    ax1.set_xlabel('Position (µm)')
    ax1.set_ylabel('Magnetization (a.u.)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for old/new DMD-relevant profiles
    ax2 = ax1.twinx()
    ax2.plot(last_profile_x, last_profile_y, color='orange', linestyle='--', label='Last DMD Profile')
    ax2.plot(profile_x, profile_y, 'r-', label='Generated Profile')
    ax2.set_ylabel('DMD Intensity (a.u.)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Create third y-axis for error profile
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x_vals, z1d_err_smooth, 'g--', alpha=0.7, label='Error Profile (Smoothed)')
    ax3.set_ylabel('Error (a.u.)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    
    title = 'Generated Load Profile for DMD (Magnetization Feedback)'
    if not can_apply_feedback:
        title += f' ⚠️ LOAD TOO LATE ({skip_reason})'
        fig1.patch.set_facecolor('#ffeeee')  # Light red background
        ax1.text(0.5, 0.95, 'LOAD FINISHED OUTSIDE TIMING WINDOW - PROFILE NOT SENT',
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=12, weight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper center')
    ax3.legend(loc='upper right')
    ax1.grid()
    plt.show()

    # -------------------------------------------------------------------------
    # 5) Update DMD only if this shot was not already consumed for feedback
    #    and timing validation allows it.
    # -------------------------------------------------------------------------
    if already_used:
        status = 'skipped_already_used_for_feedback'
        updated_now = False
    elif not can_apply_feedback:
        status = skip_reason
        updated_now = False
        print(f'DMD profile update: NO ({status})')
    else:
        status, updated_now = feedback_utils.update_dmd_profile(
            client, profile_x, profile_y, already_used
        )

    feedback_dict = {
        'program_name': program_name,
        'used_for_feedback': True,
        'dmd_profile_updated': True,
        'dmd_profile_updated_now': updated_now,
        'dmd_profile_update_status': str(status),
    }

    # Save metadata and corresponding profiles/images used for feedback
    general_lib_mod.save_data(h5file, PROGRAM_NAME_DATASET, np.bytes_(feedback_dict['program_name']))
    general_lib_mod.save_data(h5file, USED_SHOT_DATASET, np.bool_(feedback_dict['used_for_feedback']))
    general_lib_mod.save_data(h5file, FLAG_DATASET, np.bool_(feedback_dict['dmd_profile_updated']))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/dmd_profile_updated_now', np.bool_(feedback_dict['dmd_profile_updated_now']))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/dmd_profile_update_status', np.bytes_(feedback_dict['dmd_profile_update_status']))
    
    # Save the previous profile that was on the atoms (loaded before this feedback)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/last_profile_x_um', last_profile_x)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/last_profile_y', last_profile_y)
    
    # Save the new profile that was sent to DMD
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/profile_x_um', profile_x)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/profile_y_sent', profile_y)
    
    # Save derived data and images
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/magnetization_profile_z1d', z1d)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/magnetization_profile_z1d_avg', np.array(z1d_avg))
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/magnetization_profile_z1d_err_smooth', z1d_err_smooth)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/magnetization_image_2d', Z)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/image_m1', images['PTAI_m1'])
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/image_m2', images['PTAI_m2'])
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/density_profile_n1D_m1', n1D_m1)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/density_profile_n1D_m2', n1D_m2)
    general_lib_mod.save_data(h5file, RESULTS_GROUP + '/total_atom_number', np.array(N_tot))

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
