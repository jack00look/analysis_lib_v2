import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from waterfall_v2.defect_finding_lib import (
    DEFAULT_DEFECT_CONFIG,
    normalize_defect_config,
    find_defects_in_profile,
)
from waterfall_v2.plot_settings_manager import get_plot_manager
# Create a module-like object for compatibility
class DefectLib:
    DEFAULT_DEFECT_CONFIG = DEFAULT_DEFECT_CONFIG
    @staticmethod
    def normalize_defect_config(cfg=None):
        return normalize_defect_config(cfg)
    @staticmethod
    def find_defects_in_profile(*args, **kwargs):
        return find_defects_in_profile(*args, **kwargs)
defect_lib = DefectLib()


def _get_column(df, col_name):
    """
    Get a column from DataFrame handling both plain and tuple column formats.
    
    Lyse format uses tuple columns: ('col_name', '') for globals, (data_origin, 'name') for measurements.
    This helper tries both formats for compatibility.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    col_name : str
        Column name to retrieve
    
    Returns
    -------
    Series or None
        The column if found, None otherwise
    """
    # Try plain column first (for non-lyse format compatibility)
    if col_name in df.columns:
        return df[col_name]
    
    # Try tuple format (lyse format)
    if (col_name, '') in df.columns:
        return df[(col_name, '')]
    
    return None


def get_um_per_px(params, camera_name_override=None):
    camera_name = camera_name_override if camera_name_override is not None else 'cam_vert1'
    default_um_per_px = 1.0

    try:
        camera_settings_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'imaging_analysis_lib',
            'camera_settings.py',
        )
        spec = importlib.util.spec_from_file_location('imaging_analysis_lib.camera_settings', camera_settings_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load module spec from {camera_settings_path}')
        camera_settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(camera_settings)

        px_size_m = camera_settings.cameras[camera_name]['px_size']
        return float(px_size_m) * 1e6
    except Exception as err:
        print(f"Warning: could not read camera_settings for '{camera_name}' ({err}); using 1.0 µm/px.")
        return default_um_per_px


def get_scan_config(scan_type):
    if scan_type == 'ARP_Forward':
        return 'ARPF_final_set_field', 'ARP Forward'
    if scan_type == 'KZ_det_scan':
        return 'ARPKZ_final_set_field', 'KZ Detuning Scan'
    if scan_type == 'ARP_KZ_Preparation':
        return 'ARPKZ_final_set_field', 'ARP KZ Preparation'
    if scan_type == 'ARP_Backward':
        return 'ARPB_final_set_field', 'ARP Backward'
    if scan_type == 'ARP_KZ':
        return 'ARPKZ_omega_ramp_time', 'ARP Kibble-Zurek'
    if scan_type == 'ARP_KZ_reps':
        return 'rep', 'ARP KZ Repetitions'
    if scan_type == 'KZ_defect_analysis':
        return '_shot_counter', 'KZ Defect Analysis'
    if scan_type == 'ARPF_reps':
        return 'run repeat', 'ARP Forward Repetitions'
    if scan_type == 'ARPF_feedback':
        return 'run repeat', 'ARP Forward Feedback'
    if scan_type == 'DMD_density_feedback':
        return 'run repeat', 'DMD Density Feedback'
    if scan_type in ['bubbles', 'bubbles_evolution']:
        return 'bubbles_time', 'Bubbles'
    if scan_type == 'bubbles_repeat':
        return 'rep', 'Bubbles Repeat'
    return 'run repeat', 'Unknown Scan'


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return str(value)


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return _as_text(value).strip().lower() in ('1', 'true', 'yes', 'y')


def _get_first_available_values(df, candidates):
    for col in candidates:
        try:
            if col in df.columns:
                return df[col].values
        except Exception:
            continue
    return None


def _get_h5_paths_from_df(df):
    try:
        if 'filepath' in df.columns:
            return [p for p in df['filepath'].values if isinstance(p, str)]
    except Exception:
        pass
    try:
        return [p for p in df.index.values if isinstance(p, str)]
    except Exception:
        return []


def _get_global_attr_values_from_h5(df, attr_name):
    vals = []
    for h5_path in _get_h5_paths_from_df(df):
        try:
            with h5py.File(h5_path, 'r') as h5f:
                if 'globals' in h5f and attr_name in h5f['globals'].attrs:
                    vals.append(h5f['globals'].attrs[attr_name])
        except Exception:
            continue
    return vals


def _build_globals_title_lines(df, globals_in_title):
    lines = []
    for name, unit in globals_in_title:
        vals = _get_global_attr_values_from_h5(df, name)
        if len(vals) == 0:
            lines.append(f'{name} [{unit}]: not found')
        else:
            try:
                lines.append(f'{name} [{unit}]: {np.unique(vals)}')
            except Exception:
                lines.append(f'{name} [{unit}]: {vals}')
    return lines


def filter_dataframe(df, seqs, constraints=None, shot_name_filter=None):
    seq_col = _get_column(df, 'sequence_index')
    if seq_col is None:
        raise KeyError("'sequence_index' column not found (tried both plain and tuple formats)")
    
    out = df[seq_col.isin(seqs)]
    print(f"Initial shots in sequences {seqs}: {len(out)}")
    if constraints is not None:
        for key, value in constraints.items():
            before = len(out)
            try:
                key_col = _get_column(out, key)
                if key_col is None:
                    print(f"  Constraint {key}={value}: column not found, skipping")
                    continue
                    
                if not isinstance(value, (str, bytes)) and hasattr(value, '__iter__'):
                    out = out[key_col.isin(value)]
                else:
                    out = out[np.abs(key_col - value) < 1e-3]
            except Exception:
                key_col = _get_column(out, key)
                if key_col is not None:
                    out = out[key_col == value]
            after = len(out)
            rejected = before - after
            print(f"  Constraint {key}={value}: {rejected} rejected, {after} remaining")
    
    # Filter by shot name if provided
    if shot_name_filter is not None:
        before = len(out)
        try:
            # Check if 'name' column exists
            if 'name' in out.columns:
                out = out[out['name'].str.contains(shot_name_filter, na=False)]
                after = len(out)
                rejected = before - after
                print(f"  Shot name filter '{shot_name_filter}': {rejected} rejected, {after} remaining")
            else:
                print(f"  Warning: 'name' column not found for shot name filtering. Available columns: {list(out.columns)}")
        except Exception as e:
            print(f"  Warning: Error applying shot name filter: {e}")
    
    # Add a unique shot counter for plotting all shots
    out = out.copy()
    out['_shot_counter'] = np.arange(len(out))
    return out


def filter_valid_feedback_shots(df):
    status_candidates = [
        ('feedback_magnetization', 'dmd_profile_update_status'),
        ('feedback_atoms', 'dmd_profile_update_status'),
        'dmd_profile_update_status',
    ]
    updated_now_candidates = [
        ('feedback_magnetization', 'dmd_profile_updated_now'),
        ('feedback_atoms', 'dmd_profile_updated_now'),
        'dmd_profile_updated_now',
    ]

    status_vals = _get_first_available_values(df, status_candidates)
    updated_vals = _get_first_available_values(df, updated_now_candidates)

    if status_vals is None and updated_vals is None:
        print('DMD feedback: no feedback columns found; keeping all shots.')
        return df

    n = len(df)
    mask = np.ones(n, dtype=bool)
    status_txt = None

    if status_vals is not None:
        status_txt = np.array([_as_text(v) for v in status_vals])
        bad = {
            'skipped_run_time_outside_load_window',
            'skipped_load_in_progress_or_no_completion_timestamp',
            'skipped_missing_run_time_global',
        }
        mask &= ~np.isin(status_txt, list(bad))

    if updated_vals is not None:
        updated = np.array([_as_bool(v) for v in updated_vals])
        if status_txt is None:
            mask &= updated
        else:
            mask &= (updated | (status_txt == 'skipped_already_used_for_feedback'))

    print(f'DMD feedback: keeping {np.count_nonzero(mask)}/{n} shots.')
    return df[mask]


def save_sigmoid_center_interpolation(M_data, x_um, y_vals, y_axis_label, um_per_px, params):
    """
    Extract sigmoid centers from magnetization profiles and save interpolation to file.
    
    Analyzes the spatial magnetization profile at different y-values (scan parameter),
    extracts the sigmoid center position for each y, and saves interpolated profile to:
        waterfall_v2/sigmoid_center_interpolation.txt
    
    Parameters
    ----------
    M_data : ndarray
        Magnetization data, shape (n_shots, n_pixels) if averaged or (n_y_values, n_pixels)
    x_um : ndarray
        X-axis positions in micrometers
    y_vals : ndarray
        Y-axis values (scan parameter values)
    y_axis_label : str
        Label for y-axis (e.g., 'ARPB_final_set_field')
    um_per_px : float
        Micrometers per pixel
    params : dict
        Parameters dictionary (unused but kept for compatibility)
    """
    try:
        if M_data is None or len(M_data) == 0:
            print("Warning: No magnetization data to extract sigmoid centers from")
            return
        
        if len(M_data) < 2:
            print("Warning: Need at least 2 y-values to interpolate sigmoid centers")
            return
        
        # Ensure data is 2D
        M_data = np.atleast_2d(M_data)
        if M_data.shape[0] < len(y_vals):
            M_data = M_data.T
        
        # Extract sigmoid center for each profile
        sigmoid_centers_x = []
        sigmoid_centers_y = []
        
        for i, (m_profile, y_val) in enumerate(zip(M_data, y_vals)):
            if len(m_profile) < 5:
                continue
            
            # Find magnetization extrema (where derivative is closest to zero on both sides)
            m_smooth = gaussian_filter1d(m_profile, sigma=2)
            
            # Find the center as the position of maximum absolute value
            max_idx = np.argmax(np.abs(m_smooth))
            if 5 <= max_idx < len(m_profile) - 5:
                sigmoid_centers_x.append(x_um[max_idx] if max_idx < len(x_um) else max_idx * um_per_px)
                sigmoid_centers_y.append(float(y_val))
        
        if len(sigmoid_centers_x) < 2:
            print("Warning: Could not extract at least 2 sigmoid centers")
            return
        
        # Sort by x position
        sorted_pairs = sorted(zip(sigmoid_centers_x, sigmoid_centers_y))
        section_centers_x = np.array([p[0] for p in sorted_pairs])
        sigmoid_centers_y_sorted = np.array([p[1] for p in sorted_pairs])
        
        # Create interpolation function
        kind = 'cubic' if len(section_centers_x) >= 4 else 'linear'
        interp_func = interp1d(section_centers_x, sigmoid_centers_y_sorted, kind=kind, fill_value='extrapolate')
        
        # Generate smooth x values for interpolation
        x_smooth = np.linspace(min(section_centers_x), max(section_centers_x), 200)
        y_smooth = interp_func(x_smooth)
        
        # Save to text file in waterfall_v2 script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'sigmoid_center_interpolation.txt')
        
        header = f"# Interpolated sigmoid center profile\n# x (um) vs {y_axis_label} (sigmoid center)\n# Interpolation type: {kind}\n"
        data_to_save = np.column_stack((x_smooth, y_smooth))
        np.savetxt(output_file, data_to_save, header=header, fmt='%.6f', delimiter='\t', comments='')
        print(f"✓ Saved sigmoid center interpolation to: {output_file}")
        
    except Exception as e:
        print(f"Warning: Error in save_sigmoid_center_interpolation: {e}")


def _load_camera_settings_module():
    camera_settings_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'imaging_analysis_lib',
        'camera_settings.py',
    )
    spec = importlib.util.spec_from_file_location('imaging_analysis_lib.camera_settings', camera_settings_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec from {camera_settings_path}')
    camera_settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_settings)
    return camera_settings


def _legacy_1d_key_from_atom(atom_name):
    if isinstance(atom_name, str) and '_' in atom_name:
        prefix, rest = atom_name.split('_', 1)
        if prefix in ('PTAI', 'PHC'):
            return f'{rest}_1d'
    return f'{atom_name}_1d'


def _first_valid_profile_column(df, candidates):
    for col in candidates:
        if col not in df.columns:
            continue
        vals = df[col].values
        if len(vals) == 0:
            continue
        # Look for first valid array, not just the first value
        for first in vals:
            if hasattr(first, 'shape') and len(getattr(first, 'shape', ())) > 0:
                return col
    return None


def _resolve_modality(mode_cfg, params):
    default_modality_name = mode_cfg.get('magnetization_modality', 'vert1_two_component')
    modality_map = mode_cfg.get('magnetization_modalities', {}) if isinstance(mode_cfg, dict) else {}
    modality_cfg = dict(modality_map.get(default_modality_name, {})) if hasattr(modality_map, 'get') else {}

    if len(modality_cfg) == 0:
        if default_modality_name == 'vert2_phc_multicomponent':
            modality_cfg = {
                'camera_name': 'cam_vert2_PHC',
                'type': 'phc_profiles',
                'data_origin': mode_cfg.get('data_origin', 'show_ODs'),
                'atoms_images': ['PHC_1', 'PHC_2', 'PHC_3', 'PHC_4'],
                'apply_ntot_check': False,
            }
        else:
            modality_cfg = {
                'camera_name': 'cam_vert1',
                'type': 'two_component',
                'data_origin': mode_cfg.get('data_origin', 'show_ODs'),
                'atoms_images': ['PTAI_m1', 'PTAI_m2'],
                'apply_ntot_check': True,
            }

    if 'camera_name' not in modality_cfg:
        modality_cfg['camera_name'] = 'cam_vert1'
    if 'data_origin' not in modality_cfg:
        modality_cfg['data_origin'] = mode_cfg.get('data_origin', 'show_ODs')
    if 'apply_ntot_check' not in modality_cfg:
        modality_cfg['apply_ntot_check'] = bool(modality_cfg.get('type', 'two_component') == 'two_component')

    # Auto-populate atoms_images from camera settings if not explicitly provided.
    if 'atoms_images' not in modality_cfg or not modality_cfg.get('atoms_images'):
        try:
            camera_settings = _load_camera_settings_module()
            modality_cfg['atoms_images'] = list(camera_settings.cameras[modality_cfg['camera_name']]['atoms_images'])
        except Exception:
            modality_cfg['atoms_images'] = ['PTAI_m1', 'PTAI_m2']

    return modality_cfg


def get_valid_rows_mask(df, filter_cfg=None, modality_cfg=None, data_origin='show_ODs'):
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)

    # Backward compatibility: old signature get_valid_rows_mask(df, back_threshold, ntot_threshold)
    if isinstance(filter_cfg, (int, float, np.integer, np.floating)):
        back_thr = float(filter_cfg)
        ntot_thr = float(modality_cfg if isinstance(modality_cfg, (int, float, np.integer, np.floating)) else 7e5)
        filter_cfg = {
            'enabled': True,
            'apply_back_check': True,
            'apply_sat_check': False,
            'apply_probe_norm_err_check': False,
            'back_threshold': back_thr,
            'ntot_threshold': ntot_thr,
        }
        modality_cfg = {'atoms_images': ['PTAI_m1', 'PTAI_m2'], 'apply_ntot_check': True}

    if filter_cfg is None:
        filter_cfg = {
            'enabled': True,
            'apply_back_check': True,
            'apply_sat_check': False,
            'apply_probe_norm_err_check': False,
            'back_threshold': 0.05,
            'sat_threshold': 0.001,
            'probe_norm_err_threshold': 1.0,
            'ntot_threshold': 7e5,
        }
    if modality_cfg is None:
        modality_cfg = {'atoms_images': ['PTAI_m1', 'PTAI_m2'], 'apply_ntot_check': True}

    if not bool(filter_cfg.get('enabled', True)):
        return np.ones(n, dtype=bool)

    atoms = list(modality_cfg.get('atoms_images', []))
    if len(atoms) == 0:
        atoms = ['PTAI_m1', 'PTAI_m2']

    def _collect_max_metric(suffix):
        cols = []
        for atom in atoms:
            cols.extend([
                (data_origin, f'{atom}_{suffix}'),
                (data_origin, f'{atom}_SVD_{suffix}'),
            ])

        arrays = []
        for col in cols:
            if col not in df.columns:
                continue
            try:
                arr = np.asarray(df[col].values, dtype=float)
            except Exception:
                continue
            arrays.append(np.nan_to_num(arr, nan=-np.inf))

        if len(arrays) == 0:
            return None
        return np.max(np.vstack(arrays), axis=0)

    def _collect_sum_metric(suffix):
        vals = np.zeros(n, dtype=float)
        found = False
        for atom in atoms:
            candidates = [
                (data_origin, f'{atom}_{suffix}'),
                (data_origin, f'{atom}_SVD_{suffix}'),
            ]
            arr_atom = None
            for col in candidates:
                if col not in df.columns:
                    continue
                try:
                    arr_atom = np.asarray(df[col].values, dtype=float)
                    break
                except Exception:
                    continue
            if arr_atom is not None:
                found = True
                vals += np.nan_to_num(arr_atom, nan=0.0)
        return vals if found else None

    back_atoms = _collect_max_metric('cnt_rel_atoms_back')
    back_probe = _collect_max_metric('cnt_rel_probe_back')
    sat_atoms = _collect_max_metric('cnt_rel_atoms_sat')
    sat_probe = _collect_max_metric('cnt_rel_probe_sat')
    probe_norm = _collect_max_metric('probe_norm_err')
    n_tot = _collect_sum_metric('N_atoms')

    good = np.ones(n, dtype=bool)
    reject_reasons = {'background': 0, 'saturation': 0, 'probe_norm': 0, 'ntot': 0}

    if bool(filter_cfg.get('apply_back_check', True)):
        thr = float(filter_cfg.get('back_threshold', 0.05))
        back_atoms_ok = np.ones(n, dtype=bool) if back_atoms is None else (back_atoms < thr)
        back_probe_ok = np.ones(n, dtype=bool) if back_probe is None else (back_probe < thr)
        back_ok = back_atoms_ok & back_probe_ok
        reject_reasons['background'] = int(np.count_nonzero(~back_ok))
        good &= back_ok

    if bool(filter_cfg.get('apply_sat_check', False)):
        thr = float(filter_cfg.get('sat_threshold', 0.001))
        sat_atoms_ok = np.ones(n, dtype=bool) if sat_atoms is None else (sat_atoms < thr)
        sat_probe_ok = np.ones(n, dtype=bool) if sat_probe is None else (sat_probe < thr)
        sat_ok = sat_atoms_ok & sat_probe_ok
        reject_reasons['saturation'] = int(np.count_nonzero(~sat_ok))
        good &= sat_ok

    if bool(filter_cfg.get('apply_probe_norm_err_check', False)):
        thr = float(filter_cfg.get('probe_norm_err_threshold', 1.0))
        probe_norm_ok = np.ones(n, dtype=bool) if probe_norm is None else (probe_norm < thr)
        reject_reasons['probe_norm'] = int(np.count_nonzero(~probe_norm_ok))
        good &= probe_norm_ok

    if bool(modality_cfg.get('apply_ntot_check', True)):
        thr = float(filter_cfg.get('ntot_threshold', 7e5))
        n_tot_ok = np.ones(n, dtype=bool) if n_tot is None else (n_tot > thr)
        reject_reasons['ntot'] = int(np.count_nonzero(~n_tot_ok))
        good &= n_tot_ok

    print(
        'Shot quality filter summary: '
        f"keep={np.count_nonzero(good)}/{n}, "
        f"reject_background={reject_reasons['background']}, "
        f"reject_saturation={reject_reasons['saturation']}, "
        f"reject_probe_norm={reject_reasons['probe_norm']}, "
        f"reject_ntot={reject_reasons['ntot']}"
    )

    return good


def unpack_images(df, m1_lbl, m2_lbl):
    # Handle both plain and tuple column formats
    m1_col = None
    if m1_lbl in df.columns:
        m1_col = df[m1_lbl]
    elif isinstance(m1_lbl, tuple) and m1_lbl in df.columns:
        m1_col = df[m1_lbl]
    else:
        # Try to find it as a tuple with (data_origin, name)
        for col in df.columns:
            if isinstance(col, tuple) and len(col) == 2 and col[1] == m1_lbl:
                m1_col = df[col]
                break
    
    if m1_col is None:
        raise KeyError(f"Could not find m1 column: {m1_lbl}")
    
    m1_raw = m1_col.values
    if len(m1_raw) == 0:
        return None, None, 0

    dim = m1_raw[0].shape[0]
    m1 = np.zeros((len(m1_raw), dim))
    
    # Check if m2 data exists (not present in DMD density feedback)
    m2_col = None
    if m2_lbl in df.columns:
        m2_col = df[m2_lbl]
    elif isinstance(m2_lbl, tuple) and m2_lbl in df.columns:
        m2_col = df[m2_lbl]
    else:
        # Try to find it as a tuple with (data_origin, name)
        for col in df.columns:
            if isinstance(col, tuple) and len(col) == 2 and col[1] == m2_lbl:
                m2_col = df[col]
                break
    
    has_m2 = m2_col is not None
    if has_m2:
        m2_raw = m2_col.values
        # Check if m2_raw contains valid arrays (not NaN scalars)
        m2_valid = False
        try:
            if len(m2_raw) > 0 and hasattr(m2_raw[0], 'shape'):
                m2_valid = True
        except (TypeError, AttributeError):
            pass
        
        if m2_valid:
            m2 = np.zeros((len(m2_raw), dim))
            for i in range(len(m1_raw)):
                try:
                    m1[i] = m1_raw[i]
                    m2[i] = m2_raw[i]
                except Exception:
                    pass
        else:
            # m2 column exists but contains NaN/invalid data
            m2 = np.zeros((len(m1_raw), dim))
            for i in range(len(m1_raw)):
                try:
                    m1[i] = m1_raw[i]
                except Exception:
                    pass
    else:
        # For DMD density feedback: only m1 exists, set m2 to zero
        m2 = np.zeros((len(m1_raw), dim))
        for i in range(len(m1_raw)):
            try:
                m1[i] = m1_raw[i]
            except Exception:
                pass
    
    return m1, m2, dim


def _load_profile_matrix(df, data_origin, atom_name, explicit_col=None):
    candidates = []
    if explicit_col is not None:
        candidates.append(explicit_col)

    legacy_1d = _legacy_1d_key_from_atom(atom_name)
    candidates.extend([
        (data_origin, f'{atom_name}_n1D_x'),
        (data_origin, f'{atom_name}_SVD_n1D_x'),
        (data_origin, legacy_1d),
    ])

    col = _first_valid_profile_column(df, candidates)
    if col is None:
        return None, None

    raw = df[col].values
    if len(raw) == 0:
        return None, None

    # Find first valid array (skip NaN scalars or invalid data)
    dim = None
    for val in raw:
        try:
            if isinstance(val, np.ndarray) and len(val) > 0:
                dim = int(val.shape[0])
                break
        except Exception:
            continue
    
    if dim is None:
        return None, None

    arr = np.zeros((len(raw), dim), dtype=float)
    for i in range(len(raw)):
        try:
            arr[i] = np.asarray(raw[i], dtype=float)
        except Exception:
            pass
    return arr, col


def build_magnetization_inputs(sorted_df, mode_cfg, params, data_origin):
    modality_cfg = _resolve_modality(mode_cfg, params)
    modality_type = str(modality_cfg.get('type', 'two_component')).strip().lower()

    if modality_type in ('multi_component', 'phc_profiles'):
        atoms = list(modality_cfg.get('atoms_images', []))
        mag_profiles = {}
        dim = None

        for atom in atoms:
            arr, col_used = _load_profile_matrix(sorted_df, data_origin, atom)
            if arr is None:
                continue
            if dim is None:
                dim = arr.shape[1]
            if arr.shape[1] != dim:
                continue
            mag_profiles[atom] = np.asarray(arr, dtype=float)

        if len(mag_profiles) == 0:
            return None

        # Keep all PHC magnetization profiles available for downstream analysis.
        # Until a dedicated downstream consumer is defined, use their mean as the
        # representative profile for existing waterfall plots.
        profile_names = list(mag_profiles.keys())
        profile_stack = np.stack([mag_profiles[name] for name in profile_names], axis=0)
        M = np.mean(profile_stack, axis=0)
        D = np.ones_like(M)

        n_shots = M.shape[0]
        profiles_per_shot = []
        for i in range(n_shots):
            shot_dict = {name: np.asarray(mag_profiles[name][i], dtype=float) for name in profile_names}
            profiles_per_shot.append(shot_dict)

        M_full, D_full, z_fluc_full = process_profiles(
            M,
            D,
            params['SIGMA_Z_LOCAL_AVG'],
            params['SIGMA_Z_LOCAL_FLUCT'],
        )
        return {
            'M_full': M_full,
            'D_full': D_full,
            'z_fluc_full': z_fluc_full,
            'img_dims': M_full.shape[1],
            'modality_cfg': modality_cfg,
            'magnetization_profiles': mag_profiles,
            'magnetization_profile_names': profile_names,
            'magnetization_profiles_per_shot': profiles_per_shot,
        }

    # Default: two-component mode
    m1_col = modality_cfg.get('m1_column', None)
    m2_col = modality_cfg.get('m2_column', None)

    m1_arr, m1_col_used = _load_profile_matrix(sorted_df, data_origin, 'PTAI_m1', explicit_col=m1_col)
    m2_arr, m2_col_used = _load_profile_matrix(sorted_df, data_origin, 'PTAI_m2', explicit_col=m2_col)

    if m1_arr is None:
        # Fallback to legacy columns
        m1_lbl = (data_origin, 'm1_1d')
        m2_lbl = (data_origin, 'm2_1d')
        m1_arr, m2_arr, _ = unpack_images(sorted_df, m1_lbl, m2_lbl)
        m1_col_used = m1_lbl
        m2_col_used = m2_lbl
        if m1_arr is None:
            return None

    if m2_arr is None:
        m2_arr = np.zeros_like(m1_arr)

    # Optional affine recomputation of n1D profiles for two-component modality only
    affine_cfg = modality_cfg.get('affine_correction', {})
    a1 = float(affine_cfg.get('a1', 1.0))
    b1 = float(affine_cfg.get('b1', 0.0))
    c1 = float(affine_cfg.get('c1', 0.0))
    a2 = float(affine_cfg.get('a2', 1.0))
    b2 = float(affine_cfg.get('b2', 0.0))
    c2 = float(affine_cfg.get('c2', 0.0))

    if (a1, b1, c1, a2, b2, c2) != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
        m1_src = np.asarray(m1_arr, dtype=float)
        m2_src = np.asarray(m2_arr, dtype=float)
        m1_arr = a1 * m1_src + b1 * m2_src + c1
        m2_arr = a2 * m2_src + b2 * m1_src + c2

    M_full, D_full, z_fluc_full = process_magnetization(
        m1_arr,
        m2_arr,
        params['SIGMA_Z_LOCAL_AVG'],
        params['SIGMA_Z_LOCAL_FLUCT'],
    )
    return {
        'M_full': M_full,
        'D_full': D_full,
        'z_fluc_full': z_fluc_full,
        'img_dims': M_full.shape[1],
        'modality_cfg': modality_cfg,
        'm1_col_used': m1_col_used,
        'm2_col_used': m2_col_used,
    }


def process_magnetization(m1, m2, sigma_avg, sigma_fluct):
    m1 = np.clip(m1, 0, None)
    m2 = np.clip(m2, 0, None)
    d = m1 + m2
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (m2 - m1) / d
    m = np.nan_to_num(np.clip(m, -1, 1), nan=0.0)

    return process_profiles(m, d, sigma_avg, sigma_fluct)


def process_profiles(m, d, sigma_avg, sigma_fluct):
    m = np.asarray(m, dtype=float)
    d = np.asarray(d, dtype=float)
    m = np.nan_to_num(np.clip(m, -1, 1), nan=0.0)
    d = np.nan_to_num(np.clip(d, 0, None), nan=0.0)

    # Handle zero sigma: use minimal smoothing to avoid identically zero residuals
    if sigma_avg > 0:
        local_avg = gaussian_filter1d(m, sigma=sigma_avg, axis=1)
    else:
        # Use minimal smoothing (0.5) instead of raw m to avoid pw = m - m = 0
        local_avg = gaussian_filter1d(m, sigma=0.5, axis=1)
    
    pw = m - local_avg
    
    if sigma_fluct > 0:
        local_fluct = gaussian_filter1d(np.abs(pw), sigma=sigma_fluct, axis=1)
    else:
        local_fluct = np.abs(pw)

    return m, d, local_fluct


def _build_group_indices(y_raw, rounding_decimals=6):
    """
    Build stable groups for scan values.

    For floating-point scan values, we round before grouping to avoid tiny
    numerical differences preventing averaging.
    """
    y_raw = np.asarray(y_raw)
    if y_raw.size == 0:
        return np.array([]), []

    try:
        y_num = y_raw.astype(float)
        keys = np.round(y_num, rounding_decimals)
        unique_keys = np.unique(keys)
        groups = [np.where(keys == k)[0] for k in unique_keys]
        y_group_vals = np.array([np.mean(y_num[g]) for g in groups])
        return y_group_vals, groups
    except Exception:
        unique_vals = np.unique(y_raw)
        groups = [np.where(y_raw == v)[0] for v in unique_vals]
        return unique_vals, groups


def _merge_close_same_sign_peaks(peak_idx, x_axis_um, min_sep_um):
    """Merge same-sign peaks that are closer than min_sep_um into one midpoint peak."""
    idx = np.asarray(peak_idx, dtype=int)
    x_axis_um = np.asarray(x_axis_um, dtype=float)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    idx = np.unique(np.sort(idx))
    if float(min_sep_um) <= 0:
        return idx, x_axis_um[idx]

    merged_idx = []
    merged_x = []

    cluster = [int(idx[0])]
    for ii in idx[1:]:
        ii = int(ii)
        if abs(float(x_axis_um[ii]) - float(x_axis_um[cluster[-1]])) < float(min_sep_um):
            cluster.append(ii)
            continue

        if len(cluster) == 1:
            i0 = int(cluster[0])
            merged_idx.append(i0)
            merged_x.append(float(x_axis_um[i0]))
        else:
            i0 = int(cluster[0])
            i1 = int(cluster[-1])
            xmid = 0.5 * (float(x_axis_um[i0]) + float(x_axis_um[i1]))
            imid = int(np.argmin(np.abs(x_axis_um - xmid)))
            merged_idx.append(imid)
            merged_x.append(float(xmid))
        cluster = [ii]

    if len(cluster) == 1:
        i0 = int(cluster[0])
        merged_idx.append(i0)
        merged_x.append(float(x_axis_um[i0]))
    else:
        i0 = int(cluster[0])
        i1 = int(cluster[-1])
        xmid = 0.5 * (float(x_axis_um[i0]) + float(x_axis_um[i1]))
        imid = int(np.argmin(np.abs(x_axis_um - xmid)))
        merged_idx.append(imid)
        merged_x.append(float(xmid))

    return np.asarray(merged_idx, dtype=int), np.asarray(merged_x, dtype=float)


def _infer_start_sign_from_signed_walls(pos_walls, neg_walls):
    """
    Infer left-domain sign from first signed defect:
      - first negative derivative peak (down step)  => start at +1
      - first positive derivative peak (up step)    => start at -1
    Returns +1.0, -1.0, or None if unavailable.
    """
    pos = np.asarray(pos_walls if pos_walls is not None else [], dtype=float)
    neg = np.asarray(neg_walls if neg_walls is not None else [], dtype=float)

    if pos.size == 0 and neg.size == 0:
        return None
    if pos.size == 0:
        return 1.0
    if neg.size == 0:
        return -1.0

    first_pos = float(np.min(pos))
    first_neg = float(np.min(neg))
    return 1.0 if first_neg < first_pos else -1.0


def compute_fluctuation_sigma2_waterfall(M_shots, y_raw, sigma_fluct, sigma_avg):
    """
    Compute per-scan-point averaged local fluctuation profile:
      1) average magnetization profile over shots in each scan value
      2) compute shot residuals to this average (ensemble fluctuations)
      3) compute local sigma^2 profile from squared residuals
      4) smooth with sigma_fluct if > 0
      5) average sigma^2 over shots at the same scan value
    
    Note: sigma_fluct is applied to smooth the squared residuals. 
          If sigma_fluct <= 0, use a minimal smoothing (sigma=0.5) to avoid pure pixel noise.
    """
    y_group_vals, groups = _build_group_indices(y_raw)
    if len(groups) == 0:
        return y_group_vals, np.zeros((0, 0))

    n_x = M_shots.shape[1]
    sigma2_map = np.zeros((len(groups), n_x))
    
    # Use minimal smoothing if sigma_fluct <= 0 to avoid pure pixel noise
    sigma_fluct_eff = sigma_fluct if sigma_fluct > 0 else 0.5

    for i, idx in enumerate(groups):
        if len(idx) == 0:
            continue

        m_group = M_shots[idx]
        if len(idx) >= 2:
            # Ensemble fluctuations at fixed scan value: m - <m>
            m_avg = np.mean(m_group, axis=0)
            delta = m_group - m_avg[None, :]
            delta_sq = delta ** 2
            # Average delta^2 across reps FIRST, then smooth once
            avg_delta_sq = np.mean(delta_sq, axis=0)
            local_sigma2 = gaussian_filter1d(avg_delta_sq, sigma=sigma_fluct_eff, axis=0)
            sigma2_map[i] = local_sigma2
        else:
            # Single-shot fallback: use spatial fluctuations around local trend
            if sigma_avg > 0:
                local_trend = gaussian_filter1d(m_group, sigma=sigma_avg, axis=1)
            else:
                # If no sigma_avg, use minimal smoothing for the trend
                local_trend = gaussian_filter1d(m_group, sigma=0.5, axis=1)
            delta = m_group - local_trend
            delta_sq = delta ** 2
            # Average delta^2 across reps (single rep), then smooth
            avg_delta_sq = np.mean(delta_sq, axis=0)
            local_sigma2 = gaussian_filter1d(avg_delta_sq, sigma=sigma_fluct_eff, axis=0)
            sigma2_map[i] = local_sigma2

    return y_group_vals, sigma2_map


def plot_fluctuations_waterfall(sigma2_map, y_vals, dy, y_axis_label, title, params, um_per_px, mode_cfg=None, dw_fit_coeffs=None, dw_fit_x=None, dw_fit_t=None):
    """Plot local fluctuation sigma^2 as a waterfall map.
    
    If params contains DOMAIN_WALL_X_MIN and DOMAIN_WALL_X_MAX, add a side plot showing
    fluctuation profiles at 5 x-positions, shifted by domain wall crossing time.
    
    Domain wall fit (dw_fit_coeffs, dw_fit_x, dw_fit_t) can be provided to shift profiles
    so peaks align based on when the domain wall reaches each x-position.
    """
    if sigma2_map.size == 0 or len(y_vals) == 0:
        return

    # Check if we should add the side profile plot based on params
    add_profile_plot = False
    if params and isinstance(params, dict):
        x_min_dw = params.get('DOMAIN_WALL_X_MIN', None)
        x_max_dw = params.get('DOMAIN_WALL_X_MAX', None)
        add_profile_plot = (x_min_dw is not None and x_max_dw is not None)
    
    if add_profile_plot:
        fig, (ax, ax_profile) = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    
    x_plot_px = np.arange(-0.5, sigma2_map.shape[1], 1)
    x_plot_um = x_plot_px * um_per_px

    polys, colors = [], []
    for i in range(len(y_vals)):
        for j in range(sigma2_map.shape[1]):
            polys.append([
                (x_plot_um[j], y_vals[i] - dy / 2),
                (x_plot_um[j + 1], y_vals[i] - dy / 2),
                (x_plot_um[j + 1], y_vals[i] + dy / 2),
                (x_plot_um[j], y_vals[i] + dy / 2),
            ])
            colors.append(sigma2_map[i, j])

    coll = PolyCollection(polys, array=np.array(colors), cmap='magma')
    ax.add_collection(coll)

    x_min_um = params['X_MIN_INTEGRATION']
    x_max_um = params['X_MAX_INTEGRATION']

    # Auto color scale from max fluctuation in the analysis region
    # (same x-window used in the main analysis).
    x_centers_um = np.arange(sigma2_map.shape[1]) * um_per_px
    xmin_ind = int(np.argmin(np.abs(x_centers_um - x_min_um)))
    xmax_ind = int(np.argmin(np.abs(x_centers_um - x_max_um)))
    if xmax_ind < xmin_ind:
        xmin_ind, xmax_ind = xmax_ind, xmin_ind

    if xmax_ind > xmin_ind:
        region = sigma2_map[:, xmin_ind:xmax_ind]
    else:
        region = sigma2_map

    finite_region = region[np.isfinite(region)]
    if finite_region.size > 0:
        vmax = float(np.max(finite_region))
        if vmax > 0:
            coll.set_clim(0.0, vmax)

    ax.axvline(x=x_min_um, color='lime', linestyle='--', linewidth=2.0, alpha=0.95)
    ax.axvline(x=x_max_um, color='lime', linestyle='--', linewidth=2.0, alpha=0.95)
    # Clip to used region in x
    ax.set_xlim(x_min_um, x_max_um)
    ax.set_ylim(np.min(y_vals) - dy, np.max(y_vals) + dy)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel('x [μm]')
    ax.set_title(title)
    cbar = plt.colorbar(coll, ax=ax, label=r'Local $\sigma^2$ of $m - \langle m \rangle$')
    
    # Add side plot with profiles at 3 x-positions if domain wall bounds are set
    if add_profile_plot:
        x_min_dw = params.get('DOMAIN_WALL_X_MIN')
        x_max_dw = params.get('DOMAIN_WALL_X_MAX')
        x_q1_dw = x_min_dw + (x_max_dw - x_min_dw) / 4.0
        x_mid_dw = (x_min_dw + x_max_dw) / 2.0
        x_q3_dw = x_min_dw + 3 * (x_max_dw - x_min_dw) / 4.0
        
        x_positions = [x_min_dw, x_q1_dw, x_mid_dw, x_q3_dw, x_max_dw]
        colors_profile = ['blue', 'cyan', 'green', 'orange', 'red']
        labels_profile = [f'x={x_min_dw}μm', f'x={x_q1_dw:.1f}μm', f'x={x_mid_dw:.1f}μm', f'x={x_q3_dw:.1f}μm', f'x={x_max_dw}μm']
        
        # Draw colored vertical lines in main plot at profile positions
        for x_pos, color in zip(x_positions, colors_profile):
            ax.axvline(x=x_pos, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Compute shifts based on domain wall fit
        shifts = {}
        print(f"DEBUG: dw_fit_coeffs={dw_fit_coeffs}, dw_fit_t={dw_fit_t is not None}, dw_fit_x={dw_fit_x is not None}")
        if dw_fit_coeffs is not None and dw_fit_t is not None and dw_fit_x is not None:
            # Linear fit: t(x) = a*x + b, where x is in meters and t is in bubbles_time units
            # dw_fit_coeffs = [a, b] from polyfit(x_in_meters, t)
            a, b = dw_fit_coeffs if len(dw_fit_coeffs) >= 2 else (1.0, 0.0)
            print(f"DEBUG: Using fit equation: bubbles_time = {a:.6e} * x_m + {b:.6f}")
            print(f"DEBUG: Or in μm: bubbles_time = {a*1e-6:.6e} * x_um + {b:.6f}")
            for x_pos in x_positions:
                # x_pos is in micrometers, convert to meters
                x_pos_m = x_pos * 1e-6
                # Use fit equation: t = a*x + b
                t_cross = a * x_pos_m + b
                shifts[x_pos] = t_cross
                print(f"DEBUG:   x={x_pos} μm: bubbles_time_offset={t_cross:.3f}")
        else:
            print(f"DEBUG: No fit data, no shift")
            # No fit data, no shift
            for x_pos in x_positions:
                shifts[x_pos] = 0.0
        
        for x_pos, color, label in zip(x_positions, colors_profile, labels_profile):
            # Find closest x index
            x_ind = int(np.argmin(np.abs(x_centers_um - x_pos)))
            if 0 <= x_ind < sigma2_map.shape[1]:
                # Shift y_vals (bubbles_time) by the offset at this x position
                y_shifted = y_vals - shifts[x_pos]
                print(f"DEBUG: Plotting {label}, shift={shifts[x_pos]:.3f}, y_shifted range=[{y_shifted.min():.2f}, {y_shifted.max():.2f}]")
                ax_profile.plot(y_shifted, sigma2_map[:, x_ind], 'o-', color=color, label=label, linewidth=2)
        
        ax_profile.set_xlabel(f'{y_axis_label} (shifted by DW crossing time)')
        ax_profile.set_ylabel(r'Local $\sigma^2$ of $m - \langle m \rangle$')
        ax_profile.set_title('Fluctuation profiles (shifted by DW position)')
        ax_profile.legend()
        ax_profile.grid(True, alpha=0.3)
        
        # Perform exponential decay fits for y_shifted > 0
        from scipy.optimize import curve_fit
        
        def exp_decay(y, A, tau):
            """Exponential decay: A * exp(-y/tau), goes to zero at infinity"""
            return A * np.exp(-y / tau)
        
        tau_values_5pos = []
        tau_errors_5pos = []
        x_positions_for_tau_5pos = []
        
        # First fit the 5 specific positions and show fits on the plot
        for x_pos, color, label in zip(x_positions, colors_profile, labels_profile):
            x_ind = int(np.argmin(np.abs(x_centers_um - x_pos)))
            if 0 <= x_ind < sigma2_map.shape[1]:
                y_shifted = y_vals - shifts[x_pos]
                fluct_vals = sigma2_map[:, x_ind]
                
                # Select only positive shifted times
                mask = y_shifted > 0
                if np.sum(mask) >= 3:  # Need at least 3 points
                    y_fit = y_shifted[mask]
                    f_fit = fluct_vals[mask]
                    
                    try:
                        # Initial guess
                        A_guess = np.max(f_fit)
                        tau_guess = (y_fit.max() - y_fit.min()) / 3
                        p0 = [A_guess, tau_guess]
                        
                        # Fit
                        popt, pcov = curve_fit(exp_decay, y_fit, f_fit, p0=p0, maxfev=5000)
                        tau = popt[1]
                        tau_err = np.sqrt(pcov[1, 1])
                        
                        tau_values_5pos.append(tau)
                        tau_errors_5pos.append(tau_err)
                        x_positions_for_tau_5pos.append(x_pos)
                        
                        print(f"DEBUG: {label} - tau = {tau:.3f} ± {tau_err:.3f}")
                        
                        # Plot fitted curve on the profile plot
                        y_fit_smooth = np.linspace(y_fit.min(), y_fit.max(), 100)
                        f_fit_smooth = exp_decay(y_fit_smooth, *popt)
                        ax_profile.plot(y_fit_smooth, f_fit_smooth, '--', color=color, linewidth=2, alpha=0.7)
                        
                    except Exception as e:
                        print(f"DEBUG: Fit failed for {label}: {e}")
        
        # Now fit all pixels in the domain wall region
        x_min_dw_m = params.get('DOMAIN_WALL_X_MIN', 960) * 1e-6
        x_max_dw_m = params.get('DOMAIN_WALL_X_MAX', 1070) * 1e-6
        x_min_dw_um = params.get('DOMAIN_WALL_X_MIN', 960)
        x_max_dw_um = params.get('DOMAIN_WALL_X_MAX', 1070)
        
        # Find x indices for domain wall region
        x_min_ind = int(np.argmin(np.abs(x_centers_um - x_min_dw_um)))
        x_max_ind = int(np.argmin(np.abs(x_centers_um - x_max_dw_um)))
        if x_max_ind < x_min_ind:
            x_min_ind, x_max_ind = x_max_ind, x_min_ind
        
        tau_all = []
        tau_err_all = []
        x_all = []
        
        for x_ind in range(x_min_ind, x_max_ind + 1):
            x_um = x_centers_um[x_ind]
            # Get shift for this x position
            if dw_fit_coeffs is not None:
                x_m = x_um * 1e-6
                a, b = dw_fit_coeffs
                shift = a * x_m + b
            else:
                shift = 0.0
            
            y_shifted = y_vals - shift
            fluct_vals = sigma2_map[:, x_ind]
            
            # Select only positive shifted times
            mask = y_shifted > 0
            if np.sum(mask) >= 3:
                y_fit = y_shifted[mask]
                f_fit = fluct_vals[mask]
                
                try:
                    # Initial guess
                    A_guess = np.max(f_fit)
                    tau_guess = (y_fit.max() - y_fit.min()) / 3
                    p0 = [A_guess, tau_guess]
                    
                    # Fit
                    popt, pcov = curve_fit(exp_decay, y_fit, f_fit, p0=p0, maxfev=5000)
                    tau = popt[1]
                    tau_err = np.sqrt(pcov[1, 1])
                    
                    tau_all.append(tau)
                    tau_err_all.append(tau_err)
                    x_all.append(x_um)
                    
                except Exception:
                    pass
        
        # Create comprehensive tau vs x plot for all domain wall region
        if len(tau_all) >= 2:
            fig_tau_all, ax_tau_all = plt.subplots(figsize=(10, 6), tight_layout=True)
            ax_tau_all.errorbar(x_all, tau_all, yerr=tau_err_all, fmt='.', color='darkblue', 
                               ecolor='lightblue', capsize=3, linewidth=1.5, markersize=5, alpha=0.7, label='All pixels')
            # Overlay the 5 specific positions
            ax_tau_all.errorbar(x_positions_for_tau_5pos, tau_values_5pos, yerr=tau_errors_5pos, 
                               fmt='o', color='darkred', ecolor='red', capsize=5, linewidth=2, markersize=8, label='5 sampled positions')
            ax_tau_all.set_xlabel('x [μm]')
            ax_tau_all.set_ylabel(r'Decay time constant $\tau$')
            ax_tau_all.set_title(f'Fluctuation decay timescale vs position (DW region: {x_min_dw_um}-{x_max_dw_um} μm)')
            ax_tau_all.grid(True, alpha=0.3)
            ax_tau_all.legend()


def plot_used_region_density_fluctuations(D, y_unique, dy, y_axis_label, title, params, um_per_px, mode_cfg=None):
    """Plot density fluctuations (density - shot_avg_density) in the used region as a waterfall.
    
    Left panel: Sum of squared density fluctuations for each shot.
    Right panel: Waterfall of density fluctuations across the used region.
    
    Parameters:
        mode_cfg: Optional dict with 'density_fluct_y_label' to override y-axis label
    """
    x_min_um = params['X_MIN_INTEGRATION']
    x_max_um = params['X_MAX_INTEGRATION']
    x_centers_um = np.arange(D.shape[1]) * um_per_px
    xmin_ind = int(np.argmin(np.abs(x_centers_um - x_min_um)))
    xmax_ind = int(np.argmin(np.abs(x_centers_um - x_max_um)))
    
    if xmax_ind < xmin_ind:
        xmin_ind, xmax_ind = xmax_ind, xmin_ind
    
    if xmax_ind <= xmin_ind:
        print("Warning: Invalid used region bounds in plot_used_region_density_fluctuations")
        return
    
    # Extract used region
    D_used = D[:, xmin_ind:xmax_ind]
    x_used_px = np.arange(-0.5, D_used.shape[1], 1)
    x_used_um = x_used_px * um_per_px + x_min_um
    
    # Compute density fluctuations for each shot: D_fluct[i, j] = D[i, j] - mean(D[i, :])
    D_shot_avg = np.mean(D_used, axis=1, keepdims=True)
    D_fluct = D_used - D_shot_avg
    
    # Compute sum of squared fluctuations for each shot
    sum_sq_fluct = np.sum(D_fluct**2, axis=1)
    
    # Use custom y-axis label if provided in mode_cfg
    plot_y_label = y_axis_label
    if mode_cfg is not None and mode_cfg.get('density_fluct_y_label') is not None:
        plot_y_label = mode_cfg['density_fluct_y_label']
    
    # Create figure with 2 subplots (left for integrated, right for waterfall)
    fig, (ax_left, ax_right) = plt.subplots(
        ncols=2,
        tight_layout=True,
        figsize=(10, 5),
        sharey=True,
        gridspec_kw={'width_ratios': [0.3, 1]},
    )
    
    # Left panel: Plot sum of squared fluctuations vs y_unique
    ax_left.plot(sum_sq_fluct, y_unique, 'b-', linewidth=1.5, label='Sum($D_{fluct}^2$)')
    ax_left.set_xlabel(r'$\sum_x (D - \langle D \rangle)^2$ (a.u.)')
    ax_left.set_ylabel(plot_y_label)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc='best', fontsize=9)
    
    # Right panel: Waterfall plot of density fluctuations
    polys, colors = [], []
    for i in range(len(y_unique)):
        for j in range(D_fluct.shape[1]):
            polys.append([
                (x_used_um[j], y_unique[i] - dy / 2),
                (x_used_um[j + 1], y_unique[i] - dy / 2),
                (x_used_um[j + 1], y_unique[i] + dy / 2),
                (x_used_um[j], y_unique[i] + dy / 2),
            ])
            colors.append(D_fluct[i, j])
    
    coll = PolyCollection(polys, array=np.array(colors), cmap='RdBu_r')
    ax_right.add_collection(coll)
    
    # Set color limits symmetrically around zero for better visualization
    finite_colors = np.array(colors)
    finite_colors = finite_colors[np.isfinite(finite_colors)]
    if finite_colors.size > 0:
        vmax = np.max(np.abs(finite_colors))
        if vmax > 0:
            coll.set_clim(-vmax, vmax)
    
    ax_right.set_xlim(x_used_um[0], x_used_um[-1])
    ax_right.set_ylim(np.min(y_unique) - dy, np.max(y_unique) + dy)
    ax_right.set_xlabel('x [μm]')
    ax_right.set_title(title)
    cbar = plt.colorbar(coll, ax=ax_right)
    cbar.set_label(r'Density - shot avg density (a.u.)')


def _sectioned_sigmoid_analysis(M, y_unique, x_plot_um, xmin_ind, xmax_ind, num_sections):
    from scipy.optimize import curve_fit

    def sigmoid(x, x0, k, a, b):
        return a / (1.0 + np.exp(-(x - x0) / k)) + b

    section_edges = np.linspace(xmin_ind, xmax_ind, num_sections + 1, dtype=int)
    sx, sy, se = [], [], []
    avg_density_all = np.mean(M[:, xmin_ind:xmax_ind], axis=0) if xmax_ind > xmin_ind else np.mean(M, axis=0)

    for i in range(num_sections):
        s0, s1 = section_edges[i], section_edges[i + 1]
        if s1 <= s0:
            continue
        integrated = np.sum(M[:, s0:s1], axis=1) / (s1 - s0)
        p0 = [np.median(y_unique), 0.1, np.max(integrated) - np.min(integrated), np.min(integrated)]
        try:
            popt, pcov = curve_fit(sigmoid, y_unique, integrated, p0=p0, maxfev=10000)
            sx.append(x_plot_um[int((s0 + s1) / 2)])
            sy.append(popt[0])
            se.append(float(np.sqrt(max(0.0, pcov[0, 0]))))
        except Exception:
            continue

    return sx, sy, se, avg_density_all


def plot_main_waterfall(
    M,
    D,
    y_unique,
    dy,
    y_axis_label,
    title,
    scan,
    params,
    plot_flags,
    um_per_px,
    average=False,
    defect_points=None,
    defect_points_pos=None,
    defect_points_neg=None,
    defect_points_corrected=None,
    defect_points_rejected=None,
    defect_counts=None,
    defect_sequences=None,
    defect_thresholds=None,
    dw_fit_coeffs=None,
    dw_fit_x=None,
    dw_fit_t=None,
):
    fig, axs = plt.subplots(
        ncols=3,
        tight_layout=True,
        figsize=(12, 5),
        sharey=True,
        gridspec_kw={'width_ratios': [0.4, 1, 1]},
    )
    ax_int, ax_m, ax_d = axs
    x_plot_px = np.arange(-0.5, M.shape[1], 1)
    x_plot_um = x_plot_px * um_per_px
    
    # For bubbles_evolution, convert coordinates to meters and prepare for y-axis inversion
    is_bubbles_evolution = scan == 'bubbles_evolution'
    if is_bubbles_evolution:
        x_plot_plot = x_plot_um * 1e-6  # Convert to meters
    else:
        x_plot_plot = x_plot_um

    polys_M, colors_M = [], []
    for i in range(len(y_unique)):
        for j in range(M.shape[1]):
            polys_M.append([
                (x_plot_plot[j], y_unique[i] - dy / 2),
                (x_plot_plot[j + 1], y_unique[i] - dy / 2),
                (x_plot_plot[j + 1], y_unique[i] + dy / 2),
                (x_plot_plot[j], y_unique[i] + dy / 2),
            ])
            colors_M.append(M[i, j])

    collection_M = PolyCollection(polys_M, array=np.array(colors_M), cmap='RdBu')
    if params['WATERFALL_MAG_CLIM'] is not None:
        collection_M.set_clim(*params['WATERFALL_MAG_CLIM'])
    ax_m.add_collection(collection_M)
    mag_vmin, mag_vmax = collection_M.get_clim()

    polys_D, colors_D = [], []
    for i in range(len(y_unique)):
        for j in range(D.shape[1]):
            polys_D.append([
                (x_plot_plot[j], y_unique[i] - dy / 2),
                (x_plot_plot[j + 1], y_unique[i] - dy / 2),
                (x_plot_plot[j + 1], y_unique[i] + dy / 2),
                (x_plot_plot[j], y_unique[i] + dy / 2),
            ])
            colors_D.append(D[i, j])

    collection_D = PolyCollection(polys_D, array=np.array(colors_D), cmap='viridis')
    if params['WATERFALL_DENSITY_CLIM'] is not None:
        collection_D.set_clim(*params['WATERFALL_DENSITY_CLIM'])
    ax_d.add_collection(collection_D)
    ax_d.sharex(ax_m)

    x_min_um = params['X_MIN_INTEGRATION']
    x_max_um = params['X_MAX_INTEGRATION']
    x_centers_um = np.arange(M.shape[1]) * um_per_px
    xmin_ind = np.argmin(np.abs(x_centers_um - x_min_um))
    xmax_ind = np.argmin(np.abs(x_centers_um - x_max_um))
    
    # Convert to appropriate units for plot
    if is_bubbles_evolution:
        x_min_plot = x_min_um * 1e-6
        x_max_plot = x_max_um * 1e-6
    else:
        x_min_plot = x_min_um
        x_max_plot = x_max_um

    for axis in [ax_m, ax_d]:
        axis.axvline(x=x_min_plot, color='lime', linestyle='--', linewidth=2.0, alpha=0.95)
        axis.axvline(x=x_max_plot, color='lime', linestyle='--', linewidth=2.0, alpha=0.95)

    if xmax_ind > xmin_ind:
        integrated_M = np.sum(M[:, xmin_ind:xmax_ind], axis=1) / (xmax_ind - xmin_ind)
        std_M = np.std(M[:, xmin_ind:xmax_ind], axis=1)
        ax_int.plot(integrated_M, y_unique, 'b-', label='Mean Magnetization')
        ax_int_std = ax_int.twiny()
        ax_int_std.plot(std_M, y_unique, 'g-', alpha=0.7, label='Std Dev Magnetization')
        ax_int_std.set_xlabel('Std Dev (a.u.)', color='g')
        ax_int_std.tick_params(axis='x', labelcolor='g')
        ax_int.legend(loc='upper left')
        ax_int_std.legend(loc='upper right')

    section_centers_x, sigmoid_centers_y, sigmoid_centers_err = [], [], []

    if plot_flags.get('sectioned_sigmoid', False) and scan in ('ARP_Forward', 'ARP_Backward', 'bubbles', 'bubbles_evolution') and xmax_ind > xmin_ind:
        section_centers_x, sigmoid_centers_y, sigmoid_centers_err, _ = _sectioned_sigmoid_analysis(
            M, y_unique, x_plot_um, xmin_ind, xmax_ind, params['NUM_SECTIONS']
        )
        if len(section_centers_x) > 0:
            # Apply same scaling as the main plot
            plot_section_centers_x = section_centers_x
            if is_bubbles_evolution:
                plot_section_centers_x = np.array(section_centers_x) * 1e-6  # Convert to meters
            
            ax_m.plot(plot_section_centers_x, sigmoid_centers_y, color='violet', linewidth=2.5, label='Sigmoid centers')
            
            # Overlay domain wall fit line if available
            if dw_fit_coeffs is not None and dw_fit_x is not None and dw_fit_t is not None:
                # dw_fit_x is already in meters from plot_evolution_analysis
                dw_fit_x_plot = dw_fit_x
                if not is_bubbles_evolution:
                    dw_fit_x_plot = dw_fit_x * 1e6  # Convert to micrometers
                
                x_line_plot = np.array([dw_fit_x_plot.min(), dw_fit_x_plot.max()])
                # Use the fitted time values directly
                y_line_plot = np.array([dw_fit_coeffs[0] * dw_fit_x.min() + dw_fit_coeffs[1], 
                                        dw_fit_coeffs[0] * dw_fit_x.max() + dw_fit_coeffs[1]])
                
                ax_m.plot(x_line_plot, y_line_plot, 
                         'g--', linewidth=2.5, alpha=0.7, label='Domain wall fit')

            # Dedicated plot: sigmoid center (scan variable value) vs x section position
            fig_sig, ax_sig = plt.subplots(figsize=(8, 5), tight_layout=True)
            ax_sig.errorbar(
                section_centers_x,
                sigmoid_centers_y,
                yerr=sigmoid_centers_err,
                fmt='o',
                color='magenta',
                ecolor='black',
                elinewidth=1,
                capsize=3,
                markersize=5,
                label='Data points',
            )
            
            # Add interpolated curve
            if len(section_centers_x) >= 2:
                try:
                    # Create interpolation function (cubic if enough points, linear otherwise)
                    kind = 'cubic' if len(section_centers_x) >= 4 else 'linear'
                    interp_func = interp1d(section_centers_x, sigmoid_centers_y, kind=kind)
                    
                    # Generate smooth x values for interpolation
                    x_smooth = np.linspace(min(section_centers_x), max(section_centers_x), 200)
                    y_smooth = interp_func(x_smooth)
                    
                    # Plot interpolated curve
                    ax_sig.plot(x_smooth, y_smooth, '-', color='blue', linewidth=2, alpha=0.7, label='Interpolation')
                    
                    # Save interpolated profile to text file in waterfall script directory
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        output_file = os.path.join(script_dir, 'sigmoid_center_interpolation.txt')
                        
                        header = f"# Interpolated sigmoid center profile\n# x (um) vs {y_axis_label} (sigmoid center)\n# Interpolation type: {kind}\n"
                        data_to_save = np.column_stack((x_smooth, y_smooth))
                        np.savetxt(output_file, data_to_save, header=header, fmt='%.6f', delimiter='\t', comments='')
                        print(f"Saved interpolated profile to: {output_file}")
                    except Exception as save_err:
                        print(f"Warning: Could not save interpolated profile: {save_err}")
                except Exception as e:
                    print(f"Warning: Could not interpolate sigmoid centers: {e}")
            
            ax_sig.set_xlabel('x [m]' if is_bubbles_evolution else r'$\mu m$')
            ax_sig.set_ylabel(f'{y_axis_label} (sigmoid center)')
            if average:
                ax_sig.set_title('Average sigmoid center vs x')
            else:
                ax_sig.set_title('Sigmoid center vs x')
            ax_sig.legend(loc='best')
            ax_sig.grid(True, alpha=0.3)

    ax_m.set_ylabel(y_axis_label if not is_bubbles_evolution else 't [ms]')
    # For bubbles_evolution, convert x-axis from micrometers to meters
    if is_bubbles_evolution:
        x_plot_m = x_plot_um * 1e-6
        ax_m.set_xlabel('x [m]')
        ax_m.set_xlim(x_plot_m[0], x_plot_m[-1])
    else:
        ax_m.set_xlabel('x [μm]')
        ax_m.set_xlim(x_plot_um[0], x_plot_um[-1])
    ax_m.set_ylim(np.min(y_unique) - dy, np.max(y_unique) + dy)
    ax_m.set_title(title, fontsize=10)
    
    # Show y-axis tick labels on the central waterfall plot
    ax_m.tick_params(axis='y', labelleft=True)
    
    # Invert y-axis for bubbles_evolution so time increases downward
    if is_bubbles_evolution:
        ax_m.invert_yaxis()
        ax_d.invert_yaxis()
        ax_int.invert_yaxis()

    # Overlay detected defect positions for KZ_defect_analysis.
    dx_p = np.asarray(defect_points_pos[0], dtype=float) if defect_points_pos is not None else np.asarray([], dtype=float)
    dy_p = np.asarray(defect_points_pos[1], dtype=float) if defect_points_pos is not None else np.asarray([], dtype=float)
    dx_n = np.asarray(defect_points_neg[0], dtype=float) if defect_points_neg is not None else np.asarray([], dtype=float)
    dy_n = np.asarray(defect_points_neg[1], dtype=float) if defect_points_neg is not None else np.asarray([], dtype=float)
    dx_c = np.asarray(defect_points_corrected[0], dtype=float) if defect_points_corrected is not None else np.asarray([], dtype=float)
    dy_c = np.asarray(defect_points_corrected[1], dtype=float) if defect_points_corrected is not None else np.asarray([], dtype=float)

    def _plot_identified_domain_binary_profile():
        """Overlay a color-coded binary (±1) domain strip below each shot."""
        if scan not in ('KZ_defect_analysis', 'KZ_det_scan'):
            return
        if len(y_unique) == 0 or M.size == 0:
            return

        x_axis = np.arange(M.shape[1], dtype=float) * float(um_per_px)
        if x_axis.size < 2:
            return
        x_left = float(x_axis[0])
        x_right = float(x_axis[-1])
        tol_y = max(1e-9, 1e-6 * abs(float(dy)))

        y_center_offset = -0.33 * abs(float(dy))
        strip_thickness = 0.32 * abs(float(dy))
        polys_bin, colors_bin = [], []

        for i, yv in enumerate(y_unique):
            walls = []
            walls_pos = np.asarray([], dtype=float)
            walls_neg = np.asarray([], dtype=float)
            if dx_p.size > 0:
                walls_pos = np.asarray(dx_p[np.isclose(dy_p, float(yv), atol=tol_y, rtol=0.0)], dtype=float)
                walls.extend(walls_pos.tolist())
            if dx_n.size > 0:
                walls_neg = np.asarray(dx_n[np.isclose(dy_n, float(yv), atol=tol_y, rtol=0.0)], dtype=float)
                walls.extend(walls_neg.tolist())
            if dx_c.size > 0:
                walls.extend(dx_c[np.isclose(dy_c, float(yv), atol=tol_y, rtol=0.0)].tolist())

            if len(walls) > 0:
                walls = np.unique(np.round(np.asarray(walls, dtype=float), 6))
                walls = walls[(walls > x_left) & (walls < x_right)]
            else:
                walls = np.asarray([], dtype=float)

            edges = np.concatenate(([x_left], walls, [x_right]))
            if edges.size < 2:
                continue

            m_row = np.asarray(M[i], dtype=float)
            if walls.size > 0:
                left_mask = x_axis < float(walls[0])
                if not np.any(left_mask):
                    left_mask = x_axis <= float(walls[0])
            else:
                left_mask = np.ones_like(x_axis, dtype=bool)

            sign_from_defect = _infer_start_sign_from_signed_walls(walls_pos, walls_neg)
            if sign_from_defect is not None:
                sign_val = float(sign_from_defect)
            else:
                used_mask = (x_axis >= float(x_min_um)) & (x_axis <= float(x_max_um))
                avg_m = np.nanmean(m_row[used_mask]) if np.any(used_mask) else np.nanmean(m_row)
                sign_val = 1.0 if np.nan_to_num(avg_m, nan=0.0) > 0.0 else -1.0

            # Build per-pixel binary profile from detected walls.
            n_flips = np.searchsorted(walls, x_axis, side='right')
            row_bin = sign_val * ((-1.0) ** n_flips)
            used_mask = (x_axis >= float(x_min_um)) & (x_axis <= float(x_max_um))
            row_bin = np.where(used_mask, row_bin, 0.0)

            y0 = float(yv) + y_center_offset - 0.5 * strip_thickness
            y1 = float(yv) + y_center_offset + 0.5 * strip_thickness
            for j in range(M.shape[1]):
                polys_bin.append([
                    (x_plot_um[j], y0),
                    (x_plot_um[j + 1], y0),
                    (x_plot_um[j + 1], y1),
                    (x_plot_um[j], y1),
                ])
                colors_bin.append(float(row_bin[j]))

        if len(polys_bin) > 0:
            coll_bin = PolyCollection(polys_bin, array=np.asarray(colors_bin, dtype=float), cmap='RdBu')
            coll_bin.set_clim(mag_vmin, mag_vmax)
            coll_bin.set_alpha(1.0)
            coll_bin.set_zorder(14)
            ax_m.add_collection(coll_bin)

        # Delineate each shot pair (main row + binary strip) with horizontal guides.
        x_l = float(x_plot_um[0])
        x_r = float(x_plot_um[-1])
        for i in range(len(y_unique) - 1):
            y_sep = 0.5 * (float(y_unique[i]) + float(y_unique[i + 1]))
            ax_m.hlines(y_sep, x_l, x_r, colors='yellow', linestyles='--', linewidth=1.1, alpha=0.75, zorder=15)

    if defect_points is not None and defect_points_pos is None and defect_points_neg is None:
        dx, dy_pts = defect_points
        if len(dx) > 0:
            ax_m.scatter(
                dx,
                dy_pts,
                c='yellow',
                s=20,
                marker='o',
                edgecolors='black',
                linewidths=0.4,
                zorder=10,
                label='Threshold defects',
            )

    # Overlay corrected defects found from zero-crossing rule.
    if defect_points_corrected is not None:
        if len(dx_c) > 0:
            ax_m.scatter(
                dx_c,
                dy_c,
                c='lime',
                s=28,
                marker='o',
                edgecolors='black',
                linewidths=0.5,
                zorder=11,
                label='Zero-crossing corrected defects',
            )

    # Overlay rejected threshold peaks (false-positive filter).
    if defect_points_rejected is not None:
        dx_r, dy_r = defect_points_rejected
        if len(dx_r) > 0:
            ax_m.scatter(
                dx_r,
                dy_r,
                c='black',
                s=28,
                marker='x',
                linewidths=1.0,
                zorder=13,
                label='Rejected peaks (step filter)',
            )

    # Guide markers: sign of derivative peaks selected by threshold.
    if defect_points_pos is not None:
        if len(dx_p) > 0:
            ax_m.scatter(
                dx_p,
                dy_p,
                c='blue',
                s=20,
                marker='o',
                edgecolors='black',
                linewidths=0.35,
                zorder=12,
                label='Positive defects (+)',
            )

    if defect_points_neg is not None:
        if len(dx_n) > 0:
            ax_m.scatter(
                dx_n,
                dy_n,
                c='red',
                s=20,
                marker='o',
                edgecolors='black',
                linewidths=0.35,
                zorder=12,
                label='Negative defects (-)',
            )

    _plot_identified_domain_binary_profile()

    handles, labels = ax_m.get_legend_handles_labels()
    if len(handles) > 0:
        ax_m.legend(handles, labels, loc='upper right', fontsize=9)

    # For KZ defect analysis: show per-row defect count (and sequence index) on the right side.
    if scan == 'KZ_defect_analysis' and defect_counts is not None and len(defect_counts) == len(y_unique):
        ax_counts = ax_m.twinx()
        ax_counts.set_ylim(ax_m.get_ylim())
        ax_counts.set_yticks(y_unique)
        if defect_sequences is not None and len(defect_sequences) == len(y_unique):
            if defect_thresholds is not None and len(defect_thresholds) == len(y_unique):
                count_labels = [
                    f"rep {int(s)}: {int(c)} @ {t if isinstance(t, str) else f'{float(t):.3f}'}"
                    for s, c, t in zip(defect_sequences, defect_counts, defect_thresholds)
                ]
                ax_counts.set_ylabel('rep : defects @ threshold')
            else:
                count_labels = [f"rep {int(s)}: {int(c)}" for s, c in zip(defect_sequences, defect_counts)]
                ax_counts.set_ylabel('rep : defects')
        else:
            if defect_thresholds is not None and len(defect_thresholds) == len(y_unique):
                count_labels = [f"{int(c)} @ {t if isinstance(t, str) else f'{float(t):.3f}'}" for c, t in zip(defect_counts, defect_thresholds)]
                ax_counts.set_ylabel('defects @ threshold')
            else:
                count_labels = [str(int(c)) for c in defect_counts]
                ax_counts.set_ylabel('defects')
        ax_counts.set_yticklabels(count_labels, fontsize=7)
        ax_counts.grid(False)

    # Optional summary windows
    if plot_flags.get('avg_density_profile', False):
        fig_density, ax_density = plt.subplots(figsize=(10, 6))
        avg_density = np.mean(D, axis=0)
        std_density = np.std(D, axis=0)
        x_centers_um = np.arange(M.shape[1]) * um_per_px
        ax_density.plot(x_centers_um, avg_density, 'b-', linewidth=2, label='Average Density')
        ax_density.fill_between(x_centers_um, avg_density - std_density, avg_density + std_density, alpha=0.3, color='blue')
        ax_density.axvline(x=x_min_um, color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
        ax_density.axvline(x=x_max_um, color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
        if plot_flags.get('corr_sigmoid_density', False) and scan in ('ARP_Forward', 'ARP_Backward') and len(section_centers_x) > 0:
            ax2 = ax_density.twinx()
            ax2.errorbar(section_centers_x, sigmoid_centers_y, yerr=sigmoid_centers_err, fmt='ro-', capsize=3)
            ax2.set_ylabel(y_axis_label + ' (Sigmoid Center)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        ax_density.set_title(f'Average Density Profile - {scan}')
        ax_density.grid(True, alpha=0.3)

    if plot_flags.get('avg_magnetization_profile', False):
        fig_mag, ax_mag = plt.subplots(figsize=(10, 6))
        avg_m = np.mean(M, axis=0)
        std_m = np.std(M, axis=0)
        x_centers_um = np.arange(M.shape[1]) * um_per_px
        ax_mag.plot(x_centers_um, avg_m, 'r-', linewidth=2, label='Average Magnetization')
        ax_mag.fill_between(x_centers_um, avg_m - std_m, avg_m + std_m, alpha=0.3, color='red')
        ax_mag.axhline(y=0, color='k', linewidth=1, alpha=0.3)
        ax_mag.axvline(x=x_min_um, color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
        ax_mag.axvline(x=x_max_um, color='lime', linestyle='--', linewidth=2.2, alpha=0.95)

        # Auto-zoom y-range to show behavior clearly (instead of fixed [-1, 1]).
        # Use values in the analysis x-window, including uncertainty band.
        if xmax_ind > xmin_ind:
            y_band = np.concatenate([
                (avg_m - std_m)[xmin_ind:xmax_ind],
                (avg_m + std_m)[xmin_ind:xmax_ind],
            ])
        else:
            y_band = np.concatenate([avg_m - std_m, avg_m + std_m])

        y_band = y_band[np.isfinite(y_band)]
        if y_band.size > 0:
            y_lo = float(np.percentile(y_band, 1))
            y_hi = float(np.percentile(y_band, 99))
            if y_hi <= y_lo:
                y_mid = float(np.mean(y_band))
                y_lo, y_hi = y_mid - 0.05, y_mid + 0.05
            pad = max(0.02, 0.15 * (y_hi - y_lo))
            y_lo = max(-1.0, y_lo - pad)
            y_hi = min(1.0, y_hi + pad)
            if y_hi > y_lo:
                ax_mag.set_ylim(y_lo, y_hi)

        ax_mag.set_title(f'Average Magnetization Profile - {scan}')
        ax_mag.grid(True, alpha=0.3)

    if plot_flags.get('corr_mag_density', False) and xmax_ind > xmin_ind:
        avg_density = np.mean(D, axis=0)
        avg_m = np.mean(M, axis=0)
        density_region = avg_density[xmin_ind:xmax_ind]
        mag_region = avg_m[xmin_ind:xmax_ind]
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        ax_corr.scatter(density_region, mag_region, s=45, alpha=0.7, color='purple', edgecolors='black', linewidth=0.5)
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(density_region, mag_region)
            x_fit = np.linspace(np.min(density_region), np.max(density_region), 100)
            ax_corr.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2, label=f'R²={r_value**2:.3f}')
            ax_corr.legend()
        except Exception:
            pass
        ax_corr.set_title('Correlation: Magnetization vs Density (Integration Region)')
        ax_corr.set_xlabel('Density (a.u.)')
        ax_corr.set_ylabel('Magnetization (a.u.)')
        ax_corr.grid(True, alpha=0.3)


def plot_evolution_analysis(y_plot, y_axis_label, z_local_fluctuations, M, um_per_px, params=None, mode_cfg=None):
    from scipy.optimize import curve_fit

    if params is None:
        params = {}
    if mode_cfg is None:
        mode_cfg = {}
    x_min_um = params.get('X_MIN_INTEGRATION', 900)
    x_max_um = params.get('X_MAX_INTEGRATION', 1200)
    
    plot_manager = get_plot_manager()
    
    # Create named figures instead of auto-numbered ones
    fig_fluct = plt.figure('bubbles_evolution_fluctuations')
    fig_fluct.clear()
    ax_fluct = fig_fluct.add_subplot(111)
    plot_manager.register_figure('bubbles_evolution_fluctuations', fig_fluct)
    
    z_fluc_smooth = gaussian_filter1d(z_local_fluctuations, sigma=2, axis=0)
    start_idx, end_idx = int(round(x_min_um / um_per_px)), int(round(x_max_um / um_per_px))
    x_axis_pixels = np.arange(start_idx, end_idx)
    x_axis_mesh = x_axis_pixels * um_per_px * 1e-6
    im_fluct = ax_fluct.pcolormesh(x_axis_mesh, y_plot, z_fluc_smooth[:, start_idx:end_idx], cmap='inferno')
    ax_fluct.set_title('Local z fluctuations')
    ax_fluct.set_ylabel(y_axis_label)
    ax_fluct.set_xlabel('x [μm]')
    ax_fluct.invert_yaxis()
    plt.colorbar(im_fluct, ax=ax_fluct)
    
    # Apply previously saved settings if they exist and enable autosave
    plot_manager.apply_settings('bubbles_evolution_fluctuations', ax_fluct)
    plot_manager.enable_autosave('bubbles_evolution_fluctuations', ax_fluct)

    fig_evol = plt.figure('bubbles_evolution_magnetization')
    fig_evol.clear()
    ax_evol = fig_evol.add_subplot(111)
    plot_manager.register_figure('bubbles_evolution_magnetization', fig_evol)
    
    m_copy = np.copy(M)[:, start_idx:end_idx] ** 2
    m_copy = gaussian_filter1d(m_copy, sigma=2, axis=1)
    x_evol_mesh = x_axis_mesh
    im_evol = ax_evol.pcolormesh(x_evol_mesh, y_plot, m_copy, vmin=0, vmax=0.001, cmap='viridis')
    ax_evol.set_ylabel(y_axis_label)
    ax_evol.set_xlabel('x [μm]')
    ax_evol.set_title('Magnetization evolution in the used region')
    ax_evol.invert_yaxis()
    
    # Apply previously saved settings if they exist and enable autosave
    plot_manager.apply_settings('bubbles_evolution_magnetization', ax_evol)
    plot_manager.enable_autosave('bubbles_evolution_magnetization', ax_evol)
    
    plt.colorbar(im_evol, ax=ax_evol)
    
    # Compute sigmoid centers for each vertical line (time slice) in the used region
    def sigmoid(x, x0, k, a, b):
        return a / (1.0 + np.exp(-(x - x0) / k)) + b
    
    time_centers_y = []
    sigmoid_centers_x = []
    sigmoid_centers_err = []
    
    x_um_for_fit = np.arange(M.shape[0])  # y_plot indices for vertical line fits
    
    for col_idx in range(start_idx, end_idx):
        vertical_profile = M[:, col_idx]
        p0 = [np.median(x_um_for_fit), 0.1, np.max(vertical_profile) - np.min(vertical_profile), np.min(vertical_profile)]
        try:
            popt, pcov = curve_fit(sigmoid, x_um_for_fit, vertical_profile, p0=p0, maxfev=10000)
            time_centers_y.append(y_plot[int(popt[0])] if int(popt[0]) < len(y_plot) else y_plot[-1])
            sigmoid_centers_x.append((col_idx - start_idx) * um_per_px * 1e-6 + x_axis_mesh[0])
            sigmoid_centers_err.append(float(np.sqrt(max(0.0, pcov[0, 0]))))
        except Exception:
            continue
    
    if len(sigmoid_centers_x) > 0:
        ax_evol.plot(sigmoid_centers_x, time_centers_y, color='cyan', linewidth=2.5, label='Sigmoid centers')
        ax_evol.legend()
    
    plt.colorbar(im_evol, ax=ax_evol)
    
    # Domain wall velocity analysis
    dw_x_min = params.get('DOMAIN_WALL_X_MIN', None)
    dw_x_max = params.get('DOMAIN_WALL_X_MAX', None)
    
    if dw_x_min is not None and dw_x_max is not None and len(sigmoid_centers_x) > 0:
        # Convert um limits to meters for comparison
        dw_x_min_m = dw_x_min * 1e-6
        dw_x_max_m = dw_x_max * 1e-6
        
        # Filter sigmoid centers within the specified range
        mask = (np.array(sigmoid_centers_x) >= dw_x_min_m) & (np.array(sigmoid_centers_x) <= dw_x_max_m)
        filtered_x = np.array(sigmoid_centers_x)[mask]
        filtered_t = np.array(time_centers_y)[mask]
        
        # Print number of points per time value
        unique_times, counts = np.unique(filtered_t, return_counts=True)
        print(f"Domain wall analysis: {len(filtered_x)} total points")
        for t_val, count in zip(unique_times, counts):
            print(f"  Time {t_val}: {count} points")
        
        if len(filtered_x) >= 2:
            # Fit a line to extract domain wall velocity
            coeffs = np.polyfit(filtered_x, filtered_t, 1)
            y_fit = np.polyval(coeffs, filtered_x)
            residuals = filtered_t - y_fit
            sigma_residuals = np.std(residuals)
            
            # Reject points more than 1 sigma outside the linear fit
            mask_keep = np.abs(residuals) <= 1 * sigma_residuals
            filtered_x_clean = filtered_x[mask_keep]
            filtered_t_clean = filtered_t[mask_keep]
            
            # Redo fit with cleaned data
            if len(filtered_x_clean) >= 2:
                coeffs = np.polyfit(filtered_x_clean, filtered_t_clean, 1)
            
            print(f"Domain wall fit coefficients: {coeffs}")
            print(f"  polyfit result: slope (dt/dx)={coeffs[0]}, intercept={coeffs[1]}")
            print(f"  Fit equation (x in meters): bubbles_time = {coeffs[0]:.6e} * x + {coeffs[1]:.6f}")
            print(f"  Fit equation (x in μm): bubbles_time = {coeffs[0]*1e-6:.6e} * x_um + {coeffs[1]:.6f}")
            
            velocity_slope = coeffs[0]  # dt/dx in ms/meter
            velocity_um_ms = 1e6 / velocity_slope if velocity_slope != 0 else float('inf')  # um/ms
            
            # Create domain wall velocity plot with named figure
            fig_dw = plt.figure('bubbles_evolution_domain_wall')
            fig_dw.clear()
            ax_dw = fig_dw.add_subplot(111)
            ax_dw.set_figsize = (8, 5)
            plot_manager.register_figure('bubbles_evolution_domain_wall', fig_dw)
            
            ax_dw.scatter(filtered_x * 1e6, filtered_t, color='red', s=50, alpha=0.5, label='All data points')
            ax_dw.scatter(filtered_x_clean * 1e6, filtered_t_clean, color='darkred', s=50, label='Kept points (within 1σ)')
            
            # Plot fit line
            x_line = np.array([filtered_x.min(), filtered_x.max()])
            y_line = np.polyval(coeffs, x_line)
            ax_dw.plot(x_line * 1e6, y_line, 'b-', linewidth=2, label=f'Linear fit: velocity = {velocity_um_ms:.3f} μm/ms')
            
            ax_dw.set_xlabel('x [μm]')
            ax_dw.set_ylabel(y_axis_label)
            ax_dw.set_title(f'Domain Wall Velocity Analysis (x: {dw_x_min}-{dw_x_max} μm)')
            ax_dw.grid(True, alpha=0.3)
            ax_dw.legend()
            
            # Apply previously saved settings and enable autosave
            plot_manager.apply_settings('bubbles_evolution_domain_wall', ax_dw)
            plot_manager.enable_autosave('bubbles_evolution_domain_wall', ax_dw)
            
            # Calculate and print domain wall speed in um/ms
            n_rejected = len(filtered_x) - len(filtered_x_clean)
            print(f"Domain wall: rejected {n_rejected}/{len(filtered_x)} points outside 2σ (σ={sigma_residuals:.6g})")
            if velocity_slope != 0:
                print(f"Domain wall velocity: {velocity_um_ms:.3f} μm/ms")
            else:
                print("Domain wall velocity: infinite (vertical line)")
            
            # Return fit line data for overlay on main plot
            return coeffs, filtered_x_clean, filtered_t_clean
    
    return None, None, None


def plot_all_shots_waterfall(M_full, grouped_indices, grouped_y, y_axis_col, seqs, um_per_px, params, plot_manager=None):
    """
    Plot all individual shots as a waterfall with domain wall detection and bubbles_time separators.
    
    Parameters
    ----------
    M_full : array
        Full magnetization data (n_shots, n_pixels)
    grouped_indices : list
        List of indices for each group (by bubbles_time or scan variable)
    grouped_y : list
        Y-axis values for each group
    y_axis_col : str
        Name of y-axis column (e.g., 'bubbles_time')
    seqs : list
        Sequence indices
    um_per_px : float
        Micrometers per pixel
    params : dict
        Parameter dictionary with X_MIN_INTEGRATION, X_MAX_INTEGRATION, etc.
    plot_manager : PlotSettingsManager, optional
        Settings manager for persistence
    """
    if plot_manager is None:
        plot_manager = get_plot_manager()
    
    fig_raw = plt.figure('all_shots_waterfall')
    fig_raw.clear()
    ax_raw = fig_raw.add_subplot(111)
    plot_manager.register_figure('all_shots_waterfall', fig_raw)
    
    try:
        fig_raw.set_tight_layout(False)
    except Exception:
        pass
    try:
        fig_raw.set_constrained_layout(False)
    except Exception:
        pass
    fig_raw.subplots_adjust(left=0.09, right=0.84, bottom=0.10, top=0.92)
    cax_raw = fig_raw.add_axes([0.90, 0.10, 0.018, 0.82])
    
    raw_vmin = -1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][0]
    raw_vmax = 1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][1]
    
    x_centers_um_bub = np.arange(M_full.shape[1]) * um_per_px
    x_min_um_bub = float(params['X_MIN_INTEGRATION'])
    x_max_um_bub = float(params['X_MAX_INTEGRATION'])
    
    # Detect domain walls for each shot
    gaussian_sigma_um = 3.0
    deriv_threshold = 0.03
    dw_left_x = []
    dw_right_x = []
    
    for shot_idx in range(len(M_full)):
        m_profile = np.asarray(M_full[shot_idx], dtype=float)
        
        if gaussian_sigma_um > 0:
            gaussian_sigma_px = gaussian_sigma_um / um_per_px
            m_smooth = gaussian_filter1d(m_profile, sigma=gaussian_sigma_px)
        else:
            m_smooth = m_profile
        
        m_deriv = np.gradient(m_smooth)
        
        idx_min = int(np.argmin(np.abs(x_centers_um_bub - x_min_um_bub)))
        idx_max = int(np.argmin(np.abs(x_centers_um_bub - x_max_um_bub)))
        if idx_max < idx_min:
            idx_min, idx_max = idx_max, idx_min
        
        left_wall_x = None
        if idx_min < len(m_deriv):
            for i in range(idx_min, min(idx_max, len(m_deriv))):
                if m_deriv[i] <= -deriv_threshold:
                    left_wall_x = float(x_centers_um_bub[i])
                    break
        
        right_wall_x = None
        if idx_max < len(m_deriv):
            for i in range(idx_max, max(idx_min - 1, -1), -1):
                if m_deriv[i] >= deriv_threshold:
                    right_wall_x = float(x_centers_um_bub[i])
                    break
        
        dw_left_x.append(left_wall_x)
        dw_right_x.append(right_wall_x)
    
    im_raw = ax_raw.imshow(
        M_full,
        aspect='auto',
        cmap='RdBu',
        interpolation='nearest',
        resample=False,
        vmin=raw_vmin,
        vmax=raw_vmax,
        extent=[float(x_centers_um_bub[0]), float(x_centers_um_bub[-1]), float(len(M_full)) + 0.5, 0.5],
    )
    ax_raw.set_xlabel(r'$\mu m$')
    ax_raw.set_ylabel(f'shot index (ordered by {y_axis_col})')
    ax_raw.set_title(f'Raw magnetization waterfall ordered by {y_axis_col} (seqs: {seqs})')
    ax_raw.set_xlim(float(x_centers_um_bub[0]), float(x_centers_um_bub[-1]))
    ax_raw.set_ylim(float(len(M_full)) + 0.5, 0.5)
    ax_raw.set_autoscale_on(False)
    ax_raw.axvline(x=float(x_min_um_bub), color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
    ax_raw.axvline(x=float(x_max_um_bub), color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
    
    # Draw detected domain walls
    for shot_idx in range(len(M_full)):
        y_shot = float(shot_idx + 1)
        if dw_left_x[shot_idx] is not None:
            ax_raw.plot(dw_left_x[shot_idx], y_shot, 'o', color='green', markersize=4, alpha=0.8, zorder=15)
        if dw_right_x[shot_idx] is not None:
            ax_raw.plot(dw_right_x[shot_idx], y_shot, 's', color='violet', markersize=4, alpha=0.8, zorder=15)
    
    # Add separators between different scan variable values
    if len(grouped_indices) > 0:
        x_left = float(x_centers_um_bub[0])
        x_right = float(x_centers_um_bub[-1])
        x_text = x_left + 0.012 * (x_right - x_left)
        
        start = 0
        centers = []
        for time_val, g in zip(grouped_y, grouped_indices):
            end = start + len(g)
            y_start = float(start) + 0.5
            y_end = float(end) + 0.5
            y_center = 0.5 * (y_start + y_end)
            centers.append(y_center)
            
            if start > 0:
                sep_half_height = 0.42
                ax_raw.axhspan(
                    y_start - sep_half_height,
                    y_start + sep_half_height,
                    facecolor='white',
                    edgecolor='none',
                    zorder=19,
                )
            
            try:
                time_txt = f"{float(time_val):.6g}"
            except Exception:
                time_txt = str(time_val)
            
            ax_raw.text(
                x_text,
                y_center,
                f'{y_axis_col}={time_txt}',
                color='white',
                fontsize=8,
                va='center',
                ha='left',
                bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.5),
                zorder=20,
            )
            start = end
        
        ax_time_ticks = ax_raw.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        ax_time_ticks.set_yticks(centers)
        try:
            ax_time_ticks.set_yticklabels([f'{float(v):.6g}' for v in grouped_y], fontsize=8)
        except Exception:
            ax_time_ticks.set_yticklabels([str(v) for v in grouped_y], fontsize=8)
        ax_time_ticks.set_ylabel(y_axis_col)
        ax_time_ticks.grid(False)
    
    cbar_raw = fig_raw.colorbar(im_raw, cax=cax_raw)
    cbar_raw.set_label('magnetization m')
    
    # Apply previously saved settings and enable autosave
    plot_manager.apply_settings('all_shots_waterfall', ax_raw)
    plot_manager.enable_autosave('all_shots_waterfall', ax_raw)
    
    # Plot domain wall positions as a function of scan variable
    fig_dw = plt.figure('all_shots_domain_walls')
    fig_dw.clear()
    ax_dw = fig_dw.add_subplot(111)
    plot_manager.register_figure('all_shots_domain_walls', fig_dw)
    
    dw_times = []
    dw_left_mean = []
    dw_left_sem = []
    dw_right_mean = []
    dw_right_sem = []
    dw_left_all = []
    dw_left_times_all = []
    dw_right_all = []
    dw_right_times_all = []
    
    for time_val, idx_group in zip(grouped_y, grouped_indices):
        dw_left_group = [dw_left_x[i] for i in idx_group if dw_left_x[i] is not None]
        dw_right_group = [dw_right_x[i] for i in idx_group if dw_right_x[i] is not None]
        
        if len(dw_left_group) > 0:
            dw_times.append(float(time_val))
            dw_left_mean.append(np.mean(dw_left_group))
            dw_left_sem.append(np.std(dw_left_group) / np.sqrt(len(dw_left_group)))
            dw_left_all.extend(dw_left_group)
            dw_left_times_all.extend([float(time_val)] * len(dw_left_group))
        
        if len(dw_right_group) > 0:
            if float(time_val) not in dw_times:
                dw_times.append(float(time_val))
            dw_right_mean.append(np.mean(dw_right_group))
            dw_right_sem.append(np.std(dw_right_group) / np.sqrt(len(dw_right_group)))
            dw_right_all.extend(dw_right_group)
            dw_right_times_all.extend([float(time_val)] * len(dw_right_group))
    
    if len(dw_left_all) > 0:
        ax_dw.scatter(dw_left_times_all, dw_left_all, alpha=0.5, s=30, label='Left wall (individual)', color='green')
    if len(dw_right_all) > 0:
        ax_dw.scatter(dw_right_times_all, dw_right_all, alpha=0.5, s=30, label='Right wall (individual)', color='violet')
    
    if len(dw_left_mean) > 0:
        ax_dw.errorbar(dw_times[:len(dw_left_mean)], dw_left_mean, yerr=dw_left_sem, fmt='o-', label='Left wall (mean±SEM)', color='green', linewidth=2, markersize=6)
    if len(dw_right_mean) > 0:
        ax_dw.errorbar(dw_times[:len(dw_right_mean)], dw_right_mean, yerr=dw_right_sem, fmt='s-', label='Right wall (mean±SEM)', color='violet', linewidth=2, markersize=6)
    
    ax_dw.set_xlabel(y_axis_col)
    ax_dw.set_ylabel('Domain wall position (μm)')
    ax_dw.set_title('Domain wall evolution')
    ax_dw.legend()
    ax_dw.grid(True, alpha=0.3)
    
    # Apply previously saved settings and enable autosave
    plot_manager.apply_settings('all_shots_domain_walls', ax_dw)
    plot_manager.enable_autosave('all_shots_domain_walls', ax_dw)


def waterfall_plot(df, seqs, scan, data_origin='show_ODs', constraints=None, average=False, title_labels=None, globals_in_title=None, params=None, plot_flags=None, mode_cfg=None):
    if params is None:
        raise ValueError('params dict is required')
    if plot_flags is None:
        plot_flags = {}
    if globals_in_title is None:
        globals_in_title = []

    um_per_px = get_um_per_px(params)

    shot_name_filter = mode_cfg.get('shot_name_filter', None) if isinstance(mode_cfg, dict) else None
    df_subset = filter_dataframe(df, seqs, constraints, shot_name_filter=shot_name_filter)
    if scan == 'ARPF_feedback' or scan == 'DMD_density_feedback':
        skip_feedback_filter = mode_cfg.get('skip_feedback_filter', False) if isinstance(mode_cfg, dict) else False
        if not skip_feedback_filter:
            df_subset = filter_valid_feedback_shots(df_subset)
        else:
            print("DEBUG: Skipping feedback shot filtering per mode config")

    y_axis_col, title_base = get_scan_config(scan)
    
    # Allow override of scan variable from mode config
    if mode_cfg is not None and isinstance(mode_cfg, dict):
        scan_var_override = mode_cfg.get('scan_variable_override', None)
        if scan_var_override is not None:
            y_axis_col = scan_var_override
            print(f"DEBUG: Scan variable overridden to '{y_axis_col}'")
    
    y_axis_label = 'shot index' if y_axis_col == '_shot_counter' else y_axis_col
    print(f"DEBUG: Using y_axis_col = '{y_axis_col}' for scan '{scan}'")

    # Get the column to sort by (handles both plain and tuple formats)
    sort_col = _get_column(df_subset, y_axis_col)
    if sort_col is None:
        raise KeyError(f"'{y_axis_col}' column not found (tried both plain and tuple formats)")
    
    sorted_df = df_subset.sort_values(by=sort_col.name)
    modality_cfg = _resolve_modality(mode_cfg if isinstance(mode_cfg, dict) else {}, params)
    um_per_px = get_um_per_px(params, camera_name_override=modality_cfg.get('camera_name'))
    shot_filter_cfg = mode_cfg.get('shot_filter', {}) if isinstance(mode_cfg, dict) else {}
    filter_data_origin = modality_cfg.get('data_origin', data_origin)
    good_mask = get_valid_rows_mask(
        sorted_df,
        filter_cfg=shot_filter_cfg,
        modality_cfg=modality_cfg,
        data_origin=filter_data_origin,
    )
    sorted_df = sorted_df[good_mask]

    if sorted_df.empty:
        print('No valid data remaining after filtering.')
        return

    y_raw = sort_col[good_mask].values
    grouped_y, grouped_indices = _build_group_indices(y_raw)

    if scan == 'KZ_det_scan':
        print('KZ_det_scan: shots used after filtering per det:')
        for det_val, idx in zip(grouped_y, grouped_indices):
            try:
                det_txt = f"{float(det_val):.6g}"
            except Exception:
                det_txt = str(det_val)
            print(f"  det={det_txt}: {len(idx)} shots")

    if scan == 'bubbles_evolution':
        print('bubbles_evolution: shots used after filtering per time value:')
        for time_val, idx in zip(grouped_y, grouped_indices):
            try:
                time_txt = f"{float(time_val):.6g}"
            except Exception:
                time_txt = str(time_val)
            print(f"  {y_axis_col}={time_txt}: {len(idx)} shots")

    profile_inputs = build_magnetization_inputs(sorted_df, mode_cfg if isinstance(mode_cfg, dict) else {}, params, data_origin)
    if profile_inputs is None:
        print('No valid magnetization profile columns found for selected modality.')
        return
    M_full = profile_inputs['M_full']
    D_full = profile_inputs['D_full']
    z_fluc_full = profile_inputs['z_fluc_full']
    img_dims = profile_inputs['img_dims']

    rep_indices = None
    if average:
        if len(grouped_indices) == 0:
            print('No groups found for averaging.')
            return

        M_final = np.zeros((len(grouped_indices), img_dims))
        D_final = np.zeros((len(grouped_indices), img_dims))
        z_fluc_final = np.zeros((len(grouped_indices), z_fluc_full.shape[1]))
        for i, idx in enumerate(grouped_indices):
            M_final[i] = np.mean(M_full[idx], axis=0)
            D_final[i] = np.mean(D_full[idx], axis=0)
            z_fluc_final[i] = np.mean(z_fluc_full[idx], axis=0)
        y_final = grouped_y
    else:
        # Keep one representative shot per scan point (first in sorted order)
        rep_indices = np.array([idx[0] for idx in grouped_indices if len(idx) > 0], dtype=int)
        if rep_indices.size == 0:
            print('No representative rows available after grouping.')
            return
        M_final = M_full[rep_indices]
        D_final = D_full[rep_indices]
        z_fluc_final = z_fluc_full[rep_indices]
        y_final = grouped_y

    dy = np.min(np.diff(y_final)) if len(y_final) > 1 else 0.1

    defect_points = None
    defect_points_pos = None
    defect_points_neg = None
    defect_points_corrected = None
    defect_points_rejected = None
    defect_counts_for_plot = None
    defect_sequences_for_plot = None
    defect_thresholds_for_plot = None
    avg_defects = None
    stat_err = None
    if scan == 'KZ_defect_analysis':
        defect_cfg = dict(mode_cfg.get('defect_analysis_global', {}) if isinstance(mode_cfg, dict) else {})
        if isinstance(mode_cfg, dict):
            defect_cfg.update(mode_cfg.get('defect_analysis', {}))
            defect_cfg.update(mode_cfg.get('defect_analysis_override', {}))
        defect_cfg = defect_lib.normalize_defect_config(defect_cfg)
        defect_position_hist_half_window_um = max(0.0, float(defect_cfg.get('defect_position_hist_half_window_um', 4.0)))

        gauss_sigma = float(defect_cfg['gaussian_sigma'])
        deriv_threshold_pos = float(defect_cfg['derivative_threshold_pos'])
        deriv_threshold_neg = float(defect_cfg['derivative_threshold_neg'])
        peak_step_filter_enabled = bool(defect_cfg['peak_step_filter_enabled'])
        peak_step_filter_window_px = int(defect_cfg['peak_step_filter_window_px'])
        peak_step_filter_min_abs_delta_m = float(defect_cfg['peak_step_filter_min_abs_delta_m'])
        min_same_sign_peak_distance_um = float(defect_cfg['min_same_sign_peak_distance_um'])
        zero_crossing_min_distance_um = float(defect_cfg['zero_crossing_min_distance_um'])

        x_centers_um = np.arange(M_final.shape[1]) * um_per_px
        x_min_um = float(params['X_MIN_INTEGRATION'])
        x_max_um = float(params['X_MAX_INTEGRATION'])

        defect_x, defect_y = [], []
        defect_x_pos, defect_y_pos = [], []
        defect_x_neg, defect_y_neg = [], []
        defect_x_corr, defect_y_corr = [], []
        defect_x_rej, defect_y_rej = [], []
        per_shot_counts = []
        corrected_total = 0
        rejected_total = 0

        for i, yv in enumerate(y_final):
            out_i = defect_lib.find_defects_in_profile(
                M_final[i],
                x_centers_um,
                defect_cfg,
                x_min_um,
                x_max_um,
            )

            thr_all = np.asarray(out_i['threshold_x_all'], dtype=float)
            thr_pos = np.asarray(out_i['threshold_x_pos'], dtype=float)
            thr_neg = np.asarray(out_i['threshold_x_neg'], dtype=float)
            corr_x = np.asarray(out_i['corrected_x'], dtype=float)
            rej_x = np.asarray(out_i['rejected_x'], dtype=float)

            per_shot_counts.append(int(out_i['count']))

            for xx in thr_all:
                defect_x.append(float(xx))
                defect_y.append(yv)
            for xx in thr_pos:
                defect_x_pos.append(float(xx))
                defect_y_pos.append(yv)
            for xx in thr_neg:
                defect_x_neg.append(float(xx))
                defect_y_neg.append(yv)
            for xx in corr_x:
                defect_x_corr.append(float(xx))
                defect_y_corr.append(yv)
            for xx in rej_x:
                defect_x_rej.append(float(xx))
                defect_y_rej.append(yv)

            corrected_total += len(corr_x)
            rejected_total += len(rej_x)

        try:
            if rep_indices is not None:
                defect_sequences_for_plot = sorted_df['rep'].values[rep_indices]
            else:
                defect_sequences_for_plot = sorted_df['rep'].values
        except Exception:
            defect_sequences_for_plot = None

        defect_points = (np.asarray(defect_x), np.asarray(defect_y))
        defect_points_pos = (np.asarray(defect_x_pos), np.asarray(defect_y_pos))
        defect_points_neg = (np.asarray(defect_x_neg), np.asarray(defect_y_neg))
        defect_points_corrected = (np.asarray(defect_x_corr), np.asarray(defect_y_corr))
        defect_points_rejected = (np.asarray(defect_x_rej), np.asarray(defect_y_rej))
        defect_counts_for_plot = np.asarray(per_shot_counts, dtype=int)
        defect_thresholds_for_plot = None

        n_shots = len(per_shot_counts)
        total_defects = int(np.sum(per_shot_counts))
        avg_defects = (total_defects / n_shots) if n_shots > 0 else 0.0
        if n_shots > 1:
            stat_err = float(np.std(per_shot_counts, ddof=1) / np.sqrt(n_shots))
        else:
            stat_err = 0.0
        print(
            f"KZ_defect_analysis: gaussian_sigma={gauss_sigma}, "
            f"base_thresholds=(+{deriv_threshold_pos:.4f}, {deriv_threshold_neg:.4f}), "
            f"missed_defect_method=opposite_peak, "
            f"peak_step_filter={'on' if peak_step_filter_enabled else 'off'}"
            f"(N={peak_step_filter_window_px}, min|Δm|={peak_step_filter_min_abs_delta_m:.3f}), "
            f"x_window_um=[{x_min_um:.1f}, {x_max_um:.1f}], "
            f"shots={n_shots}, total_defects={total_defects}, corrected_missed={corrected_total}, rejected_peaks={rejected_total}, "
            f"avg_defects_per_shot={avg_defects:.3f} ± {stat_err:.3f} (SEM)"
        )

        # Histogram 1: number of defects per shot.
        if len(per_shot_counts) > 0:
            fig_hn, ax_hn = plt.subplots(figsize=(7.0, 4.2), tight_layout=True)
            c_arr = np.asarray(per_shot_counts, dtype=int)
            n_bins = max(1, int(np.sqrt(len(c_arr))))
            bins = n_bins
            ax_hn.hist(c_arr, bins=bins, color='tab:blue', alpha=0.75, edgecolor='black')
            ax_hn.set_xlabel('Defects per shot')
            ax_hn.set_ylabel('Number of shots')
            ax_hn.set_title(f'KZ_defect_analysis: histogram of defects/shot (seqs: {seqs}, bins=sqrt(N_shots))')
            ax_hn.grid(True, alpha=0.25)

        # Histogram 2: defect position accumulation over all shots.
        # 1 um center spacing; each center counts defects within ±2 um.
        all_defect_x = np.concatenate([
            np.asarray(defect_x, dtype=float),
            np.asarray(defect_x_corr, dtype=float),
        ])
        all_defect_x = all_defect_x[np.isfinite(all_defect_x)]
        if all_defect_x.size > 0:
            x0 = int(np.floor(float(x_min_um)))
            x1 = int(np.ceil(float(x_max_um)))
            x_centers = np.arange(x0, x1 + 1, 1, dtype=float)
            counts_pm2 = np.array([
                int(np.count_nonzero(np.abs(all_defect_x - xc) <= defect_position_hist_half_window_um)) for xc in x_centers
            ], dtype=float)

            fig_hx, ax_hx = plt.subplots(figsize=(9.0, 4.5), tight_layout=True)
            ax_hx.bar(x_centers, counts_pm2, width=0.9, color='tab:orange', alpha=0.75, edgecolor='black')
            ax_hx.set_xlabel(r'x ($\mu m$)')
            ax_hx.set_ylabel(rf'Counts in $\pm {defect_position_hist_half_window_um:g}\,\mu m$ window')
            ax_hx.set_title(f'KZ_defect_analysis: defect-position accumulation (all shots, seqs: {seqs})')
            ax_hx.grid(True, alpha=0.25)

        # Histogram 3: defect size (distance between paired pos and neg peaks).
        # For each shot, compute distances between all pos/neg peak pairs within that shot.
        all_defect_sizes = []
        for i, yv in enumerate(y_final):
            if i < len(defect_x_pos) and i < len(defect_x_neg):
                # Collect all pos and neg x positions for this shot (y value)
                pos_x_shot = [defect_x_pos[j] for j in range(len(defect_x_pos)) if defect_y_pos[j] == yv] if len(defect_y_pos) > 0 else []
                neg_x_shot = [defect_x_neg[j] for j in range(len(defect_x_neg)) if defect_y_neg[j] == yv] if len(defect_y_neg) > 0 else []
                
                # For each positive peak, find nearest negative peak and record distance
                if len(pos_x_shot) > 0 and len(neg_x_shot) > 0:
                    for px in pos_x_shot:
                        distances = np.abs(np.asarray(neg_x_shot) - float(px))
                        all_defect_sizes.append(float(np.min(distances)))
        
        all_defect_sizes = np.asarray(all_defect_sizes, dtype=float)
        if all_defect_sizes.size > 0:
            fig_hs, ax_hs = plt.subplots(figsize=(7.0, 4.2), tight_layout=True)
            # Read bin size from config with fallback to 'sqrt'
            bin_size_config = defect_cfg.get('defect_size_hist_bin_size', 'sqrt')
            if isinstance(bin_size_config, str):
                n_bins_size = bin_size_config  # 'sqrt', 'fd', 'sturges', etc.
            else:
                # bin_size_config is a number (int or float), treat as bin width
                n_bins_size = max(3, int(np.ceil((all_defect_sizes.max() - all_defect_sizes.min()) / float(bin_size_config)))) if float(bin_size_config) > 0 else 'sqrt'
            ax_hs.hist(all_defect_sizes, bins=n_bins_size, color='tab:green', alpha=0.75, edgecolor='black')
            ax_hs.set_xlabel(r'Defect size ($\mu m$, distance pos-to-neg)')
            ax_hs.set_ylabel('Count')
            mean_size = float(np.mean(all_defect_sizes))
            std_size = float(np.std(all_defect_sizes))
            ax_hs.set_title(f'KZ_defect_analysis: defect-size distribution (seqs: {seqs})\nmean={mean_size:.2f} ± {std_size:.2f} µm')
            ax_hs.grid(True, alpha=0.25)
            print(f"Defect sizes: mean={mean_size:.3f} µm, std={std_size:.3f} µm, median={float(np.median(all_defect_sizes)):.3f} µm, N={len(all_defect_sizes)}")

        # Threshold-scan diagnostics were removed for KZ_defect_analysis.

    if scan == 'KZ_det_scan':
        # Plot defects/shot vs ARPKZ_final_set_field using the same logic and parameters as KZ_defect_analysis.
        gauss_sigma_det = 0.3
        deriv_threshold_det = 0.055
        deriv_thr_pos_det = 0.055
        deriv_thr_neg_det = -0.035
        threshold_scan_det = False
        threshold_scan_min_det = 0.02
        threshold_scan_max_det = 0.20
        threshold_scan_pos_min_det = 0.02
        threshold_scan_pos_max_det = 0.20
        threshold_scan_neg_min_det = -0.20
        threshold_scan_neg_max_det = -0.02
        threshold_scan_points_det = 50
        plateau_delta_defects_det = 0.05
        plateau_min_points_det = 3
        zero_cross_min_dist_um_det = 5.0
        peak_step_filter_enabled_det = False
        peak_step_filter_window_px_det = 4
        peak_step_filter_min_abs_delta_m_det = 0.3
        min_same_sign_peak_distance_um_det = 4.0
        missed_defect_correction_method_det = 'opposite_peak'
        try:
            from waterfall_config import DEFECT_ANALYSIS_PARAMS as _wf_defect_cfg
            kz_ref_cfg = dict(_wf_defect_cfg) if isinstance(_wf_defect_cfg, dict) else {}
            det_cfg_override = {}
            if isinstance(mode_cfg, dict):
                det_cfg_override.update(mode_cfg.get('defect_analysis_global', {}))
                det_cfg_override.update(mode_cfg.get('defect_analysis', {}))
                det_cfg_override.update(mode_cfg.get('defect_analysis_override', {}))
            det_cfg = dict(kz_ref_cfg)
            det_cfg.update(det_cfg_override)

            gauss_sigma_det = float(det_cfg.get('gaussian_sigma', gauss_sigma_det))
            deriv_threshold_det = float(det_cfg.get('derivative_threshold', deriv_threshold_det))
            deriv_thr_pos_det = float(det_cfg.get('derivative_threshold_pos', deriv_thr_pos_det))
            deriv_thr_neg_det = float(det_cfg.get('derivative_threshold_neg', -abs(deriv_thr_pos_det)))
            threshold_scan_det = bool(det_cfg.get('threshold_scan', threshold_scan_det))
            zero_cross_min_dist_um_det = float(det_cfg.get('zero_crossing_min_distance_um', zero_cross_min_dist_um_det))
            peak_step_filter_enabled_det = bool(det_cfg.get('peak_step_filter_enabled', peak_step_filter_enabled_det))
            peak_step_filter_window_px_det = int(det_cfg.get('peak_step_filter_window_px', peak_step_filter_window_px_det))
            peak_step_filter_min_abs_delta_m_det = float(det_cfg.get('peak_step_filter_min_abs_delta_m', peak_step_filter_min_abs_delta_m_det))
            min_same_sign_peak_distance_um_det = float(det_cfg.get('min_same_sign_peak_distance_um', min_same_sign_peak_distance_um_det))
            missed_defect_correction_method_det = str(
                det_cfg.get('missed_defect_correction_method', missed_defect_correction_method_det)
            ).strip().lower()
        except Exception:
            pass
        peak_step_filter_window_px_det = max(1, int(peak_step_filter_window_px_det))
        peak_step_filter_min_abs_delta_m_det = max(0.0, float(peak_step_filter_min_abs_delta_m_det))
        min_same_sign_peak_distance_um_det = max(0.0, float(min_same_sign_peak_distance_um_det))
        if missed_defect_correction_method_det not in ('zero_crossing', 'opposite_peak', 'both'):
            missed_defect_correction_method_det = 'opposite_peak'

        x_centers_um_det = np.arange(M_full.shape[1]) * um_per_px
        x_min_um_det = float(params['X_MIN_INTEGRATION'])
        x_max_um_det = float(params['X_MAX_INTEGRATION'])
        xmin_det = int(np.argmin(np.abs(x_centers_um_det - x_min_um_det)))
        xmax_det = int(np.argmin(np.abs(x_centers_um_det - x_max_um_det)))
        if xmax_det < xmin_det:
            xmin_det, xmax_det = xmax_det, xmin_det
        x_roi_um_det = x_centers_um_det[xmin_det:xmax_det + 1]

        def _count_peak_indices_pos_det(d_signal, threshold):
            if d_signal.size < 3:
                return np.array([], dtype=int)
            local_max = (d_signal[1:-1] >= d_signal[:-2]) & (d_signal[1:-1] > d_signal[2:])
            above = d_signal[1:-1] >= threshold
            return np.where(local_max & above)[0] + 1

        def _count_peak_indices_neg_det(d_signal, threshold):
            if d_signal.size < 3:
                return np.array([], dtype=int)
            local_min = (d_signal[1:-1] <= d_signal[:-2]) & (d_signal[1:-1] < d_signal[2:])
            below = d_signal[1:-1] <= threshold
            return np.where(local_min & below)[0] + 1

        def _find_zero_crossing_x_det(m_roi, x_roi_um, i_start, i_end):
            if m_roi.size < 2:
                return None
            i0 = int(max(0, min(i_start, i_end)))
            i1 = int(min(m_roi.size - 1, max(i_start, i_end)))
            if i1 <= i0:
                return None
            for k in range(i0, i1):
                y0 = float(m_roi[k])
                y1 = float(m_roi[k + 1])
                if y0 == 0.0:
                    return float(x_roi_um[k])
                if y1 == 0.0:
                    return float(x_roi_um[k + 1])
                if y0 * y1 < 0.0:
                    x0 = float(x_roi_um[k])
                    x1 = float(x_roi_um[k + 1])
                    if y1 != y0:
                        return float(x0 - y0 * (x1 - x0) / (y1 - y0))
                    return float(0.5 * (x0 + x1))
            return None

        def _find_opposite_peak_x_det(d_roi, x_roi_um, sign_run, i_start, i_end):
            if d_roi.size < 3:
                return None
            i0 = int(max(1, min(i_start, i_end)))
            i1 = int(min(d_roi.size - 2, max(i_start, i_end)))
            if i1 < i0:
                return None

            seg = d_roi[i0:i1 + 1]
            if seg.size == 0:
                return None

            if sign_run == 'pos':
                rel = int(np.argmin(seg))
                idx = i0 + rel
                is_local = bool((d_roi[idx] <= d_roi[idx - 1]) and (d_roi[idx] < d_roi[idx + 1]))
                passes = bool(d_roi[idx] < 0.0)
            else:
                rel = int(np.argmax(seg))
                idx = i0 + rel
                is_local = bool((d_roi[idx] >= d_roi[idx - 1]) and (d_roi[idx] > d_roi[idx + 1]))
                passes = bool(d_roi[idx] > 0.0)

            if not (is_local and passes):
                return None
            return float(x_roi_um[idx])

        def _passes_peak_step_filter_det(m_roi, peak_idx, window_px, min_abs_delta):
            i = int(peak_idx)
            l0 = i - int(window_px)
            l1 = i
            r0 = i + 1
            r1 = i + 1 + int(window_px)
            if l0 < 0 or r1 > m_roi.size:
                return False
            left_avg = float(np.mean(m_roi[l0:l1]))
            right_avg = float(np.mean(m_roi[r0:r1]))
            return abs(right_avg - left_avg) >= float(min_abs_delta)

        def _suggest_threshold_from_plateau_det(thr_values, avg_curve, current_thr, delta_tol, min_points):
            if len(thr_values) == 0 or len(avg_curve) == 0:
                return float(current_thr), None, 'fallback-empty'
            if len(thr_values) == 1:
                return float(thr_values[0]), (float(thr_values[0]), float(thr_values[0])), 'single-point'

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

        def _domain_balance_in_used_region(m_roi, x_roi_um, defect_edges_x, x_left, x_right, start_sign=None):
            """Compute (L_pos - L_neg) / L_used from domain segments in one shot."""
            if m_roi.size == 0 or x_roi_um.size == 0:
                return np.nan

            x_left = float(min(x_left, x_right))
            x_right = float(max(x_left, x_right))
            used_len = float(x_right - x_left)
            if used_len <= 0:
                return np.nan

            edges = [float(x_left)]
            if defect_edges_x is not None and len(defect_edges_x) > 0:
                for xe in np.asarray(defect_edges_x, dtype=float):
                    if x_left < float(xe) < x_right:
                        edges.append(float(xe))
            edges.append(float(x_right))
            edges = np.asarray(sorted(edges), dtype=float)

            if edges.size < 2:
                return np.nan

            l_pos = 0.0
            l_neg = 0.0
            for i in range(len(edges) - 1):
                xa = float(edges[i])
                xb = float(edges[i + 1])
                seg_len = xb - xa
                if seg_len <= 0:
                    continue

                if start_sign is not None:
                    seg_sign = float(start_sign) * ((-1.0) ** i)
                    if seg_sign >= 0.0:
                        l_pos += seg_len
                    else:
                        l_neg += seg_len
                else:
                    mask = (x_roi_um >= xa) & (x_roi_um <= xb)
                    if np.any(mask):
                        m_avg = float(np.mean(m_roi[mask]))
                    else:
                        x_mid = 0.5 * (xa + xb)
                        idx_mid = int(np.argmin(np.abs(x_roi_um - x_mid)))
                        m_avg = float(m_roi[idx_mid])

                    if m_avg >= 0.0:
                        l_pos += seg_len
                    else:
                        l_neg += seg_len

            return float((l_pos - l_neg) / used_len)

        det_x_raw = np.asarray(y_raw, dtype=float)
        # Group nearly identical detuning values to avoid duplicate mean±SEM
        # points caused by floating-point noise.
        det_group_vals, det_group_indices = _build_group_indices(det_x_raw, rounding_decimals=6)
        d_rois_det = []
        m_rois_det = []
        for i in range(len(det_x_raw)):
            m_sig = np.asarray(M_full[i], dtype=float)
            if gauss_sigma_det > 0:
                m_sig = gaussian_filter1d(m_sig, sigma=gauss_sigma_det)
            d_sig = np.gradient(m_sig)
            if d_sig.size < 3 or xmax_det - xmin_det < 2:
                d_rois_det.append(np.array([], dtype=float))
                m_rois_det.append(np.array([], dtype=float))
                continue

            d_rois_det.append(d_sig[xmin_det:xmax_det + 1])
            m_rois_det.append(m_sig[xmin_det:xmax_det + 1])

        if threshold_scan_points_det < 2:
            threshold_scan_points_det = 2
        if threshold_scan_pos_max_det < threshold_scan_pos_min_det:
            threshold_scan_pos_min_det, threshold_scan_pos_max_det = threshold_scan_pos_max_det, threshold_scan_pos_min_det
        if threshold_scan_neg_max_det < threshold_scan_neg_min_det:
            threshold_scan_neg_min_det, threshold_scan_neg_max_det = threshold_scan_neg_max_det, threshold_scan_neg_min_det

        thr_vals_pos_det = np.linspace(threshold_scan_pos_min_det, threshold_scan_pos_max_det, threshold_scan_points_det)
        thr_vals_neg_det = np.linspace(threshold_scan_neg_min_det, threshold_scan_neg_max_det, threshold_scan_points_det)

        thr_pos_per_shot = np.full(det_x_raw.shape, float(deriv_thr_pos_det), dtype=float)
        thr_neg_per_shot = np.full(det_x_raw.shape, float(deriv_thr_neg_det), dtype=float)
        print(
            "KZ_det_scan settings: "
            f"gaussian_sigma={gauss_sigma_det:g}, "
            f"base_thresholds=(+{deriv_thr_pos_det:.4f}, {deriv_thr_neg_det:.4f}), "
            f"threshold_scan={'on' if threshold_scan_det else 'off'}, "
            f"missed_defect_method={missed_defect_correction_method_det}, "
            f"peak_step_filter={'on' if peak_step_filter_enabled_det else 'off'}"
            f"(N={peak_step_filter_window_px_det}, min|Δm|={peak_step_filter_min_abs_delta_m_det:.3f})"
        )
        if threshold_scan_det:
            for det_val, idx_det in zip(det_group_vals, det_group_indices):
                d_rois_group = [d_rois_det[ii] for ii in idx_det]
                if len(d_rois_group) == 0:
                    continue

                avg_counts_pos = []
                avg_counts_neg = []
                for thr_pos in thr_vals_pos_det:
                    c_pos = [len(_count_peak_indices_pos_det(d_roi, thr_pos)) for d_roi in d_rois_group]
                    avg_counts_pos.append(float(np.mean(c_pos)) if len(c_pos) > 0 else 0.0)
                for thr_neg in thr_vals_neg_det:
                    c_neg = [len(_count_peak_indices_neg_det(d_roi, thr_neg)) for d_roi in d_rois_group]
                    avg_counts_neg.append(float(np.mean(c_neg)) if len(c_neg) > 0 else 0.0)

                sug_pos, _, mode_pos = _suggest_threshold_from_plateau_det(
                    thr_vals_pos_det,
                    np.asarray(avg_counts_pos, dtype=float),
                    deriv_thr_pos_det,
                    plateau_delta_defects_det,
                    plateau_min_points_det,
                )
                sug_neg, _, mode_neg = _suggest_threshold_from_plateau_det(
                    thr_vals_neg_det,
                    np.asarray(avg_counts_neg, dtype=float),
                    deriv_thr_neg_det,
                    plateau_delta_defects_det,
                    plateau_min_points_det,
                )
                thr_pos_per_shot[idx_det] = float(sug_pos)
                thr_neg_per_shot[idx_det] = float(sug_neg)
                print(
                    f"KZ_det_scan det={float(det_val):.6g}: +thr={sug_pos:.4f} ({mode_pos}), "
                    f"-thr={sug_neg:.4f} ({mode_neg})"
                )

        det_counts_raw = []
        det_defect_x = []
        det_defect_y = []
        det_defect_x_pos = []
        det_defect_y_pos = []
        det_defect_x_neg = []
        det_defect_y_neg = []
        det_defect_x_rej = []
        det_defect_y_rej = []
        det_defect_x_corr = []
        det_defect_y_corr = []
        det_domain_balance_raw = []
        for i in range(len(det_x_raw)):
            d_roi = d_rois_det[i] if i < len(d_rois_det) else np.array([], dtype=float)
            m_roi = m_rois_det[i] if i < len(m_rois_det) else np.array([], dtype=float)
            if d_roi.size < 3:
                det_counts_raw.append(0)
                det_domain_balance_raw.append(np.nan)
                continue

            thr_pos_i = float(thr_pos_per_shot[i])
            thr_neg_i = float(thr_neg_per_shot[i])
            pos_idx_raw = _count_peak_indices_pos_det(d_roi, thr_pos_i)
            neg_idx_raw = _count_peak_indices_neg_det(d_roi, thr_neg_i)
            pos_idx = []
            neg_idx = []
            rejected_idx = []
            for idxp in pos_idx_raw:
                if peak_step_filter_enabled_det and not _passes_peak_step_filter_det(m_roi, idxp, peak_step_filter_window_px_det, peak_step_filter_min_abs_delta_m_det):
                    rejected_idx.append(int(idxp))
                else:
                    pos_idx.append(int(idxp))
            for idxn in neg_idx_raw:
                if peak_step_filter_enabled_det and not _passes_peak_step_filter_det(m_roi, idxn, peak_step_filter_window_px_det, peak_step_filter_min_abs_delta_m_det):
                    rejected_idx.append(int(idxn))
                else:
                    neg_idx.append(int(idxn))
            pos_idx, pos_x = _merge_close_same_sign_peaks(
                pos_idx,
                x_roi_um_det,
                min_same_sign_peak_distance_um_det,
            )
            neg_idx, neg_x = _merge_close_same_sign_peaks(
                neg_idx,
                x_roi_um_det,
                min_same_sign_peak_distance_um_det,
            )
            peak_x = np.concatenate([pos_x, neg_x]).astype(float)

            events = [(int(p), 'pos') for p in pos_idx] + [(int(n), 'neg') for n in neg_idx]
            events.sort(key=lambda e: e[0])

            corrected_x = []
            j = 0
            while j < len(events):
                sign_j = events[j][1]
                k = j
                while k < len(events) and events[k][1] == sign_j:
                    k += 1
                run = events[j:k]
                if len(run) >= 2:
                    candidates = []
                    # Search ONLY between adjacent same-sign peaks in the run.
                    for p in range(len(run) - 1):
                        i_start = int(run[p][0])
                        i_end = int(run[p + 1][0])
                        sign_pair = run[p][1]
                        if missed_defect_correction_method_det in ('zero_crossing', 'both'):
                            zx = _find_zero_crossing_x_det(m_roi, x_roi_um_det, i_start, i_end)
                            if zx is not None:
                                candidates.append(float(zx))
                        if missed_defect_correction_method_det in ('opposite_peak', 'both'):
                            ox = _find_opposite_peak_x_det(
                                d_roi,
                                x_roi_um_det,
                                sign_pair,
                                i_start,
                                i_end,
                            )
                            if ox is not None:
                                candidates.append(float(ox))

                    for cx in candidates:
                        if peak_x.size == 0:
                            corrected_x.append(float(cx))
                            continue
                        min_dist = float(np.min(np.abs(peak_x - float(cx))))
                        if min_dist >= zero_cross_min_dist_um_det:
                            corrected_x.append(float(cx))
                j = k

            if len(corrected_x) > 1:
                corrected_x = list(np.unique(np.round(np.asarray(corrected_x, dtype=float), 6)))

            # Domain-balance metric per shot:
            # (total positive-domain size - total negative-domain size) / used-region size
            defect_edges_x_shot = np.concatenate([pos_x, neg_x, np.asarray(corrected_x, dtype=float)]).astype(float)
            if defect_edges_x_shot.size > 1:
                defect_edges_x_shot = np.unique(np.round(defect_edges_x_shot, 6))
            start_sign_shot = _infer_start_sign_from_signed_walls(pos_x, neg_x)
            det_domain_balance_raw.append(
                _domain_balance_in_used_region(
                    m_roi,
                    x_roi_um_det,
                    defect_edges_x_shot,
                    float(x_min_um_det),
                    float(x_max_um_det),
                    start_sign=start_sign_shot,
                )
            )

            for x_thr in np.concatenate([pos_x, neg_x]).astype(float):
                det_defect_x.append(float(x_thr))
                det_defect_y.append(float(i + 1))
            for x_p in pos_x:
                det_defect_x_pos.append(float(x_p))
                det_defect_y_pos.append(float(i + 1))
            for x_n in neg_x:
                det_defect_x_neg.append(float(x_n))
                det_defect_y_neg.append(float(i + 1))
            for idxr in rejected_idx:
                det_defect_x_rej.append(float(x_roi_um_det[int(idxr)]))
                det_defect_y_rej.append(float(i + 1))
            for xcorr in corrected_x:
                det_defect_x_corr.append(float(xcorr))
                det_defect_y_corr.append(float(i + 1))

            det_counts_raw.append(int(len(pos_x) + len(neg_x) + len(corrected_x)))

        det_counts_raw = np.asarray(det_counts_raw, dtype=float)
        det_domain_balance_raw = np.asarray(det_domain_balance_raw, dtype=float)
        # Normalize by the used x-region length (between xmin and xmax).
        used_region_um = float(abs(x_max_um_det - x_min_um_det))
        if used_region_um <= 0:
            used_region_um = 1.0
        det_norm_raw = det_counts_raw / used_region_um

        if det_counts_raw.size > 0:
            det_unique = np.asarray(det_group_vals, dtype=float)
            det_mean = []
            det_sem = []
            det_n = []
            for idx_det in det_group_indices:
                vals = det_norm_raw[idx_det]
                det_mean.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
                if vals.size > 1:
                    det_sem.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)))
                else:
                    det_sem.append(0.0)
                det_n.append(int(vals.size))
            det_mean = np.asarray(det_mean, dtype=float)
            det_sem = np.asarray(det_sem, dtype=float)

            fig_det, ax_det = plt.subplots(figsize=(8.5, 5.0), tight_layout=True)
            ax_det.scatter(det_x_raw, det_norm_raw, s=18, color='0.6', alpha=0.65, label='single shot')
            ax_det.errorbar(
                det_unique,
                det_mean,
                yerr=det_sem,
                fmt='o-',
                color='tab:blue',
                ecolor='tab:blue',
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                linewidth=1.5,
                label='mean ± SEM',
            )
            ax_det.set_xlabel('ARPKZ_final_set_field')
            ax_det.set_ylabel(r'Defects / used region ($\mu m^{-1}$)')
            ax_det.set_title(f'Defects per used region vs ARPKZ_final_set_field (seqs: {seqs})')
            ax_det.grid(True, alpha=0.25)
            ax_det.legend(loc='best', fontsize=9)

            print(f"\nKZ_det_scan defects/used-region summary (after filtering, used region = {used_region_um:.3f} um):")
            for xv, m, s, n in zip(det_unique, det_mean, det_sem, det_n):
                print(f"  det={xv:.6g}: defects/region={m:.5f} ± {s:.5f} um^-1 (SEM), shots={n}")

            # Domain magnetization balance vs det:
            # balance = (L_pos - L_neg) / L_used for each shot, then avg over shots at same det.
            det_dom_mean = []
            det_dom_sem = []
            for idx_det in det_group_indices:
                vals = det_domain_balance_raw[idx_det]
                vals = vals[np.isfinite(vals)]
                det_dom_mean.append(float(np.mean(vals)) if vals.size > 0 else np.nan)
                if vals.size > 1:
                    det_dom_sem.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)))
                else:
                    det_dom_sem.append(0.0 if vals.size == 1 else np.nan)
            det_dom_mean = np.asarray(det_dom_mean, dtype=float)
            det_dom_sem = np.asarray(det_dom_sem, dtype=float)

            # Report defects/used-region at the det where domain balance is closest to zero.
            valid_zero = np.isfinite(det_dom_mean) & np.isfinite(det_mean) & np.isfinite(det_sem)
            if np.any(valid_zero):
                idx_candidates = np.where(valid_zero)[0]
                idx0 = int(idx_candidates[int(np.argmin(np.abs(det_dom_mean[idx_candidates])) )])
                print(
                    "KZ_det_scan point closest to domain_balance=0: "
                    f"det={det_unique[idx0]:.6g}, "
                    f"domain_balance={det_dom_mean[idx0]:.5f} ± {det_dom_sem[idx0]:.5f}, "
                    f"defects/region={det_mean[idx0]:.5f} ± {det_sem[idx0]:.5f} um^-1"
                )

            # Requested fit-based estimate around domain_balance~0:
            # 1) linear fit to domain-balance points with |balance| <= fit_window
            # 2) parabolic fit to defects/region for the SAME det points
            # 3) evaluate defects/region at det where fitted domain balance is zero.
            fit_window = 0.3
            if isinstance(mode_cfg, dict):
                fit_window = float(mode_cfg.get('domain_balance_fit_window', 0.3))
            fit_mask = (
                np.isfinite(det_unique)
                & np.isfinite(det_dom_mean)
                & np.isfinite(det_mean)
                & (np.abs(det_dom_mean) <= fit_window)
            )

            det_zero_fit = None
            def_at_zero_fit = None
            def_at_zero_fit_err = np.nan
            fit_text = None
            dom_lin_coeffs = None
            def_par_coeffs = None
            x_fit_span = None

            if np.count_nonzero(fit_mask) >= 3:
                x_fit = np.asarray(det_unique[fit_mask], dtype=float)
                y_dom_fit = np.asarray(det_dom_mean[fit_mask], dtype=float)
                y_def_fit = np.asarray(det_mean[fit_mask], dtype=float)
                y_def_sem_fit = np.asarray(det_sem[fit_mask], dtype=float)

                order = np.argsort(x_fit)
                x_fit = x_fit[order]
                y_dom_fit = y_dom_fit[order]
                y_def_fit = y_def_fit[order]
                y_def_sem_fit = y_def_sem_fit[order]

                # Linear fit: domain balance vs det
                dom_lin = np.polyfit(x_fit, y_dom_fit, 1)
                dom_slope = float(dom_lin[0])
                dom_intercept = float(dom_lin[1])
                dom_lin_coeffs = np.asarray(dom_lin, dtype=float)
                x_fit_span = (float(np.min(x_fit)), float(np.max(x_fit)))

                if abs(dom_slope) > 1e-12:
                    det_zero_fit = float(-dom_intercept / dom_slope)

                    # Constrained parabolic fit with forced vertex at det_zero_fit:
                    # y = a * (x - det_zero_fit)^2 + c
                    # Then the parabolic peak/minimum is exactly at det_zero_fit.
                    u = (x_fit - det_zero_fit) ** 2
                    Xq = np.column_stack([u, np.ones_like(u)])
                    beta, _, rank, _ = np.linalg.lstsq(Xq, y_def_fit, rcond=None)
                    a_q = float(beta[0])
                    c_q = float(beta[1])
                    def_par_coeffs = (a_q, c_q, float(det_zero_fit))

                    def_at_zero_fit = c_q

                    # Requested uncertainty: average error bar (SEM) of the
                    # defect-density points used in the parabolic fit.
                    sem_used = y_def_sem_fit[np.isfinite(y_def_sem_fit)]
                    if sem_used.size > 0:
                        def_at_zero_fit_err = float(np.mean(sem_used))

                    if np.isfinite(def_at_zero_fit_err):
                        fit_text = (
                            rf"fit @ domain balance$=0$: "
                            rf"$\rho_d={def_at_zero_fit:.5f}\pm{def_at_zero_fit_err:.5f}\,\mu m^{{-1}}$"
                        )
                    else:
                        fit_text = (
                            rf"fit @ domain balance$=0$: "
                            rf"$\rho_d={def_at_zero_fit:.5f}\,\mu m^{{-1}}$"
                        )

                    if np.isfinite(def_at_zero_fit_err):
                        print(
                            "KZ_det_scan fit-based domain_balance=0: "
                            f"det={det_zero_fit:.6g}, "
                            f"defects/region={def_at_zero_fit:.5f} ± {def_at_zero_fit_err:.5f} um^-1 "
                            f"(fit window |domain_balance| <= {fit_window}, constrained parabola vertex at this det)"
                        )
                    else:
                        print(
                            "KZ_det_scan fit-based domain_balance=0: "
                            f"det={det_zero_fit:.6g}, "
                            f"defects/region={def_at_zero_fit:.5f} um^-1 "
                            f"(fit window |domain_balance| <= {fit_window}; constrained parabola vertex at this det; fit uncertainty unavailable)"
                        )
                else:
                    print(
                        "KZ_det_scan: linear fit slope is ~0 for domain balance; "
                        "cannot compute det at domain_balance=0."
                    )
            else:
                print(
                    "KZ_det_scan: not enough points with |domain_balance| <= 0.3 "
                    "for requested linear/parabolic fit."
                )

            # Overlay domain-balance metric on the same figure with a second y-axis.
            ax_dom = ax_det.twinx()
            valid_single = np.isfinite(det_domain_balance_raw)
            if np.any(valid_single):
                ax_dom.scatter(
                    det_x_raw[valid_single],
                    det_domain_balance_raw[valid_single],
                    s=18,
                    color='tab:purple',
                    alpha=0.25,
                    label='domain balance (single shot)',
                )
            valid_mean = np.isfinite(det_dom_mean)
            if np.any(valid_mean):
                ax_dom.errorbar(
                    det_unique[valid_mean],
                    det_dom_mean[valid_mean],
                    yerr=det_dom_sem[valid_mean],
                    fmt='o-',
                    color='tab:purple',
                    ecolor='tab:purple',
                    elinewidth=1.2,
                    capsize=3,
                    markersize=5,
                    linewidth=1.5,
                    label='domain balance mean ± SEM',
                )
            ax_dom.axhline(0.0, color='0.5', linestyle=':', linewidth=1.0)
            ax_dom.set_ylabel(r'Domain balance $(L_{+}-L_{-})/L_{used}$')
            ax_dom.grid(False)

            # Show fit curves explicitly in the same x-range used for fitting.
            if x_fit_span is not None:
                x_fit_curve = np.linspace(x_fit_span[0], x_fit_span[1], 300)
                if dom_lin_coeffs is not None:
                    y_dom_curve = np.polyval(dom_lin_coeffs, x_fit_curve)
                    ax_dom.plot(
                        x_fit_curve,
                        y_dom_curve,
                        color='purple',
                        linestyle='--',
                        linewidth=1.7,
                        alpha=0.95,
                        label=r'linear fit ($|balance|\leq0.3$)',
                    )
                if def_par_coeffs is not None:
                    a_q, c_q, x0_q = def_par_coeffs
                    y_def_curve = a_q * (x_fit_curve - x0_q) ** 2 + c_q
                    ax_det.plot(
                        x_fit_curve,
                        y_def_curve,
                        color='tab:blue',
                        linestyle='--',
                        linewidth=1.9,
                        alpha=0.95,
                        label=r'parabolic fit (vertex fixed at domain-balance $=0$)',
                    )

            if det_zero_fit is not None and np.isfinite(det_zero_fit):
                ax_det.axvline(
                    det_zero_fit,
                    color='red',
                    linestyle='--',
                    linewidth=1.4,
                    alpha=0.9,
                    label='fit: domain balance = 0',
                    zorder=20,
                )
                if fit_text is not None:
                    ax_det.text(
                        0.02,
                        0.93,
                        fit_text,
                        transform=ax_det.transAxes,
                        ha='left',
                        va='top',
                        fontsize=9,
                        color='red',
                        bbox=dict(facecolor='white', edgecolor='red', alpha=0.75, boxstyle='round,pad=0.25'),
                    )

            # Merge legends from both y-axes.
            h1, l1 = ax_det.get_legend_handles_labels()
            h2, l2 = ax_dom.get_legend_handles_labels()
            ax_det.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8)

            # Second waterfall: all raw magnetization profiles (single shots) ordered by det.
            # Keep layout static for interactive zoom/pan: tight_layout can
            # continuously reflow the axes as tick labels change on redraw.
            fig_raw, ax_raw = plt.subplots(figsize=(10.0, 6.5), constrained_layout=False)
            # Fully disable dynamic layout engines to keep axes geometry fixed
            # during interactive redraws (zoom/pan).
            try:
                fig_raw.set_tight_layout(False)
            except Exception:
                pass
            try:
                fig_raw.set_constrained_layout(False)
            except Exception:
                pass
            fig_raw.subplots_adjust(left=0.09, right=0.84, bottom=0.10, top=0.92)
            cax_raw = fig_raw.add_axes([0.90, 0.10, 0.018, 0.82])
            raw_vmin = -1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][0]
            raw_vmax = 1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][1]
            im_raw = ax_raw.imshow(
                M_full,
                aspect='auto',
                cmap='RdBu',
                interpolation='nearest',
                resample=False,
                vmin=raw_vmin,
                vmax=raw_vmax,
                extent=[float(x_centers_um_det[0]), float(x_centers_um_det[-1]), float(len(M_full)) + 0.5, 0.5],
            )
            ax_raw.set_xlabel(r'$\mu m$')
            ax_raw.set_ylabel('shot index (ordered by det)')
            ax_raw.set_title(f'Raw magnetization waterfall ordered by ARPKZ_final_set_field (seqs: {seqs})')
            ax_raw.set_xlim(float(x_centers_um_det[0]), float(x_centers_um_det[-1]))
            ax_raw.set_ylim(float(len(M_full)) + 0.5, 0.5)
            ax_raw.set_autoscale_on(False)
            ax_raw.axvline(x=float(x_min_um_det), color='lime', linestyle='--', linewidth=2.2, alpha=0.95)
            ax_raw.axvline(x=float(x_max_um_det), color='lime', linestyle='--', linewidth=2.2, alpha=0.95)

            det_x_pos_arr = np.asarray(det_defect_x_pos, dtype=float)
            det_y_pos_arr = np.asarray(det_defect_y_pos, dtype=float)
            det_x_neg_arr = np.asarray(det_defect_x_neg, dtype=float)
            det_y_neg_arr = np.asarray(det_defect_y_neg, dtype=float)
            det_x_corr_arr = np.asarray(det_defect_x_corr, dtype=float)
            det_y_corr_arr = np.asarray(det_defect_y_corr, dtype=float)

            def _plot_raw_identified_domain_binary_profile():
                """Overlay a color-coded binary (±1) domain strip below each raw shot."""
                if len(M_full) == 0:
                    return
                x_axis = np.asarray(x_centers_um_det, dtype=float)
                if x_axis.size < 2:
                    return
                x_left = float(x_axis[0])
                x_right = float(x_axis[-1])

                # Inverted y-axis (N -> 1): add offset to draw visually below the shot row.
                y_center_offset = 0.33
                strip_thickness = 0.32
                polys_bin, colors_bin = [], []
                x_edges = np.arange(-0.5, M_full.shape[1], 1.0) * float(um_per_px)

                for shot_idx in range(len(M_full)):
                    y_shot = float(shot_idx + 1)
                    walls = []
                    walls_pos = np.asarray([], dtype=float)
                    walls_neg = np.asarray([], dtype=float)
                    if det_x_pos_arr.size > 0:
                        walls_pos = np.asarray(det_x_pos_arr[np.isclose(det_y_pos_arr, y_shot, atol=1e-9, rtol=0.0)], dtype=float)
                        walls.extend(walls_pos.tolist())
                    if det_x_neg_arr.size > 0:
                        walls_neg = np.asarray(det_x_neg_arr[np.isclose(det_y_neg_arr, y_shot, atol=1e-9, rtol=0.0)], dtype=float)
                        walls.extend(walls_neg.tolist())
                    if det_x_corr_arr.size > 0:
                        walls.extend(det_x_corr_arr[np.isclose(det_y_corr_arr, y_shot, atol=1e-9, rtol=0.0)].tolist())

                    if len(walls) > 0:
                        walls = np.unique(np.round(np.asarray(walls, dtype=float), 6))
                        walls = walls[(walls > x_left) & (walls < x_right)]
                    else:
                        walls = np.asarray([], dtype=float)

                    edges = np.concatenate(([x_left], walls, [x_right]))
                    if edges.size < 2:
                        continue

                    m_row = np.asarray(M_full[shot_idx], dtype=float)
                    if walls.size > 0:
                        left_mask = x_axis < float(walls[0])
                        if not np.any(left_mask):
                            left_mask = x_axis <= float(walls[0])
                    else:
                        left_mask = np.ones_like(x_axis, dtype=bool)

                    sign_from_defect = _infer_start_sign_from_signed_walls(walls_pos, walls_neg)
                    if sign_from_defect is not None:
                        sign_val = float(sign_from_defect)
                    else:
                        used_mask_sign = (x_axis >= float(x_min_um_det)) & (x_axis <= float(x_max_um_det))
                        avg_m = np.nanmean(m_row[used_mask_sign]) if np.any(used_mask_sign) else np.nanmean(m_row)
                        sign_val = 1.0 if np.nan_to_num(avg_m, nan=0.0) > 0.0 else -1.0
                    n_flips = np.searchsorted(walls, x_axis, side='right')
                    row_bin = sign_val * ((-1.0) ** n_flips)
                    used_mask = (x_axis >= float(x_min_um_det)) & (x_axis <= float(x_max_um_det))
                    row_bin = np.where(used_mask, row_bin, 0.0)

                    y0 = y_shot + y_center_offset - 0.5 * strip_thickness
                    y1 = y_shot + y_center_offset + 0.5 * strip_thickness
                    for j in range(M_full.shape[1]):
                        polys_bin.append([
                            (x_edges[j], y0),
                            (x_edges[j + 1], y0),
                            (x_edges[j + 1], y1),
                            (x_edges[j], y1),
                        ])
                        colors_bin.append(float(row_bin[j]))

                if len(polys_bin) > 0:
                    coll_bin = PolyCollection(polys_bin, array=np.asarray(colors_bin, dtype=float), cmap='RdBu')
                    coll_bin.set_clim(raw_vmin, raw_vmax)
                    coll_bin.set_alpha(1.0)
                    coll_bin.set_zorder(13)
                    ax_raw.add_collection(coll_bin)

                # Delineate each shot pair (main row + binary strip) with horizontal guides.
                x_l = float(x_centers_um_det[0])
                x_r = float(x_centers_um_det[-1])
                for shot_idx in range(len(M_full) - 1):
                    yv = float(shot_idx + 1)
                    y_sep = yv + 0.50
                    ax_raw.plot([x_l, x_r], [y_sep, y_sep], color='yellow', linestyle='--', linewidth=1.0, alpha=0.72, zorder=14)

            if len(det_defect_x_pos) > 0:
                ax_raw.scatter(
                    det_x_pos_arr,
                    det_y_pos_arr,
                    c='blue',
                    s=16,
                    edgecolors='black',
                    linewidths=0.3,
                    zorder=10,
                    label='positive defects (+)',
                )
            if len(det_defect_x_neg) > 0:
                ax_raw.scatter(
                    det_x_neg_arr,
                    det_y_neg_arr,
                    c='red',
                    s=16,
                    edgecolors='black',
                    linewidths=0.3,
                    zorder=10,
                    label='negative defects (-)',
                )
            if len(det_defect_x_corr) > 0:
                ax_raw.scatter(
                    det_x_corr_arr,
                    det_y_corr_arr,
                    c='lime',
                    s=18,
                    edgecolors='black',
                    linewidths=0.35,
                    zorder=11,
                    label='zero-crossing corrected',
                )
            if len(det_defect_x_rej) > 0:
                ax_raw.plot(
                    np.asarray(det_defect_x_rej, dtype=float),
                    np.asarray(det_defect_y_rej, dtype=float),
                    linestyle='None',
                    marker='x',
                    color='black',
                    markersize=5,
                    markeredgewidth=0.9,
                    zorder=12,
                    label='rejected peaks',
                )

            _plot_raw_identified_domain_binary_profile()

            if len(grouped_indices) > 0:
                x_left = float(x_centers_um_det[0])
                x_right = float(x_centers_um_det[-1])
                x_text = x_left + 0.012 * (x_right - x_left)

                start = 0
                centers = []
                for det_val, g in zip(grouped_y, grouped_indices):
                    end = start + len(g)
                    y_start = float(start) + 0.5
                    y_end = float(end) + 0.5
                    y_center = 0.5 * (y_start + y_end)
                    centers.append(y_center)

                    # Draw an empty horizontal separator between different detunings.
                    # Skip the very first start boundary; keep separators only between blocks.
                    if start > 0:
                        sep_half_height = 0.42
                        ax_raw.axhspan(
                            y_start - sep_half_height,
                            y_start + sep_half_height,
                            facecolor='white',
                            edgecolor='none',
                            zorder=19,
                        )

                    # Print det value inside the block.
                    ax_raw.text(
                        x_text,
                        y_center,
                        f'det={float(det_val):.6g}',
                        color='white',
                        fontsize=8,
                        va='center',
                        ha='left',
                        bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.5),
                        zorder=20,
                    )
                    start = end

                # Use a secondary y-axis instead of twinx, so y-limits stay
                # synchronized during interactive zoom/pan.
                ax_det_ticks = ax_raw.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
                ax_det_ticks.set_yticks(centers)
                ax_det_ticks.set_yticklabels([f'{float(v):.6g}' for v in grouped_y], fontsize=8)
                ax_det_ticks.set_ylabel('ARPKZ_final_set_field')
                ax_det_ticks.grid(False)

            if len(det_defect_x_pos) > 0 or len(det_defect_x_neg) > 0 or len(det_defect_x_corr) > 0 or len(det_defect_x_rej) > 0:
                ax_raw.legend(loc='upper right', fontsize=8)
            cbar_raw = fig_raw.colorbar(im_raw, cax=cax_raw)
            cbar_raw.set_label('magnetization m')

    # Get domain wall fit data before creating main plot
    dw_fit_coeffs, dw_fit_x, dw_fit_t = None, None, None
    if plot_flags.get('evolution_plots', False) and scan == 'bubbles_evolution':
        dw_fit_coeffs, dw_fit_x, dw_fit_t = plot_evolution_analysis(y_final, y_axis_col, z_fluc_final, M_final, um_per_px, params, mode_cfg)

    if plot_flags.get('main_waterfall', True):
        title_full = f"{title_base}\n{'AVERAGED' if average else 'UNIQUE'}\n(seqs: {seqs})"
        if scan == 'KZ_defect_analysis' and avg_defects is not None and stat_err is not None:
            title_full += f"\n⟨defects/shot⟩ = {avg_defects:.3f} ± {stat_err:.3f} (SEM)"
        try:
            seq_col = _get_column(sorted_df, 'sequence_index')
            if seq_col is not None:
                seq_used = np.unique(seq_col.values)
                title_full += f"\nsequence_index used: {seq_used}"
        except Exception:
            pass

        g_lines = _build_globals_title_lines(sorted_df, globals_in_title)
        if len(g_lines) > 0:
            title_full += '\n' + '\n'.join(g_lines)

        if title_labels:
            title_full += '\n' + ', '.join([f"{l}: {np.unique(_get_column(sorted_df, l).values)}" for l in title_labels if _get_column(sorted_df, l) is not None])

        plot_main_waterfall(
            M_final,
            D_final,
            y_final,
            dy,
            y_axis_label,
            title_full,
            scan,
            params,
            plot_flags,
            um_per_px,
            average=average,
            defect_points=defect_points,
            defect_points_pos=defect_points_pos,
            defect_points_neg=defect_points_neg,
            defect_points_corrected=defect_points_corrected,
            defect_points_rejected=defect_points_rejected,
            defect_counts=defect_counts_for_plot,
            defect_sequences=defect_sequences_for_plot,
            defect_thresholds=defect_thresholds_for_plot,
            dw_fit_coeffs=dw_fit_coeffs,
            dw_fit_x=dw_fit_x,
            dw_fit_t=dw_fit_t,
        )
        
        # For bubbles_evolution, plot fluctuations waterfall immediately after main waterfall
        if scan == 'bubbles_evolution' and plot_flags.get('fluctuations_waterfall', False):
            y_fluc, sigma2_map = compute_fluctuation_sigma2_waterfall(
                M_full,
                y_raw,
                params['SIGMA_Z_LOCAL_FLUCT'],
                params['SIGMA_Z_LOCAL_AVG'],
            )
            dy_fluc = np.min(np.diff(y_fluc)) if len(y_fluc) > 1 else 0.1
            fluc_title = f"{title_base}\nAverage local fluctuations $\\sigma^2$ (m - <m>)\n(seqs: {seqs})"
            plot_fluctuations_waterfall(
                sigma2_map,
                y_fluc,
                dy_fluc,
                y_axis_label,
                fluc_title,
                params,
                um_per_px,
                mode_cfg=mode_cfg if isinstance(mode_cfg, dict) else {},
                dw_fit_coeffs=dw_fit_coeffs,
                dw_fit_x=dw_fit_x,
                dw_fit_t=dw_fit_t,
            )
        
        # For bubbles_evolution, plot all individual shots with domain wall detection
        if scan == 'bubbles_evolution' and plot_flags.get('all_shots_waterfall', False) and len(M_full) > 0:
            plot_all_shots_waterfall(
                M_full,
                grouped_indices,
                grouped_y,
                y_axis_col,
                seqs,
                um_per_px,
                params,
                plot_manager=plot_manager if 'plot_manager' in locals() else None,
            )
        
        # For bubbles_evolution, create additional waterfall with minimum magnetization selection
        if scan == 'bubbles_evolution' and plot_flags.get('main_waterfall', True):
            # Select shot with minimum average magnetization per time value
            M_min_mag = np.zeros((len(grouped_indices), img_dims))
            D_min_mag = np.zeros((len(grouped_indices), img_dims))
            z_fluc_min_mag = np.zeros((len(grouped_indices), z_fluc_full.shape[1]))
            
            x_min_um = float(params['X_MIN_INTEGRATION'])
            x_max_um = float(params['X_MAX_INTEGRATION'])
            x_centers_um = np.arange(img_dims) * um_per_px
            xmin_ind = int(np.argmin(np.abs(x_centers_um - x_min_um)))
            xmax_ind = int(np.argmin(np.abs(x_centers_um - x_max_um)))
            if xmax_ind < xmin_ind:
                xmin_ind, xmax_ind = xmax_ind, xmin_ind
            
            for i, idx in enumerate(grouped_indices):
                # Compute average magnetization in used region for each shot in this group
                avg_mag_per_shot = np.mean(M_full[idx, xmin_ind:xmax_ind+1], axis=1)
                # Find shot with minimum average magnetization
                min_mag_shot_idx = idx[np.argmin(avg_mag_per_shot)]
                M_min_mag[i] = M_full[min_mag_shot_idx]
                D_min_mag[i] = D_full[min_mag_shot_idx]
                z_fluc_min_mag[i] = z_fluc_full[min_mag_shot_idx]
            
            title_min_mag = f"{title_base}\nMinimum average magnetization shot per time\n(seqs: {seqs})"
            plot_main_waterfall(
                M_min_mag,
                D_min_mag,
                y_final,
                dy,
                y_axis_label,
                title_min_mag,
                scan,
                params,
                plot_flags,
                um_per_px,
                average=False,  # Already selected, not averaged
                dw_fit_coeffs=dw_fit_coeffs,
                dw_fit_x=dw_fit_x,
                dw_fit_t=dw_fit_t,
            )

    # Clean waterfall for KZ_defect_analysis without defect markers
    if scan == 'KZ_defect_analysis':
        title_clean = f"{title_base} (clean, no markers)\n(seqs: {seqs})"
        plot_main_waterfall(
            M_final,
            D_final,
            y_final,
            dy,
            y_axis_label,
            title_clean,
            scan,
            params,
            plot_flags,
            um_per_px,
            average=average,
            defect_points=None,
            defect_points_pos=None,
            defect_points_neg=None,
            defect_points_corrected=None,
            defect_points_rejected=None,
            defect_counts=None,
            defect_sequences=None,
            defect_thresholds=None,
        )

    # Plot fluctuations waterfall for non-bubbles_evolution modes
    if plot_flags.get('fluctuations_waterfall', False) and scan != 'bubbles_evolution':
        y_fluc, sigma2_map = compute_fluctuation_sigma2_waterfall(
            M_full,
            y_raw,
            params['SIGMA_Z_LOCAL_FLUCT'],
            params['SIGMA_Z_LOCAL_AVG'],
        )
        dy_fluc = np.min(np.diff(y_fluc)) if len(y_fluc) > 1 else 0.1
        fluc_title = f"{title_base}\nAverage local fluctuations $\\sigma^2$ (m - <m>)\n(seqs: {seqs})"
        plot_fluctuations_waterfall(
            sigma2_map,
            y_fluc,
            dy_fluc,
            y_axis_label,
            fluc_title,
            params,
            um_per_px,
            mode_cfg=mode_cfg if isinstance(mode_cfg, dict) else {},
            dw_fit_coeffs=dw_fit_coeffs,
            dw_fit_x=dw_fit_x,
            dw_fit_t=dw_fit_t,
        )

    if plot_flags.get('used_region_density_fluctuations', False):
        used_region_title = f"{title_base}\nDensity fluctuations in used region (density - shot avg)\n(seqs: {seqs})"
        plot_used_region_density_fluctuations(
            D_final,
            y_final,
            dy,
            y_axis_label,
            used_region_title,
            params,
            um_per_px,
            mode_cfg=mode_cfg,
        )

    # Save sigmoid center interpolation (extract from averaged magnetization profile)
    # Only save if sectioned_sigmoid plot is enabled
    if plot_flags.get('sectioned_sigmoid', False):
        try:
            save_sigmoid_center_interpolation(
                M_final if average else M_full,
                x_plot_um,
                y_final if average else y_raw,
                y_axis_label,
                um_per_px,
                params,
            )
        except Exception as e:
            print(f"Warning: Could not save sigmoid center interpolation: {e}")

    plt.show()


def run_mode(df, mode_cfg, params):
    waterfall_plot(
        df=df,
        seqs=mode_cfg['seqs'],
        scan=mode_cfg['scan'],
        data_origin=mode_cfg.get('data_origin', 'show_ODs'),
        constraints=mode_cfg.get('constraints', None),
        average=mode_cfg.get('average', False),
        title_labels=mode_cfg.get('title_labels', None),
        globals_in_title=mode_cfg.get('globals_in_title', []),
        params=params,
        plot_flags=mode_cfg.get('plots', {}),
        mode_cfg=mode_cfg,
    )