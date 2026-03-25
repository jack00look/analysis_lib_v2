import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    c = result[result.size // 2]
    if c == 0:
        return np.zeros_like(result[result.size // 2:])
    return result[result.size // 2:] / c


def get_um_per_px(params):
    camera_name = params.get('CAMERA_NAME', 'cam_vert1')
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


def filter_dataframe(df, seqs, constraints=None):
    out = df[df['sequence_index'].isin(seqs)]
    print(f"Initial shots in sequences {seqs}: {len(out)}")
    if constraints is not None:
        for key, value in constraints.items():
            before = len(out)
            try:
                if not isinstance(value, (str, bytes)) and hasattr(value, '__iter__'):
                    out = out[out[key].isin(value)]
                else:
                    out = out[np.abs(out[key] - value) < 1e-3]
            except Exception:
                out = out[out[key] == value]
            after = len(out)
            rejected = before - after
            print(f"  Constraint {key}={value}: {rejected} rejected, {after} remaining")
    
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


def get_valid_rows_mask(df, back_threshold, ntot_threshold):
    n = len(df)

    # Try to get background and N_atoms columns - they may not exist or have different names
    # Try multiple possible column names
    back_m1_candidates = [
        ('show_ODs', 'PTAI_m1_cnt_rel_atoms_back'),
        ('show_ODs', 'PTAI_m1_SVD_cnt_rel_atoms_back'),
    ]
    back_m1 = None
    for col in back_m1_candidates:
        if col in df.columns:
            temp = df[col].values
            if not np.all(np.isnan(temp)):
                back_m1 = temp
                break
    if back_m1 is None or np.all(np.isnan(back_m1)):
        back_m1 = np.zeros(n)
    
    # Check if m2 data exists (not present in DMD density feedback)
    has_m2 = ('show_ODs', 'PTAI_m2_cnt_rel_atoms_back') in df.columns
    if has_m2:
        try:
            back_m2 = df[('show_ODs', 'PTAI_m2_cnt_rel_atoms_back')].values
            # Replace NaN with zeros (DMD feedback has m2 columns but they're all NaN)
            back_m2 = np.nan_to_num(back_m2, nan=0.0)
        except (KeyError, TypeError):
            back_m2 = np.zeros(n)
    else:
        back_m2 = np.zeros(n)
    
    back_probe = back_m1  # Use same as m1
    
    n_m1_candidates = [
        ('show_ODs', 'PTAI_m1_N_atoms'),
        ('show_ODs', 'PTAI_m1_SVD_N_atoms'),
    ]
    n_m1 = None
    for col in n_m1_candidates:
        if col in df.columns:
            temp = df[col].values
            if not np.all(np.isnan(temp)):
                n_m1 = temp
                break
    if n_m1 is None or np.all(np.isnan(n_m1)):
        n_m1 = np.ones(n) * 1e6  # Set to large value so ntot check passes
    
    if has_m2:
        try:
            n_m2 = df[('show_ODs', 'PTAI_m2_N_atoms')].values
            # Replace NaN with zeros (DMD feedback has m2 columns but they're all NaN)
            n_m2 = np.nan_to_num(n_m2, nan=0.0)
        except (KeyError, TypeError):
            n_m2 = np.zeros(n)
    else:
        n_m2 = np.zeros(n)

    # Calculate max background and total N for each shot
    back_max = np.maximum(np.maximum(back_m1, back_m2), back_probe)
    n_tot = n_m1 + n_m2

    good = np.zeros(n, dtype=bool)
    reject_reasons = {'background': 0, 'ntot': 0, 'exception': 0}
    
    for i in range(n):
        try:
            back_ok = (back_m1[i] < back_threshold) and (back_m2[i] < back_threshold) and (back_probe[i] < back_threshold)
            ntot_ok = (n_m1[i] + n_m2[i]) > ntot_threshold
            good[i] = back_ok and ntot_ok
            
            if not good[i]:
                if not back_ok:
                    reject_reasons['background'] += 1
                if not ntot_ok:
                    reject_reasons['ntot'] += 1
        except Exception:
            good[i] = False
            reject_reasons['exception'] += 1
    
    n_valid = np.count_nonzero(good)
    
    return good


def unpack_images(df, m1_lbl, m2_lbl):
    m1_raw = df[m1_lbl].values
    if len(m1_raw) == 0:
        return None, None, 0

    dim = m1_raw[0].shape[0]
    m1 = np.zeros((len(m1_raw), dim))
    
    # Check if m2 data exists (not present in DMD density feedback)
    has_m2 = m2_lbl in df.columns
    if has_m2:
        m2_raw = df[m2_lbl].values
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


def process_magnetization(m1, m2, sigma_avg, sigma_fluct, autocorr_center, autocorr_window):
    m1 = np.clip(m1, 0, None)
    m2 = np.clip(m2, 0, None)
    d = m1 + m2
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (m2 - m1) / d
    m = np.nan_to_num(np.clip(m, -1, 1), nan=0.0)

    local_avg = gaussian_filter1d(m, sigma=sigma_avg, axis=1)
    pw = m - local_avg
    local_fluct = gaussian_filter1d(np.abs(pw), sigma=sigma_fluct, axis=1)

    c = autocorr_center
    w = autocorr_window
    corr = np.zeros((m.shape[0], 2 * w))
    for i in range(m.shape[0]):
        seg = local_fluct[i, c - w:c + w]
        if len(seg) > 0:
            corr[i, :] = autocorr(seg)

    return m, d, local_fluct, corr


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


def compute_fluctuation_sigma2_waterfall(M_shots, y_raw, sigma_fluct, sigma_avg):
    """
    Compute per-scan-point averaged local fluctuation profile:
      1) average magnetization profile over shots in each scan value
      2) compute shot residuals to this average
      3) compute local sigma^2 profile from residuals (smoothed delta^2)
      4) average sigma^2 over shots at the same scan value
    """
    y_group_vals, groups = _build_group_indices(y_raw)
    if len(groups) == 0:
        return y_group_vals, np.zeros((0, 0))

    n_x = M_shots.shape[1]
    sigma2_map = np.zeros((len(groups), n_x))

    for i, idx in enumerate(groups):
        if len(idx) == 0:
            continue

        m_group = M_shots[idx]
        if len(idx) >= 2:
            # Ensemble fluctuations at fixed scan value: m - <m>
            m_avg = np.mean(m_group, axis=0)
            delta = m_group - m_avg[None, :]
            local_sigma2_each = gaussian_filter1d(delta ** 2, sigma=sigma_fluct, axis=1)
        else:
            # Single-shot fallback: use spatial fluctuations around local trend,
            # otherwise m - <m> is identically zero for one sample.
            local_trend = gaussian_filter1d(m_group, sigma=sigma_avg, axis=1)
            delta = m_group - local_trend
            local_sigma2_each = gaussian_filter1d(delta ** 2, sigma=sigma_fluct, axis=1)

        sigma2_map[i] = np.mean(local_sigma2_each, axis=0)

    return y_group_vals, sigma2_map


def plot_fluctuations_waterfall(sigma2_map, y_vals, dy, y_axis_label, title, params, um_per_px):
    """Plot local fluctuation sigma^2 as a waterfall map."""
    if sigma2_map.size == 0 or len(y_vals) == 0:
        return

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

    ax.axvline(x=x_min_um, color='white', linestyle='--', alpha=0.8)
    ax.axvline(x=x_max_um, color='white', linestyle='--', alpha=0.8)
    ax.set_xlim(x_plot_um[0], x_plot_um[-1])
    ax.set_ylim(np.min(y_vals) - dy, np.max(y_vals) + dy)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(r'$\mu m$')
    ax.set_title(title)
    plt.colorbar(coll, ax=ax, label=r'Local $\sigma^2$ of $m - \langle m \rangle$')


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

    polys_M, colors_M = [], []
    for i in range(len(y_unique)):
        for j in range(M.shape[1]):
            polys_M.append([
                (x_plot_um[j], y_unique[i] - dy / 2),
                (x_plot_um[j + 1], y_unique[i] - dy / 2),
                (x_plot_um[j + 1], y_unique[i] + dy / 2),
                (x_plot_um[j], y_unique[i] + dy / 2),
            ])
            colors_M.append(M[i, j])

    collection_M = PolyCollection(polys_M, array=np.array(colors_M), cmap='RdBu')
    if params['WATERFALL_MAG_CLIM'] is not None:
        collection_M.set_clim(*params['WATERFALL_MAG_CLIM'])
    ax_m.add_collection(collection_M)

    polys_D, colors_D = [], []
    for i in range(len(y_unique)):
        for j in range(D.shape[1]):
            polys_D.append([
                (x_plot_um[j], y_unique[i] - dy / 2),
                (x_plot_um[j + 1], y_unique[i] - dy / 2),
                (x_plot_um[j + 1], y_unique[i] + dy / 2),
                (x_plot_um[j], y_unique[i] + dy / 2),
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

    for axis in [ax_m, ax_d]:
        axis.axvline(x=x_min_um, color='black', linestyle='--')
        axis.axvline(x=x_max_um, color='black', linestyle='--')

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
    if plot_flags.get('sectioned_sigmoid', False) and scan in ('ARP_Forward', 'ARP_Backward') and xmax_ind > xmin_ind:
        section_centers_x, sigmoid_centers_y, sigmoid_centers_err, _ = _sectioned_sigmoid_analysis(
            M, y_unique, x_plot_um, xmin_ind, xmax_ind, params['NUM_SECTIONS']
        )
        if len(section_centers_x) > 0:
            ax_m.scatter(section_centers_x, sigmoid_centers_y, c='magenta', s=70, edgecolors='white', linewidths=1)
            ax_m.plot(section_centers_x, sigmoid_centers_y, 'w--', linewidth=2, alpha=0.7)
            
            ax_m.legend(['Sigmoid centers'], loc='upper right', fontsize=10)

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
            
            ax_sig.set_xlabel(r'$\mu m$')
            ax_sig.set_ylabel(f'{y_axis_label} (sigmoid center)')
            if average:
                ax_sig.set_title('Average sigmoid center vs x')
            else:
                ax_sig.set_title('Sigmoid center vs x')
            ax_sig.legend(loc='best')
            ax_sig.grid(True, alpha=0.3)

    ax_m.set_ylabel(y_axis_label)
    ax_m.set_xlabel(r'$\mu m$')
    ax_m.set_xlim(x_plot_um[0], x_plot_um[-1])
    ax_m.set_ylim(np.min(y_unique) - dy, np.max(y_unique) + dy)
    ax_m.set_title(title, fontsize=10)

    # Overlay detected defect positions for KZ_defect_analysis.
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
        dx_c, dy_c = defect_points_corrected
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
        dx_p, dy_p = defect_points_pos
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
        dx_n, dy_n = defect_points_neg
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
        ax_density.axvline(x=x_min_um, color='red', linestyle='--', linewidth=2)
        ax_density.axvline(x=x_max_um, color='red', linestyle='--', linewidth=2)
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
        ax_mag.axvline(x=x_min_um, color='blue', linestyle='--', linewidth=2)
        ax_mag.axvline(x=x_max_um, color='blue', linestyle='--', linewidth=2)

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


def plot_evolution_analysis(y_plot, y_axis_label, z_local_fluctuations, central_autocorr, M, params, um_per_px):
    w = params['AUTOCORR_WINDOW']

    fig_fluct, ax_fluct = plt.subplots()
    z_fluc_smooth = gaussian_filter1d(z_local_fluctuations, sigma=2, axis=0)
    start_idx, end_idx = 950, 1220
    x_axis_mesh = np.linspace(930 * um_per_px, 1240 * um_per_px, z_fluc_smooth[:, start_idx:end_idx].shape[1])
    im_fluct = ax_fluct.pcolormesh(x_axis_mesh, y_plot, z_fluc_smooth[:, start_idx:end_idx], cmap='inferno')
    ax_fluct.set_title('Local z fluctuations')
    ax_fluct.set_ylabel(y_axis_label)
    ax_fluct.set_xlabel(r'$\mu m$')
    plt.colorbar(im_fluct, ax=ax_fluct)

    fig_corr, ax_corr = plt.subplots()
    central_corr_smooth = gaussian_filter1d(central_autocorr, sigma=2, axis=0)
    im_corr = ax_corr.pcolormesh(np.linspace(0, 2 * w - 1, 2 * w), y_plot, central_corr_smooth, vmin=-1.0, vmax=1.0, cmap='viridis')
    plt.colorbar(im_corr, ax=ax_corr)

    fig_evol, ax_evol = plt.subplots()
    m_copy = np.copy(M)[:, 920:1240] ** 2
    m_copy = gaussian_filter1d(m_copy, sigma=2, axis=1)
    x_evol_mesh = np.linspace(920 * um_per_px, 1240 * um_per_px, m_copy.shape[1])
    im_evol = ax_evol.pcolormesh(x_evol_mesh, y_plot, m_copy, vmin=0, vmax=0.001, cmap='viridis')
    ax_evol.set_ylabel(y_axis_label)
    ax_evol.set_xlabel(r'$\mu m$')
    ax_evol.set_title('Magnetization evolution in the bubble region')
    plt.colorbar(im_evol, ax=ax_evol)


def waterfall_plot(df, seqs, scan, data_origin='show_ODs', constraints=None, average=False, title_labels=None, globals_in_title=None, params=None, plot_flags=None, mode_cfg=None):
    if params is None:
        raise ValueError('params dict is required')
    if plot_flags is None:
        plot_flags = {}
    if globals_in_title is None:
        globals_in_title = []

    um_per_px = get_um_per_px(params)

    df_subset = filter_dataframe(df, seqs, constraints)
    if scan == 'ARPF_feedback' or scan == 'DMD_density_feedback':
        df_subset = filter_valid_feedback_shots(df_subset)

    y_axis_col, title_base = get_scan_config(scan)
    y_axis_label = 'shot index' if y_axis_col == '_shot_counter' else y_axis_col
    print(f"DEBUG: Using y_axis_col = '{y_axis_col}' for scan '{scan}'")
    m1_lbl = (data_origin, 'm1_1d')
    m2_lbl = (data_origin, 'm2_1d')

    sorted_df = df_subset.sort_values(y_axis_col)
    good_mask = get_valid_rows_mask(sorted_df, params['BACK_CHECK_THRESHOLD'], params['NTOT_CHECK_THRESHOLD'])
    sorted_df = sorted_df[good_mask]

    if sorted_df.empty:
        print('No valid data remaining after filtering.')
        return

    y_raw = sorted_df[y_axis_col].values
    grouped_y, grouped_indices = _build_group_indices(y_raw)

    if scan == 'KZ_det_scan':
        print('KZ_det_scan: shots used after filtering per det:')
        for det_val, idx in zip(grouped_y, grouped_indices):
            try:
                det_txt = f"{float(det_val):.6g}"
            except Exception:
                det_txt = str(det_val)
            print(f"  det={det_txt}: {len(idx)} shots")

    m1_unpacked, m2_unpacked, img_dims = unpack_images(sorted_df, m1_lbl, m2_lbl)
    if m1_unpacked is None:
        return

    M_full, D_full, z_fluc_full, corr_full = process_magnetization(
        m1_unpacked,
        m2_unpacked,
        params['SIGMA_Z_LOCAL_AVG'],
        params['SIGMA_Z_LOCAL_FLUCT'],
        params['AUTOCORR_CENTER'],
        params['AUTOCORR_WINDOW'],
    )

    rep_indices = None
    if average:
        if len(grouped_indices) == 0:
            print('No groups found for averaging.')
            return

        m1_final = np.zeros((len(grouped_indices), img_dims))
        m2_final = np.zeros((len(grouped_indices), img_dims))
        z_fluc_final = np.zeros((len(grouped_indices), z_fluc_full.shape[1]))
        corr_final = np.zeros((len(grouped_indices), corr_full.shape[1]))
        for i, idx in enumerate(grouped_indices):
            m1_final[i] = np.mean(m1_unpacked[idx], axis=0)
            m2_final[i] = np.mean(m2_unpacked[idx], axis=0)
            z_fluc_final[i] = np.mean(z_fluc_full[idx], axis=0)
            corr_final[i] = np.mean(corr_full[idx], axis=0)
        M_final, D_final, _, _ = process_magnetization(
            m1_final,
            m2_final,
            params['SIGMA_Z_LOCAL_AVG'],
            params['SIGMA_Z_LOCAL_FLUCT'],
            params['AUTOCORR_CENTER'],
            params['AUTOCORR_WINDOW'],
        )
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
        corr_final = corr_full[rep_indices]
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
        # Optional per-mode settings live in waterfall_config MODE_CONFIGS['KZ_defect_analysis']['defect_analysis']
        # with keys for signed-derivative peak detection.
        gauss_sigma = 2.0
        deriv_threshold = 0.09
        deriv_threshold_pos = 0.09
        deriv_threshold_neg = -0.09
        threshold_scan = False
        threshold_scan_min = 0.02
        threshold_scan_max = 0.20
        threshold_scan_pos_min = 0.02
        threshold_scan_pos_max = 0.20
        threshold_scan_neg_min = -0.20
        threshold_scan_neg_max = -0.02
        threshold_scan_points = 50
        target_avg_peaks = 1.0
        threshold_scan_first_n_sequences = 3
        plateau_delta_defects = 0.05
        plateau_min_points = 3
        single_rep_shot_number = 3
        zero_crossing_min_distance_um = 5.0
        peak_step_filter_enabled = False
        peak_step_filter_window_px = 4
        peak_step_filter_min_abs_delta_m = 0.3
        min_same_sign_peak_distance_um = 4.0
        defect_position_hist_half_window_um = 2.0
        missed_defect_correction_method = 'zero_crossing'
        try:
            defect_cfg = mode_cfg.get('defect_analysis', {}) if isinstance(mode_cfg, dict) else {}
            gauss_sigma = float(defect_cfg.get('gaussian_sigma', gauss_sigma))
            deriv_threshold = float(defect_cfg.get('derivative_threshold', deriv_threshold))
            deriv_threshold_pos = float(defect_cfg.get('derivative_threshold_pos', deriv_threshold))
            deriv_threshold_neg = float(defect_cfg.get('derivative_threshold_neg', -abs(deriv_threshold_pos)))
            threshold_scan = bool(defect_cfg.get('threshold_scan', threshold_scan))
            threshold_scan_min = float(defect_cfg.get('threshold_scan_min', threshold_scan_min))
            threshold_scan_max = float(defect_cfg.get('threshold_scan_max', threshold_scan_max))
            threshold_scan_pos_min = float(defect_cfg.get('threshold_scan_pos_min', threshold_scan_min))
            threshold_scan_pos_max = float(defect_cfg.get('threshold_scan_pos_max', threshold_scan_max))
            threshold_scan_neg_min = float(defect_cfg.get('threshold_scan_neg_min', -threshold_scan_pos_max))
            threshold_scan_neg_max = float(defect_cfg.get('threshold_scan_neg_max', -threshold_scan_pos_min))
            threshold_scan_points = int(defect_cfg.get('threshold_scan_points', threshold_scan_points))
            target_avg_peaks = float(defect_cfg.get('target_avg_peaks', target_avg_peaks))
            threshold_scan_first_n_sequences = int(defect_cfg.get('threshold_scan_first_n_sequences', threshold_scan_first_n_sequences))
            plateau_delta_defects = float(defect_cfg.get('plateau_delta_defects', plateau_delta_defects))
            plateau_min_points = int(defect_cfg.get('plateau_min_points', plateau_min_points))
            single_rep_shot_number = int(defect_cfg.get('single_rep_shot_number', single_rep_shot_number))
            zero_crossing_min_distance_um = float(defect_cfg.get('zero_crossing_min_distance_um', zero_crossing_min_distance_um))
            peak_step_filter_enabled = bool(defect_cfg.get('peak_step_filter_enabled', peak_step_filter_enabled))
            peak_step_filter_window_px = int(defect_cfg.get('peak_step_filter_window_px', peak_step_filter_window_px))
            peak_step_filter_min_abs_delta_m = float(defect_cfg.get('peak_step_filter_min_abs_delta_m', peak_step_filter_min_abs_delta_m))
            min_same_sign_peak_distance_um = float(defect_cfg.get('min_same_sign_peak_distance_um', min_same_sign_peak_distance_um))
            defect_position_hist_half_window_um = float(defect_cfg.get('defect_position_hist_half_window_um', defect_position_hist_half_window_um))
            missed_defect_correction_method = str(
                defect_cfg.get('missed_defect_correction_method', missed_defect_correction_method)
            ).strip().lower()
        except Exception:
            pass
        peak_step_filter_window_px = max(1, int(peak_step_filter_window_px))
        peak_step_filter_min_abs_delta_m = max(0.0, float(peak_step_filter_min_abs_delta_m))
        min_same_sign_peak_distance_um = max(0.0, float(min_same_sign_peak_distance_um))
        defect_position_hist_half_window_um = max(0.0, float(defect_position_hist_half_window_um))
        if missed_defect_correction_method not in ('zero_crossing', 'opposite_peak', 'both'):
            missed_defect_correction_method = 'zero_crossing'

        x_centers_um = np.arange(M_final.shape[1]) * um_per_px
        x_min_um = float(params['X_MIN_INTEGRATION'])
        x_max_um = float(params['X_MAX_INTEGRATION'])
        xmin_ind = int(np.argmin(np.abs(x_centers_um - x_min_um)))
        xmax_ind = int(np.argmin(np.abs(x_centers_um - x_max_um)))
        if xmax_ind < xmin_ind:
            xmin_ind, xmax_ind = xmax_ind, xmin_ind

        defect_x, defect_y = [], []
        defect_x_corr, defect_y_corr = [], []
        per_shot_counts = []
        d_rois = []
        m_rois = []

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

        def _find_zero_crossing_x(m_roi, x_roi_um, i_start, i_end):
            if m_roi.size < 2:
                return None
            i0 = int(max(0, min(i_start, i_end)))
            i1 = int(min(m_roi.size - 1, max(i_start, i_end)))
            if i1 <= i0:
                return None

            # Search sign changes in [i0, i1]
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

        def _find_opposite_peak_x(d_roi, x_roi_um, sign_run, i_start, i_end):
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

        def _passes_peak_step_filter(m_roi, peak_idx, window_px, min_abs_delta):
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

        for i, yv in enumerate(y_final):
            m_sig = M_final[i]
            if gauss_sigma > 0:
                m_sig = gaussian_filter1d(m_sig, sigma=gauss_sigma)

            d_sig = np.gradient(m_sig)
            if d_sig.size < 3 or xmax_ind - xmin_ind < 2:
                d_rois.append(np.array([], dtype=float))
                m_rois.append(np.array([], dtype=float))
                continue

            # Store derivative signal inside configured integration window (in um).
            d_roi = d_sig[xmin_ind:xmax_ind + 1]
            m_roi = m_sig[xmin_ind:xmax_ind + 1]
            d_rois.append(d_roi)
            m_rois.append(m_roi)
        try:
            if rep_indices is not None:
                defect_sequences_for_plot = sorted_df['rep'].values[rep_indices]
                defect_reps_for_plot = sorted_df['rep'].values[rep_indices]
            else:
                defect_sequences_for_plot = sorted_df['rep'].values
                defect_reps_for_plot = sorted_df['rep'].values
        except Exception:
            defect_sequences_for_plot = None
            defect_reps_for_plot = None
        rep_opt_threshold_pos_by_rep = {}
        rep_opt_threshold_neg_by_rep = {}
        rep_opt_plateau_bounds_pos_by_rep = {}
        rep_opt_plateau_bounds_neg_by_rep = {}
        rep_opt_selection_mode_pos_by_rep = {}
        rep_opt_selection_mode_neg_by_rep = {}

        def _suggest_threshold_from_plateau(thr_values, avg_curve, current_thr, delta_tol, min_points):
            """Choose threshold from the flattest (plateau) region of avg defects vs threshold."""
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
                end_idx = j + 1  # edge run [i..j] maps to point run [i..j+1]
                width_pts = end_idx - start_idx + 1
                if width_pts >= min_points:
                    mid_idx = (start_idx + end_idx) // 2
                    dist_to_current = abs(float(thr_values[mid_idx]) - float(current_thr))
                    candidates.append((width_pts, -dist_to_current, start_idx, end_idx, mid_idx))
                i = j + 1

            if len(candidates) > 0:
                # widest plateau first, tie-breaker: closest to current threshold
                candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
                _, _, start_idx, end_idx, mid_idx = candidates[0]
                return (
                    float(thr_values[mid_idx]),
                    (float(thr_values[start_idx]), float(thr_values[end_idx])),
                    'plateau',
                )

            # Fallback: pick point next to minimum local change
            min_diff_idx = int(np.argmin(diffs))
            mid_idx = min_diff_idx + 1
            start_idx = min_diff_idx
            end_idx = min_diff_idx + 1
            return (
                float(thr_values[mid_idx]),
                (float(thr_values[start_idx]), float(thr_values[end_idx])),
                'fallback-min-diff',
            )

        # Initialize threshold arrays (used conditionally below)
        thr_vals_pos = np.array([], dtype=float)
        thr_vals_neg = np.array([], dtype=float)
        
        # Only perform threshold scan if explicitly enabled
        if threshold_scan:
            if threshold_scan_points < 2:
                threshold_scan_points = 2
            if threshold_scan_pos_max < threshold_scan_pos_min:
                threshold_scan_pos_min, threshold_scan_pos_max = threshold_scan_pos_max, threshold_scan_pos_min
            if threshold_scan_neg_max < threshold_scan_neg_min:
                threshold_scan_neg_min, threshold_scan_neg_max = threshold_scan_neg_max, threshold_scan_neg_min
            thr_vals_pos = np.linspace(threshold_scan_pos_min, threshold_scan_pos_max, threshold_scan_points)
            thr_vals_neg = np.linspace(threshold_scan_neg_min, threshold_scan_neg_max, threshold_scan_points)

            # Analyze optimal thresholds for each repetition (rep): one for positive peaks, one for negative peaks.
            if defect_reps_for_plot is not None and len(defect_reps_for_plot) == len(d_rois):
                rep_vals_all = np.asarray(defect_reps_for_plot)
                for rep_val in np.sort(np.unique(rep_vals_all)):
                    idx_rep = np.where(rep_vals_all == rep_val)[0]
                    d_rois_rep = [d_rois[i] for i in idx_rep]
                    if len(d_rois_rep) == 0:
                        continue

                    avg_counts_pos_rep, avg_counts_neg_rep = [], []
                    for thr_pos in thr_vals_pos:
                        c_pos = [len(_count_peak_indices_pos(d_roi, thr_pos)) for d_roi in d_rois_rep]
                        avg_counts_pos_rep.append(float(np.mean(c_pos)) if len(c_pos) > 0 else 0.0)
                    for thr_neg in thr_vals_neg:
                        c_neg = [len(_count_peak_indices_neg(d_roi, thr_neg)) for d_roi in d_rois_rep]
                        avg_counts_neg_rep.append(float(np.mean(c_neg)) if len(c_neg) > 0 else 0.0)
                    avg_counts_pos_rep = np.asarray(avg_counts_pos_rep, dtype=float)
                    avg_counts_neg_rep = np.asarray(avg_counts_neg_rep, dtype=float)

                    suggested_thr_pos_rep, plateau_bounds_pos_rep, selection_mode_pos_rep = _suggest_threshold_from_plateau(
                        thr_vals_pos,
                        avg_counts_pos_rep,
                        deriv_threshold_pos,
                        plateau_delta_defects,
                        plateau_min_points,
                    )
                    suggested_thr_neg_rep, plateau_bounds_neg_rep, selection_mode_neg_rep = _suggest_threshold_from_plateau(
                        thr_vals_neg,
                        avg_counts_neg_rep,
                        deriv_threshold_neg,
                        plateau_delta_defects,
                        plateau_min_points,
                    )

                    rep_opt_threshold_pos_by_rep[rep_val] = float(suggested_thr_pos_rep)
                    rep_opt_threshold_neg_by_rep[rep_val] = float(suggested_thr_neg_rep)
                    rep_opt_plateau_bounds_pos_by_rep[rep_val] = plateau_bounds_pos_rep
                    rep_opt_plateau_bounds_neg_by_rep[rep_val] = plateau_bounds_neg_rep
                    rep_opt_selection_mode_pos_by_rep[rep_val] = selection_mode_pos_rep
                    rep_opt_selection_mode_neg_by_rep[rep_val] = selection_mode_neg_rep

                    bi_pos = int(np.argmin(np.abs(thr_vals_pos - suggested_thr_pos_rep)))
                    bi_neg = int(np.argmin(np.abs(thr_vals_neg - suggested_thr_neg_rep)))
                    print(
                        f"KZ rep {int(rep_val)} optimal thresholds: pos={suggested_thr_pos_rep:.4f} "
                        f"({selection_mode_pos_rep}, avg+={avg_counts_pos_rep[bi_pos]:.2f}), "
                        f"neg={suggested_thr_neg_rep:.4f} "
                        f"({selection_mode_neg_rep}, avg-={avg_counts_neg_rep[bi_neg]:.2f})"
                    )

        # Recount defects using optimal positive/negative thresholds per rep.
        per_shot_thresholds = []
        per_shot_counts = []
        defect_x, defect_y = [], []
        defect_x_pos, defect_y_pos = [], []
        defect_x_neg, defect_y_neg = [], []
        defect_x_corr, defect_y_corr = [], []
        defect_x_rej, defect_y_rej = [], []
        corrected_total = 0
        rejected_total = 0
        x_roi_um = x_centers_um[xmin_ind:xmax_ind + 1]
        rep_vals_for_count = np.asarray(defect_reps_for_plot) if defect_reps_for_plot is not None and len(defect_reps_for_plot) == len(d_rois) else None
        for i, yv in enumerate(y_final):
            d_roi = d_rois[i] if i < len(d_rois) else np.array([], dtype=float)
            m_roi = m_rois[i] if i < len(m_rois) else np.array([], dtype=float)
            if rep_vals_for_count is not None:
                rep_val_i = rep_vals_for_count[i]
                thr_pos_i = float(rep_opt_threshold_pos_by_rep.get(rep_val_i, deriv_threshold_pos))
                thr_neg_i = float(rep_opt_threshold_neg_by_rep.get(rep_val_i, deriv_threshold_neg))
            else:
                thr_pos_i = float(deriv_threshold_pos)
                thr_neg_i = float(deriv_threshold_neg)

            per_shot_thresholds.append(f"+{thr_pos_i:.3f}/{thr_neg_i:.3f}")
            if d_roi.size < 3:
                per_shot_counts.append(0)
                continue

            peak_pos_raw = _count_peak_indices_pos(d_roi, thr_pos_i)
            peak_neg_raw = _count_peak_indices_neg(d_roi, thr_neg_i)
            peak_pos_roi = []
            peak_neg_roi = []
            rejected_idx = []
            for idxp in peak_pos_raw:
                if peak_step_filter_enabled and not _passes_peak_step_filter(m_roi, idxp, peak_step_filter_window_px, peak_step_filter_min_abs_delta_m):
                    rejected_idx.append(int(idxp))
                else:
                    peak_pos_roi.append(int(idxp))
            for idxn in peak_neg_raw:
                if peak_step_filter_enabled and not _passes_peak_step_filter(m_roi, idxn, peak_step_filter_window_px, peak_step_filter_min_abs_delta_m):
                    rejected_idx.append(int(idxn))
                else:
                    peak_neg_roi.append(int(idxn))
            peak_pos_roi, peak_pos_x = _merge_close_same_sign_peaks(
                peak_pos_roi,
                x_roi_um,
                min_same_sign_peak_distance_um,
            )
            peak_neg_roi, peak_neg_x = _merge_close_same_sign_peaks(
                peak_neg_roi,
                x_roi_um,
                min_same_sign_peak_distance_um,
            )
            peak_thr_x = np.concatenate([peak_pos_x, peak_neg_x]).astype(float)
            events = []
            for idxp in peak_pos_roi:
                events.append((int(idxp), 'pos'))
            for idxn in peak_neg_roi:
                events.append((int(idxn), 'neg'))
            events.sort(key=lambda e: e[0])

            # Keep ALL threshold-selected peaks (user-requested behavior).
            kept_thr_x = np.concatenate([peak_pos_x, peak_neg_x]).astype(float)
            corrected_x_shot = []
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
                    # This guarantees correction is local to regions bracketed by
                    # two detected peaks of the same sign.
                    for p in range(len(run) - 1):
                        i_start = int(run[p][0])
                        i_end = int(run[p + 1][0])
                        sign_pair = run[p][1]
                        if missed_defect_correction_method in ('zero_crossing', 'both'):
                            zx = _find_zero_crossing_x(m_roi, x_roi_um, i_start, i_end)
                            if zx is not None:
                                candidates.append(float(zx))
                        if missed_defect_correction_method in ('opposite_peak', 'both'):
                            ox = _find_opposite_peak_x(
                                d_roi,
                                x_roi_um,
                                sign_pair,
                                i_start,
                                i_end,
                            )
                            if ox is not None:
                                candidates.append(float(ox))

                    for cx in candidates:
                        if peak_thr_x.size == 0:
                            corrected_x_shot.append(float(cx))
                            continue
                        min_dist = float(np.min(np.abs(peak_thr_x - float(cx))))
                        if min_dist >= zero_crossing_min_distance_um:
                            corrected_x_shot.append(float(cx))
                j = k

            if len(corrected_x_shot) > 1:
                corrected_x_shot = list(np.unique(np.round(np.asarray(corrected_x_shot, dtype=float), 6)))

            per_shot_counts.append(int(len(kept_thr_x) + len(corrected_x_shot)))

            # Keep yellow markers for all threshold defects.
            for x_thr in kept_thr_x:
                defect_x.append(float(x_thr))
                defect_y.append(yv)

            # Add signed guide markers.
            for x_p in peak_pos_x:
                x_p = float(x_p)
                defect_x_pos.append(x_p)
                defect_y_pos.append(yv)
            for x_n in peak_neg_x:
                x_n = float(x_n)
                defect_x_neg.append(x_n)
                defect_y_neg.append(yv)
            for idxr in rejected_idx:
                defect_x_rej.append(float(x_roi_um[int(idxr)]))
                defect_y_rej.append(yv)
            rejected_total += len(rejected_idx)

            for xcorr in corrected_x_shot:
                defect_x_corr.append(float(xcorr))
                defect_y_corr.append(yv)
            corrected_total += len(corrected_x_shot)

        defect_points = (np.asarray(defect_x), np.asarray(defect_y))
        defect_points_pos = (np.asarray(defect_x_pos), np.asarray(defect_y_pos))
        defect_points_neg = (np.asarray(defect_x_neg), np.asarray(defect_y_neg))
        defect_points_corrected = (np.asarray(defect_x_corr), np.asarray(defect_y_corr))
        defect_points_rejected = (np.asarray(defect_x_rej), np.asarray(defect_y_rej))
        defect_counts_for_plot = np.asarray(per_shot_counts, dtype=int)
        defect_thresholds_for_plot = np.asarray(per_shot_thresholds, dtype=object)

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
            f"per-rep-optimal={'on' if len(rep_opt_threshold_pos_by_rep) > 0 else 'off'}, "
            f"missed_defect_method={missed_defect_correction_method}, "
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
            n_bins_size = max(3, int(np.sqrt(len(all_defect_sizes))))
            ax_hs.hist(all_defect_sizes, bins=n_bins_size, color='tab:green', alpha=0.75, edgecolor='black')
            ax_hs.set_xlabel(r'Defect size ($\mu m$, distance pos-to-neg)')
            ax_hs.set_ylabel('Count')
            mean_size = float(np.mean(all_defect_sizes))
            std_size = float(np.std(all_defect_sizes))
            ax_hs.set_title(f'KZ_defect_analysis: defect-size distribution (seqs: {seqs})\nmean={mean_size:.2f} ± {std_size:.2f} µm')
            ax_hs.grid(True, alpha=0.25)
            print(f"Defect sizes: mean={mean_size:.3f} µm, std={std_size:.3f} µm, median={float(np.median(all_defect_sizes)):.3f} µm, N={len(all_defect_sizes)}")

        if threshold_scan and len(d_rois) > 0:
            # Plot this process in the selected single-repetition analyses.
            if defect_reps_for_plot is not None and len(defect_reps_for_plot) == len(d_rois):
                try:
                    rep_vals = np.asarray(defect_reps_for_plot)
                    rep_unique = np.sort(np.unique(rep_vals))
                    n_reps = max(1, threshold_scan_first_n_sequences)
                    rep_selected = rep_unique[:n_reps]
                    print(f"\nPer-repetition (+/-) threshold analysis (first {len(rep_selected)} repetitions):")

                    for rep_val in rep_selected:
                        idx = np.where(rep_vals == rep_val)[0]
                        d_rois_rep = [d_rois[i] for i in idx]
                        if len(d_rois_rep) == 0:
                            continue

                        counts_per_rep_pos, counts_per_rep_neg = [], []
                        for thr_pos in thr_vals_pos:
                            counts_per_rep_pos.append([len(_count_peak_indices_pos(d_roi, thr_pos)) for d_roi in d_rois_rep])
                        for thr_neg in thr_vals_neg:
                            counts_per_rep_neg.append([len(_count_peak_indices_neg(d_roi, thr_neg)) for d_roi in d_rois_rep])

                        mat_pos = np.asarray(counts_per_rep_pos, dtype=float).T
                        mat_neg = np.asarray(counts_per_rep_neg, dtype=float).T
                        if mat_pos.ndim != 2 or mat_pos.shape[0] == 0 or mat_neg.ndim != 2 or mat_neg.shape[0] == 0:
                            continue

                        shot_number_used = max(1, single_rep_shot_number)
                        shot_idx_used = min(shot_number_used - 1, mat_pos.shape[0] - 1)
                        curve_pos = mat_pos[shot_idx_used]
                        curve_neg = mat_neg[shot_idx_used]

                        sug_pos = float(rep_opt_threshold_pos_by_rep.get(rep_val, deriv_threshold_pos))
                        sug_neg = float(rep_opt_threshold_neg_by_rep.get(rep_val, deriv_threshold_neg))
                        pb_pos = rep_opt_plateau_bounds_pos_by_rep.get(rep_val, None)
                        pb_neg = rep_opt_plateau_bounds_neg_by_rep.get(rep_val, None)
                        mode_pos = rep_opt_selection_mode_pos_by_rep.get(rep_val, 'fallback-current-threshold')
                        mode_neg = rep_opt_selection_mode_neg_by_rep.get(rep_val, 'fallback-current-threshold')

                        i_pos = int(np.argmin(np.abs(thr_vals_pos - sug_pos)))
                        i_neg = int(np.argmin(np.abs(thr_vals_neg - sug_neg)))

                        shot_global_idx = int(idx[shot_idx_used])
                        x_roi_um_local = x_centers_um[xmin_ind:xmax_ind + 1]

                        m_raw_sel = np.asarray(M_final[shot_global_idx], dtype=float)
                        if gauss_sigma > 0:
                            m_smooth_sel = gaussian_filter1d(m_raw_sel, sigma=gauss_sigma)
                        else:
                            m_smooth_sel = m_raw_sel.copy()
                        m_raw_roi_sel = m_raw_sel[xmin_ind:xmax_ind + 1]
                        m_smooth_roi_sel = m_smooth_sel[xmin_ind:xmax_ind + 1]
                        d_roi_sel = np.gradient(m_smooth_sel)[xmin_ind:xmax_ind + 1]

                        peak_pos_sel = _count_peak_indices_pos(d_roi_sel, sug_pos)
                        peak_neg_sel = _count_peak_indices_neg(d_roi_sel, sug_neg)

                        fig_rep, axs_rep = plt.subplots(nrows=3, figsize=(11.0, 9.0), tight_layout=True)
                        ax_map, ax_mag, ax_der = axs_rep

                        im = None
                        cbar_label = 'Magnetization m'
                        raw_row_idx = int(rep_indices[shot_global_idx]) if rep_indices is not None else int(shot_global_idx)
                        try:
                            row = sorted_df.iloc[raw_row_idx]
                            m1_2d = None
                            m2_2d = None
                            cam_candidates = [
                                (data_origin, 'PTAI_m1'),
                                ('show_ODs', 'PTAI_m1'),
                                (data_origin, 'm1_2d'),
                            ]
                            for c in cam_candidates:
                                if c in sorted_df.columns:
                                    arr = np.asarray(row[c])
                                    if arr.ndim == 2:
                                        m1_2d = arr.astype(float)
                                        break
                            cam2_candidates = [
                                (data_origin, 'PTAI_m2'),
                                ('show_ODs', 'PTAI_m2'),
                                (data_origin, 'm2_2d'),
                            ]
                            for c in cam2_candidates:
                                if c in sorted_df.columns:
                                    arr = np.asarray(row[c])
                                    if arr.ndim == 2:
                                        m2_2d = arr.astype(float)
                                        break

                            if m1_2d is not None and m2_2d is not None:
                                d2d = m1_2d + m2_2d
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    z2d = np.divide(m2_2d - m1_2d, d2d, out=np.zeros_like(d2d), where=d2d != 0)
                                im = ax_map.imshow(z2d, aspect='auto', cmap='RdBu', vmin=-1.0, vmax=1.0)
                                cbar_label = 'Magnetization m'
                                ax_map.set_title('Camera image of selected rep (2D magnetization)')
                            elif m1_2d is not None:
                                im = ax_map.imshow(m1_2d, aspect='auto', cmap='gist_stern')
                                cbar_label = 'OD (m1)'
                                ax_map.set_title('Camera image of selected rep (m1)')

                            if im is not None:
                                ax_map.axvline(float(xmin_ind), color='w', linestyle='--', linewidth=1.0, alpha=0.9)
                                ax_map.axvline(float(xmax_ind), color='w', linestyle='--', linewidth=1.0, alpha=0.9)
                                ax_map.set_ylabel('y (px)')
                        except Exception:
                            im = None

                        if im is None:
                            m_roi_stack = np.asarray(
                                [M_final[int(ii)][xmin_ind:xmax_ind + 1] for ii in idx],
                                dtype=float,
                            )
                            im = ax_map.imshow(
                                m_roi_stack,
                                aspect='auto',
                                cmap='RdBu',
                                vmin=-1.0,
                                vmax=1.0,
                                extent=[float(x_roi_um_local[0]), float(x_roi_um_local[-1]), float(len(idx)), 1.0],
                            )
                            ax_map.axhline(float(shot_idx_used + 1), color='k', linestyle='--', linewidth=1.0, alpha=0.8)
                            ax_map.set_ylabel('Shot in repetition')
                            ax_map.set_title('Relative magnetization colormap (KZ ROI, fallback)')
                        cbar = fig_rep.colorbar(im, ax=ax_map, fraction=0.025, pad=0.015)
                        cbar.set_label(cbar_label)

                        ax_mag.plot(x_roi_um_local, m_raw_roi_sel, color='0.35', linewidth=1.3, label='magnetization')
                        ax_mag.plot(
                            x_roi_um_local,
                            m_smooth_roi_sel,
                            color='k',
                            linewidth=1.5,
                            label=f'smoothed magnetization (σ={gauss_sigma:g})',
                        )
                        ax_mag.axhline(0.0, color='0.6', linestyle=':', linewidth=1.0)
                        ax_mag.set_ylabel('m')
                        ax_mag.set_title('Magnetization and smoothed signal')
                        ax_mag.grid(True, alpha=0.25)
                        ax_mag.legend(loc='best', fontsize=8)

                        ax_der.plot(x_roi_um_local, d_roi_sel, color='0.2', linewidth=1.2, label=r'$\partial m / \partial x$')
                        if peak_pos_sel.size > 0:
                            ax_der.scatter(
                                x_roi_um_local[peak_pos_sel],
                                d_roi_sel[peak_pos_sel],
                                color='tab:blue',
                                marker='x',
                                s=36,
                                linewidths=1.2,
                                label='positive peaks',
                            )
                        if peak_neg_sel.size > 0:
                            ax_der.scatter(
                                x_roi_um_local[peak_neg_sel],
                                d_roi_sel[peak_neg_sel],
                                color='tab:red',
                                marker='x',
                                s=36,
                                linewidths=1.2,
                                label='negative peaks',
                            )
                        ax_der.axhline(sug_pos, color='tab:blue', linestyle='--', linewidth=1.2, label=f'+ threshold = {sug_pos:.3f}')
                        ax_der.axhline(sug_neg, color='tab:red', linestyle='--', linewidth=1.2, label=f'- threshold = {sug_neg:.3f}')
                        ax_der.axhline(0.0, color='0.6', linestyle=':', linewidth=1.0)
                        ax_der.set_xlabel('x (µm)')
                        ax_der.set_ylabel(r'$\partial m / \partial x$')
                        ax_der.set_title('Derivative with detected peaks and thresholds')
                        ax_der.grid(True, alpha=0.25)
                        ax_der.legend(loc='best', fontsize=8)

                        fig_rep.suptitle(
                            f'KZ threshold diagnostics - Repetition {int(rep_val)} (shot {shot_idx_used + 1} of {len(d_rois_rep)})',
                            fontsize=10,
                        )

                        print(
                            f"  Rep {int(rep_val)}: shots={len(d_rois_rep)}, curve_shot={shot_idx_used + 1}, "
                            f"+thr={sug_pos:.4f} ({mode_pos}, N+={curve_pos[i_pos]:.1f}), "
                            f"-thr={sug_neg:.4f} ({mode_neg}, N-={curve_neg[i_neg]:.1f})"
                        )
                except Exception as err:
                    print(f"Warning: could not compute per-repetition +/- threshold scan: {err}")

    if scan == 'KZ_det_scan':
        # Plot defects/shot vs ARPKZ_final_set_field using the same logic and parameters as KZ_defect_analysis.
        gauss_sigma_det = 2.0
        deriv_threshold_det = 0.09
        deriv_thr_pos_det = 0.09
        deriv_thr_neg_det = -0.09
        threshold_scan_det = True
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
        missed_defect_correction_method_det = 'zero_crossing'

        # Primary source of criteria: MODE_CONFIGS['KZ_defect_analysis']['defect_analysis']
        # Optional per-mode override: current mode_cfg['defect_analysis']
        try:
            kz_ref_cfg = {}
            try:
                from . import waterfall_config as _wf_cfg  # type: ignore
            except Exception:
                import waterfall_config as _wf_cfg  # type: ignore
            kz_ref_cfg = (
                _wf_cfg.MODE_CONFIGS.get('KZ_defect_analysis', {}).get('defect_analysis', {})
                if hasattr(_wf_cfg, 'MODE_CONFIGS') else {}
            )

            det_cfg_override = mode_cfg.get('defect_analysis', {}) if isinstance(mode_cfg, dict) else {}
            det_cfg = dict(kz_ref_cfg)
            det_cfg.update(det_cfg_override)

            gauss_sigma_det = float(det_cfg.get('gaussian_sigma', gauss_sigma_det))
            deriv_threshold_det = float(det_cfg.get('derivative_threshold', deriv_threshold_det))
            deriv_thr_pos_det = float(det_cfg.get('derivative_threshold_pos', deriv_thr_pos_det))
            deriv_thr_neg_det = float(det_cfg.get('derivative_threshold_neg', -abs(deriv_thr_pos_det)))
            threshold_scan_det = bool(det_cfg.get('threshold_scan', threshold_scan_det))
            threshold_scan_min_det = float(det_cfg.get('threshold_scan_min', threshold_scan_min_det))
            threshold_scan_max_det = float(det_cfg.get('threshold_scan_max', threshold_scan_max_det))
            threshold_scan_pos_min_det = float(det_cfg.get('threshold_scan_pos_min', threshold_scan_pos_min_det))
            threshold_scan_pos_max_det = float(det_cfg.get('threshold_scan_pos_max', threshold_scan_pos_max_det))
            threshold_scan_neg_min_det = float(det_cfg.get('threshold_scan_neg_min', -threshold_scan_pos_max_det))
            threshold_scan_neg_max_det = float(det_cfg.get('threshold_scan_neg_max', -threshold_scan_pos_min_det))
            threshold_scan_points_det = int(det_cfg.get('threshold_scan_points', threshold_scan_points_det))
            plateau_delta_defects_det = float(det_cfg.get('plateau_delta_defects', plateau_delta_defects_det))
            plateau_min_points_det = int(det_cfg.get('plateau_min_points', plateau_min_points_det))
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
            missed_defect_correction_method_det = 'zero_crossing'

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

        def _domain_balance_in_used_region(m_roi, x_roi_um, defect_edges_x, x_left, x_right):
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
            det_domain_balance_raw.append(
                _domain_balance_in_used_region(
                    m_roi,
                    x_roi_um_det,
                    defect_edges_x_shot,
                    float(x_min_um_det),
                    float(x_max_um_det),
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

            # Interpolate det where domain balance crosses zero, then interpolate
            # defects/region at that det. Error is the average SEM of the two
            # bracketing points.
            valid_interp = np.isfinite(det_unique) & np.isfinite(det_dom_mean) & np.isfinite(det_mean)
            if np.count_nonzero(valid_interp) >= 2:
                x_det = det_unique[valid_interp]
                y_dom = det_dom_mean[valid_interp]
                y_dom_sem = det_dom_sem[valid_interp]
                y_def = det_mean[valid_interp]
                y_def_sem = det_sem[valid_interp]

                order = np.argsort(x_det)
                x_det = x_det[order]
                y_dom = y_dom[order]
                y_dom_sem = y_dom_sem[order]
                y_def = y_def[order]
                y_def_sem = y_def_sem[order]

                i_cross = None
                for ii in range(len(x_det) - 1):
                    y0 = float(y_dom[ii])
                    y1 = float(y_dom[ii + 1])
                    if y0 == 0.0 or y1 == 0.0 or (y0 * y1 < 0.0):
                        i_cross = ii
                        break

                if i_cross is not None:
                    x0 = float(x_det[i_cross])
                    x1 = float(x_det[i_cross + 1])
                    yd0 = float(y_dom[i_cross])
                    yd1 = float(y_dom[i_cross + 1])
                    yf0 = float(y_def[i_cross])
                    yf1 = float(y_def[i_cross + 1])

                    if yd1 == yd0:
                        det_zero = 0.5 * (x0 + x1)
                    else:
                        det_zero = float(x0 + (0.0 - yd0) * (x1 - x0) / (yd1 - yd0))

                    if x1 == x0:
                        def_at_zero = 0.5 * (yf0 + yf1)
                    else:
                        def_at_zero = float(yf0 + (det_zero - x0) * (yf1 - yf0) / (x1 - x0))

                    err_dom = 0.5 * (float(y_dom_sem[i_cross]) + float(y_dom_sem[i_cross + 1]))
                    err_def = 0.5 * (float(y_def_sem[i_cross]) + float(y_def_sem[i_cross + 1]))

                    print(
                        "KZ_det_scan interpolated domain_balance=0 crossing: "
                        f"det={det_zero:.6g}, "
                        f"domain_balance=0 ± {err_dom:.5f}, "
                        f"defects/region={def_at_zero:.5f} ± {err_def:.5f} um^-1 "
                        f"(between det={x0:.6g} and {x1:.6g})"
                    )
                else:
                    print("KZ_det_scan: no sign change in averaged domain balance; cannot interpolate domain_balance=0 crossing.")

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

            # Merge legends from both y-axes.
            h1, l1 = ax_det.get_legend_handles_labels()
            h2, l2 = ax_dom.get_legend_handles_labels()
            ax_det.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)

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
            im_raw = ax_raw.imshow(
                M_full,
                aspect='auto',
                cmap='RdBu',
                vmin=-1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][0],
                vmax=1.0 if params.get('WATERFALL_MAG_CLIM') is None else params['WATERFALL_MAG_CLIM'][1],
                extent=[float(x_centers_um_det[0]), float(x_centers_um_det[-1]), float(len(M_full)), 1.0],
            )
            ax_raw.set_xlabel(r'$\mu m$')
            ax_raw.set_ylabel('shot index (ordered by det)')
            ax_raw.set_title(f'Raw magnetization waterfall ordered by ARPKZ_final_set_field (seqs: {seqs})')
            ax_raw.set_xlim(float(x_centers_um_det[0]), float(x_centers_um_det[-1]))
            ax_raw.set_ylim(float(len(M_full)), 1.0)
            ax_raw.set_autoscale_on(False)
            ax_raw.axvline(x=float(x_min_um_det), color='k', linestyle='--', linewidth=1.0, alpha=0.7)
            ax_raw.axvline(x=float(x_max_um_det), color='k', linestyle='--', linewidth=1.0, alpha=0.7)

            if len(det_defect_x_pos) > 0:
                ax_raw.scatter(
                    np.asarray(det_defect_x_pos, dtype=float),
                    np.asarray(det_defect_y_pos, dtype=float),
                    c='blue',
                    s=16,
                    edgecolors='black',
                    linewidths=0.3,
                    zorder=10,
                    label='positive defects (+)',
                )
            if len(det_defect_x_neg) > 0:
                ax_raw.scatter(
                    np.asarray(det_defect_x_neg, dtype=float),
                    np.asarray(det_defect_y_neg, dtype=float),
                    c='red',
                    s=16,
                    edgecolors='black',
                    linewidths=0.3,
                    zorder=10,
                    label='negative defects (-)',
                )
            if len(det_defect_x_corr) > 0:
                ax_raw.scatter(
                    np.asarray(det_defect_x_corr, dtype=float),
                    np.asarray(det_defect_y_corr, dtype=float),
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

                    # Draw horizontal boundaries where this det block starts/ends.
                    ax_raw.axhline(y_start, color='white', linestyle='-', linewidth=0.9, alpha=0.75)
                    ax_raw.axhline(y_end, color='white', linestyle='-', linewidth=0.9, alpha=0.75)

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

    if plot_flags.get('evolution_plots', False) and scan == 'bubbles_evolution':
        plot_evolution_analysis(y_final, y_axis_col, z_fluc_final, corr_final, M_final, params, um_per_px)

    if plot_flags.get('main_waterfall', True):
        title_full = f"{title_base}\n{'AVERAGED' if average else 'UNIQUE'}\n(seqs: {seqs})"
        if scan == 'KZ_defect_analysis' and avg_defects is not None and stat_err is not None:
            title_full += f"\n⟨defects/shot⟩ = {avg_defects:.3f} ± {stat_err:.3f} (SEM)"
        try:
            seq_used = np.unique(sorted_df['sequence_index'].values)
            title_full += f"\nsequence_index used: {seq_used}"
        except Exception:
            pass

        g_lines = _build_globals_title_lines(sorted_df, globals_in_title)
        if len(g_lines) > 0:
            title_full += '\n' + '\n'.join(g_lines)

        if title_labels:
            title_full += '\n' + ', '.join([f"{l}: {np.unique(sorted_df[l].values)}" for l in title_labels])

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
        )

    if plot_flags.get('fluctuations_waterfall', False):
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
        )

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