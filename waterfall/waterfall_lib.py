import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    c = result[result.size // 2]
    if c == 0:
        return np.zeros_like(result[result.size // 2:])
    return result[result.size // 2:] / c


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
    if constraints is not None:
        for key, value in constraints.items():
            try:
                if not isinstance(value, (str, bytes)) and hasattr(value, '__iter__'):
                    out = out[out[key].isin(value)]
                else:
                    out = out[np.abs(out[key] - value) < 1e-3]
            except Exception:
                out = out[out[key] == value]
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
    
    # Debug: print available columns
    show_od_cols = [col for col in df.columns if isinstance(col, tuple) and col[0] == 'show_ODs']
    print(f"Available show_ODs columns: {[c[1] for c in show_od_cols[:10]]}{'...' if len(show_od_cols) > 10 else ''}")
    
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
                print(f"Using {col[1]} for m1 background, values: {temp[:3]}")
                back_m1 = temp
                break
    if back_m1 is None or np.all(np.isnan(back_m1)):
        print("Warning: Valid m1 background data not found, skipping background check")
        back_m1 = np.zeros(n)
    
    # Check if m2 data exists (not present in DMD density feedback)
    has_m2 = ('show_ODs', 'PTAI_m2_cnt_rel_atoms_back') in df.columns
    if has_m2:
        try:
            back_m2 = df[('show_ODs', 'PTAI_m2_cnt_rel_atoms_back')].values
            # Replace NaN with zeros (DMD feedback has m2 columns but they're all NaN)
            back_m2 = np.nan_to_num(back_m2, nan=0.0)
            print(f"m2 background found, values (NaN->0): {back_m2[:3]}")
        except (KeyError, TypeError):
            back_m2 = np.zeros(n)
            print("m2 background exception, using zeros")
    else:
        back_m2 = np.zeros(n)
        print("m2 background not found, using zeros")
    
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
                print(f"Using {col[1]} for m1 atom count, values: {temp[:3]}")
                n_m1 = temp
                break
    if n_m1 is None or np.all(np.isnan(n_m1)):
        print("Warning: Valid N_atoms data not found, skipping N_tot check")
        n_m1 = np.ones(n) * 1e6  # Set to large value so ntot check passes
    
    if has_m2:
        try:
            n_m2 = df[('show_ODs', 'PTAI_m2_N_atoms')].values
            # Replace NaN with zeros (DMD feedback has m2 columns but they're all NaN)
            n_m2 = np.nan_to_num(n_m2, nan=0.0)
            print(f"m2 atom count found, values (NaN->0): {n_m2[:3]}")
        except (KeyError, TypeError):
            n_m2 = np.zeros(n)
            print("m2 atom count exception, using zeros")
    else:
        n_m2 = np.zeros(n)
        print("m2 atom count not found, using zeros")

    # Calculate max background and total N for each shot
    print(f"Before calculation - back_m1[:3]: {back_m1[:3]}, back_m2[:3]: {back_m2[:3]}")
    print(f"Before calculation - n_m1[:3]: {n_m1[:3]}, n_m2[:3]: {n_m2[:3]}")
    back_max = np.maximum(np.maximum(back_m1, back_m2), back_probe)
    n_tot = n_m1 + n_m2
    print(f"After calculation - back_max[:3]: {back_max[:3]}, n_tot[:3]: {n_tot[:3]}")

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
    print(f'Valid rows after quality checks: {n_valid}/{n}')
    print(f'  Background (max of m1, m2, probe): {back_max}')
    print(f'  N_tot (m1 + m2): {n_tot}')
    if n_valid < n:
        print(f'  Rejected: {reject_reasons["background"]} for background > {back_threshold}')
        print(f'  Rejected: {reject_reasons["ntot"]} for N_tot < {ntot_threshold}')
        if reject_reasons['exception'] > 0:
            print(f'  Rejected: {reject_reasons["exception"]} due to exceptions')
    
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
            print("m2 column found but contains invalid data, using zeros")
            m2 = np.zeros((len(m1_raw), dim))
            for i in range(len(m1_raw)):
                try:
                    m1[i] = m1_raw[i]
                except Exception:
                    pass
    else:
        # For DMD density feedback: only m1 exists, set m2 to zero
        print("m2 column not found, using zeros")
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


def plot_fluctuations_waterfall(sigma2_map, y_vals, dy, y_axis_label, title, params):
    """Plot local fluctuation sigma^2 as a waterfall map."""
    if sigma2_map.size == 0 or len(y_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    x_plot = np.arange(-0.5, sigma2_map.shape[1], 1)

    polys, colors = [], []
    for i in range(len(y_vals)):
        for j in range(sigma2_map.shape[1]):
            polys.append([
                (x_plot[j], y_vals[i] - dy / 2),
                (x_plot[j + 1], y_vals[i] - dy / 2),
                (x_plot[j + 1], y_vals[i] + dy / 2),
                (x_plot[j], y_vals[i] + dy / 2),
            ])
            colors.append(sigma2_map[i, j])

    coll = PolyCollection(polys, array=np.array(colors), cmap='magma')
    ax.add_collection(coll)

    x_min = params['X_MIN_INTEGRATION']
    x_max = params['X_MAX_INTEGRATION']

    # Auto color scale from max fluctuation in the analysis region
    # (same x-window used in the main analysis).
    x_centers = np.arange(sigma2_map.shape[1])
    xmin_ind = int(np.argmin(np.abs(x_centers - x_min)))
    xmax_ind = int(np.argmin(np.abs(x_centers - x_max)))
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

    ax.axvline(x=x_min, color='white', linestyle='--', alpha=0.8)
    ax.axvline(x=x_max, color='white', linestyle='--', alpha=0.8)

    ax.set_xlabel(r'$\mu m$')
    ax.set_ylabel(y_axis_label)
    ax.set_xlim(x_plot[0], x_plot[-1])
    ax.set_ylim(np.min(y_vals) - dy, np.max(y_vals) + dy)
    ax.set_title(title)
    plt.colorbar(coll, ax=ax, label=r'Local $\sigma^2$ of $m - \langle m \rangle$')


def _sectioned_sigmoid_analysis(M, y_unique, x_plot, xmin_ind, xmax_ind, num_sections):
    from scipy.optimize import curve_fit

    def sigmoid(y, center, width, amp, offset):
        return amp * (np.tanh((y - center) / width)) + offset

    avg_density_all = None
    section_edges = np.linspace(xmin_ind, xmax_ind, num_sections + 1, dtype=int)
    sx, sy, se = [], [], []

    for i in range(num_sections):
        s0, s1 = section_edges[i], section_edges[i + 1]
        if s1 <= s0:
            continue
        integrated = np.sum(M[:, s0:s1], axis=1) / (s1 - s0)
        p0 = [np.median(y_unique), 0.1, 0.5, 0.0]
        try:
            popt, pcov = curve_fit(sigmoid, y_unique, integrated, p0=p0, maxfev=10000)
            sx.append(x_plot[int((s0 + s1) / 2)])
            sy.append(popt[0])
            se.append(np.sqrt(pcov[0, 0]))
        except Exception:
            continue

    return sx, sy, se, avg_density_all


def plot_main_waterfall(M, D, y_unique, dy, y_axis_label, title, scan, params, plot_flags, average=False):
    fig, axs = plt.subplots(
        ncols=3,
        tight_layout=True,
        figsize=(12, 5),
        sharey=True,
        gridspec_kw={'width_ratios': [0.4, 1, 1]},
    )
    ax_int, ax_m, ax_d = axs
    x_plot = np.arange(-0.5, M.shape[1], 1)

    polys_M, colors_M = [], []
    for i in range(len(y_unique)):
        for j in range(M.shape[1]):
            polys_M.append([
                (x_plot[j], y_unique[i] - dy / 2),
                (x_plot[j + 1], y_unique[i] - dy / 2),
                (x_plot[j + 1], y_unique[i] + dy / 2),
                (x_plot[j], y_unique[i] + dy / 2),
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
                (x_plot[j], y_unique[i] - dy / 2),
                (x_plot[j + 1], y_unique[i] - dy / 2),
                (x_plot[j + 1], y_unique[i] + dy / 2),
                (x_plot[j], y_unique[i] + dy / 2),
            ])
            colors_D.append(D[i, j])

    collection_D = PolyCollection(polys_D, array=np.array(colors_D), cmap='viridis')
    if params['WATERFALL_DENSITY_CLIM'] is not None:
        collection_D.set_clim(*params['WATERFALL_DENSITY_CLIM'])
    ax_d.add_collection(collection_D)
    ax_d.sharex(ax_m)

    x_min = params['X_MIN_INTEGRATION']
    x_max = params['X_MAX_INTEGRATION']
    xmin_ind = np.argmin(np.abs(x_plot - x_min))
    xmax_ind = np.argmin(np.abs(x_plot - x_max))

    for axis in [ax_m, ax_d]:
        axis.axvline(x=x_min, color='black', linestyle='--')
        axis.axvline(x=x_max, color='black', linestyle='--')

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
            M, y_unique, x_plot, xmin_ind, xmax_ind, params['NUM_SECTIONS']
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
                        import os
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
    ax_m.set_xlim(x_plot[0], x_plot[-1])
    ax_m.set_ylim(np.min(y_unique) - dy, np.max(y_unique) + dy)
    ax_m.set_title(title, fontsize=10)

    # Optional summary windows
    if plot_flags.get('avg_density_profile', False):
        fig_density, ax_density = plt.subplots(figsize=(10, 6))
        avg_density = np.mean(D, axis=0)
        std_density = np.std(D, axis=0)
        ax_density.plot(x_plot[:-1], avg_density, 'b-', linewidth=2, label='Average Density')
        ax_density.fill_between(x_plot[:-1], avg_density - std_density, avg_density + std_density, alpha=0.3, color='blue')
        ax_density.axvline(x=x_min, color='red', linestyle='--', linewidth=2)
        ax_density.axvline(x=x_max, color='red', linestyle='--', linewidth=2)
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
        ax_mag.plot(x_plot[:-1], avg_m, 'r-', linewidth=2, label='Average Magnetization')
        ax_mag.fill_between(x_plot[:-1], avg_m - std_m, avg_m + std_m, alpha=0.3, color='red')
        ax_mag.axhline(y=0, color='k', linewidth=1, alpha=0.3)
        ax_mag.axvline(x=x_min, color='blue', linestyle='--', linewidth=2)
        ax_mag.axvline(x=x_max, color='blue', linestyle='--', linewidth=2)

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


def plot_evolution_analysis(y_plot, y_axis_label, z_local_fluctuations, central_autocorr, M, params):
    w = params['AUTOCORR_WINDOW']

    fig_fluct, ax_fluct = plt.subplots()
    z_fluc_smooth = gaussian_filter1d(z_local_fluctuations, sigma=2, axis=0)
    start_idx, end_idx = 950, 1220
    x_axis_mesh = np.linspace(930, 1240, z_fluc_smooth[:, start_idx:end_idx].shape[1])
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
    x_evol_mesh = np.linspace(920, 1240, m_copy.shape[1])
    im_evol = ax_evol.pcolormesh(x_evol_mesh, y_plot, m_copy, vmin=0, vmax=0.001, cmap='viridis')
    ax_evol.set_ylabel(y_axis_label)
    ax_evol.set_xlabel(r'$\mu m$')
    ax_evol.set_title('Magnetization evolution in the bubble region')
    plt.colorbar(im_evol, ax=ax_evol)


def waterfall_plot(df, seqs, scan, data_origin='show_ODs', constraints=None, average=False, title_labels=None, globals_in_title=None, params=None, plot_flags=None):
    if params is None:
        raise ValueError('params dict is required')
    if plot_flags is None:
        plot_flags = {}
    if globals_in_title is None:
        globals_in_title = []

    df_subset = filter_dataframe(df, seqs, constraints)
    if scan == 'ARPF_feedback' or scan == 'DMD_density_feedback':
        df_subset = filter_valid_feedback_shots(df_subset)

    y_axis_col, title_base = get_scan_config(scan)
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

    if plot_flags.get('evolution_plots', False) and scan == 'bubbles_evolution':
        plot_evolution_analysis(y_final, y_axis_col, z_fluc_final, corr_final, M_final, params)

    if plot_flags.get('main_waterfall', True):
        title_full = f"{title_base}\n{'AVERAGED' if average else 'UNIQUE'}\n(seqs: {seqs})"
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
            y_axis_col,
            title_full,
            scan,
            params,
            plot_flags,
            average=average,
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
            y_axis_col,
            fluc_title,
            params,
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
    )