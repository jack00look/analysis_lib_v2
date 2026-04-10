import importlib.util
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import defect_finding_lib as defect_lib


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSISLIB_V2_DIR = os.path.dirname(THIS_DIR)
GENERAL_LIB_DIR = os.path.join(ANALYSISLIB_V2_DIR, 'general_lib')
CFG_PATH = os.path.join(THIS_DIR, 'waterfall_config.py')
LIB_PATH = os.path.join(THIS_DIR, 'waterfall_lib.py')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

if GENERAL_LIB_DIR not in sys.path:
    sys.path.insert(0, GENERAL_LIB_DIR)

from main_lyse import get_day_data  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec from: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dataframe(cfg):
    hdf_cfg = getattr(cfg, 'HDF_CONFIG', {})
    today_flag = bool(hdf_cfg.get('today', True))

    df_orig = get_day_data(
        today=today_flag,
        year=hdf_cfg.get('year'),
        month=hdf_cfg.get('month'),
        day=hdf_cfg.get('day'),
        include_lyse_live=False,
    )

    if df_orig is None:
        raise RuntimeError('get_day_data returned None')

    # Robust deduplication: handle flat columns, MultiIndex columns, or filepath in index.
    try:
        filepath_col = None
        for col in df_orig.columns:
            if col == 'filepath':
                filepath_col = col
                break
            if isinstance(col, tuple) and len(col) > 0 and col[0] == 'filepath':
                filepath_col = col
                break

        if filepath_col is not None:
            df_orig = df_orig.drop_duplicates(subset=[filepath_col], keep='last')
        else:
            # Fallback: deduplicate by index only.
            df_orig = df_orig[~df_orig.index.duplicated(keep='last')]
    except Exception as err:
        print(f'Warning: could not deduplicate rows ({err}); continuing without dedup.')

    return df_orig


def _prepare_profiles(df, lib, mode_cfg, params):
    seqs = mode_cfg['seqs']
    scan = mode_cfg['scan']
    data_origin = mode_cfg.get('data_origin', 'show_ODs')
    constraints = mode_cfg.get('constraints', None)
    average = bool(mode_cfg.get('average', False))

    modality_cfg = lib._resolve_modality(mode_cfg, params)
    um_per_px = lib.get_um_per_px(params, camera_name_override=modality_cfg.get('camera_name'))

    df_subset = lib.filter_dataframe(df, seqs, constraints)
    y_axis_col, _ = lib.get_scan_config(scan)
    sorted_df = df_subset.sort_values(y_axis_col)

    good_mask = lib.get_valid_rows_mask(
        sorted_df,
        filter_cfg=mode_cfg.get('shot_filter', {}),
        modality_cfg=modality_cfg,
        data_origin=modality_cfg.get('data_origin', data_origin),
    )
    sorted_df = sorted_df[good_mask]
    if sorted_df.empty:
        raise RuntimeError('No valid data remaining after filtering.')

    y_raw = sorted_df[y_axis_col].values
    grouped_y, grouped_indices = lib._build_group_indices(y_raw)

    profile_inputs = lib.build_magnetization_inputs(sorted_df, mode_cfg, params, data_origin)
    if profile_inputs is None:
        raise RuntimeError('No valid magnetization profile columns found for selected modality.')

    M_full = profile_inputs['M_full']
    D_full = profile_inputs['D_full']

    if average:
        if len(grouped_indices) == 0:
            raise RuntimeError('No groups found for averaging.')

        M_final = np.zeros((len(grouped_indices), M_full.shape[1]))
        D_final = np.zeros((len(grouped_indices), D_full.shape[1]))
        for i, idx in enumerate(grouped_indices):
            M_final[i] = np.mean(M_full[idx], axis=0)
            D_final[i] = np.mean(D_full[idx], axis=0)
        y_final = grouped_y
    else:
        rep_indices = np.array([idx[0] for idx in grouped_indices if len(idx) > 0], dtype=int)
        if rep_indices.size == 0:
            raise RuntimeError('No representative rows available after grouping.')
        M_final = M_full[rep_indices]
        D_final = D_full[rep_indices]
        y_final = grouped_y

    x_centers_um = np.arange(M_final.shape[1]) * um_per_px
    x_min_um = float(params['X_MIN_INTEGRATION'])
    x_max_um = float(params['X_MAX_INTEGRATION'])
    xmin_ind = int(np.argmin(np.abs(x_centers_um - x_min_um)))
    xmax_ind = int(np.argmin(np.abs(x_centers_um - x_max_um)))
    if xmax_ind < xmin_ind:
        xmin_ind, xmax_ind = xmax_ind, xmin_ind

    return {
        'M_final': M_final,
        'D_final': D_final,
        'y_final': y_final,
        'x_centers_um': x_centers_um,
        'xmin_ind': xmin_ind,
        'xmax_ind': xmax_ind,
        'um_per_px': um_per_px,
    }


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


def detect_counts(profile_data, lib, defect_cfg):
    M_final = profile_data['M_final']
    y_final = profile_data['y_final']
    x_centers_um = profile_data['x_centers_um']
    xmin_ind = int(profile_data['xmin_ind'])
    xmax_ind = int(profile_data['xmax_ind'])

    x_min_um = float(x_centers_um[xmin_ind])
    x_max_um = float(x_centers_um[xmax_ind])
    cfg = defect_lib.normalize_defect_config(defect_cfg)

    per_shot_counts = []
    corrected_total = 0
    rejected_total = 0

    for i, _ in enumerate(y_final):
        out_i = defect_lib.find_defects_in_profile(
            M_final[i],
            x_centers_um,
            cfg,
            x_min_um,
            x_max_um,
        )
        per_shot_counts.append(int(out_i['count']))
        corrected_total += int(len(out_i['corrected_x']))
        rejected_total += int(len(out_i['rejected_x']))

    return {
        'counts': np.asarray(per_shot_counts, dtype=int),
        'total_defects': int(np.sum(per_shot_counts)),
        'corrected_total': int(corrected_total),
        'rejected_total': int(rejected_total),
    }


def _sample_local_params(base, rng):
    p = dict(base)

    # log-normal style perturbation around base values
    def _mul(base_val, sigma_frac=0.18, lo=None, hi=None):
        b = float(base_val)
        if b == 0:
            v = float(rng.normal(0.0, sigma_frac))
        else:
            v = b * float(np.exp(rng.normal(0.0, sigma_frac)))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    p['gaussian_sigma'] = _mul(base['gaussian_sigma'], sigma_frac=0.22, lo=0.0, hi=2.5)
    p['derivative_threshold_pos'] = _mul(base['derivative_threshold_pos'], sigma_frac=0.18, lo=0.005, hi=0.30)

    # keep negative threshold negative; perturb magnitude
    neg_mag = _mul(abs(base['derivative_threshold_neg']), sigma_frac=0.18, lo=0.005, hi=0.30)
    p['derivative_threshold_neg'] = -float(neg_mag)

    p['peak_step_filter_min_abs_delta_m'] = _mul(
        base['peak_step_filter_min_abs_delta_m'],
        sigma_frac=0.20,
        lo=0.0,
        hi=0.8,
    )
    p['min_same_sign_peak_distance_um'] = _mul(base['min_same_sign_peak_distance_um'], sigma_frac=0.20, lo=0.0, hi=20.0)
    p['zero_crossing_min_distance_um'] = _mul(base['zero_crossing_min_distance_um'], sigma_frac=0.20, lo=0.0, hi=20.0)

    # integer local perturbation
    w0 = int(base['peak_step_filter_window_px'])
    p['peak_step_filter_window_px'] = int(np.clip(w0 + rng.integers(-2, 3), 1, 16))

    # boolean unchanged in local stability test
    p['peak_step_filter_enabled'] = bool(base['peak_step_filter_enabled'])

    return p


def _relative_param_distance(base, p):
    keys = [
        'gaussian_sigma',
        'derivative_threshold_pos',
        'derivative_threshold_neg',
        'peak_step_filter_min_abs_delta_m',
        'min_same_sign_peak_distance_um',
        'zero_crossing_min_distance_um',
    ]

    vals = []
    for k in keys:
        b = float(base[k])
        v = float(p[k])
        denom = max(1e-9, abs(b))
        vals.append(abs(v - b) / denom)

    w0 = float(base['peak_step_filter_window_px'])
    w1 = float(p['peak_step_filter_window_px'])
    vals.append(abs(w1 - w0) / max(1.0, w0))

    return float(np.sqrt(np.mean(np.square(vals))))


def _extract_safe_box(res_df, defect_cfg):
    """
    Build a robust local 'safe box' around the baseline parameters using
    low-instability samples.
    """
    if len(res_df) == 0:
        return {
            'n_safe_samples': 0,
            'criteria': {},
            'ranges': {},
            'recommended_center': dict(defect_cfg),
        }

    mae_thr = float(np.percentile(res_df['mae_counts_per_shot'], 25))
    flip_thr = float(np.percentile(res_df['flip_rate'], 25))
    rmse_thr = float(np.percentile(res_df['rmse_counts_per_shot'], 25))
    shift_thr = float(np.percentile(np.abs(res_df['total_shift_vs_baseline'].values), 50))

    safe = res_df[
        (res_df['mae_counts_per_shot'] <= mae_thr)
        & (res_df['flip_rate'] <= flip_thr)
        & (res_df['rmse_counts_per_shot'] <= rmse_thr)
        & (np.abs(res_df['total_shift_vs_baseline']) <= shift_thr)
    ].copy()

    if len(safe) < 8:
        # Fallback to top 10% best by combined score.
        score = (
            0.50 * res_df['mae_counts_per_shot']
            + 0.35 * res_df['flip_rate']
            + 0.15 * res_df['rmse_counts_per_shot']
        )
        n_top = max(8, int(np.ceil(0.10 * len(res_df))))
        safe = res_df.loc[np.argsort(score.values)[:n_top]].copy()

    tunable_keys = [
        'gaussian_sigma',
        'derivative_threshold_pos',
        'derivative_threshold_neg',
        'peak_step_filter_window_px',
        'peak_step_filter_min_abs_delta_m',
        'min_same_sign_peak_distance_um',
        'zero_crossing_min_distance_um',
    ]

    ranges = {}
    recommended = dict(defect_cfg)
    for k in tunable_keys:
        c = f'param__{k}'
        if c not in safe.columns:
            continue

        lo = float(safe[c].min())
        hi = float(safe[c].max())
        med = float(safe[c].median())

        if k == 'peak_step_filter_window_px':
            lo = int(np.floor(lo))
            hi = int(np.ceil(hi))
            med = int(np.round(med))

        ranges[k] = {
            'min': lo,
            'max': hi,
            'median': med,
            'baseline': defect_cfg.get(k),
        }
        recommended[k] = med

    return {
        'n_safe_samples': int(len(safe)),
        'criteria': {
            'mae_q25_threshold': mae_thr,
            'flip_q25_threshold': flip_thr,
            'rmse_q25_threshold': rmse_thr,
            'abs_total_shift_q50_threshold': shift_thr,
            'fallback_used': bool(len(safe) < 8),
        },
        'ranges': ranges,
        'recommended_center': recommended,
    }


def _save_stability_plots(res_df, out_png_path):
    if len(res_df) == 0:
        return

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), tight_layout=True)

    ax = axs[0, 0]
    ax.scatter(
        res_df['rel_param_dist'],
        res_df['mae_counts_per_shot'],
        s=18,
        alpha=0.7,
        c='tab:blue',
        edgecolors='none',
    )
    ax.set_xlabel('Relative parameter distance from baseline')
    ax.set_ylabel('MAE (defects/shot)')
    ax.set_title('Distance vs MAE')
    ax.grid(True, alpha=0.25)

    ax = axs[0, 1]
    ax.scatter(
        res_df['rel_param_dist'],
        res_df['flip_rate'],
        s=18,
        alpha=0.7,
        c='tab:orange',
        edgecolors='none',
    )
    ax.set_xlabel('Relative parameter distance from baseline')
    ax.set_ylabel('Flip rate')
    ax.set_title('Distance vs flip rate')
    ax.grid(True, alpha=0.25)

    ax = axs[1, 0]
    ax.hist(res_df['mae_counts_per_shot'], bins=max(10, int(np.sqrt(len(res_df)))), color='tab:green', alpha=0.75, edgecolor='black')
    ax.set_xlabel('MAE (defects/shot)')
    ax.set_ylabel('Count')
    ax.set_title('MAE distribution')
    ax.grid(True, alpha=0.25)

    ax = axs[1, 1]
    ax.hist(res_df['flip_rate'], bins=max(10, int(np.sqrt(len(res_df)))), color='tab:red', alpha=0.75, edgecolor='black')
    ax.set_xlabel('Flip rate')
    ax.set_ylabel('Count')
    ax.set_title('Flip-rate distribution')
    ax.grid(True, alpha=0.25)

    fig.suptitle('KZ defect parameter local-stability diagnostics', fontsize=12)
    fig.savefig(out_png_path, dpi=160)
    plt.close(fig)


def main(n_samples=300, seed=7):
    cfg = _load_module(CFG_PATH, 'waterfall_config')
    lib = _load_module(LIB_PATH, 'waterfall_lib')

    mode_cfg = dict(cfg.MODE_CONFIGS['KZ_defect_analysis'])
    if hasattr(cfg, 'SEQS') and cfg.SEQS is not None:
        mode_cfg['seqs'] = cfg.SEQS
    mode_cfg['shot_filter'] = dict(getattr(cfg, 'SHOT_FILTER_CONFIG', {}))
    mode_cfg['magnetization_modalities'] = dict(getattr(cfg, 'MAGNETIZATION_MODALITIES', {}))
    mode_cfg['defect_analysis_global'] = dict(getattr(cfg, 'DEFECT_ANALYSIS_PARAMS', {}))

    if not mode_cfg.get('seqs'):
        raise RuntimeError('KZ_defect_analysis seqs is empty. Set SEQS in waterfall_config.py first.')

    defect_cfg = dict(getattr(cfg, 'DEFECT_ANALYSIS_PARAMS', {}))
    defect_cfg.update(mode_cfg.get('defect_analysis', {}))
    defect_cfg.update(mode_cfg.get('defect_analysis_override', {}))
    defect_cfg = defect_lib.normalize_defect_config(defect_cfg)
    required = [
        'gaussian_sigma',
        'derivative_threshold_pos',
        'derivative_threshold_neg',
        'peak_step_filter_enabled',
        'peak_step_filter_window_px',
        'peak_step_filter_min_abs_delta_m',
        'min_same_sign_peak_distance_um',
        'zero_crossing_min_distance_um',
    ]
    for k in required:
        if k not in defect_cfg:
            raise KeyError(f'Missing defect_analysis parameter: {k}')

    df = _load_dataframe(cfg)
    profile_data = _prepare_profiles(df, lib, mode_cfg, cfg.PARAMS)

    baseline = detect_counts(profile_data, lib, defect_cfg)
    c0 = baseline['counts']

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_samples)):
        p = _sample_local_params(defect_cfg, rng)
        out = detect_counts(profile_data, lib, p)
        c = out['counts']

        if c.shape != c0.shape:
            raise RuntimeError('Count vectors shape mismatch. Check grouping logic.')

        diff = c - c0
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff.astype(float) ** 2)))
        flip_rate = float(np.mean(diff != 0))
        total_shift = int(np.sum(c) - np.sum(c0))
        rel_param_dist = _relative_param_distance(defect_cfg, p)

        rows.append({
            'sample_idx': i,
            'rel_param_dist': rel_param_dist,
            'mae_counts_per_shot': mae,
            'rmse_counts_per_shot': rmse,
            'flip_rate': flip_rate,
            'total_shift_vs_baseline': total_shift,
            'total_defects': int(out['total_defects']),
            'corrected_total': int(out['corrected_total']),
            'rejected_total': int(out['rejected_total']),
            **{f'param__{k}': p[k] for k in p.keys()},
        })

    res_df = pd.DataFrame(rows).sort_values(['mae_counts_per_shot', 'flip_rate', 'rmse_counts_per_shot'])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'kz_param_stability_samples_{stamp}.csv')
    json_path = os.path.join(RESULTS_DIR, f'kz_param_stability_summary_{stamp}.json')
    safe_box_path = os.path.join(RESULTS_DIR, f'kz_param_stability_safe_box_{stamp}.json')
    plot_path = os.path.join(RESULTS_DIR, f'kz_param_stability_plots_{stamp}.png')

    res_df.to_csv(csv_path, index=False)
    safe_box = _extract_safe_box(res_df, defect_cfg)
    _save_stability_plots(res_df, plot_path)

    with open(safe_box_path, 'w', encoding='utf-8') as f:
        json.dump(safe_box, f, indent=2)

    summary = {
        'n_samples': int(n_samples),
        'seed': int(seed),
        'baseline': {
            'n_shots': int(len(c0)),
            'total_defects': int(baseline['total_defects']),
            'mean_defects_per_shot': float(np.mean(c0) if len(c0) else 0.0),
            'std_defects_per_shot': float(np.std(c0, ddof=1) if len(c0) > 1 else 0.0),
            'corrected_total': int(baseline['corrected_total']),
            'rejected_total': int(baseline['rejected_total']),
            'params': defect_cfg,
        },
        'stability_metrics': {
            'median_mae_counts_per_shot': float(res_df['mae_counts_per_shot'].median()),
            'p90_mae_counts_per_shot': float(np.percentile(res_df['mae_counts_per_shot'], 90)),
            'median_flip_rate': float(res_df['flip_rate'].median()),
            'p90_flip_rate': float(np.percentile(res_df['flip_rate'], 90)),
        },
        'best_10_samples': res_df.head(10).to_dict(orient='records'),
        'safe_box': safe_box,
        'files': {
            'samples_csv': csv_path,
            'summary_json': json_path,
            'safe_box_json': safe_box_path,
            'plots_png': plot_path,
        },
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('=== KZ defect parameter local-stability scan complete ===')
    print(f"Samples: {n_samples}, Seed: {seed}")
    print(f"Baseline mean defects/shot: {summary['baseline']['mean_defects_per_shot']:.4f}")
    print(f"Median MAE (counts/shot): {summary['stability_metrics']['median_mae_counts_per_shot']:.4f}")
    print(f"P90 MAE (counts/shot): {summary['stability_metrics']['p90_mae_counts_per_shot']:.4f}")
    print(f"Median flip rate: {summary['stability_metrics']['median_flip_rate']:.4f}")
    print(f"P90 flip rate: {summary['stability_metrics']['p90_flip_rate']:.4f}")
    print(f"Safe-box samples used: {safe_box['n_safe_samples']}")
    print(f'Saved samples CSV: {csv_path}')
    print(f'Saved summary JSON: {json_path}')
    print(f'Saved safe-box JSON: {safe_box_path}')
    print(f'Saved plots PNG: {plot_path}')


if __name__ == '__main__':
    main(n_samples=300, seed=7)
