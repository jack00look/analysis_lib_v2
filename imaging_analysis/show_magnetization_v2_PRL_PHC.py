"""
PRL-style phase-contrast magnetization analysis.

This routine mirrors the defect-analysis view used by
show_magnetization_v2_PRL.py, but reads phase-contrast images from
cam_vert2_PHC instead of spin-selective absorption images.
"""

import os
import sys
import time
import traceback
import importlib.util

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


ANALYSISLIB_V2 = '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2'
sys.path.insert(0, ANALYSISLIB_V2)

CAMERA_NAME = 'cam_vert2_PHC'
PREFERRED_IMAGE_KEYS = ('PHC_m1_1', 'PHC_1', 'PHC_2', 'PHC_3', 'PHC_4')
DEFAULT_X_MIN_UM = 100.0
DEFAULT_X_MAX_UM = 350.0
PLOT_Y_MIN_UM = 20.0
PLOT_Y_MAX_UM = 100.0


def _load_module(name, rel_path):
    path = os.path.join(ANALYSISLIB_V2, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera_settings = _load_module(
    'imaging_analysis_lib.camera_settings',
    'imaging_analysis_lib/camera_settings.py',
)
imaging_analysis_lib_mod = _load_module(
    'imaging_analysis_lib_v2.main',
    'imaging_analysis_lib_v2/main.py',
)
general_lib_mod = _load_module(
    'general_lib.main',
    'general_lib/main.py',
)
defect_finding_lib = _load_module(
    'defect_finding_lib',
    'waterfall/defect_finding_lib.py',
)

CAMERAS = camera_settings.cameras


def _load_phc_mode_config():
    try:
        mod = _load_module(
            'waterfall_v2.mode_configs.KZ_det_scan_PHC',
            'waterfall_v2/mode_configs/KZ_det_scan_PHC.py',
        )
        return dict(getattr(mod, 'MODE_CONFIG', {}))
    except Exception as exc:
        print(f"WARNING: Could not load KZ_det_scan_PHC config: {exc}")
        return {}


def _load_common_config():
    try:
        mod = _load_module('waterfall_v2.common_config', 'waterfall_v2/common_config.py')
        return getattr(mod, 'MAGNETIZATION_MODALITIES', {}), getattr(mod, 'PARAMS', {})
    except Exception as exc:
        print(f"WARNING: Could not load waterfall_v2 common_config: {exc}")
        return {}, {}


def _get_phc_defect_config():
    mode_cfg = _load_phc_mode_config()
    defect_cfg = dict(mode_cfg.get('defect_analysis', {}))
    if len(defect_cfg) == 0:
        defect_cfg = {
            'gaussian_sigma': 2.0,
            'derivative_threshold': 0.4e-6,
            'derivative_threshold_pos': 0.4e-6,
            'derivative_threshold_neg': -0.4e-6,
            'threshold_scan': False,
            'zero_crossing_min_distance_um': 2.0,
            'peak_step_filter_enabled': False,
            'peak_step_filter_window_px': 3,
            'peak_step_filter_min_abs_delta_m': 0.1e-5,
            'min_same_sign_peak_distance_um': 6.0,
            'missed_defect_correction_method': 'opposite_peak',
        }
    print(
        "PHC defect config: "
        f"sigma={defect_cfg.get('gaussian_sigma')}, "
        f"+thr={defect_cfg.get('derivative_threshold_pos')}, "
        f"-thr={defect_cfg.get('derivative_threshold_neg')}"
    )
    return defect_cfg


def _get_phc_roi_um():
    modalities, _ = _load_common_config()
    phc = dict(modalities.get('vert2_phc_multicomponent', {}))
    x_min = float(phc.get('x_min_integration', DEFAULT_X_MIN_UM))
    x_max = float(phc.get('x_max_integration', DEFAULT_X_MAX_UM))
    return x_min, x_max


def _choose_image_key(images):
    for key in PREFERRED_IMAGE_KEYS:
        if key in images:
            return key
    normal_keys = [key for key in images.keys() if 'SVD' not in key]
    if normal_keys:
        return normal_keys[0]
    return None


def _get_title(h5file, cam):
    try:
        return general_lib_mod.get_title(h5file, cam)
    except Exception:
        return f"CAM {cam.get('name', CAMERA_NAME)}"


def _domain_state_from_defects(x_axis_um, results, x_min_um, x_max_um):
    all_walls = np.concatenate([
        np.asarray(results.get('threshold_x_pos', []), dtype=float),
        np.asarray(results.get('threshold_x_neg', []), dtype=float),
        np.asarray(results.get('corrected_x', []), dtype=float),
    ])
    all_walls = np.unique(np.round(all_walls, 6))
    all_walls = np.sort(all_walls[(all_walls > x_min_um) & (all_walls < x_max_um)])

    m_roi = np.asarray(results.get('m_roi', []), dtype=float)
    x_roi = np.asarray(results.get('x_roi_um', []), dtype=float)
    if m_roi.size == 0 or x_roi.size == 0:
        return np.zeros_like(x_axis_um, dtype=float)

    avg_m = float(np.nanmean(m_roi))
    start_sign = 1.0 if np.nan_to_num(avg_m, nan=0.0) > 0.0 else -1.0
    n_flips = np.searchsorted(all_walls, x_axis_um, side='right')
    domain = start_sign * ((-1.0) ** n_flips)
    used = (x_axis_um >= x_min_um) & (x_axis_um <= x_max_um)
    return np.where(used, domain, np.nan)


def _plot_prl_phc_analysis(Z, profile, x_axis_um, y_roi, x_min_um, x_max_um, defect_results, title):
    px_size_um = float(np.mean(np.diff(x_axis_um))) if len(x_axis_um) > 1 else 1.019
    x_roi = np.asarray(defect_results['x_roi_um'], dtype=float)
    m_roi = np.asarray(defect_results['m_roi'], dtype=float)
    d_roi = np.asarray(defect_results['d_roi'], dtype=float)

    x_rel = x_axis_um - x_min_um
    x_roi_rel = x_roi - x_min_um
    x_min_rel = 0.0
    x_max_rel = float(x_max_um - x_min_um)

    fig = plt.figure(figsize=(3.39, 8.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.0, 1.0, 0.55], hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    ax = axes[0]
    y_extent_um = Z.shape[0] * px_size_um
    im = ax.imshow(
        Z,
        cmap='RdBu',
        aspect='auto',
        vmin=np.nanpercentile(Z, 2),
        vmax=np.nanpercentile(Z, 98),
        extent=[x_axis_um[0] - x_min_um, x_axis_um[-1] - x_min_um, y_extent_um, 0],
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="8%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")
    cbar.set_label("PHC signal", fontsize=9)
    cax.tick_params(labelsize=7, pad=1)
    ax.axvline(x_min_rel, color='g', linestyle=':', linewidth=1.5)
    ax.axvline(x_max_rel, color='g', linestyle=':', linewidth=1.5)
    if y_roi is not None:
        y0, y1 = y_roi
        ax.axhline(y0 * px_size_um, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axhline(y1 * px_size_um, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_xlim(x_min_rel, x_max_rel)
    ax.set_ylim(PLOT_Y_MAX_UM, PLOT_Y_MIN_UM)
    ax.set_ylabel('y (um)', fontsize=10)

    ax = axes[1]
    smooth_full = gaussian_filter1d(np.asarray(profile, dtype=float), sigma=2.0)
    ax.plot(x_rel, profile, color='0.25', linewidth=1.0, alpha=0.65, label='raw PHC 1D')
    ax.plot(x_rel, smooth_full, color='tab:orange', linewidth=1.5, label='smoothed')
    ax.axvline(x_min_rel, color='g', linestyle=':', linewidth=1.5)
    ax.axvline(x_max_rel, color='g', linestyle=':', linewidth=1.5)
    ax.set_ylabel('PHC 1D', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc='best')

    ax = axes[2]
    ax.plot(x_roi_rel, d_roi, color='k', linewidth=1.2, label='dZ/dx')
    thr_pos = float(defect_finding_lib.normalize_defect_config({}).get('derivative_threshold_pos', 0.0))
    thr_neg = float(defect_finding_lib.normalize_defect_config({}).get('derivative_threshold_neg', 0.0))
    try:
        defect_cfg = _get_phc_defect_config()
        cfg_norm = defect_finding_lib.normalize_defect_config(defect_cfg)
        thr_pos = float(cfg_norm.get('derivative_threshold_pos', thr_pos))
        thr_neg = float(cfg_norm.get('derivative_threshold_neg', thr_neg))
    except Exception as exc:
        print(f"WARNING: Could not load PHC thresholds for PRL plot: {exc}")
    ax.axhline(
        thr_pos,
        color='tab:blue',
        linestyle='--',
        linewidth=1.0,
        alpha=0.75,
        label=f'+ threshold {thr_pos:.3g}',
    )
    ax.axhline(
        thr_neg,
        color='tab:red',
        linestyle='--',
        linewidth=1.0,
        alpha=0.75,
        label=f'- threshold {thr_neg:.3g}',
    )
    threshold_x_pos = np.asarray(defect_results['threshold_x_pos'], dtype=float)
    threshold_x_neg = np.asarray(defect_results['threshold_x_neg'], dtype=float)
    corrected_x = np.asarray(defect_results['corrected_x'], dtype=float)
    rejected_x = np.asarray(defect_results['rejected_x'], dtype=float)

    if threshold_x_pos.size > 0:
        d_pos_vals = np.interp(threshold_x_pos, x_roi, d_roi)
        ax.scatter(
            threshold_x_pos - x_min_um,
            d_pos_vals,
            color='black',
            s=34,
            marker='^',
            linewidths=1.4,
            edgecolors='black',
            zorder=5,
            label=f'Pos peaks ({threshold_x_pos.size})',
        )
    if threshold_x_neg.size > 0:
        d_neg_vals = np.interp(threshold_x_neg, x_roi, d_roi)
        ax.scatter(
            threshold_x_neg - x_min_um,
            d_neg_vals,
            color='black',
            s=34,
            marker='v',
            linewidths=1.4,
            edgecolors='black',
            zorder=5,
            label=f'Neg peaks ({threshold_x_neg.size})',
        )
    if corrected_x.size > 0:
        d_corr_vals = np.interp(corrected_x, x_roi, d_roi)
        ax.scatter(
            corrected_x - x_min_um,
            d_corr_vals,
            color='green',
            s=70,
            marker='*',
            linewidths=1.0,
            edgecolors='black',
            zorder=6,
            label=f'Corrected ({corrected_x.size})',
        )
    if rejected_x.size > 0:
        d_rej_vals = np.interp(rejected_x, x_roi, d_roi)
        ax.scatter(
            rejected_x - x_min_um,
            d_rej_vals,
            color='orange',
            s=55,
            marker='x',
            linewidths=1.4,
            zorder=6,
            label=f'Rejected ({rejected_x.size})',
        )
    ax.axhline(0.0, color='0.5', linestyle=':', linewidth=0.8)
    ax.set_ylabel('dZ/dx', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc='best')
    ax.set_title(
        f"Defects: {defect_results['count']} "
        f"(+{len(defect_results['threshold_x_pos'])}, -{len(defect_results['threshold_x_neg'])}, "
        f"corr={len(defect_results['corrected_x'])})",
        fontsize=9,
    )

    ax = axes[3]
    domain_state = _domain_state_from_defects(x_axis_um, defect_results, x_min_um, x_max_um)
    used = np.isfinite(domain_state)
    ax.step(x_rel[used], domain_state[used], where='mid', color='k', linewidth=1.1)
    ax.axhline(0.0, color='0.5', linestyle=':', linewidth=0.8)
    ax.set_ylim(-1.25, 1.25)
    ax.set_ylabel('domain', fontsize=10)
    ax.set_xlabel('x - ROI start (um)', fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=9)
    return fig


def show_magnetization_v2_PRL_PHC(h5file, show=True):
    infos_analysis = {}
    try:
        print("DEBUG: Entering show_magnetization_v2_PRL_PHC")
        if CAMERA_NAME not in CAMERAS:
            print(f"ERROR: {CAMERA_NAME} not found in camera settings")
            return infos_analysis

        cam = CAMERAS[CAMERA_NAME]
        dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
        if dict_images is None:
            print(f"ERROR: Could not load images for {CAMERA_NAME}")
            return infos_analysis

        images = dict_images.images
        images_odlog = dict_images.images_ODlog
        image_key = _choose_image_key(images)
        if image_key is None:
            print("ERROR: No PHC image found")
            return infos_analysis

        print(f"DEBUG: Using PHC image key: {image_key}")
        phc_image = np.asarray(images[image_key], dtype=float)
        phc_odlog = np.asarray(images_odlog.get(image_key, phc_image), dtype=float)
        Z = phc_odlog - 1.0 if cam.get('method') == 'phase_contrast' else phc_odlog

        from imaging_analysis_lib_v2.main import get_n1Ds, get_axes

        n1d_x, n1d_y = get_n1Ds(phc_image[cam['roi_integration']], cam)
        x_vals, _, _, _, inverted = get_axes(dict_images.axes)
        if inverted:
            Z = Z.T
            n1d_x, n1d_y = n1d_y, n1d_x

        px_size_um = float(cam.get('px_size', 1.019e-6)) * 1e6
        if len(x_vals) == len(n1d_x):
            x_axis_um = np.arange(len(n1d_x), dtype=float) * px_size_um
        else:
            x_axis_um = np.arange(len(n1d_x), dtype=float) * px_size_um

        x_min_um, x_max_um = _get_phc_roi_um()
        defect_cfg = _get_phc_defect_config()
        defect_results = defect_finding_lib.find_defects_in_profile(
            n1d_x,
            x_axis_um,
            defect_cfg=defect_cfg,
            x_min_um=x_min_um,
            x_max_um=x_max_um,
        )

        title = _get_title(h5file, cam)
        y_roi = None
        roi = cam.get('roi_integration')
        if isinstance(roi, tuple) and len(roi) >= 1 and isinstance(roi[0], slice):
            y_roi = (roi[0].start or 0, roi[0].stop or Z.shape[0])

        fig = _plot_prl_phc_analysis(
            Z,
            n1d_x,
            x_axis_um,
            y_roi,
            x_min_um,
            x_max_um,
            defect_results,
            title,
        )

        h5_base = os.path.splitext(os.path.basename(getattr(h5file, 'filename', 'phc')))[0]
        out_path = os.path.join(os.path.expanduser('~'), f"{h5_base}_PRL_PHC_defect_analysis.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved PRL PHC defect-analysis figure to: {out_path}")

        infos_analysis[f'{image_key}_n1D_x'] = np.asarray(n1d_x, dtype=float)
        infos_analysis[f'{image_key}_n1D_y'] = np.asarray(n1d_y, dtype=float)
        infos_analysis[f'{image_key}_defect_count'] = int(defect_results['count'])
        infos_analysis[f'{image_key}_defect_count_pos'] = int(len(defect_results['threshold_x_pos']))
        infos_analysis[f'{image_key}_defect_count_neg'] = int(len(defect_results['threshold_x_neg']))
        infos_analysis[f'{image_key}_defect_count_corrected'] = int(len(defect_results['corrected_x']))
        infos_analysis[f'{image_key}_defect_density_um'] = (
            float(defect_results['count']) / abs(float(x_max_um - x_min_um))
            if abs(float(x_max_um - x_min_um)) > 0 else np.nan
        )
        infos_analysis[f'{image_key}_x_min_um'] = float(x_min_um)
        infos_analysis[f'{image_key}_x_max_um'] = float(x_max_um)

        print(
            "PRL PHC defect result: "
            f"count={defect_results['count']}, "
            f"density={infos_analysis[f'{image_key}_defect_density_um']:.6g} um^-1, "
            f"ROI=[{x_min_um:.3f}, {x_max_um:.3f}] um"
        )

        if show and 'lyse' not in sys.modules:
            plt.show()

        return infos_analysis

    except Exception:
        print(traceback.format_exc())
        return infos_analysis


def show_magnetization_v2(h5file, show=True):
    return show_magnetization_v2_PRL_PHC(h5file, show=show)


if __name__ == '__main__':
    time_0 = time.time()
    try:
        h5_file_path = sys.argv[1] if len(sys.argv) > 1 else None
        if h5_file_path:
            with h5py.File(h5_file_path, 'r') as h5file:
                show_magnetization_v2_PRL_PHC(h5file, show=True)
        else:
            import lyse
            run = lyse.Run(lyse.path)
            with h5py.File(lyse.path, 'r+') as h5file:
                res_dict = show_magnetization_v2_PRL_PHC(h5file, show=False)
            if res_dict is not None:
                for key, value in res_dict.items():
                    try:
                        run.save_result(key, value)
                    except Exception as exc:
                        print(f"WARNING: Could not save result {key}: {exc}")
        print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
    except Exception:
        print('DEBUG: Top level exception in show_magnetization_v2_PRL_PHC')
        print('Elapsed time: {:.2f} s'.format(time.time() - time_0))
        print(traceback.format_exc())
