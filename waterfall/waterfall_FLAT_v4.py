import importlib.util
import json
import traceback
import lyse
import sys
import os

# Add analysislib_v2 to path so waterfall_v2 plotting helpers are available.
sys.path.insert(0, "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2")
# Add path to general_lib
sys.path.insert(0, "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib")
from main_lyse import get_day_data


CFG_PATH = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_config.py"
LIB_PATH = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_lib.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_recommended_center_if_enabled(cfg, mode_cfg):
    if not bool(getattr(cfg, 'RECOMMENDED_CENTER', False)):
        return mode_cfg

    results_dir = os.path.join(os.path.dirname(CFG_PATH), 'results')
    if not os.path.isdir(results_dir):
        print(f"RECOMMENDED_CENTER=True but results dir not found: {results_dir}")
        return mode_cfg

    safe_box_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('kz_param_stability_safe_box_') and f.endswith('.json')
    ])
    if len(safe_box_files) == 0:
        print('RECOMMENDED_CENTER=True but no safe-box file found in results/.')
        return mode_cfg

    latest_path = os.path.join(results_dir, safe_box_files[-1])
    try:
        with open(latest_path, 'r', encoding='utf-8') as f:
            safe_box = json.load(f)
        rec = safe_box.get('recommended_center', {})
        if not isinstance(rec, dict) or len(rec) == 0:
            print(f'RECOMMENDED_CENTER=True but recommended_center missing/empty in {latest_path}')
            return mode_cfg

        out = dict(mode_cfg)
        # Keep mode-level override for backward compatibility.
        current_defect = dict(out.get('defect_analysis', {}))
        current_defect.update(rec)
        out['defect_analysis'] = current_defect
        # Also store explicit override key for the new global-defect-config workflow.
        out['defect_analysis_override'] = dict(rec)
        print(f"Applied recommended_center from: {latest_path}")
        print(f"Updated defect_analysis keys: {sorted(list(rec.keys()))}")
        return out
    except Exception as err:
        print(f'Failed to load recommended_center from {latest_path}: {err}')
        return mode_cfg


def _run_kz_det_scan_window_analysis(df, mode_cfg, params):
    print()
    print(f"{'='*80}")
    print("KZ_det_scan_window EXTENDED ANALYSIS")
    print(f"{'='*80}")

    from waterfall.domain_extraction_lib import create_domain_info_per_shot
    from waterfall.window_analysis_lib import (
        sliding_window_analysis,
        save_window_analysis_results,
        save_window_analysis_csv,
    )
    from waterfall_v2.plot_domains_validation import plot_domains_validation
    from waterfall_v2.plot_window_analysis import (
        plot_window_analysis,
        plot_window_analysis_with_domains,
        plot_raw_magnetization_by_field,
    )

    seqs = mode_cfg.get('seqs')
    data_origin = mode_cfg.get('data_origin', 'show_ODs_v2')
    output_dir = os.path.join(os.path.dirname(CFG_PATH), 'results', 'window_analysis')
    if seqs is not None:
        df_window = df[df['sequence_index'].isin(seqs)].copy().reset_index(drop=True)
    else:
        df_window = df.copy().reset_index(drop=True)
    print(f"Window analysis shots: {len(df_window)}")

    print("Step 1: extracting domain information for all shots...")
    domain_info_dict = create_domain_info_per_shot(
        df_window,
        seqs=seqs,
        scan='KZ_det_scan_window',
        data_origin=data_origin,
        mode_cfg=mode_cfg,
        params=params,
    )
    if not domain_info_dict:
        print("No domain info extracted; skipping window analysis.")
        return

    try:
        plot_domains_validation(
            df_window,
            domain_info_dict,
            params,
            mode_cfg,
            output_dir=output_dir,
            figname='domains_validation.png',
        )
    except Exception as err:
        print(f"Warning: could not create domain validation plot: {err}")

    window_size_um = float(mode_cfg.get('window_size', 80.0))
    window_step_um = float(mode_cfg.get('window_step', 5.0))
    x_min_um = float(params.get('X_MIN_INTEGRATION', params.get('DOMAIN_WALL_X_MIN', 920.0)))
    x_max_um = float(params.get('X_MAX_INTEGRATION', params.get('DOMAIN_WALL_X_MAX', 1180.0)))
    print(
        "Step 2: sliding window analysis "
        f"(size={window_size_um:.1f} um, step={window_step_um:.1f} um, "
        f"range=[{x_min_um:.1f}, {x_max_um:.1f}] um)..."
    )

    results = sliding_window_analysis(
        df_window,
        x_min_um=x_min_um,
        x_max_um=x_max_um,
        window_size_um=window_size_um,
        window_step_um=window_step_um,
        mode_cfg=mode_cfg,
        params=params,
        domain_info_dict=domain_info_dict,
    )
    if not results:
        print("No valid windows analyzed.")
        return

    print(f"Step 3: saving {len(results)} windows to {output_dir}")
    save_window_analysis_results(results, output_dir=output_dir)
    save_window_analysis_csv(results, output_dir=output_dir)

    print("Step 4: creating window-analysis plots...")
    try:
        plot_window_analysis(
            results,
            window_size_um,
            output_dir=output_dir,
            figname='window_analysis_twin_axis.png',
        )
        plot_window_analysis_with_domains(
            results,
            window_size_um,
            output_dir=output_dir,
            figname='window_analysis_detailed.png',
        )

        field_summary = {}
        for result in results:
            for field_val_str, field_data in result.get('field_summary', {}).items():
                try:
                    field_summary[float(field_val_str)] = field_data
                except (TypeError, ValueError):
                    continue
        if field_summary:
            plot_raw_magnetization_by_field(
                df_window,
                field_summary,
                domain_info_dict,
                params,
                output_dir=output_dir,
                figname='raw_magnetization_by_field.png',
            )
    except Exception as err:
        print(f"Warning: could not create window-analysis plots: {err}")

    print(f"KZ_det_scan_window analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    try:
        cfg = _load_module(CFG_PATH, "waterfall_config")
        lib = _load_module(LIB_PATH, "waterfall_lib")

        # Load HDF data using HDF_CONFIG
        if hasattr(cfg, 'HDF_CONFIG'):
            hdf_cfg = cfg.HDF_CONFIG
            print(f"HDF_CONFIG: today={hdf_cfg.get('today', True)}, "
                  f"year={hdf_cfg.get('year')}, "
                  f"month={hdf_cfg.get('month')}, "
                  f"day={hdf_cfg.get('day')}")
            
            today_flag = hdf_cfg.get('today', True)
            df_orig = get_day_data(
                today=today_flag,
                year=hdf_cfg.get('year'),
                month=hdf_cfg.get('month'),
                day=hdf_cfg.get('day'),
                include_lyse_live=False,  # Avoid double-counting shots (HDF day files already contain today's data)
            )
            
            if df_orig is None:
                print("ERROR: get_day_data returned None. Falling back to lyse.data()")
                df_orig = lyse.data()
            else:
                print(f"Successfully loaded HDF data with get_day_data: {type(df_orig)}")
        else:
            df_orig = lyse.data()
            print("Using default lyse.data()")

        # Final safety dedup in case a source already contains repeated rows.
        try:
            n_before = len(df_orig)
            if 'filepath' in df_orig.columns:
                df_orig = df_orig.drop_duplicates(subset=['filepath'], keep='last')
            else:
                df_orig = df_orig[~df_orig.index.duplicated(keep='last')]
            n_after = len(df_orig)
            if n_after != n_before:
                print(f"Deduplicated loaded data: {n_before} -> {n_after} rows")
        except Exception as err:
            print(f"Warning: deduplication step failed ({err})")

        # Print shots per sequence
        print("\nShots per sequence in loaded HDF:")
        seq_counts = df_orig['sequence_index'].value_counts().sort_index()
        for seq, count in seq_counts.items():
            print(f"  Sequence {seq}: {count} shots")
        print(f"  Total: {len(df_orig)} shots\n")

        mode_name = cfg.ACTIVE_MODE
        if mode_name not in cfg.MODE_CONFIGS:
            raise ValueError(f"Unknown ACTIVE_MODE '{mode_name}'. Available: {list(cfg.MODE_CONFIGS.keys())}")

        mode_cfg = dict(cfg.MODE_CONFIGS[mode_name])
        if hasattr(cfg, 'SEQS') and cfg.SEQS is not None:
            mode_cfg['seqs'] = cfg.SEQS

        # Inject shared configs into mode runtime payload.
        mode_cfg['shot_filter'] = dict(getattr(cfg, 'SHOT_FILTER_CONFIG', {}))
        mode_cfg['magnetization_modalities'] = dict(getattr(cfg, 'MAGNETIZATION_MODALITIES', {}))
        mode_cfg['defect_analysis_global'] = dict(getattr(cfg, 'DEFECT_ANALYSIS_PARAMS', {}))

        mode_cfg = _apply_recommended_center_if_enabled(cfg, mode_cfg)
        lib.run_mode(df_orig, mode_cfg, cfg.PARAMS)

        if mode_cfg.get('scan') == 'KZ_det_scan_window':
            _run_kz_det_scan_window_analysis(df_orig, mode_cfg, cfg.PARAMS)

    except Exception as e:
        print(f"FAIL: waterfall_FLAT_v4 failed: {e}")
        print(traceback.format_exc())
