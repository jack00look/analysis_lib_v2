import importlib.util
import traceback
import lyse
import sys
import os

# Add path to general_lib
sys.path.insert(0, "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib")
from main_lyse import get_day_data


CFG_PATH = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_config.py"
LIB_PATH = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_lib.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
            
            df_orig = get_day_data(
                today=hdf_cfg.get('today', True),
                year=hdf_cfg.get('year'),
                month=hdf_cfg.get('month'),
                day=hdf_cfg.get('day')
            )
            
            if df_orig is None:
                print("ERROR: get_day_data returned None. Falling back to lyse.data()")
                df_orig = lyse.data()
            else:
                print(f"Successfully loaded HDF data with get_day_data: {type(df_orig)}")
        else:
            df_orig = lyse.data()
            print("Using default lyse.data()")

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
        lib.run_mode(df_orig, mode_cfg, cfg.PARAMS)

    except Exception as e:
        print(f"FAIL: waterfall_FLAT_v4 failed: {e}")
        print(traceback.format_exc())
