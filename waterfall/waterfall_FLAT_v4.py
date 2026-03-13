import importlib.util
import traceback
import lyse


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

        df_orig = lyse.data()

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
