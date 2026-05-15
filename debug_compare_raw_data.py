"""
Debug script to save raw data from both waterfall_FLAT_v4 and waterfall_v2_example
for comparison.

This extracts:
1. bubbles_time values
2. m1_1d profiles (full arrays)
3. m2_1d profiles (full arrays)
4. Shot names for matching

And saves them to CSV files for analysis.
"""

import sys
import os
import importlib.util
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')


def load_module(path, name):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_raw_data_from_lyse(df, data_origin='show_ODs'):
    """
    Extract raw data from Lyse DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Lyse DataFrame (from waterfall_FLAT_v4)
    data_origin : str
        Column prefix for data (e.g., 'show_ODs', 'show_ODs_v2')
    
    Returns
    -------
    dict
        Dictionary with 'bubbles_time', 'm1_profiles', 'm2_profiles', 'shot_names'
    """
    logger.info(f"\nExtracting raw data from Lyse DataFrame (origin: {data_origin})...")
    
    # Look for bubbles_time column
    bubbles_time_col = None
    for col in df.columns:
        if 'bubbles_time' in str(col).lower():
            bubbles_time_col = col
            break
    
    if bubbles_time_col is None:
        logger.error("Could not find bubbles_time column")
        return None
    
    logger.info(f"Found bubbles_time column: {bubbles_time_col}")
    
    # Look for m1_1d and m2_1d columns
    m1_col = None
    m2_col = None
    
    for col in df.columns:
        col_str = str(col)
        if 'm1_1d' in col_str:
            m1_col = col
        if 'm2_1d' in col_str:
            m2_col = col
    
    logger.info(f"m1_1d column: {m1_col}")
    logger.info(f"m2_1d column: {m2_col}")
    
    if m1_col is None or m2_col is None:
        logger.error("Could not find m1_1d or m2_1d columns")
        return None
    
    data = {
        'bubbles_time': df[bubbles_time_col].values,
        'm1_profiles': np.array([np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x]) for x in df[m1_col]]),
        'm2_profiles': np.array([np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x]) for x in df[m2_col]]),
        'shot_names': df.index.astype(str).values if 'shot_name' not in df.columns else df['shot_name'].values,
    }
    
    logger.info(f"Extracted {len(data['bubbles_time'])} shots")
    logger.info(f"  bubbles_time: {data['bubbles_time'].shape}")
    logger.info(f"  m1_profiles: {data['m1_profiles'].shape}")
    logger.info(f"  m2_profiles: {data['m2_profiles'].shape}")
    
    return data


def extract_raw_data_from_multishot(df):
    """
    Extract raw data from multishot_lib DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from get_and_filter_shots (waterfall_v2_example)
    
    Returns
    -------
    dict
        Dictionary with 'bubbles_time', 'm1_profiles', 'm2_profiles', 'shot_names'
    """
    logger.info(f"\nExtracting raw data from multishot DataFrame...")
    
    # Look for bubbles_time in columns
    bubbles_time_col = None
    for col in df.columns:
        if isinstance(col, tuple):
            if 'bubbles_time' in col[0]:
                bubbles_time_col = col
        elif 'bubbles_time' in str(col).lower():
            bubbles_time_col = col
    
    if bubbles_time_col is None:
        logger.error("Could not find bubbles_time column")
        logger.info(f"Available columns: {list(df.columns)[:20]}")
        return None
    
    logger.info(f"Found bubbles_time column: {bubbles_time_col}")
    
    # Look for m1_1d and m2_1d columns (with MultiIndex)
    m1_col = None
    m2_col = None
    
    for col in df.columns:
        if isinstance(col, tuple):
            if col[0] == 'show_ODs_v2' and col[1] == 'm1_1d':
                m1_col = col
            if col[0] == 'show_ODs_v2' and col[1] == 'm2_1d':
                m2_col = col
    
    logger.info(f"m1_1d column: {m1_col}")
    logger.info(f"m2_1d column: {m2_col}")
    
    if m1_col is None or m2_col is None:
        logger.error("Could not find m1_1d or m2_1d columns")
        logger.info(f"Available columns: {list(df.columns)[:20]}")
        return None
    
    # Extract shot names from 'shot_name' column if available
    shot_names = df['shot_name'].values if ('show_ODs_v2', 'shot_name') in df.columns or 'shot_name' in df.columns else df.index.astype(str).values
    
    data = {
        'bubbles_time': df[bubbles_time_col].values,
        'm1_profiles': np.array([np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x]) for x in df[m1_col]]),
        'm2_profiles': np.array([np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x]) for x in df[m2_col]]),
        'shot_names': shot_names,
    }
    
    logger.info(f"Extracted {len(data['bubbles_time'])} shots")
    logger.info(f"  bubbles_time: {data['bubbles_time'].shape}")
    logger.info(f"  m1_profiles: {data['m1_profiles'].shape}")
    logger.info(f"  m2_profiles: {data['m2_profiles'].shape}")
    
    return data


def save_raw_data(data, output_prefix):
    """
    Save raw data to CSV files.
    
    Parameters
    ----------
    data : dict
        Dictionary with bubbles_time, m1_profiles, m2_profiles, shot_names
    output_prefix : str
        Prefix for output files
    """
    if data is None:
        logger.error(f"No data to save for {output_prefix}")
        return
    
    # Save bubbles_time
    df_time = pd.DataFrame({
        'shot_name': data['shot_names'],
        'bubbles_time': data['bubbles_time']
    })
    time_file = f"{output_prefix}_bubbles_time.csv"
    df_time.to_csv(time_file, index=False)
    logger.info(f"Saved: {time_file}")
    
    # Save m1 profiles (one row per shot, columns are pixel indices)
    max_len = max(len(p) for p in data['m1_profiles'])
    m1_data = np.full((len(data['m1_profiles']), max_len), np.nan)
    for i, profile in enumerate(data['m1_profiles']):
        m1_data[i, :len(profile)] = profile
    
    df_m1 = pd.DataFrame(m1_data)
    df_m1.insert(0, 'shot_name', data['shot_names'])
    m1_file = f"{output_prefix}_m1_profiles.csv"
    df_m1.to_csv(m1_file, index=False)
    logger.info(f"Saved: {m1_file} (shape: {m1_data.shape})")
    
    # Save m2 profiles
    max_len = max(len(p) for p in data['m2_profiles'])
    m2_data = np.full((len(data['m2_profiles']), max_len), np.nan)
    for i, profile in enumerate(data['m2_profiles']):
        m2_data[i, :len(profile)] = profile
    
    df_m2 = pd.DataFrame(m2_data)
    df_m2.insert(0, 'shot_name', data['shot_names'])
    m2_file = f"{output_prefix}_m2_profiles.csv"
    df_m2.to_csv(m2_file, index=False)
    logger.info(f"Saved: {m2_file} (shape: {m2_data.shape})")


def main():
    """Main function."""
    logger.info("\n" + "="*80)
    logger.info("DEBUG: Comparing raw data from waterfall_FLAT_v4 vs waterfall_v2_example")
    logger.info("="*80)
    
    # Suppress matplotlib interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    # Load waterfall_FLAT_v4 data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA FROM WATERFALL_FLAT_V4")
    logger.info("="*80)
    
    cfg_flat = load_module(
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_config.py",
        "waterfall_config_flat"
    )
    lib_flat = load_module(
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_lib.py",
        "waterfall_lib_flat"
    )
    
    # Load HDF data
    sys.path.insert(0, "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib")
    from main_lyse import get_day_data
    
    hdf_cfg = cfg_flat.HDF_CONFIG
    logger.info(f"HDF_CONFIG: {hdf_cfg}")
    
    df_flat = get_day_data(
        today=hdf_cfg.get('today', True),
        year=hdf_cfg.get('year'),
        month=hdf_cfg.get('month'),
        day=hdf_cfg.get('day'),
        include_lyse_live=False,
    )
    
    if df_flat is None:
        logger.error("Failed to load waterfall_FLAT_v4 data")
        return
    
    logger.info(f"Loaded {len(df_flat)} shots from waterfall_FLAT_v4")
    logger.info(f"Columns: {list(df_flat.columns)[:10]}...")
    
    # Filter by sequences and scan
    mode_name = cfg_flat.ACTIVE_MODE
    mode_cfg = cfg_flat.MODE_CONFIGS[mode_name]
    seqs = cfg_flat.SEQS if cfg_flat.SEQS else mode_cfg['seqs']
    
    logger.info(f"Filtering for sequences: {seqs}")
    df_flat_filtered = lib_flat.filter_dataframe(df_flat, seqs, constraints=mode_cfg.get('constraints'))
    logger.info(f"After filtering: {len(df_flat_filtered)} shots")
    
    # Extract raw data from waterfall_FLAT_v4
    data_flat = extract_raw_data_from_lyse(df_flat_filtered, data_origin=mode_cfg.get('data_origin', 'show_ODs'))
    save_raw_data(data_flat, "raw_data_waterfall_flat_v4")
    
    # Load waterfall_v2_example data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA FROM WATERFALL_V2_EXAMPLE")
    logger.info("="*80)
    
    cfg_v2 = load_module(
        "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2/waterfall_config.py",
        "waterfall_config_v2"
    )
    
    from multishot_lib import get_and_filter_shots
    from waterfall_v2.common_config import MAGNETIZATION_MODALITIES
    
    hdf_cfg_v2 = cfg_v2.HDF_CONFIG
    logger.info(f"HDF_CONFIG: {hdf_cfg_v2}")
    
    mode_name_v2 = cfg_v2.ACTIVE_MODE
    import importlib
    mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{mode_name_v2}')
    MODE_CONFIG_V2 = mode_config_module.MODE_CONFIG
    
    mag_modality_name = MODE_CONFIG_V2['magnetization_modality']
    mag_modality = MAGNETIZATION_MODALITIES[mag_modality_name]
    camera_name = mag_modality['camera_name']
    
    logger.info(f"Camera: {camera_name}")
    
    seqs_v2 = cfg_v2.SEQS if cfg_v2.SEQS is not None else None
    logger.info(f"Sequences: {seqs_v2}")
    
    df_v2 = get_and_filter_shots(
        hdf_config=hdf_cfg_v2,
        camera_name=camera_name,
        seqs=seqs_v2
    )
    
    if df_v2 is None or df_v2.empty:
        logger.error("Failed to load waterfall_v2_example data")
        return
    
    logger.info(f"Loaded {len(df_v2)} shots from waterfall_v2_example")
    logger.info(f"Columns: {list(df_v2.columns)[:10]}...")
    
    # Extract raw data from waterfall_v2_example
    data_v2 = extract_raw_data_from_multishot(df_v2)
    save_raw_data(data_v2, "raw_data_waterfall_v2_example")
    
    # Compare
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    
    if data_flat and data_v2:
        logger.info(f"\nwaterfall_FLAT_v4: {len(data_flat['bubbles_time'])} shots")
        logger.info(f"waterfall_v2_example: {len(data_v2['bubbles_time'])} shots")
        
        logger.info(f"\nbubbles_time values (FLAT_v4):\n  {sorted(np.unique(data_flat['bubbles_time']))}")
        logger.info(f"bubbles_time values (v2_example):\n  {sorted(np.unique(data_v2['bubbles_time']))}")
        
        # Check if shot names match
        shots_flat = set(data_flat['shot_names'])
        shots_v2 = set(data_v2['shot_names'])
        
        common = shots_flat & shots_v2
        only_flat = shots_flat - shots_v2
        only_v2 = shots_v2 - shots_flat
        
        logger.info(f"\nShot name comparison:")
        logger.info(f"  Common shots: {len(common)}")
        logger.info(f"  Only in FLAT_v4: {len(only_flat)}")
        logger.info(f"  Only in v2_example: {len(only_v2)}")
        
        if only_flat:
            logger.info(f"    Sample: {list(only_flat)[:5]}")
        if only_v2:
            logger.info(f"    Sample: {list(only_v2)[:5]}")
    
    logger.info("\n✓ Raw data files saved. You can now compare:")
    logger.info("  - raw_data_waterfall_flat_v4_bubbles_time.csv")
    logger.info("  - raw_data_waterfall_v2_example_bubbles_time.csv")
    logger.info("  - raw_data_waterfall_flat_v4_m1_profiles.csv")
    logger.info("  - raw_data_waterfall_v2_example_m1_profiles.csv")
    logger.info("  - raw_data_waterfall_flat_v4_m2_profiles.csv")
    logger.info("  - raw_data_waterfall_v2_example_m2_profiles.csv")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
