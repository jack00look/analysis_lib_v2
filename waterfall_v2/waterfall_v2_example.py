"""
Waterfall v2 - Multishot analysis using refactored v2 libraries

This script demonstrates how to use the new multishot_lib to:
1. Load shots from a specific day
2. Check parameter consistency across all shots
3. Filter shots by quality metrics (saturation, background, norm_err, DMD update)
4. Return a DataFrame ready for further analysis

Example usage:
    python waterfall_v2_example.py
"""

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import multishot_lib
from multishot_lib import get_and_filter_shots, get_day_data

# Import main config
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG

# Import common config
from waterfall_v2.common_config import MAGNETIZATION_MODALITIES

# Auto-load mode config based on ACTIVE_MODE
import importlib
mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
MODE_CONFIG = mode_config_module.MODE_CONFIG

# Import plotting library
from waterfall.waterfall_lib import waterfall_plot
from waterfall_v2.common_config import PARAMS


def main():
    """
    Main waterfall analysis function.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"WATERFALL V2 - MULTISHOT ANALYSIS")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Mode: {ACTIVE_MODE}")
    logger.info(f"Scan: {MODE_CONFIG['scan']}")
    
    # Get magnetization modality
    mag_modality_name = MODE_CONFIG['magnetization_modality']
    mag_modality = MAGNETIZATION_MODALITIES[mag_modality_name]
    camera_name = mag_modality['camera_name']
    atoms_images = mag_modality['atoms_images']
    
    logger.info(f"Camera: {camera_name}")
    logger.info(f"Atom images: {atoms_images}")
    
    # Determine sequences to load
    seqs_to_load = SEQS if SEQS is not None else None
    if seqs_to_load:
        logger.info(f"Sequences: {seqs_to_load}")
    else:
        logger.info("Sequences: All available")
    
    # Load shots using get_day_data (same as waterfall_FLAT_v4)
    # This loads raw data without quality filtering, for comparison with waterfall_FLAT_v4
    logger.info(f"\n{'='*80}")
    logger.info("LOADING SHOTS (RAW DATA - NO QUALITY FILTERING)")
    logger.info(f"{'='*80}\n")
    
    df = get_day_data(HDF_CONFIG)
    
    if df is None or df.empty:
        logger.error("\nNo shots loaded. Exiting.")
        return
    
    logger.info(f"\n✓ Loaded {len(df)} total shots from HDF files")
    
    logger.info(f"\n{'='*80}")
    logger.info("SHOTS LOADED SUCCESSFULLY")
    logger.info(f"{'='*80}\n")
    
    # Show summary
    logger.info(f"Shape of DataFrame: {df.shape}")
    
    # Display shot information if available
    shot_name_col = None
    for col in df.columns:
        if isinstance(col, tuple) and col[1] == 'shot_name':
            shot_name_col = col
            break
        elif col == 'shot_name':
            shot_name_col = col
            break
    
    if shot_name_col:
        logger.info(f"\nShot names (first 10):")
        for shot_name in df[shot_name_col].head(10).values:
            logger.info(f"  - {shot_name}")
        if len(df) > 10:
            logger.info(f"  ... and {len(df) - 10} more")
    else:
        logger.info(f"\nLoaded {len(df)} shots from the DataFrame")
    
    # At this point, the multishot script can use df to extract:
    # - n1D projections (if available in results)
    # - Processing parameters (cam_vert1_param_*)
    # - Quality metrics (cam_vert1_*_cnt_rel_atoms_sat, etc.)
    # - Any other analysis results stored in HDF5
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING WATERFALL PLOTS")
    logger.info(f"{'='*80}\n")
    
    try:
        logger.info("Preparing mode config...")
        
        # Set seqs in mode config (required by run_mode)
        MODE_CONFIG['seqs'] = seqs_to_load
        
        # Inject shared configs into mode runtime payload (like waterfall_FLAT_v4 does)
        from waterfall_v2.common_config import SHOT_FILTER_CONFIG, DEFECT_ANALYSIS_PARAMS
        MODE_CONFIG['shot_filter'] = dict(SHOT_FILTER_CONFIG)
        MODE_CONFIG['magnetization_modalities'] = dict(MAGNETIZATION_MODALITIES)
        MODE_CONFIG['defect_analysis_global'] = dict(DEFECT_ANALYSIS_PARAMS)
        
        logger.info("Calling run_mode to generate waterfall plots...")
        from waterfall.waterfall_lib import run_mode
        run_mode(df, MODE_CONFIG, PARAMS)
        logger.info("✓ Waterfall plots generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.info(f"\n{'='*80}")
    logger.info("Ready for further analysis (e.g., waterfall plots, fitting, etc.)")
    logger.info(f"{'='*80}\n")
    
    return df


if __name__ == '__main__':
    df = main()
