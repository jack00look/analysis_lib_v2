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
from multishot_lib import get_and_filter_shots

# Import main config
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG

# Import common config
from waterfall_v2.common_config import MAGNETIZATION_MODALITIES

# Auto-load mode config based on ACTIVE_MODE
import importlib
mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
MODE_CONFIG = mode_config_module.MODE_CONFIG

# Import plotting library
from waterfall_v2.waterfall_lib import waterfall_plot
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
    
    # Load and filter shots using multishot_lib
    logger.info(f"\n{'='*80}")
    logger.info("LOADING AND FILTERING SHOTS")
    logger.info(f"{'='*80}\n")
    
    df = get_and_filter_shots(
        hdf_config=HDF_CONFIG,
        camera_name=camera_name,
        seqs=seqs_to_load
    )
    
    if df.empty:
        logger.error("\nNo shots passed filtering. Exiting.")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("SHOTS LOADED SUCCESSFULLY")
    logger.info(f"{'='*80}\n")
    
    # Show summary
    logger.info(f"Shape of DataFrame: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:10]}... (showing first 10)")
    
    # Display shot information if available
    if 'shot_name' in df.columns:
        logger.info(f"\nShot names:")
        for shot_name in df['shot_name'].head(10).values:
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
    
    # Get plot configuration
    plot_flags = MODE_CONFIG['plots']
    constraints = MODE_CONFIG.get('constraints')
    average = MODE_CONFIG.get('average', False)
    globals_in_title = MODE_CONFIG.get('globals_in_title', [])
    
    try:
        logger.info("Preparing data for waterfall_plot...")
        
        # Data is already in correct lyse format.
        # waterfall_lib now handles tuple column format natively.
        df_waterfall = df.copy()
        
        logger.info(f"Calling waterfall_plot...")
        
        waterfall_plot(
            df=df_waterfall,
            seqs=seqs_to_load if seqs_to_load else None,
            scan=MODE_CONFIG['scan'],
            data_origin=MODE_CONFIG['data_origin'],
            constraints=constraints,
            average=average,
            title_labels=None,
            globals_in_title=globals_in_title,
            params=PARAMS,
            plot_flags=plot_flags,
            mode_cfg=MODE_CONFIG
        )
        logger.info("✓ Waterfall plots generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"\n{'='*80}")
    logger.info("Ready for further analysis (e.g., waterfall plots, fitting, etc.)")
    logger.info(f"{'='*80}\n")
    
    return df


if __name__ == '__main__':
    df = main()
