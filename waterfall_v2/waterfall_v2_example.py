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

# Import config
from waterfall_v2.waterfall_config_v2 import (
    HDF_CONFIG, ACTIVE_MODE, SEQS, MODE_CONFIGS, MAGNETIZATION_MODALITIES
)

# Import imaging lib for further processing
import importlib.util
spec = importlib.util.spec_from_file_location(
    "imaging_analysis_lib.camera_settings",
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py"
)
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)


def main():
    """
    Main waterfall analysis function.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"WATERFALL V2 - MULTISHOT ANALYSIS")
    logger.info(f"{'='*80}\n")
    
    # Get mode configuration
    if ACTIVE_MODE not in MODE_CONFIGS:
        logger.error(f"Unknown mode: {ACTIVE_MODE}")
        return
    
    mode_config = MODE_CONFIGS[ACTIVE_MODE]
    logger.info(f"Mode: {ACTIVE_MODE}")
    logger.info(f"Scan: {mode_config['scan']}")
    
    # Get magnetization modality
    mag_modality_name = mode_config['magnetization_modality']
    mag_modality = MAGNETIZATION_MODALITIES[mag_modality_name]
    camera_name = mag_modality['camera_name']
    atoms_images = mag_modality['atoms_images']
    
    logger.info(f"Camera: {camera_name}")
    logger.info(f"Atom images: {atoms_images}")
    
    # Determine sequences to load
    seqs_to_load = SEQS if SEQS is not None else mode_config.get('seqs', None)
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
    logger.info(f"\nShot names:")
    for shot_name in df['shot_name'].head(10).values:
        logger.info(f"  - {shot_name}")
    if len(df) > 10:
        logger.info(f"  ... and {len(df) - 10} more")
    
    # At this point, the multishot script can use df to extract:
    # - n1D projections (if available in results)
    # - Processing parameters (cam_vert1_param_*)
    # - Quality metrics (cam_vert1_*_cnt_rel_atoms_sat, etc.)
    # - Any other analysis results stored in HDF5
    
    logger.info(f"\n{'='*80}")
    logger.info("Ready for further analysis (e.g., waterfall plots, fitting, etc.)")
    logger.info(f"{'='*80}\n")
    
    return df


if __name__ == '__main__':
    df = main()
