"""
Example script: Sliding window analysis for KZ_det_scan_window mode.

This script demonstrates how to:
1. Load shots from HDF files
2. Extract domain information
3. Perform sliding window analysis
4. Find field where magnetization is closest to zero in each window
5. Track average number of domains across field values
6. Save results for further analysis
"""

import sys
import logging
import numpy as np

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
from multishot_lib import get_day_data
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG
from waterfall_v2.mode_configs.KZ_det_scan_window import MODE_CONFIG
from waterfall.domain_extraction_lib import create_domain_info_per_shot
from waterfall.window_analysis_lib import (
    sliding_window_analysis,
    save_window_analysis_results,
    save_window_analysis_csv,
    results_to_dataframe
)
from waterfall_v2.common_config import PARAMS


def main():
    """
    Main function to perform sliding window analysis.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"SLIDING WINDOW ANALYSIS - KZ_det_scan_window MODE")
    logger.info(f"{'='*80}\n")
    
    # Load shots
    logger.info("Loading shots from HDF files...")
    df = get_day_data(HDF_CONFIG)
    
    if df is None or df.empty:
        logger.error("No shots loaded. Exiting.")
        return
    
    logger.info(f"✓ Loaded {len(df)} shots\n")
    
    # Extract domain information
    logger.info("Extracting domain information for all shots...")
    domain_info_dict = create_domain_info_per_shot(
        df,
        seqs=SEQS,
        scan='KZ_det_scan_window',
        data_origin='show_ODs_v2',
        mode_cfg=MODE_CONFIG,
        params=PARAMS
    )
    
    if domain_info_dict:
        logger.info(f"✓ Extracted domain info for {len(domain_info_dict)} shots\n")
    else:
        logger.warning("No domain info extracted, continuing without it.\n")
    
    # Get window parameters from config
    window_size_um = float(MODE_CONFIG.get('window_size', 80.0))
    window_step_um = float(MODE_CONFIG.get('window_step', 5.0))
    x_min_um = float(PARAMS.get('DOMAIN_WALL_X_MIN', 0.0))
    x_max_um = float(PARAMS.get('DOMAIN_WALL_X_MAX', 200.0))
    
    logger.info(f"Window parameters:")
    logger.info(f"  X range: [{x_min_um:.1f}, {x_max_um:.1f}] μm")
    logger.info(f"  Window size: {window_size_um:.1f} μm")
    logger.info(f"  Window step: {window_step_um:.1f} μm\n")
    
    # Perform sliding window analysis
    logger.info("Performing sliding window analysis...\n")
    results = sliding_window_analysis(
        df,
        x_min_um=x_min_um,
        x_max_um=x_max_um,
        window_size_um=window_size_um,
        window_step_um=window_step_um,
        mode_cfg=MODE_CONFIG,
        params=PARAMS,
        domain_info_dict=domain_info_dict
    )
    
    if not results:
        logger.error("No valid windows analyzed.")
        return
    
    logger.info(f"\n✓ Analyzed {len(results)} windows\n")
    
    # Display sample results
    logger.info("Sample results:")
    for i, result in enumerate(results[:3]):  # Show first 3 windows
        logger.info(f"\nWindow {result['window_num']}:")
        logger.info(f"  Position: [{result['window_start_um']:.1f}, {result['window_end_um']:.1f}] μm")
        logger.info(f"  Center: {result['window_center_um']:.1f} μm")
        logger.info(f"  Field at mag=0: {result['field_at_zero']:.6g}")
        logger.info(f"  Mag at zero: {result['mag_mean_at_zero']:.5f} ± {result['mag_sem_at_zero']:.5f}")
        logger.info(f"  Domains at zero: {result['domain_mean_at_zero']:.2f} ± {result['domain_std_at_zero']:.2f}")
        logger.info(f"  Shots: {result['n_shots_at_zero']}")
    
    # Save results
    output_dir = './results/window_analysis'
    logger.info(f"\nSaving results to {output_dir}...")
    
    json_path = save_window_analysis_results(results, output_dir=output_dir)
    csv_path = save_window_analysis_csv(results, output_dir=output_dir)
    
    # Create summary DataFrame
    df_results = results_to_dataframe(results)
    
    logger.info(f"\n{'='*80}")
    logger.info("Sliding window analysis complete!")
    logger.info(f"Results saved to:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV:  {csv_path}")
    logger.info(f"{'='*80}\n")
    
    logger.info("\nResults summary:")
    logger.info(df_results.to_string())
    
    return results, df_results


if __name__ == '__main__':
    results, df_results = main()
