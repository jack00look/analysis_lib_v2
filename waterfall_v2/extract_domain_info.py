"""
Example script: Extract and save domain information for KZ_det_scan_window mode.

This script shows how to use the domain_extraction_lib to:
1. Load shots from HDF files
2. Extract domain boundaries and signs for each shot
3. Save domain information to CSV and JSON files
"""

import sys
import logging
import numpy as np
import pandas as pd

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
from waterfall.domain_extraction_lib import (
    create_domain_info_per_shot,
    save_domain_info_to_csv,
    save_domain_info_to_json
)
from waterfall_v2.common_config import PARAMS


def main():
    """
    Main function to extract and save domain information.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"DOMAIN EXTRACTION - KZ_det_scan_window MODE")
    logger.info(f"{'='*80}\n")
    
    # Load shots
    logger.info("Loading shots from HDF files...")
    df = get_day_data(HDF_CONFIG)
    
    if df is None or df.empty:
        logger.error("No shots loaded. Exiting.")
        return
    
    logger.info(f"✓ Loaded {len(df)} shots\n")
    
    # Determine output directory
    output_dir = './results/domain_info'
    
    # Extract domain information for all shots
    logger.info("Extracting domain information for each shot...")
    domain_info_dict = create_domain_info_per_shot(
        df,
        seqs=SEQS,
        scan='KZ_det_scan_window',
        data_origin='show_ODs_v2',
        mode_cfg=MODE_CONFIG,
        params=PARAMS
    )
    
    if not domain_info_dict:
        logger.warning("No domain information extracted.")
        return
    
    logger.info(f"✓ Extracted domain info for {len(domain_info_dict)} shots\n")
    
    # Display sample
    logger.info("Sample domain info (first shot):")
    first_shot_idx = min(domain_info_dict.keys())
    info = domain_info_dict[first_shot_idx]
    logger.info(f"  Shot {first_shot_idx}:")
    logger.info(f"    Field: {info['field']:.6g}")
    logger.info(f"    Domains: {info['n_domains']}")
    logger.info(f"    Defects: {info['defect_count']} (pos: {info['n_pos_defects']}, neg: {info['n_neg_defects']})")
    for i, domain in enumerate(info['domains']):
        logger.info(f"      Domain {i+1}: [{domain['x_start_um']:.1f}, {domain['x_end_um']:.1f}] μm, sign={domain['sign']:+d}, color={domain['color']}")
    
    # Save to files
    logger.info(f"\nSaving domain information...")
    csv_path = save_domain_info_to_csv(domain_info_dict, output_dir=output_dir)
    json_path = save_domain_info_to_json(domain_info_dict, output_dir=output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("Domain extraction complete!")
    logger.info(f"CSV output:  {csv_path}")
    logger.info(f"JSON output: {json_path}")
    logger.info(f"{'='*80}\n")
    
    return domain_info_dict


if __name__ == '__main__':
    domain_info = main()
