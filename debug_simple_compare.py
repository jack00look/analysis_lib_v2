"""
Simple debug script to directly compare raw data from both sources.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib')

import matplotlib
matplotlib.use('Agg')

def main():
    """Load data from multishot_lib with different settings and compare."""
    logger.info("\n" + "="*80)
    logger.info("COMPARING MULTISHOT_LIB LOADS WITH DIFFERENT CONFIGS")
    logger.info("="*80)
    
    # Load 1: With reanalyzed=False (from NAS)
    logger.info("\n" + "="*80)
    logger.info("LOADING 1: reanalyzed=False (from NAS)")
    logger.info("="*80)
    
    from multishot_lib import get_and_filter_shots
    
    df1 = get_and_filter_shots(
        hdf_config={
            'today': False,
            'year': 2026,
            'month': 5,
            'day': 7,
            'reanalyzed': False,
        },
        camera_name='cam_vert1',
        seqs=[29, 30, 34, 35]
    )
    
    logger.info(f"✓ Loaded {len(df1)} shots")
    logger.info(f"  Shape: {df1.shape}")
    
    # Find bubbles_time
    bubbles_col1 = None
    for col in df1.columns:
        if isinstance(col, tuple):
            if 'bubbles_time' in col[0]:
                bubbles_col1 = col
                break
        else:
            if 'bubbles_time' in str(col).lower():
                bubbles_col1 = col
                break
    
    if bubbles_col1:
        logger.info(f"\nBubbles_time column: {bubbles_col1}")
        times1 = sorted(df1[bubbles_col1].unique())
        logger.info(f"Unique times: {times1}")
        
        # Save for comparison
        df_times1 = pd.DataFrame({'bubbles_time': times1, 'source': 'NAS'})
        df_times1.to_csv('raw_data_nas_bubbles_time.csv', index=False)
        logger.info(f"Saved: raw_data_nas_bubbles_time.csv ({len(times1)} unique times)")
    
    # Load 2: With reanalyzed=True (from data_reanalyzed)
    logger.info("\n" + "="*80)
    logger.info("LOADING 2: reanalyzed=True (from data_reanalyzed)")
    logger.info("="*80)
    
    df2 = get_and_filter_shots(
        hdf_config={
            'today': False,
            'year': 2026,
            'month': 5,
            'day': 7,
            'reanalyzed': True,
        },
        camera_name='cam_vert1',
        seqs=[29, 30, 34, 35]
    )
    
    logger.info(f"✓ Loaded {len(df2)} shots")
    logger.info(f"  Shape: {df2.shape}")
    
    # Find bubbles_time
    bubbles_col2 = None
    for col in df2.columns:
        if isinstance(col, tuple):
            if 'bubbles_time' in col[0]:
                bubbles_col2 = col
                break
        else:
            if 'bubbles_time' in str(col).lower():
                bubbles_col2 = col
                break
    
    if bubbles_col2:
        logger.info(f"\nBubbles_time column: {bubbles_col2}")
        times2 = sorted(df2[bubbles_col2].unique())
        logger.info(f"Unique times: {times2}")
        
        # Save for comparison
        df_times2 = pd.DataFrame({'bubbles_time': times2, 'source': 'reanalyzed'})
        df_times2.to_csv('raw_data_reanalyzed_bubbles_time.csv', index=False)
        logger.info(f"Saved: raw_data_reanalyzed_bubbles_time.csv ({len(times2)} unique times)")
    
    # Compare
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    if bubbles_col1 and bubbles_col2:
        logger.info(f"\nNAS shots: {len(df1)}")
        logger.info(f"Reanalyzed shots: {len(df2)}")
        
        set1 = set(times1)
        set2 = set(times2)
        
        logger.info(f"\nBubbles_time ranges:")
        logger.info(f"  NAS: {min(times1):.1f} - {max(times1):.1f} ({len(times1)} unique values)")
        logger.info(f"  Reanalyzed: {min(times2):.1f} - {max(times2):.1f} ({len(times2)} unique values)")
        
        common = set1 & set2
        only_nas = set1 - set2
        only_rean = set2 - set1
        
        logger.info(f"\nTime value overlap:")
        logger.info(f"  Common: {len(common)} values")
        logger.info(f"  Only in NAS: {len(only_nas)} values - {sorted(only_nas) if only_nas else 'none'}")
        logger.info(f"  Only in reanalyzed: {len(only_rean)} values - {sorted(only_rean) if only_rean else 'none'}")
    
    logger.info(f"\n✓ Comparison complete. Check:")
    logger.info(f"  - raw_data_nas_bubbles_time.csv")
    logger.info(f"  - raw_data_reanalyzed_bubbles_time.csv")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
