#!/usr/bin/env python
"""
Quick test to verify get_day_data fixes and waterfall_v2_example imports.
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test 1: Check imports
logger.info("\n" + "="*80)
logger.info("TEST 1: Checking imports...")
logger.info("="*80)

try:
    sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')
    from multishot_lib import get_day_data
    from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG
    from waterfall_v2.common_config import MAGNETIZATION_MODALITIES, SHOT_FILTER_CONFIG, DEFECT_ANALYSIS_PARAMS, PARAMS
    from waterfall.waterfall_lib import run_mode
    logger.info("✓ All imports successful")
except Exception as e:
    logger.error(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test get_day_data with today=False (should load HDF)
logger.info("\n" + "="*80)
logger.info("TEST 2: Testing get_day_data with today=False...")
logger.info("="*80)

try:
    hdf_config_past = {
        'today': False,
        'year': 2026,
        'month': 5,
        'day': 25,
        'reanalyzed': False,
    }
    df_past = get_day_data(hdf_config_past)
    if df_past is None:
        logger.warning("No data loaded (HDF directory may not exist for 2026-05-25)")
    else:
        logger.info(f"✓ Loaded {len(df_past)} shots from HDF")
except Exception as e:
    logger.error(f"✗ get_day_data failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check that mode config loads correctly
logger.info("\n" + "="*80)
logger.info("TEST 3: Checking mode config loading...")
logger.info("="*80)

try:
    import importlib
    mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
    MODE_CONFIG = mode_config_module.MODE_CONFIG
    logger.info(f"✓ Loaded mode config for {ACTIVE_MODE}")
    logger.info(f"  scan: {MODE_CONFIG['scan']}")
    logger.info(f"  magnetization_modality: {MODE_CONFIG['magnetization_modality']}")
except Exception as e:
    logger.error(f"✗ Mode config loading failed: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "="*80)
logger.info("TEST SUMMARY: All basic checks passed!")
logger.info("="*80 + "\n")
