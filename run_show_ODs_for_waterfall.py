"""
Quick runner for show_ODs_v2 analysis on waterfall_config_v2 sequences.

This script uses the SEQS and HDF_CONFIG from waterfall_config_v2 directly,
without requiring interactive prompts.

Run as:
    python run_show_ODs_for_waterfall.py
"""

import sys

# Load waterfall config
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2')
import waterfall_config_v2

# Copy config to run_show_ODs_batch
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')
import run_show_ODs_batch

# Set configuration from waterfall_config_v2
run_show_ODs_batch.SEQS = waterfall_config_v2.SEQS
run_show_ODs_batch.HDF_CONFIG = waterfall_config_v2.HDF_CONFIG

print(f"Using configuration from waterfall_config_v2:")
print(f"  SEQS = {run_show_ODs_batch.SEQS}")
print(f"  HDF_CONFIG = {run_show_ODs_batch.HDF_CONFIG}")

# Run main
if __name__ == '__main__':
    try:
        success = run_show_ODs_batch.main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
