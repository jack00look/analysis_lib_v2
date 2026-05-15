#!/usr/bin/env python3
"""Compare magnetization sign between two analyses"""
import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib')

import numpy as np
from main_lyse import get_day_data as get_day_data_v4

# Load data like waterfall_FLAT_v4 does
df = get_day_data_v4(today=False, year=2026, month=5, day=7, include_lyse_live=False)
print("Loaded", len(df), "shots")

# Filter to sequence 29 (has 17 shots)
df_seq29 = df[df['sequence_index'] == 29]
print(f"Sequence 29: {len(df_seq29)} shots")

if len(df_seq29) > 0:
    shot = df_seq29.iloc[0]
    print(f"\nFirst shot in sequence 29:")
    print(f"  bubbles_time: {shot.get(('bubbles_time', ''), 'N/A')}")
    
    # Try to get m1_1d and m2_1d
    m1_col = ('show_ODs_v2', 'm1_1d')
    m2_col = ('show_ODs_v2', 'm2_1d')
    
    if m1_col in shot.index:
        m1 = shot[m1_col]
        print(f"  {m1_col} shape: {getattr(m1, 'shape', len(m1))}")
    if m2_col in shot.index:
        m2 = shot[m2_col]
        print(f"  {m2_col} shape: {getattr(m2, 'shape', len(m2))}")
        
    # Check what m1/m2 columns actually exist
    print(f"\n  All columns in shot (first 30):")
    cols = list(shot.index)[:30]
    for col in cols:
        val = shot[col]
        if isinstance(val, (list, np.ndarray)):
            print(f"    {col}: array-like")
        else:
            print(f"    {col}: {val}")
