#!/usr/bin/env python3
"""Compare magnetization extraction between waterfall_FLAT_v4 and waterfall_v2"""
import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from multishot_lib import get_day_data
import numpy as np

HDF_CONFIG = {
    'raw_or_processed': 'raw',
    'day': 7,
    'month': 5,
    'year': 2026,
}

# Load raw data
df = get_day_data(HDF_CONFIG)
print("Loaded DataFrame shape:", df.shape)

# Filter to the sequences we're analyzing
seqs = [29, 30, 34, 35]
filtered_df = df[df[('sequence', '')].isin(seqs)]
print(f"\nFiltered to sequences {seqs}: {len(filtered_df)} shots")

# Get one shot and examine m1_1d, m2_1d
shot = filtered_df.iloc[0]
print("\nFirst shot data:")
print(f"  sequence: {shot.get('sequence', 'N/A')}")
print(f"  bubbles_time: {shot.get(('bubbles_time', ''), 'N/A')}")

# Check column names
m1_col = ('show_ODs_v2', 'm1_1d')
m2_col = ('show_ODs_v2', 'm2_1d')

print(f"\n{m1_col} shape:", np.array(shot[m1_col]).shape if m1_col in shot.index else "NOT FOUND")
print(f"{m2_col} shape:", np.array(shot[m2_col]).shape if m2_col in shot.index else "NOT FOUND")

if m1_col in shot.index:
    m1_data = np.array(shot[m1_col])
    print(f"\n{m1_col}[:5] = {m1_data[:5] if len(m1_data) > 0 else 'empty'}")
    print(f"  min={m1_data.min():.3e}, max={m1_data.max():.3e}, mean={m1_data.mean():.3e}")

if m2_col in shot.index:
    m2_data = np.array(shot[m2_col])
    print(f"\n{m2_col}[:5] = {m2_data[:5] if len(m2_data) > 0 else 'empty'}")
    print(f"  min={m2_data.min():.3e}, max={m2_data.max():.3e}, mean={m2_data.mean():.3e}")

    # Check magnetization (m1 - m2)
    mag = m1_data - m2_data
    print(f"\n(m1 - m2)[:5] = {mag[:5]}")
    print(f"  min={mag.min():.3e}, max={mag.max():.3e}, mean={mag.mean():.3e}")
