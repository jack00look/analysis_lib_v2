#!/usr/bin/env python3
"""Check what columns exist in the 2026-05-07 data"""
import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from multishot_lib import get_day_data

HDF_CONFIG = {
    'raw_or_processed': 'raw',
    'day': 7,
    'month': 5,
    'year': 2026,
}

df = get_day_data(HDF_CONFIG)
print("DataFrame shape:", df.shape)
print("\nColumn names (first 30):")
for i, col in enumerate(df.columns[:30]):
    print(f"  {i}: {col}")

print("\n\nSearching for m1_1d and m2_1d columns...")
m1_cols = [c for c in df.columns if 'm1_1d' in str(c)]
m2_cols = [c for c in df.columns if 'm2_1d' in str(c)]
print(f"m1_1d columns: {m1_cols}")
print(f"m2_1d columns: {m2_cols}")

print("\n\nSearching for bubbles columns...")
bubbles_cols = [c for c in df.columns if 'bubbles' in str(c)]
for col in bubbles_cols[:20]:
    print(f"  {col}")
