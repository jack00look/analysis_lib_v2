#!/usr/bin/env python3
"""Check sequence column"""
import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from multishot_lib import get_day_data

HDF_CONFIG = {
    'raw_or_processed': 'raw',
    'day': 7,
    'month': 5,
    'year': 2026,
}

# Load raw data
df = get_day_data(HDF_CONFIG)
print("DataFrame columns (first 20):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i}: {col}")

print("\n\nLooking for 'sequence' columns:")
seq_cols = [c for c in df.columns if 'sequence' in str(c).lower()]
print(seq_cols)

print("\n\nChecking what's in df index:")
print("Index name:", df.index.name)
print("Index values (first 10):", df.index.values[:10])
