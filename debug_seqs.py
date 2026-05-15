#!/usr/bin/env python3
"""Check available sequences"""
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
print("Total shots:", len(df))

# Check sequences
seqs_in_df = df[('sequence', '')].unique()
print(f"\nUnique sequences: {sorted(seqs_in_df)}")

# Check if [29,30,34,35] are in there
target_seqs = [29, 30, 34, 35]
for seq in target_seqs:
    count = len(df[df[('sequence', '')] == seq])
    print(f"  seq {seq}: {count} shots")
