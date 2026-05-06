#!/usr/bin/env python
"""Benchmark consolidation loading performance."""

import sys
import h5py
import os
import time
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import glob

sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

NUM_WORKERS = max(1, cpu_count() - 1)

# Find test HDF files
date_str = '2026-05-04'
seq_num = 45
hdf_root = f'/home/rick/NAS542_dataBEC2/{os.getenv("USER", "rick")}/data/{date_str[:4]}/{date_str[5:7]}/{date_str[8:10]}'
seq_pattern = f'{hdf_root}/seq_{seq_num:04d}/*.h5'
all_files = sorted(glob.glob(seq_pattern))[:300]  # Limit to first 300

print(f"Found {len(all_files)} files")

if not all_files:
    print("No files found!")
    sys.exit(1)

# Test SEQUENTIAL consolidation loading
print("\n" + "="*70)
print("SEQUENTIAL CONSOLIDATION LOADING")
print("="*70)

def load_consolidation_row_seq(hdf_file):
    """Load a single file's consolidation data."""
    try:
        with h5py.File(hdf_file, 'r') as h5file:
            if 'results' in h5file and 'show_ODs_v2' in h5file['results']:
                show_ODs_group = h5file['results']['show_ODs_v2']
                
                # Extract result values
                dict_results = {}
                for key in show_ODs_group.attrs.keys():
                    dict_results[key] = show_ODs_group.attrs[key]
                for key in show_ODs_group.keys():
                    try:
                        dict_results[key] = show_ODs_group[key][()]
                    except:
                        dict_results[key] = show_ODs_group[key][...]
                return True
    except:
        pass
    return False

start = time.time()
count = 0
for f in all_files:
    if load_consolidation_row_seq(f):
        count += 1
sequential_time = time.time() - start

print(f"Loaded: {count}/{len(all_files)}")
print(f"Time: {sequential_time:.2f}s")
print(f"Throughput: {len(all_files)/sequential_time:.1f} files/sec")

# Test PARALLEL consolidation loading
print("\n" + "="*70)
print("PARALLEL CONSOLIDATION LOADING (Workers: {})".format(NUM_WORKERS))
print("="*70)

start = time.time()
with Pool(NUM_WORKERS) as pool:
    results = list(pool.imap_unordered(load_consolidation_row_seq, all_files))
    count = sum(1 for r in results if r)
parallel_time = time.time() - start

print(f"Loaded: {count}/{len(all_files)}")
print(f"Time: {parallel_time:.2f}s")
print(f"Throughput: {len(all_files)/parallel_time:.1f} files/sec")

# Summary
print("\n" + "="*70)
print("SPEEDUP ANALYSIS")
print("="*70)
speedup = sequential_time / parallel_time
print(f"Speedup: {speedup:.1f}x faster with parallel")
print(f"Time saved: {sequential_time - parallel_time:.1f}s ({100*(sequential_time-parallel_time)/sequential_time:.0f}%)")
