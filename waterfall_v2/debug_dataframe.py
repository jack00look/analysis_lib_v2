"""
Debug script to inspect DataFrame structure and column names.

Use this to understand the column layout of your HDF data.
"""

import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from multishot_lib import get_day_data
from waterfall_v2.waterfall_config import HDF_CONFIG

print("\n" + "="*80)
print("DATAFRAME STRUCTURE DEBUG")
print("="*80 + "\n")

# Load data
df = get_day_data(HDF_CONFIG)

if df is None or df.empty:
    print("ERROR: No data loaded!")
    sys.exit(1)

print(f"Shape: {df.shape}")
print(f"Data type: {type(df)}\n")

print("Column names (first 30):")
for i, col in enumerate(df.columns[:30]):
    col_type = type(df[col].iloc[0]) if len(df) > 0 else "unknown"
    col_shape = df[col].iloc[0].shape if hasattr(df[col].iloc[0], 'shape') else "scalar"
    print(f"  {i:2d}. {col} (type: {col_type.__name__}, shape: {col_shape})")

print(f"\nTotal columns: {len(df.columns)}")

# Search for specific columns
print("\n" + "-"*80)
print("SEARCHING FOR KEY COLUMNS:")
print("-"*80 + "\n")

# Magnetization
mag_cols = [col for col in df.columns if 'mag' in str(col).lower()]
print(f"Magnetization columns ({len(mag_cols)}):")
for col in mag_cols[:5]:
    print(f"  - {col}")

# Field
field_cols = [col for col in df.columns if 'arpkz' in str(col).lower() or 'field' in str(col).lower()]
print(f"\nField columns ({len(field_cols)}):")
for col in field_cols[:5]:
    print(f"  - {col}")

# Shot info
shot_cols = [col for col in df.columns if 'shot' in str(col).lower()]
print(f"\nShot/Name columns ({len(shot_cols)}):")
for col in shot_cols[:5]:
    print(f"  - {col}")

print("\n" + "="*80)
