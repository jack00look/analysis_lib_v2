#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Saves dataframe
"""

import pandas
import lyse
from pathlib import Path
import h5py

# I want to always save the dataframes to the NAS, in the correct folder for the day
save_path = Path('/home/rick/NAS542_dataBEC2')

df = lyse.data()
filepath = Path(df['filepath'][0])

# I know, it's hardcoding but I didn't find a smarter way
#year, month, day = filepath.parts[-2:]

df_dir = filepath.parent.parent

# create directory if not existing
if not df_dir.exists():
    df_dir.mkdir(parents = True, exist_ok = True)

#get sequence_index from the first file
with h5py.File(filepath) as h5file:
    attrs = h5file.attrs
    sequence_id = attrs['sequence_index']

df_path = df_dir / f'seq_{sequence_id}.hdf'
#df_path = df_dir / 'data_demler_modulation.hdf'

#Drop first level of Multiindex, which is just
#2023-01-01 00:00:00+01:00 and equal for all shots
df = df.droplevel(0)


df.to_hdf(df_path,
          key = str(filepath.name[:-8]),
          mode = 'w',
          format = 'fixed')

print(f'saving DataFrame to : {df_path}')
