import lyse
import os
import importlib
import datetime
import pandas as pd

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

def get_day_data(year,month,day,path = settings.bec2path):
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    year = str(year)
    path = path + '/' + year + '/' + month + '/' + day
    names = os.listdir(path)
    df_s = []
    for name in names:
        if name[-3:] == 'hdf':
            df = pd.read_hdf(path + '/' + name)
            df_s.append(df)
    current_time = datetime.datetime.now()
    year_now = str(current_time.year).zfill(4)
    month_now = str(current_time.month).zfill(2)
    day_now = str(current_time.day).zfill(2)
    if year == year_now and month == month_now and day == day_now:
        df = lyse.data()
        df_s.append(df)
    if len(df_s) == 0:
        print('No data found for this day')
        return
    elif len(df_s) == 1:
        return df_s[0]
    else:
        return pd.concat(df_s)