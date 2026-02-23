import lyse
import os
import importlib
import datetime
import pandas as pd
import getpass
username = getpass.getuser()
analysis_lib_v2_folder = "/home/{}/labscript-suite/userlib/analysislib/analysis_lib_v2".format(username)

# spec = importlib.util.spec_from_file_location("general_lib.settings", "{}/general_lib/settings.py".format(analysis_lib_v2_folder))
# settings = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(settings)

bec_data_path = "/home/{}/NAS542_dataBEC2".format(username)

def get_day_data(today = True, year = None, month = None, day = None, path =bec_data_path,bec = 2):
    if today:
        current_time = datetime.datetime.now()
        year = current_time.year
        month = current_time.month
        day = current_time.day 
    if year is None or month is None or day is None:
        print('Please specify year, month and day')
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