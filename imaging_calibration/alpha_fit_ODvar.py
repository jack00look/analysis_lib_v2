"""
fit of alpha
"""
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
import lyse
import h5py
from scipy.ndimage import gaussian_filter

spec = importlib.util.spec_from_file_location("general_lib.main_lyse", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyse_mod)

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

cam = camera_settings.cam_vert1

year = 2024
month = 12
day = 17
seqs = [13,14]
filenames = general_lib_mod.get_ordered_multiple_sequences(year, month, day, seqs,settings.bec2path)

alpha_arr = np.linspace(1.5,3.5,21)
w_y_arr = np.linspace(10,60,11)

ODvar_arr = np.zeros((len(alpha_arr),len(w_y_arr)))
for i in range(len(alpha_arr)):
    alpha = alpha_arr[i]
    print(alpha)
    ODs = []
    for j in range(len(filenames)):
        with h5py.File(filenames[j],'r+') as h5file:
            raws = imaging_lib_mod.get_raws_camera(h5file,cam)
            filter_radius = 20
            atom_image = gaussian_filter(raws['PTAI_m1'],filter_radius)
            probe_image = gaussian_filter(raws[cam['probe_image']],filter_radius)
            back_image = gaussian_filter(raws[cam['background_image']],filter_radius)
            OD = imaging_lib_mod.calculate_OD(atom_image, probe_image, cam, back_image,alpha = alpha,roi_background=cam['roi_back'])
            ODs.append(OD)
    ODs = np.array(ODs)
    ODs_vars = np.var(ODs,axis=0)
    OD_vars = ODs_vars/np.mean(ODs_vars)
    for k in range(len(w_y_arr)):
        w_y = w_y_arr[k]
        w_x = w_y*10
        ODvar_arr[i,k] = np.sum(OD_vars[140-int(w_y/2):140+int(w_y/2),1090-int(w_x/2):1090+int(w_x/2)])

fig, ax = plt.subplots()
for i in range(len(w_y_arr)):
    OD_var_temp = ODvar_arr[:,i]
    OD_var_temp = OD_var_temp-np.min(OD_var_temp)
    OD_var_temp = OD_var_temp/np.max(OD_var_temp)
    ax.plot(alpha_arr,OD_var_temp,label = str(w_y_arr[i]))
ax.set_xlabel('alpha')
ax.set_ylabel('ODvar')
ax.legend()
title = 'ODvar vs alpha'
title += '\n' + 'year: ' + str(year) + ' month: ' + str(month) + ' day: ' + str(day) + ' seqs: ' + str(seqs)

