"""
fit of alpha
"""
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
import lyse

spec = importlib.util.spec_from_file_location("general_lib.main_lyse", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyse_mod)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

cam = camera_settings.cam_vert1


df = lyse_mod.get_day_data(today=False,year=2024, month=12, day=17)

seqs = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]
# select only images without saturation
df_orig = df[df['sequence_index'].isin(seqs)]
tof = df_orig['tof'].values
tof_unique = np.unique(tof)
fig,ax = plt.subplots()
fig1, ax1 = plt.subplots()

for tof_val in tof_unique:
    df = df_orig[df_orig['tof'] == tof_val]

    Clin = np.array(df[('Clin_Clog_measure', 'Clin_array')].values)
    Clog = np.array(df[('Clin_Clog_measure', 'Clog_array')].values)
    Clog_int = np.array(df[('Clin_Clog_measure', 'Clog_int_array')].values)
    Clin_int = np.array(df[('Clin_Clog_measure', 'Clin_int_array')].values)
    N_shots = len(Clin)
    w_y_array = np.array(df[('Clin_Clog_measure', 'w_y_array')].values[0])

    Clog_table = np.zeros((N_shots, len(w_y_array)))
    Clin_table = np.zeros((N_shots, len(w_y_array)))
    Clin_int_table = np.zeros((N_shots, len(w_y_array)))
    Clog_int_table = np.zeros((N_shots, len(w_y_array)))
    for i in range(N_shots):
        Clog_table[i,:] = Clog[i]
        Clin_table[i,:] = Clin[i]
        Clin_int_table[i,:] = Clin_int[i]
        Clog_int_table[i,:] = Clog_int[i]
    chi_sat = cam['chi_sat']

    alphas_array = np.zeros(len(w_y_array))
    alphas_array_err = np.zeros(len(w_y_array))
    alphas_array_int = np.zeros(len(w_y_array))
    alphas_array_err_int = np.zeros(len(w_y_array))
    for i in range(len(w_y_array)):
        # linear fit for alpha
        linmodel = LinearModel(nan_policy='omit')
        params=linmodel.make_params()
        fit = linmodel.fit(Clog_table[:,i], x=Clin_table[:,i], params=params)
        print(fit.fit_report())
        slope = fit.params['slope'].value
        # alpha with uncertainty
        alpha = -1/(slope*chi_sat)
        sigma_alpha = np.abs(-1/(slope**2*chi_sat)*fit.params['slope'].stderr)
        alphas_array[i] = alpha
        alphas_array_err[i] = sigma_alpha

        # linear fit for alpha
        linmodel = LinearModel(nan_policy='omit')
        params=linmodel.make_params()
        fit = linmodel.fit(Clog_int_table[:,i], x=Clin_int_table[:,i], params=params)
        print(fit.fit_report())
        slope = fit.params['slope'].value
        # alpha with uncertainty
        alpha = -1/(slope*chi_sat)
        sigma_alpha = np.abs(-1/(slope**2*chi_sat)*fit.params['slope'].stderr)
        alphas_array_int[i] = alpha
        alphas_array_err_int[i] = sigma_alpha
    
    w_y_array_temp = w_y_array

    ax.errorbar(w_y_array_temp, alphas_array, yerr=alphas_array_err, fmt='o', label=f'tof: {tof_val}')
    ax.set_xlabel('w_y')
    ax.set_ylabel('alpha')
    ax.set_title(f'seqs: {seqs} \n AVG')

    ax1.errorbar(w_y_array_temp, alphas_array_int, yerr=alphas_array_err_int, fmt='o',label=f'tof: {tof_val}')
    ax1.set_xlabel('w_y')
    ax1.set_ylabel('alpha')
    ax1.set_title(f'seqs: {seqs} \n INT')
ax.legend()
ax1.legend()

