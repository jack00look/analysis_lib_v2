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

cam = camera_settings.cameras['cam_vert1']


df = lyse_mod.get_day_data(today=True,year=2024, month=12, day=17)
df = lyse.data()

seqs = [49,50]
# select only images without saturation
df = df[df['sequence_index'].isin(seqs)]
Clin = df[('Clin_Clog_measure', 'Clin')].values
Clog = df[('Clin_Clog_measure', 'Clog')].values
print(Clin)
print(Clog)

chi_sat = cam['chi_sat']

# linear fit for alpha
linmodel = LinearModel(nan_policy='omit')
params=linmodel.make_params()
fit = linmodel.fit(Clog, x=Clin, params=params)
print(fit.fit_report())
slope = fit.params['slope'].value

sum_Clin2 = np.sum(Clin**2)
sum_Clin = np.sum(Clin)
sum_Clog = np.sum(Clog)
sum_Clin_Clog = np.sum(Clin*Clog)
sum_Clog2 = np.sum(Clog**2)
N = len(Clin)
delta = N*sum_Clin2 - sum_Clin**2
slope_1 = (N*sum_Clin_Clog - sum_Clin*sum_Clog)/delta
intercept_1 = (sum_Clin2*sum_Clog - sum_Clin*sum_Clin_Clog)/delta
Clog_fit = slope_1*Clin + intercept_1
var_post = np.sum((Clog - Clog_fit)**2)/(N-2)
print(f"err_post: {np.sqrt(var_post)}")
sigma_slope = np.sqrt(var_post/delta*N)
print(f"slope: {slope}; sigma_slope: {sigma_slope}")
alpha_1 = -1/(slope_1*chi_sat)
sigma_alpha_1 = np.abs(-1/(slope_1**2*chi_sat))*sigma_slope
print(f"alpha: {alpha_1}; sigma_alpha: {sigma_alpha_1}")
OD_1 = intercept_1*alpha_1
sigma_intercept_1 = np.sqrt(var_post*sum_Clin2/delta)
sigma_OD_1 = np.abs(OD_1)*np.sqrt((sigma_intercept_1/intercept_1)**2 + (sigma_alpha_1/alpha_1)**2)
print(f"OD: {OD_1}; sigma_OD: {sigma_OD_1}")


# alpha with uncertainty
alpha = -1/(slope*chi_sat)
sigma_alpha = -1/(slope**2*chi_sat)*fit.params['slope'].stderr

print(f"alpha: {alpha}")

fig, ax = plt.subplots()
fit.plot_fit(ax = ax, numpoints=100)
ax.set_xlabel('Clin')
ax.set_ylabel('Clog')
title = f'seqs: {seqs}' + '\n' + f'$\\alpha$ = {alpha:.2f}+- {-sigma_alpha:.2f}' + '\n' + f'OD: {OD_1:.2f}+- {sigma_OD_1:.2f}'
ax.set_title(title)

