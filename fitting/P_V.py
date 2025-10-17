import numpy as np
import matplotlib.pyplot as plt
import h5py
import lyse
import pandas as pd
#from fitting.models1D import Gaussian1DModel, SkewedGaussian1DModel
from settings import cameras_db
#from settings import cameras_db
import matplotlib as mpl
from scipy.special import spence, erf
from lmfit import Model

plt.style.use('seaborn-colorblind')
mpl.rcParams['savefig.dpi'] = 50

df = lyse.data()
#df = pd.read_hdf('/backup/img/2019/04/16/0108/dataframe.hdf')
# df['atom_counts'] =  df[('spin_waves', 'PTAI_m1_sum')] + df[('spin_waves', 'PTAI_p1_sum')]
# df['magnetization'] =  (df[('spin_waves', 'PTAI_p1_sum')] - df[('spin_waves', 'PTAI_m1_sum')]) /(df[('spin_waves', 'PTAI_m1_sum')] + df[('spin_waves', 'PTAI_p1_sum')])
df = df.droplevel(0)
df = df.reset_index(drop=True)
cam = cameras_db['cam_vert1']
seqs=[[58,59]]
same_axis = True
#df = df[(df['tof']==4)]



x_var = 'probe_det'

y_vars =   [ \
            #[('focusing', 'amp')]
            #[('focusing', 'DMD_focus_delta1'), ('focusing', 'DMD_focus_delta2')],
            #[('focusing', 'DMD_focus_min_sx'), ('focusing', 'DMD_focus_min_dx')],
            #[('focusing', 'DMD_focus_max_cen')]
            #[('spin_waves','RTF')],[('spin_waves', 'mx')],
            #[('DMD_dmd_vs_atoms_calibration_analysis', 'a')], [('DMD_dmd_vs_atoms_calibration_analysis', 'b')],
        #  [[('spin_waves','mx')]]

          # [('spin_waves','mx')],

          # [('atom_counts')],
          # [('magnetization')],
#         # [('field_lock',f'lock2_ODsum_1_{i}') for i in range(5)],
#
#         # [('two_states_position', 'PTAI_m1_mx'), ('two_states_position', 'PTAI_p1_mx')],
#         # [('two_states_position', 'PTAI_m1_mean'), ('two_states_position', 'PTAI_p1_mean')],
#         # [('two_states_position', f'PTAI_{s}_rx') for s in ['m1', 'p1']],
#         # [('two_states_position', f'PTAI_{s}_mx') for s in ['m1', 'p1']],
#         # [('two_states_position', f'PTAI_{s}_std') for s in ['m1', 'p1']],
          # [('two_states_position', f'PTAI_{s}_sum') for s in ['m1', 'p1']],

          # [('spin_waves', 'PTAI_m1_sum'), ('spin_waves', 'PTAI_m2_sum')],
          # [('spin_waves', 'yaxis_rx')],
          # [('spin_waves', 'PTAI_m1_1_sum'), ('spin_waves', 'PTAI_p1_1_sum')],
            [('spin_waves', 'N_bec'),('spin_waves', 'N_th')],
          # [('spin_waves', 'mu_spin')],
          # [('spin_waves', 'm1_0'), ('spin_waves', 'p1_0')],
          # [('spin_waves', 'm1_-300'), ('spin_waves', 'p1_-300')],
          # [('spin_waves', 'm1_300'), ('spin_waves', 'p1_300')]
          # [('spin_waves', 'rx')],atom

          
          # [('od_1DBimodal', x) for x in ['m1x_sx']],


        #   [('od_1DBimodal', x) for x in ['m1y_N_bec']],
        #   [('od_1DBimodal', x) for x in ['m1y_N_th']],
        # #  [('atoms_sum','-1')],

          # [('od_Bimodal_Hor1','rad_x')],
          # [('od_Bimodal_Hor1','sigma_x')],
          # [('od_Bimodal_Hor1','Nbec')],
          # [('od_Bimodal_Hor1','Nth')],


          ##[('od_1Dgaussian', x) for x in ['y_m']],
          # [('od_1Dgaussian', 'skewed_X_N'), ('od_1Dgaussian', 'skewed_Y_N')],
          # [('od_1Dgaussian', 'skewed_X_sigma'), ('od_1Dgaussian', 'skewed_Y_sigma')],


         # [('spin_waves', 'sum_y')],
#         # [('show_twostates', 'p_peak'), ('show_twostates', 'n_peak')]
#
#         # [('phase_imprint_focus', 'BEC center')],
#         # [('phase_imprint_focus', 'BEC sigma')],
        # [('od_remove_thpart', 'N_bec')],
#
        # [('atoms_sum', 'atoms_sum')],
        # [('atoms_sum', '-2'),('atoms_sum', '-1')],
        # [('hor_sloshing', 'x0')]
        # [('1CompBECHamam', 'N_bec')],
        # [('1CompBECHamam', 'PTAI_m1_sum')]
        # [('od_remove_thpart', 'N_bec')],
        # [('atoms_sum', '+2')],# fft_ampm = np.fft.fftshift(np.fft.fft(np.array(ampm_list)/max(ampm_list)))

        # [('atoms_sum', '0')],
        # [('atoms_sum', 'MOT_counts')],
        # [('od_2Dgaussian', 'N')],
        # [('od_2Dgaussian', 'mx'), ('od_2Dgaussian', 'my')],

        # [('Plot_raws', 'cigar_x'), ('Plot_raws', 'cigar_y')]
       # # #
       # [("atom_number_count", "cam_hor3_ODsum")],
       # [('focusing', 'PTAI_m1_sum')]

       #
       # [('plot_scope_GM_peak', 'GM_peak')],
       # [('plot_scope_GM_peak', 'MOT_fluo')],
       # # [('plot_scope_GM_peak', 'Cigar_errorsignal')],
       # # [('pl            [('focusing', 'DMD_focus_cen')],ot_scope_GM_peak', 'GM_intensity')],
       # [('plot_scope_GM_peak', 'GM_polarization')],
       #
       # # [('plot_scope_GM_peak', 'Cigar_errorsignal')],
       # # # [('plot_scope_GM_peak', 'Cigar_mean')],
       # [('plot_scope_traces',print(result.fit_report())CigarV_error')],
       #[('plot_scope_traces', 'CigarV_RelativeErr')],

       # [('plot_scope_traces', 'CigarV_Polarization')],

       #
       # # [('plot_scope_traces', 'DS_mean')],
       # # [('plot_scope_traces', 'DS_std')],
       #
       # # [('plot_scope_traces', '2DMOT_mean')],
       # # [('plot_scope_traces', '2DMOT_std')],
       #
       # [('plot_scope_traces', '3DMOT_mean')],
       # [('plot_scope_traces', '3DMOT_std')],
       # #
       # [('cigar_position', 'cigar_x')],
       # [('cigar_position', 'cigar_y')],
       # # [('cigar_position', 'cigar_sum')],
       # # # ['CigarV']
       # # [('od_2Dgaussian','my')],
       # # [('od_2Dgaussian','N')],
       #
       #[("atom_number_count", "cam_hor3_ODsum")],
       # # [("atom_number_count", "cam_hor3_ODmax")],
       # [('od_1Dgaussian', "y_N")],
        # [('240808_od_remove_th_part', 'thermal_sx')],
        # [('240808_od_remove_th_part', 'thermal_sy')],
        #[('240808_od_remove_th_part', 'N_bec')],
        #[('240808_od_remove_th_part', 'N_th')],
        #[('SingleBecOD', 'N_bec'), ('SingleBecOD', 'N_th')],
        #[('SingleBecOD', 'rx'), ('SingleBecOD', 'sx')],
        #[('SingleBecOD', 'n_bec')],
        # [('SingleBecOD', 'fft_res'), ('SingleBecOD', 'freqr')],
        # [('SingleBecOD', 'mtfx'), ('SingleBecOD', 'mgx')],
        #[('OD_Fit_Vert_1D_1pic', 'rx'), ('OD_Fit_Vert_1D_1pic', 'sx')],
        #[('OD_Fit_Vert_1D_1pic', 'mtfx'), ('OD_Fit_Vert_1D_1pic', 'mgx')],
        #[('OD_Fit_Vert_1D_1pic', 'mx')],
        #[('OD_Fit_Vert_1D_1pic', 'rx')],
        #[('OD_Fit_Vert_1D_1pic', 'std/n0')],
        # [('OD_Fit_Vert_1D_1pic', 'N_bec')],
        # [('OD_Fit_Vert_1D_1pic', 'n_bec')],
        ]
#
# #

y_var =  [[('spin_waves', 'PTAI_m1_sum')]]

OD_tot = df[y_var].values

probe_dets = df[x_var].values

plt.plot(probe_dets, OD_tot)

def skewedgaussian1d(X, amp, m, s, gamma, offset = 0):
    return amp*np.exp(-(X-m)**2/(2*s**2))*(1+erf(gamma*(X-m)/(s*np.sqrt(2)))) + offset

skew_gauss = Model(skewedgaussian1d, independent_vars = ['X'], nan_policy = 'omit')

params = skew_gauss.make_params()
print(params)
params['m'].set(value=6)
params['amp'].set(value=1200000)
params['s'].set(value=10)
params['gamma'].set(value =1)# 0.25/time)
params['offset'].set(value = 100000)


fit = skew_gauss.fit(OD_tot, X=probe, params=params)
print(fit.best_fit)
#
#p0['skewed_Y_amp'].set(value = fittedPY['y_amp'].value)
#p0['skewed_Y_m'].set(value = fittedPY['y_m'].value)
#p0['skewed_Y_s'].set(value = fittedPY['y_s'].value)

# resultS = model.fit(N,f=x,params = p0)
# resultS.plot(show_init = True)
# #run.save_results_dict(parameters_dict_with_errors(resultS.params))
# print(resultS.fit_report())
print(fit.fit_report())
fig,ax=plt.subplots()
ax.plot(x,N,'o')
ax.plot(x, fit.best_fit)
ax.plot(x,fit.init_fit)
print(x[fit.best_fit.argmax()])


plt.tight_layout()