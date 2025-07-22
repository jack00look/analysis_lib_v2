import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lyse
from lmfit import Model
from lmfit.models import ExpressionModel, ExponentialModel, ConstantModel
import importlib
import traceback

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
general_lib_lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_lyse_mod)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

def F_rabi_oscillations(t_pulse,delta,f_rabi,tau_decoherence,amp,offset, phi):
    t_pulse = t_pulse*1e-3  # Convert t_pulse from ms to s
    delta = delta*2.1e3  # Convert delta from mG to Hz
    f_rabi_gen = np.sqrt(f_rabi**2 + delta**2)
    y = f_rabi**2/f_rabi_gen**2*(np.cos(2*np.pi*f_rabi_gen*t_pulse + phi))*np.exp(-t_pulse/tau_decoherence)
    return y*amp + offset

rabi_model = Model(F_rabi_oscillations, independent_vars=['t_pulse'],nan_policy='omit' )

seqs = [[42, 43]]

if __name__ == '__main__':
    try:
        df_orig = general_lib_lyse_mod.get_day_data(today = True)
        df_orig = lyse.data()

        x_var = 'uW_pulse'

        y_m1 = ('spectroscopy_atom_count', 'm1_fit')
        y_m2 = ('spectroscopy_atom_count', 'm2_fit')
        # y_m1 = ('atoms_sum', '-1')
        # y_m2 = ('atoms_sum', '-2')

        y_Bcompzfine = 'Bcompzfine_spectroscopy'

        y_uW_amp_m1m2 = 'uW_amp_m1m2'
        fig, ax = plt.subplots(nrows = 1, tight_layout=True)
        #fig, axes = plt.subplots(nrows = 4, tight_layout=True)
        #ax, ax1, ax2, ax3 = axes
        ax.set_xlabel(x_var)
        ax.set_ylabel('Magnetization')
        ax.set_title('uW Rabi Oscillations Analysis')

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        if len(seqs) > len(colors):
            raise ValueError(f"Too many sequences ({len(seqs)}) for available colors ({len(colors)})")

        for s in seqs:

            df = df_orig[(df_orig['sequence_index'].isin(s))]
            df = df.sort_values(by=x_var)

            t_pulse_vals = df[x_var].values

        
            y_m1_vals = df[y_m1].values
            y_m2_vals = df[y_m2].values
            y_Ntot_vals = y_m1_vals + y_m2_vals
            y_M_vals = (y_m2_vals - y_m1_vals) / y_Ntot_vals

        
            Bcompzfine_vals = df[y_Bcompzfine].values
            Bcompzfine_unique = np.unique(Bcompzfine_vals)
            if Bcompzfine_unique.size != 1:
                raise ValueError(f"Multiple unique values found for {y_Bcompzfine}: {Bcompzfine_unique}, analysing sequences {s}")
            Bcompzfine_val = Bcompzfine_unique[0]

            uW_amp_m1m2_vals = df[y_uW_amp_m1m2].values
            uW_amp_m1m2_unique = np.unique(uW_amp_m1m2_vals)
            if uW_amp_m1m2_unique.size != 1:
                raise ValueError(f"Multiple unique values found for {y_uW_amp_m1m2}: {uW_amp_m1m2_unique}, analysing sequences {s}")
            uW_amp_m1m2_val = uW_amp_m1m2_unique[0]

            params = rabi_model.make_params()
            params['delta'].set(value=0.00,vary=False)
            params['f_rabi'].set(value=692.135,vary=True, min=0,max = 2e5)
            params['amp'].set(value=1., vary=False, min=0., max=2.)
            params['offset'].set(value=0., vary=False,min = -1., max=1.)
            params['phi'].set(value=-np.pi, vary=False)
            params['tau_decoherence'].set(value=30e-3, vary=True,min=1e-4, max=1e8)
            ind_sorted = np.argsort(t_pulse_vals)
            t_pulse_vals = t_pulse_vals[ind_sorted]
            y_M_vals = y_M_vals[ind_sorted]

            data_label = 'Seq {} - Bcompzfine: {:.2f} ms, uW amp m1m2: {:.2f}'.format(s, Bcompzfine_val, uW_amp_m1m2_val)
            ax.scatter(t_pulse_vals, y_M_vals, color=colors[seqs.index(s)], s=20,label= data_label)
            # ax1.plot(t_pulse_vals, y_m1_vals, 'o',  label = 'M1')
            # ax2.plot(t_pulse_vals, y_m2_vals, 'o', label = 'M2')
            # ax3.plot(t_pulse_vals, y_Ntot_vals, 'o', label = 'Ntot')

            tpulse_min = t_pulse_vals.min()
            tpulse_max = t_pulse_vals.max()
            tpulse_plot = np.linspace(tpulse_min, tpulse_max, 1000)
          
            
            ax.plot(tpulse_plot, rabi_model.eval(t_pulse=tpulse_plot, params=params), '--',color = colors[seqs.index(s)], alpha=0.5)

            fit = rabi_model.fit(y_M_vals, t_pulse=t_pulse_vals, params=params)
            best_params = fit.params
            print(fit.fit_report()) 
            fit_label = 'FIT f_rabi: {:.2f} Hz, tau: {:.2f}, delta: {:.2f} mG'.format(fit.params['f_rabi'].value, fit.params['tau_decoherence'].value,fit.params['delta'].value)

            ax.plot(tpulse_plot, fit.eval(t_pulse=tpulse_plot,params=best_params), ls = '-', color=colors[seqs.index(s)],label = fit_label)
        
        ax.legend()

        
    except Exception as e:
        print(traceback.format_exc())