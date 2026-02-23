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

def F_rabi_spectroscopy(field,f0,f_rabi,t_pulse,amp,offset):
    delta = (field - f0)*2.1e3  # Convert from mG to Hz
    f_rabi_gen = np.sqrt(f_rabi**2 + delta**2)
    y = offset - amp * (delta**2 + f_rabi**2*np.cos(2*np.pi*f_rabi_gen*t_pulse))/ (f_rabi_gen**2)
    return y

rabi_model = Model(F_rabi_spectroscopy, independent_vars=['field'],nan_policy='omit' )

seqs = [[21,22,23]]


if __name__ == '__main__':
    try:
        #df_orig = general_lib_lyse_mod.get_day_data(today = True)
        
        df_orig = lyse.data()
        x_var = 'BCompZ_spectroscopy'

        y_m1 = ('spectroscopy_atom_count', 'm1_Nfit')
        y_m2 = ('spectroscopy_atom_count', 'm2_Nfit')

        y_uw_pulse = 'uW_pulse'

        y_DAC100W = 'uW_100WDAC'

        y_uW_amp = 'uW_amp'

        fig, ax = plt.subplots(nrows = 1, tight_layout=True)
        ax.set_xlabel(x_var)
        ax.set_ylabel('Magnetization')
        ax.set_title('uW Spectroscopy Analysis')

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        if len(seqs) > len(colors):
            raise ValueError(f"Too many sequences ({len(seqs)}) for available colors ({len(colors)})")

        for s in seqs:
            df = df_orig[(df_orig['sequence_index'].isin(s))]
            df = df.sort_values(by=x_var)
            Bcompzfine_values = df[x_var].values

        
            y_m1_vals = df[y_m1].values
            y_m2_vals = df[y_m2].values
            y_Ntot_vals = y_m1_vals + y_m2_vals
            y_M_vals = (y_m2_vals - y_m1_vals) / y_Ntot_vals

            # Filter out values where Ntot is too low
            mask = y_Ntot_vals > 0
            Bcompzfine_values = Bcompzfine_values[mask]
            y_M_vals = y_M_vals[mask]
            y_m1_vals = y_m1_vals[mask]
            y_m2_vals = y_m2_vals[mask]
            y_Ntot_vals = y_Ntot_vals[mask]


        
            uW_pulse_vals = df[y_uw_pulse].values
            uW_pulse_unique = np.unique(uW_pulse_vals)
            if uW_pulse_unique.size != 1:
                raise ValueError(f"Multiple unique values found for {y_uw_pulse}: {uW_pulse_unique}, analysing sequences {s}")
            uW_pulse_val = uW_pulse_unique[0]

            uW_amp_vals = df[y_uW_amp].values
            uW_amp_unique = np.unique(uW_amp_vals)
            if uW_amp_unique.size != 1:
                raise ValueError(f"Multiple unique values found for {y_uW_amp}: {uW_amp_unique}, analysing sequences {s}")
            uW_amp_val = uW_amp_unique[0]

            DAC100W_vals = df[y_DAC100W].values
            DAC100W_unique = np.unique(DAC100W_vals)
            if DAC100W_unique.size != 1:
                raise ValueError(f"Multiple unique values found for {y_DAC100W}: {DAC100W_unique}, analysing sequences {s}")
            DAC100W_val = DAC100W_unique[0]

            params = rabi_model.make_params()
            params['f0'].set(value=Bcompzfine_values[np.argmax(y_M_vals)], vary=True,min = 1,max = 142)
            params['f_rabi'].set(value=0.5/(uW_pulse_val*1e-3),vary=True, min=0,max = 1e5)
            params['t_pulse'].set(value=uW_pulse_val/1000., vary=False)
            params['amp'].set(value=1., vary=False, min=0., max=1.)
            params['offset'].set(value=0., vary=False,min = 0., max=2.)
            f0_D2_Na = 1.7716261288 * 1e9  # Hz
            uW_freq = ((df['uW_freq'].values[0]*1e3 + 1.670*1e9) - f0_D2_Na)*1e-3 # kHz
            data_label = 'Seq {} - uW freq = {:.2f} kHz, uW pulse: {:.3f} ms, uW amp: {:.2f}, DAC100W: {:.4f}'.format(s, uW_freq, uW_pulse_val, uW_amp_val,DAC100W_val)
            ax.plot(Bcompzfine_values, y_M_vals, 'o',color=colors[seqs.index(s)], label= data_label)

            ax.set_ylim(-1.05, 1.05)

            Bcomp_min = Bcompzfine_values.min()
            Bcomp_max = Bcompzfine_values.max()
            Bcompzfine_plot = np.linspace(Bcomp_min, Bcomp_max, 1000)
            
            ax.plot(Bcompzfine_plot, rabi_model.eval(field=Bcompzfine_plot, params=params), '--',color = colors[seqs.index(s)], alpha=0.5)

            try:
                fit = rabi_model.fit(y_M_vals, field=Bcompzfine_values, params=params)
                print(fit.fit_report()) 
                fit_label = 'FIT f_rabi: {:.2f} Hz, f0: {:.3f} mG'.format(fit.params['f_rabi'].value, fit.params['f0'].value)

                ax.plot(Bcompzfine_plot, fit.eval(field=Bcompzfine_plot,params = fit.params), ls = '--', color=colors[seqs.index(s)],label = fit_label)

            except Exception as e:
                print(f"Fit failed for sequence {s}: {e}")
                ax.plot(Bcompzfine_values, y_M_vals, 'o-', color=colors[seqs.index(s)], label=f"{data_label} (fit failed)")
        
        # I want the legend to be outside the plot, below the x-axis    
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)

        
    except Exception as e:
        print(traceback.format_exc())