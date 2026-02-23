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

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

### Define the Models for the Analysis ###

def F_tof_temp(sigma2,tof2,T,offset):
    kB = 1.38e-23  # Boltzmann constant in J/K
    m_Na = 23 * 1.66e-27  # Mass of sodium atom in kg
    T = T*1e-6  # Convert T from µK to K
    tof2 = tof2*1e-6  # Convert tof2 from ms^2 to s^2
    y = kB*T/m_Na * tof2 + offset
    return y

tof_temp_model = Model(F_tof_temp, independent_vars=['tof2'],nan_policy='omit' )

### Define the Multishot Parameters ###

multishot_parameters = {
    'day': 'today',
    'seqs': [[3]],
    'xvar': {
        'name': 'tof',
        'single_shot_script': None,
        'var': 'tof_debug'
    },
    'yvars': {
        'thermal_width': {
        'single_shot_script': 'Fit_Gaussian_1D',
        'var': 'atoms_xfit_s'
        },
        'N_atoms': {
        'single_shot_script': 'Fit_Gaussian_1D',
        'var': 'atoms_xfit_N'
        }
    }
}

### Main Analysis Code ###

if __name__ == '__main__':

    # Get the day data
    try:
        if multishot_parameters['day'] == 'today':
            df_orig = general_lib_lyse_mod.get_day_data(today = True)
        else:
            day = multishot_parameters['day']
            year, month, day = day.split('-')
            df_orig = general_lib_lyse_mod.get_day_data(today = False, year = int(year), month = int(month), day = int(day))
    except Exception as e:
        print(f"Failed to get day data: {e}")


    # create plots
    fig,ax = plt.subplots(nrows = 2, tight_layout=True)
    fig.suptitle('TOF Temperature Measure of a Thermal Cloud', fontsize=16)

    # extract single plots
    ax_temp = ax[0]
    ax_N = ax[1]

    # define axes labels
    ax_N.set_xlabel('TOF2 [s^2]')
    ax_N.set_ylabel('N_atoms')
    ax_N.set_title('Atom Number Stability')

    ax_temp.set_xlabel('TOF2 [s^2]')
    ax_temp.set_ylabel('Thermal_width^2 [m^2]')
    ax_temp.set_title('TOF Temperature')

    # loop through all sequences
    for (s,i) in zip(multishot_parameters['seqs'], range(len(multishot_parameters['seqs']))):

        color_number = i % 10
        color = f'C{color_number}'

        # extract df for the sequences
        df_temp = df_orig[(df_orig['sequence_index'].isin(s))]

        # get x and y values
        dict_xy = general_lib_mod.get_xy_values(df_temp, multishot_parameters)

        # get x vals
        xvar_name = multishot_parameters['xvar']['name']
        tof_vals = dict_xy[xvar_name]

        # get y vals
        thermal_width_vals = dict_xy['thermal_width']
        N_atoms_vals = dict_xy['N_atoms']

        # compute squared values of tof and thermal width which are used for the analysis
        tof2_vals = (tof_vals)**2
        thermal_width2_vals = (thermal_width_vals)**2

        ## Atom number stability plot ##
        ax_N.scatter(tof2_vals, N_atoms_vals, label=f'Seq {s}', color=color)
        avg_N = np.mean(N_atoms_vals)
        ax_N.axhline(y=avg_N, color=color, linestyle='--', label=f'Avg N Seq {s}: {avg_N:.2e}')

        ## TOF Temperature plot ##

        # raw data
        ax_temp.scatter(tof2_vals, thermal_width2_vals, label=f'Seq {s}', color=color)

        # define initial parameters
        params = tof_temp_model.make_params()
        params['T'].set(value=15, min=1e-4, max=1e4)
        params['offset'].set(value=np.min(thermal_width2_vals), min=0., max=1e-3)

        # plot initial guess
        tof2_plot = np.linspace(np.min(tof2_vals), np.max(tof2_vals), 100)
        ax_temp.plot(tof2_plot, tof_temp_model.eval(params=params, tof2=tof2_plot), '--', color=color, alpha=0.5)

        # perform the fit
        try:
            fit = tof_temp_model.fit(thermal_width2_vals, params, tof2=tof2_vals)
            print(fit.fit_report())
            ax_temp.plot(tof2_plot, fit.eval(tof2=tof2_plot, params=fit.params), '-', color=color, label=f'Fit Seq {s}: T={fit.params["T"].value:.2f} uK')
        except Exception as e:
            print(f"Fit failed for sequence {s}: {e}")
    
    # add legends and grids to all axes
    for axis in ax:
        axis.legend()
        axis.grid()