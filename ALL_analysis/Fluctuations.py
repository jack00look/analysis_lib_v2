import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lyse
from lmfit.models import ExpressionModel, ExponentialModel, ConstantModel
import importlib
import traceback
from pathlib import Path

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
general_lib_lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_lyse_mod)

try:
    df = general_lib_lyse_mod.get_day_data(today = True)
    #df = general_lib_lyse_mod.get_day_data(year = 2024, month = 9,day=24)

    seqs = [2]
    df = df[(df['sequence_index'].isin(seqs))]

    y_var = ('BEC_fit_hor', 'PTAI_m1xfit_c_tf')
    y= df[y_var].values*1e6
    y_var_1 = ('plo')
    y_repeats = df['run time'].values
    y_repeats_local = pd.to_datetime(y_repeats).tz_localize('UTC').tz_convert('Europe/Rome')
    print(y_repeats[0])
    print(type(y_repeats[0]))
    # want to add 2hr to the time
    # add 2hr to datetime time
    y_repeats = y_repeats + 2*60*60

    fig,ax = plt.subplots(nrows = 3,ncols = 1, tight_layout=True)
    ax_hist = ax[0]
    ax_scatter = ax[1]
    ax_scatter_1 = ax[2]
    n_bins = len(y)//8 # what does it do? 
    if n_bins < 3:
        n_bins = 3
    # // is the 
    ax_hist.hist(y, bins=n_bins)
    ax_hist.set_xlabel(y_var)
    ax_hist.set_ylabel('Counts')

    ax_scatter.scatter(y_repeats_local, y, alpha=0.5)
    ax_scatter.set_xlabel('Repeat Number')
    ax_scatter.set_ylabel(y_var)
    #ax_scatter.axhline(2e6, color='r', linestyle='--', label='Mean')


    title = Path(__file__).name
    title += '\n' + 'seqs: ' + str(seqs)
    title += '\n' + 'y_var: ' + str(y_var)
    title += '\n' + 'Avg Y value: {:.3e}'.format(y.mean())
    title += '\n' + 'Std Y value: {:.3e}'.format(y.std())
    fig.suptitle(title)
except Exception as e:
        print(traceback.format_exc())