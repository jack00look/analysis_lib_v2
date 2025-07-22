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

    seqs = [23,24,27]
    df = df[(df['sequence_index'].isin(seqs))]

    y_var = 'Magnetization'
    y_m1 = ('atoms_sum', '-1')
    y_m2 = ('atoms_sum', '-2')
    y_m1_vals = df[y_m1].values
    y_m2_vals = df[y_m2].values
    y = (y_m2_vals - y_m1_vals) / (y_m2_vals + y_m1_vals)
    y_repeats = df['run time'].values

    fig,ax = plt.subplots(nrows = 2,ncols = 1, tight_layout=True)
    ax_hist = ax[0]
    ax_scatter = ax[1]

    ax_hist.hist(y, bins=5)
    ax_hist.set_xlabel(y_var)
    ax_hist.set_ylabel('Counts')

    ax_scatter.scatter(y_repeats, y, alpha=0.5)
    ax_scatter.set_xlabel('Repeat Number')
    ax_scatter.set_ylabel(y_var)


    title = Path(__file__).name
    title += '\n' + 'seqs: ' + str(seqs)
    fig.suptitle(title)
except Exception as e:
        print(traceback.format_exc())