import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lyse
from lmfit.models import ExpressionModel, ExponentialModel, ConstantModel
import importlib
import traceback

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
general_lib_lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_lyse_mod)

if __name__ == '__main__':
    try:
        df = general_lib_lyse_mod.get_day_data(today = True)

        seqs = [19,20]
        x_var = 'hold_time_debug'
        y_var = ('BEC_fit_hor', 'atomsyfit_c_tf')

        f_init = 2 #kHz

        df = df[(df['sequence_index'].isin(seqs))]

        x = df[x_var].values
        y = df[y_var].values*1e6

        fig,ax = plt.subplots(tight_layout=True) 
        ax.plot(x, y, 'o')
    

        model = ExpressionModel(expr="amp * cos(2*pi*freq * x + phi) + offset",
                                independent_vars=['x'],
                                nan_policy='omit',
                                )
        
        params = model.make_params(amp = 1.414*y.std(),
                                   freq = f_init,
                                   phi=np.pi,
                                   offset = y.mean())
        
        x_fit = np.linspace(x.min(), x.max(), 1000)
        ax.plot(x_fit, model.eval(x=x_fit, params=params), 'k--')
        
        params['amp'].set(min=0, max=None, vary=True)
        params['freq'].set(vary=True, min=0)
        params['phi'].set(vary=True, min=0, max=2*np.pi)

        fit = model.fit(y, x=x, params=params)
        x_fit = np.linspace(x.min(), x.max(), 1000)
        ax.plot(x_fit, fit.eval(x=x_fit), 'r-')
        dict = fit.params.valuesdict()

        title = __name__
        title += '\n' + 'seqs: ' + str(seqs)
        title += '\n' + 'x_var: ' + x_var
        title += '\n' + 'y_var: ' + str(y_var)
        title += '\n freq: {:.3f} kHz     amp: {:.3f}     offset: {:.3f}     phi: {:.3f}'.format(dict['freq'], dict['amp'], dict['offset'], dict['phi'])
        ax.set_title(title)
        print(fit.fit_report())
    except Exception as e:
        print(traceback.format_exc())
