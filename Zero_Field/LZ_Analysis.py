import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lyse
from lmfit.models import ExpressionModel, ExponentialModel, ConstantModel
import importlib
import traceback
from pathlib import Path
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, report_fit,Model


spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main_lyse.py")
general_lib_lyse_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_lyse_mod)

fitParams=Parameters()

hbar = 1.05e-34 #J*s
muB = 9.27e-24 #J/T
gF = 1./2.

muB_mG_s = muB*1e-7
def get_th_sigmaB(speed_LZramp):
    sigmaB = np.sqrt((hbar*speed_LZramp)/(2*np.pi*gF*muB_mG_s))
    return sigmaB

# hbar = 1.054571817*10**(-34)
# mb = 9.2740100783*10**(-24)
# gf = 1/2


def pm1(x,params):
    yy=params['Bscan_center']
    xx=params['Bother_res']
    sigma2=params['sigma']**2
    cm1=params['cm1']
    Am1 = params['Am1']
    P=np.exp(-((x-yy)**2+(xx)**2)/2./(2*sigma2))
    Pm1=Am1*(1-P)**2+cm1
    return Pm1

def p0(x,params):
    yy=params['Bscan_center']
    xx=params['Bother_res']
    sigma2=params['sigma']**2
    c0=params['c0']
    A0=params['A0']
    P=np.exp(-((x-yy)**2+(xx)**2)/2./(2*sigma2))
    P0=A0*(1-P)*(2*P)+c0
    return P0

def pp1(x,params):
    yy=params['Bscan_center']
    xx=params['Bother_res']
    sigma2=params['sigma']**2
    cp1=params['cp1']
    Ap1=params['Ap1']
    P=np.exp(-((x-yy)**2+(xx)**2)/2./(2*sigma2))
    Pp1=Ap1*P**2+cp1
    return Pp1

def objective(params,x,data):
    resid = 0.0*data[:]
    resid[0,:] = data[0,:]-pm1(x,params)
    resid[1,:] = data[1,:]-p0(x,params)
    resid[2,:] = data[2,:]-pp1(x,params)
    return resid.flatten()

def Bcomp_complete(X,Bscan_center,Bother_res,sigma,cm1,Am1,c0,A0,cp1,Ap1):
    P = np.exp(-((X-Bscan_center)**2+Bother_res**2)/2./(2*sigma**2))
    L = int(P.size/3)
    P[0:L] = Am1 * (1-P[0:L])**2 + cm1
    P[L:2*L] = A0 * (1-P[L:2*L])*2*P[L:2*L] + c0
    P[2*L:] = Ap1 * P[2*L:]**2 + cp1
    return P

def Bcomp_m1(X,Bscan_center,Bother_res,sigma,Am1,cm1):
    P = np.exp(-((X-Bscan_center)**2+Bother_res**2)/2./(2*sigma**2))
    P = Am1*(1-P)**2 + cm1
    return P

def BZ_complete(X,A,B,x0,xc,tau):
    L = int(X.size/3.)
    P = np.zeros(3*L)
    P[:L] = A/(1+np.exp(-(X[:L]+x0-xc)/tau))
    P[-L:] = B * (1- 1./(1+np.exp(-(X[-L:]-x0-xc)/tau)))
    P[L:2*L] = 1 - P[:L] - P[-L:]
    return P

def pm1_Bz(x,params):
    A = params['A']
    x0 = params['x0']
    xc = params['xc']
    tau = params['tau']
    return A/(1+np.exp(-(x+x0-xc)/tau))

def pp1_Bz(x,params):
    B = params['B']
    x0 = params['x0']
    xc = params['xc']
    tau = params['tau']
    return B * (1- 1./(1+np.exp(-(x-x0-xc)/tau)))

def p0_Bz(x,params):
    A = params['A']
    B = params['B']
    x0 = params['x0']
    xc = params['xc']
    tau = params['tau']
    P_m1 = A/(1+np.exp(-(x+x0-xc)/tau))
    P_p1 = B * (1- 1./(1+np.exp(-(x-x0-xc)/tau)))
    return 1 - P_m1 - P_p1

Bcomp_model = Model(Bcomp_complete,independent_vars=['X'],nan_policy='omit')
Bz_model = Model(BZ_complete,independent_vars=['X'],nan_policy='omit')
Bcomp_m1_model = Model(Bcomp_m1,independent_vars=['X'],nan_policy='omit')

if __name__ == '__main__':
    try:
        df = general_lib_lyse_mod.get_day_data(today = True)

        seqs = [[74]]

        x_var_BCompY = 'BCompY'
        x_var_BCompZ = 'BCompZ_LZramp_end'
        x_var_BGradY = 'BGradY'
        x_var_BCompX = 'BCompX'

        y_m1 = ('spinor_atom_count', 'm1_Nsum')
        y_0 = ('spinor_atom_count', '_0_Nsum')
        y_p1 = ('spinor_atom_count', 'p1_Nsum')

        y_m1 = ('spinor_atom_count', 'm1_Nfit')
        y_0 = ('spinor_atom_count', '_0_Nfit')
        y_p1 = ('spinor_atom_count', 'p1_Nfit')

        # y_m1 = ('atoms_sum', '-1')
        # y_0 = ('atoms_sum', '0')
        # y_p1 = ('atoms_sum', '+1')

        # StartBz = df['BCompZ_LZramp'].values[0]
        # StopBz =  df['BCompZ_LZramp_end'].values[0]
        # DeltaBz = StartBz-StopBz
        # RampTime = df['t_LZramp'].values[0]
        # alpha = DeltaBz/RampTime
        # s = np.sqrt(((alpha*hbar)/(2*np.pi*mb*gf)))

       


        fig,ax = plt.subplots(3,tight_layout=True,sharex=True)
        ax[0].set_ylabel('$n_{-1}$',fontsize = 15)
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylabel('$n_0$', fontsize = 15)
        ax[1].set_ylim(-0.1, 1.1)
        ax[2].set_ylabel('$n_{+1}$', fontsize = 15)
        ax[2].set_ylim(-0.1, 1.1)

        

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        x_var_name = None

        for (index,seq) in enumerate(seqs):
            df_temp = df[(df['sequence_index'].isin(seq))]

            BCompX_vals = df_temp[x_var_BCompX].values
            BCompY_vals = df_temp[x_var_BCompY].values
            BCompZ_vals = df_temp[x_var_BCompZ].values
            BGradY_vals = df_temp[x_var_BGradY].values
            BCompX_unique_val = None
            BCompY_unique_val = None
            BGradY_unique_val = None

            if np.unique(BCompX_vals).size > 1:
                x_vals = BCompX_vals
                if x_var_name is None:
                    x_var_name = x_var_BCompX
                elif x_var_name != x_var_BCompX:
                    print(f'Warning: x_var_name changed from {x_var_name} to {x_var_BCompX} from seq {seqs[index-1]} to {seqs[index]}')
                    pass
            else:
                BCompX_unique_val = np.unique(BCompX_vals)[0]
            if np.unique(BCompY_vals).size > 1:
                x_vals = BCompY_vals
                if x_var_name is None:
                    x_var_name = x_var_BCompY
                elif x_var_name != x_var_BCompY:
                    print(f'Warning: x_var_name changed from {x_var_name} to {x_var_BCompY} from seq {seqs[index-1]} to {seqs[index]}')
                    pass
            else:
                BCompY_unique_val = np.unique(BCompY_vals)[0]
            if np.unique(BCompZ_vals).size > 1:
                x_vals = BCompZ_vals
                if x_var_name is None:
                    x_var_name = x_var_BCompZ
                elif x_var_name != x_var_BCompZ:
                    print(f'Warning: x_var_name changed from {x_var_name} to {x_var_BCompZ} from seq {seqs[index-1]} to {seqs[index]}')
                    pass
            if np.unique(BGradY_vals).size > 1:
                x_vals = BGradY_vals
                if x_var_name is None:
                    x_var_name = x_var_BGradY
                elif x_var_name != x_var_BGradY:
                    print(f'Warning: x_var_name changed from {x_var_name} to {x_var_BGradY} from seq {seqs[index-1]} to {seqs[index]}')
                    pass
            else:
                BGradY_unique_val = np.unique(BGradY_vals)[0]
        


            y_m1_vals = df_temp[y_m1].values
            y_0_vals = df_temp[y_0].values
            y_p1_vals = df_temp[y_p1].values
            y_Ntot_vals = y_m1_vals + y_0_vals + y_p1_vals
            p_m1_vals = y_m1_vals / y_Ntot_vals
            p_0_vals = y_0_vals / y_Ntot_vals
            p_p1_vals = y_p1_vals / y_Ntot_vals
            #sort
            sort_index = np.argsort(x_vals)
            x_vals = x_vals[sort_index]
            p_m1_vals = p_m1_vals[sort_index]
            p_0_vals = p_0_vals[sort_index]
            p_p1_vals = p_p1_vals[sort_index]

            label = f'Seq {seqs[index]}  '
            Bz_start = df_temp['BCompZ_LZramp'].values[0]
            time_Bz_ramp = df_temp['t_LZramp'].values[0]
            speed_Bz = df_temp['speed_LZramp'].values[0]

            if x_var_name == 'BCompZ_LZramp_end':
                Bz_end_min = df_temp['BCompZ_LZramp_end'].min()
                Bz_end_max = df_temp['BCompZ_LZramp_end'].max()
                # label +=  f'Bz: {Bz_start:.2f} -> ({Bz_end_min:.2f}, {Bz_end_max:.2f}) mG, @ {speed_Bz:.2f} mG/s'
                label +=  f'Bz: {Bz_start:.3f} -> ({Bz_end_min:.3f}, {Bz_end_max:.3f}) mG, @ {speed_Bz:.3f} mG/s'
            else:
                Bz_end = df_temp['BCompZ_LZramp_end'].values[0]
                #label += f'Bz: {Bz_start:.2f} -> {Bz_end:.2f} mG, @ {speed_Bz:.2f} mG/s'
                label += f'Bz: {Bz_start:.3f} -> {Bz_end:.3f} mG, @ {speed_Bz:.3f} mG/s in {time_Bz_ramp:.3f} ms'
                label += '\n'
                label += 'BCompX = {:.3f}'.format(BCompX_unique_val) if BCompX_unique_val is not None else ''
                label += 'BCompY = {:.3f}'.format(BCompY_unique_val) if BCompY_unique_val is not None else ''
                label += 'BGradY = {:.3f}'.format(BGradY_unique_val) if BGradY_unique_val is not None else ''
            ax[0].plot(x_vals, p_m1_vals, 'o', color=colors[index % len(colors)], label=label)
            ax[1].plot(x_vals, p_0_vals, 'o', color=colors[index % len(colors)], label=label)
            ax[2].plot(x_vals, p_p1_vals, 'o', color=colors[index % len(colors)], label=label)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
            params_start = Parameters()
            data = np.array([p_m1_vals, p_0_vals, p_p1_vals])
            x_vals_complete = np.array([x_vals,x_vals,x_vals])

            if x_var_name != 'BCompZ_LZramp_end':
                params_start.add('Bscan_center', value=x_vals[np.argmax(p_p1_vals)], vary=True, min=-3., max=3.)
                params_start.add('Bother_res', value=0.0123, vary=True, min=0.00001, max=1.)
                params_start.add('sigma', value=get_th_sigmaB(speed_Bz), vary=True, min=0.001, max=2.)
                params_start.add('cm1', value=0.0, vary=False, min=-0.3, max=0.3)
                params_start.add('c0', value=0.0, vary=False, min=-0.3, max=0.3)
                params_start.add('cp1', value=0.0, vary=False, min=-0.3, max=0.3)
                params_start.add('A0', value=1., vary=False, min=0.1, max=1.5)
                params_start.add('Am1', value=1., vary=False, min=0.1, max=1.5)
                params_start.add('Ap1', value=1., vary=False, min=0.1, max=1.5)
                params_start_m1 = Parameters()
                params_start_m1.add('Bscan_center', value=x_vals[np.argmax(p_p1_vals)], vary=True, min=-1., max=1.)
                params_start_m1.add('Bother_res', value=0.0123, vary=True, min=0.00001, max=1.)
                params_start_m1.add('sigma', value=get_th_sigmaB(speed_Bz), vary=True, min=0.001, max=2.)
                params_start_m1.add('cm1', value=0.0, vary=False, min=-0.3, max=0.3)
                params_start_m1.add('Am1', value=1., vary=False, min=0.1, max=1.5)
                ax[0].plot(x_fit, pm1(x_fit, params_start), '--', color=colors[index % len(colors)], label='Initial guess')
                ax[1].plot(x_fit, p0(x_fit, params_start), '--', color=colors[index % len(colors)])
                ax[2].plot(x_fit, pp1(x_fit, params_start), '--', color=colors[index % len(colors)])
                try:
                    out = Bcomp_model.fit(data,params_start,X = x_vals_complete,method='leastsq')
                    params_dict = out.params.valuesdict()
                    label_p1 = 'Fit: Bscan_res={:.3f}, Bother_res={:.3f}, sigma_B={:.3f}'.format(params_dict["Bscan_center"],params_dict["Bother_res"],params_dict["sigma"])
                    ax[0].plot(x_fit, pm1(x_fit, out.params), '-', color=colors[index % len(colors)])
                    ax[1].plot(x_fit, p0(x_fit, out.params), '-', color=colors[index % len(colors)])
                    ax[2].plot(x_fit, pp1(x_fit, out.params), '-', color=colors[index % len(colors)],label = label_p1)
                    print(out.fit_report())
                    out_m1 = Bcomp_m1_model.fit(p_m1_vals,params_start_m1,X = x_vals, method = 'leastsq')
                    #ax[0].plot(x_fit,pm1(x_fit,out_m1.params),'.-')
                except Exception as e:
                    print(f'Fit failed for seq {seqs[index]}: {e}')
                    print(traceback.format_exc())
                    continue

            if x_var_name == 'BCompZ_LZramp_end':
                params_start.add('A', value=0.9, vary=True, min=0.0, max=1.0)
                params_start.add('B', value=0.1, vary=True, min=0.0, max=1.0)
                params_start.add('x0', value=0.1, vary=True, min=-10., max=10.)
                params_start.add('xc', value=4.5, vary=True, min=-10., max=10.)
                params_start.add('tau', value=0.1, vary=True, min=0.001, max=2.)
                ax[0].plot(x_fit, pm1_Bz(x_fit, params_start), '--', color=colors[index % len(colors)], label='Initial guess')
                ax[1].plot(x_fit, p0_Bz(x_fit, params_start), '--', color=colors[index % len(colors)])
                ax[2].plot(x_fit, pp1_Bz(x_fit, params_start), '--', color=colors[index % len(colors)])
                try:
                    out = Bz_model.fit(data,params_start,X = x_vals_complete,method='leastsq')
                    params_dict = out.params.valuesdict()
                    print(params_dict.keys())
                    label_p1 = 'Fit: A={:.3f}, B={:.3f}, x0={:.3f}, xc={:.3f}, tau={:.3f}'.format(params_dict["A"],params_dict["B"],params_dict["x0"],params_dict["xc"],params_dict["tau"])
                    ax[0].plot(x_fit, pm1_Bz(x_fit, out.params), '-', color=colors[index % len(colors)])
                    ax[1].plot(x_fit, p0_Bz(x_fit, out.params), '-', color=colors[index % len(colors)])
                    ax[2].plot(x_fit, pp1_Bz(x_fit, out.params), '-', color=colors[index % len(colors)],label = label_p1)
                    print(out.fit_report())
                except Exception as e:
                    print(f'Fit failed for seq {seqs[index]}: {e}')
                    print(traceback.format_exc())
                    continue

                
    
        ax[2].set_xlabel(x_var_name, fontsize = 15)
        #ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        #ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax[2].legend(loc='upper left', bbox_to_anchor=(0., -0.3))

        ax[0].tick_params(direction = 'in', which = 'both')
        ax[1].tick_params(direction = 'in', which = 'both')
        ax[2].tick_params(direction = 'in', which = 'both')

        title = Path(__file__).name
        title += '\n' + 'seqs: ' + str(seqs)
        title += '\n' + 'x_var: ' + x_var_name
        title += '\n'
        title += 'BCompX: {:.3f}   '.format(BCompX_unique_val) if BCompX_unique_val is not None else ''
        title += 'BCompY: {:.3f}   '.format(BCompY_unique_val) if BCompY_unique_val is not None else ''
        title += 'BGradY: {:.3f}   '.format(BGradY_unique_val) if BGradY_unique_val is not None else ''
        fig.suptitle(title)
        
        fig.suptitle(title)
    except Exception as e:
        print(traceback.format_exc())
